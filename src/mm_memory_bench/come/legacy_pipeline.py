from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ..methods.backends import AnswerModel, MultiModalEmbedder, estimate_multimodal_input_tokens
from ..methods.base import GenerationConfig
from ..methods.vector_index import FaissFlatIPIndex, VectorIndex
from .legacy_models import MODEL_PROTOCOL, CoMEModels, ResponseValidationError, WorkBudgetExceeded
from .packet import PacketBuilder
from .retrieval import HybridRetriever, deduplicate
from .selection import Selection, closure, select_evidence
from .store import CropCache, ObservationStore, SourceCatalog, atomic_json
from .types import (Candidate, CoMEConfig, EvidenceDelta, Gap, PROTOCOL, QUERY_PROTOCOL, QueryState,
                    Verification, digest)


class GraphMemixPipeline:
    """Historical query experiment, retained only for regression comparison."""

    def __init__(self, catalog: SourceCatalog, root: Path, embedder: MultiModalEmbedder,
                 model: AnswerModel, reader: AnswerModel, generation: GenerationConfig,
                 config: CoMEConfig | None = None, *,
                 index_factory: Callable[[int], VectorIndex] = FaissFlatIPIndex) -> None:
        self.catalog, self.root, self.embedder = catalog, root, embedder
        self.model, self.reader, self.generation = model, reader, generation
        self.config = config or CoMEConfig()
        self.index_factory = index_factory
        self.crops = CropCache(root / "crops", catalog, max_edge=self.config.image_max_edge)
        self.packet = PacketBuilder(catalog, self.crops, self.config)
        self.models = CoMEModels(model, catalog, self.packet, self.config)
        self.base_store = ObservationStore(root / "observations.jsonl", catalog)
        self.ingest_contract = digest([PROTOCOL, MODEL_PROTOCOL, catalog.signature, asdict(self.config),
            asdict(getattr(model, "config", generation)),
            getattr(embedder, "model_identity", type(embedder).__name__)])
        manifest = root / "ingestion.json"
        self.ingestion = {"contract": self.ingest_contract, "completed_images": [], "usage": {}}
        if manifest.exists():
            self.ingestion = json.loads(manifest.read_text())
            if self.ingestion.get("contract") != self.ingest_contract:
                raise ValueError("incompatible CoME source/model/config snapshot; use a new artifact root")
        elif self.base_store.latest:
            raise ValueError("observations exist without their ingestion contract")

    def ingest(self) -> dict:
        # Persist the contract before any model call; partial ingestion resumes per image.
        manifest = self.root / "ingestion.json"
        atomic_json(manifest, self.ingestion)
        completed = set(self.ingestion["completed_images"])
        previous_usage = dict(self.ingestion.get("usage", {}))
        outcomes = self.ingestion.setdefault("image_outcomes", {})
        total_images = sum(node.kind == "image" for node in self.catalog.nodes.values())
        for node in self.catalog.nodes.values():
            if node.kind != "image" or node.node_id in completed:
                continue
            for observation in self.models.observe(node, self.base_store):
                self.base_store.put(observation)
            outcomes[node.node_id] = dict(self.models.last_observation_report)
            completed.add(node.node_id)
            self.ingestion["completed_images"] = sorted(completed)
            self.ingestion["usage"] = {key: previous_usage.get(key, 0) + value
                                       for key, value in self.models.ingest_usage.items()}
            atomic_json(manifest, self.ingestion)
            if len(completed) % 10 == 0 or outcomes[node.node_id].get("errors"):
                print(json.dumps({"event": "ingestion_progress", "completed_images": len(completed),
                    "total_images": total_images, "observations": len(self.base_store.latest),
                    "last_image": node.node_id, "last_outcome": outcomes[node.node_id]}), flush=True)
        return {"images": len(completed), "observations": len(self.base_store.latest),
                "usage": self.ingestion["usage"], "quality": self.ingestion_quality()}

    def ingestion_quality(self) -> dict:
        outcomes = self.ingestion.get("image_outcomes", {})
        return {"completed_images": len(self.ingestion["completed_images"]),
            "images_with_outcome_details": len(outcomes),
            "images_without_observations": sum(not row.get("accepted", 0) for row in outcomes.values()),
            "images_with_validation_errors": sum(bool(row.get("errors")) for row in outcomes.values()),
            "rejected_observations": sum(row.get("rejected", 0) for row in outcomes.values())}

    def _query_components(self, visible_memory_ids: set[str]):
        catalog = self.catalog.subset(visible_memory_ids)
        initial = []
        for observation in self.base_store.latest.values():
            try:
                catalog.validate(observation)
            except (ValueError, StopIteration):
                continue
            initial.append(observation)
        # Query inspection reads original sources; only ingestion creates observations.
        store = ObservationStore(None, catalog, initial)
        packet = PacketBuilder(catalog, self.crops, self.config)
        models = CoMEModels(self.model, catalog, packet, self.config)
        retriever = HybridRetriever(self.embedder, catalog, self.crops, self.config,
            root=self.root / "indices" / catalog.signature, index_factory=self.index_factory)
        retriever.rebuild(store.candidates())
        return catalog, store, packet, models, retriever

    def _fit(self, question: dict, candidates: dict[str, Candidate], verification: Verification,
             packet: PacketBuilder, gaps: list[Gap]) -> tuple[Selection, list[dict], str]:
        limit = self.generation.max_model_len - self.generation.max_output_tokens
        cache: dict[str, int] = {}

        def cost(ids, claims):
            key = digest([sorted(ids), [c.id for c in claims]])
            if key not in cache:
                cache[key] = packet.estimate(question, [candidates[nid] for nid in sorted(ids)], claims, gaps)
            return cache[key]

        # Search with local costs, then preflight the exact payload. A measured
        # underestimate tightens the local budget and repacks dependency groups.
        search_budget = limit
        for _ in range(len(candidates) + 2):
            selected = select_evidence(candidates, verification, cost, search_budget, self.config)
            messages = packet.messages(question, [candidates[nid] for nid in selected.node_ids], selected.claims, gaps)
            count = getattr(self.reader, "count_input_tokens", None)
            actual = count(messages, packet.tools(question)) if count else estimate_multimodal_input_tokens(messages, packet.tools(question))
            source = getattr(self.reader, "last_preflight_source", None) or "local_estimate"
            if actual <= limit:
                selected.tokens = actual
                selected.diagnostics["token_count_source"] = source
                return selected, messages, source
            if not selected.node_ids:
                raise ValueError("reader question and unresolved gaps exceed context capacity")
            search_budget = min(search_budget - 1, int(selected.tokens * limit / actual) - 1)
        raise ValueError("unable to fit reader evidence within context")

    def _inspect(self, question: dict, gaps: list[Gap], state: QueryState,
                 catalog: SourceCatalog, retriever: HybridRetriever, candidates: dict[str, Candidate],
                 round_id: int) -> EvidenceDelta:
        delta = EvidenceDelta()
        # Expand all three routes; ambiguity and missing targets need text as well as pixels.
        expanded = retriever.retrieve(question, expansion=round_id + 2)
        available = {**retriever.nodes, **candidates}
        # Resolve a gap's observation/crop to its full original, even if that
        # image did not rank in retrieval. All referenced targets precede fallbacks.
        targets = [available[nid] for gap in gaps for nid in gap.node_ids if nid in available]
        targets.extend(expanded)
        image_limit = min(self.config.inspection_images_per_round, self.config.candidate_limit)
        for node in targets:
            if not node.asset_id or catalog.assets[node.asset_id]["type"] != "image":
                continue
            image = catalog.image_node(node.asset_id, node.memory_id)
            if image.node_id in candidates or image.node_id in state.attempts:
                continue
            delta.candidates.append(image)
            delta.image_ids.append(image.node_id)
            state.attempts.add(image.node_id)
            if len(delta.image_ids) >= image_limit:
                break
        # Existing observations and source text may be recalled, but inspection
        # creates no observations or crop coordinates and never updates indexes.
        delta.candidates.extend(n for n in expanded if n.node_id not in candidates and
            (n.kind != "image" or catalog.assets[n.asset_id]["type"] != "image"))
        return delta

    def answer(self, question: dict, *, visible_memory_ids: set[str] | None = None) -> dict:
        started = time.perf_counter()
        visible = set(self.catalog.memories) if visible_memory_ids is None else visible_memory_ids
        catalog, store, packet, models, retriever = self._query_components(visible)
        initial_versions = {oid: o.version for oid, o in store.latest.items()}
        state = QueryState()
        candidates = {n.node_id: n for n in retriever.retrieve(question)}
        verification = Verification()
        gaps: list[Gap] = []
        delta = EvidenceDelta()
        for round_id in range(self.config.max_inspection_rounds + 1):
            candidates = {n.node_id: n for n in deduplicate(candidates.values())}
            if round_id and len(candidates) > self.config.candidate_limit:
                # Ensure requested originals reach verification before applying
                # the cap to previous selections and other recalled candidates.
                priority = list(dict.fromkeys([
                    *delta.image_ids,
                    *selected.node_ids,
                    *(n.node_id for n in delta.candidates),
                    *candidates]))
                keep = priority[:self.config.candidate_limit]
                candidates = {nid: candidates[nid] for nid in keep if nid in candidates}
            # Required original text bypasses retrieval ranking, but stays in scope.
            required = closure(candidates, {nid: n.requires for nid, n in retriever.nodes.items()})
            for nid in required - candidates.keys():
                if nid in retriever.nodes:
                    candidates[nid] = retriever.nodes[nid]
            try:
                verification = models.verify(question, list(candidates.values()), state)
            except (WorkBudgetExceeded, ResponseValidationError) as exc:
                state.stop_reason = "invalid_model_output" if isinstance(exc, ResponseValidationError) else "work_budget"
                # Keep only actually verified direct scores; a missing ECV call is not support.
                query_key = digest(question)
                verification.prizes.update({nid: state.verification_cache.get(
                    digest(["node", query_key, nid, node.version]), 0.0) for nid, node in candidates.items()})
            selected, messages, count_source = self._fit(question, candidates, verification, packet, [])
            if state.stop_reason in {"work_budget", "invalid_model_output"}:
                gaps = gaps or [Gap("target_missing", "Some evidence could not be verified; facts depending on it remain unknown.")]
            else:
                try:
                    gaps = models.gaps(question, [candidates[nid] for nid in selected.node_ids], selected.claims, state)
                except (WorkBudgetExceeded, ResponseValidationError) as exc:
                    state.stop_reason = "invalid_model_output" if isinstance(exc, ResponseValidationError) else "work_budget"
                    gaps = [Gap("target_missing", "The available evidence could not be checked for completeness.")]
            state.rounds.append({"round": round_id, "candidates": len(candidates),
                "selected": list(selected.node_ids), "gaps": [asdict(g) for g in gaps],
                "inspected_images": delta.image_ids})
            if not gaps:
                state.stop_reason = "complete"
                break
            if state.stop_reason in {"work_budget", "invalid_model_output"} or state.exhausted(self.config):
                state.stop_reason = state.stop_reason or "work_budget"
                break
            if round_id == self.config.max_inspection_rounds:
                state.stop_reason = "inspection_limit"
                break
            delta = self._inspect(question, gaps, state, catalog, retriever, candidates, round_id)
            if not delta.has_progress:
                state.stop_reason = state.stop_reason or "no_progress"
                break
            candidates.update({n.node_id: n for n in delta.candidates})

        # Unresolved descriptions themselves consume input tokens. If adding them
        # changes selection, retain a conservative gap for the newly removed facts.
        before = set(selected.node_ids)
        selected, messages, count_source = self._fit(question, candidates, verification, packet, gaps)
        if before - set(selected.node_ids):
            gaps = [*gaps, Gap("attribute_missing", "Some evidence was removed to fit the final reader budget; facts depending on it remain unknown.")]
            selected, messages, count_source = self._fit(question, candidates, verification, packet, gaps)
        prediction = self.reader.complete(messages, tools=packet.tools(question))
        query_contract = digest([PROTOCOL, QUERY_PROTOCOL, self.ingest_contract, question, sorted(visible),
                                initial_versions, asdict(self.generation)])
        query_root = self.root / "queries" / digest([question.get("question_id"), query_contract])
        # Keep the artifact shape for existing report readers; the delta is always empty.
        atomic_json(query_root / "observations.json", {"protocol": PROTOCOL, "observations": []})
        diagnostics = {"protocol": PROTOCOL, "query_protocol": QUERY_PROTOCOL, "contract": query_contract,
            "selected_nodes": [asdict(candidates[nid]) for nid in selected.node_ids],
            "claims": [{**asdict(c), "supports": sorted(c.supports)} for c in selected.claims],
            "requires": {nid: sorted(candidates[nid].requires | verification.requires.get(nid, frozenset()))
                         for nid in selected.node_ids},
            "unresolved": [asdict(g) for g in gaps], "rounds": state.rounds,
            "stop_reason": state.stop_reason, "selection": selected.diagnostics,
            "reader_input_tokens": selected.tokens, "token_count_source": count_source,
            "work_calls": state.work_calls, "work_tokens": state.work_tokens,
            "reader_completion_tokens": getattr(self.reader, "last_completion_tokens", None),
            "work_completion_tokens": state.completion_tokens,
            "new_observations": 0, "latency_seconds": time.perf_counter() - started}
        diagnostics["ingestion_quality"] = self.ingestion_quality()
        for node in diagnostics["selected_nodes"]:
            node["requires"] = sorted(node["requires"])
        diagnostics["requires"].update({c.id: sorted(c.supports) for c in selected.claims})
        atomic_json(query_root / "trace.json", diagnostics)
        return {"question_id": question.get("question_id"), "context_id": question.get("context_id"),
            "prediction": prediction,
            "retrieved_memory_ids": list(dict.fromkeys(candidates[nid].memory_id for nid in selected.node_ids)),
            "metadata": diagnostics}
