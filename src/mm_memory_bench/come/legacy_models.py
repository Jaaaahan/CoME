"""Historical CoME query experiment; never imported by the release pipeline."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, replace
from typing import Any, Callable

from ..evidence_verifier import ALL_ROLES, POSITIVE_ROLES, parse_json_object, validate_candidate_rows
from ..methods.backends import AnswerModel, estimate_multimodal_input_tokens
from ..methods.media import openai_content_from_parts, question_text
from .packet import PacketBuilder
from .models import ObservationExtractor, MODEL_PROTOCOL, ResponseValidationError, VerificationInputTooLarge, WorkBudgetExceeded
from .store import ObservationStore, SourceCatalog
from .types import (Candidate, Claim, CoMEConfig, Edge, Fact, Gap, Observation,
                    QueryState, Verification, digest)


def object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def fixed_rows_schema(field: str, rows: list[dict]) -> dict:
    # Positional schemas force every supplied ID exactly once. A generic JSON
    # object constraint cannot prevent a model from inventing candidate IDs.
    return object_schema({field: {"type": "array", "prefixItems": rows,
        "items": False, "minItems": len(rows), "maxItems": len(rows)}})


class CoMEModels(ObservationExtractor):
    """Bounded multimodal extraction, direct verification, and GraphMemix ECV."""

    def __init__(self, model: AnswerModel, catalog: SourceCatalog, packet: PacketBuilder,
                 config: CoMEConfig) -> None:
        self.model, self.catalog, self.packet, self.config = model, catalog, packet, config
        self.ingest_usage = {"calls": 0, "input_tokens": 0, "completion_tokens": 0}
        self.last_observation_report: dict = {}

    def call(self, task: str, instruction: str, nodes: list[Candidate], payload: dict,
             validate: Callable[[dict], Any], state: QueryState | None = None,
             *, question: dict | None = None, response_schema: dict | None = None,
             repair_schema: dict | None = None) -> Any:
        query_content = openai_content_from_parts([
            part for part in (question or {}).get("prompt", []) if part.get("type") in {"image", "video"}])
        messages = [{"role": "system", "content": f"CoME task={task}. {instruction} Return a JSON object only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
                        *query_content,
                        *self.packet.node_content(nodes)]}]
        last_error = None
        invalid_responses = []
        for attempt in range(3):
            estimate = estimate_multimodal_input_tokens(messages)
            # Reserve the configured completion allowance before submitting work.
            output_limit = getattr(getattr(self.model, "config", None), "max_output_tokens", 4096)
            counter = getattr(self.model, "count_input_tokens", None)
            if counter:
                estimate = counter(messages)
            if estimate + output_limit > getattr(getattr(self.model, "config", None), "max_model_len", 32768):
                raise VerificationInputTooLarge("verification input exceeds model capacity")
            if state and (state.exhausted(self.config) or
                          state.work_tokens + estimate + output_limit > self.config.max_work_tokens):
                raise WorkBudgetExceeded("query verification/inspection budget exhausted")
            if state:
                state.work_calls += 1
            schema = response_schema if response_schema is not None else repair_schema if attempt else None
            response_format = ({"type": "json_schema", "json_schema": {
                "name": task, "strict": True, "schema": schema}}
                if schema is not None else {"type": "json_object"})
            raw = self.model.complete(messages, response_format=response_format)
            input_tokens = getattr(self.model, "last_input_tokens", None) or estimate
            output_tokens = getattr(self.model, "last_completion_tokens", None) or (len(raw) + 2) // 3
            if state:
                state.work_tokens += input_tokens + output_tokens
                state.completion_tokens += output_tokens
            else:
                self.ingest_usage["calls"] += 1
                self.ingest_usage["input_tokens"] += input_tokens
                self.ingest_usage["completion_tokens"] += output_tokens
            try:
                return validate(parse_json_object(raw))
            except (ValueError, TypeError, KeyError) as exc:
                last_error = exc
                invalid_responses.append(raw)
                diagnostic = self.packet.crops.root.parent / "model_validation_errors.jsonl"
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                with diagnostic.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"task": task, "attempt": attempt + 1,
                        "node_ids": [node.node_id for node in nodes],
                        "question_id": (question or {}).get("question_id"),
                        "error": str(exc), "response": raw}, ensure_ascii=False) + "\n")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"Invalid response: {exc}. Return a fresh complete object matching the schema."})
        raise ResponseValidationError(f"{task} failed response validation three times: {last_error}",
                                      responses=tuple(invalid_responses))

    def verify(self, question: dict, candidates: list[Candidate], state: QueryState) -> Verification:
        result = Verification(requires={n.node_id: n.requires for n in candidates})
        by_id = {n.node_id: n for n in candidates}
        query = question_text(question)
        query_context = {"question": query, "instruction": question.get("instruction", ""),
                         "tools": question.get("tools", [])}
        query_key = digest(question)
        unscored = []
        for node in candidates:
            key = digest(["node", query_key, node.node_id, node.version])
            if key in state.verification_cache:
                result.prizes[node.node_id] = state.verification_cache[key]
            else:
                unscored.append(node)
        batches = [unscored[start:start + self.config.verifier_batch_size]
                   for start in range(0, len(unscored), self.config.verifier_batch_size)]
        while batches:
            batch = batches.pop(0)
            expected = {n.node_id for n in batch}

            def score_rows(row: dict) -> dict[str, float]:
                values = row.get("nodes")
                if not isinstance(values, list) or len(values) != len(expected):
                    raise ValueError("node scores must cover the batch once")
                scores = {str(v["id"]): float(v["direct_support"]) for v in values}
                if set(scores) != expected or any(not math.isfinite(v) or not 0 <= v <= 5 for v in scores.values()):
                    raise ValueError("invalid node IDs or support scores")
                return {nid: v / 5 for nid, v in scores.items()}

            try:
                scores = self.call("node_verify",
                    "Score direct support for the question, 0 to 5. Inspect visual evidence. "
                    "Do not credit identity/ownership inferred without original text. "
                    "Output {nodes:[{id,direct_support}]} for every ID in node_ids, in that order, "
                    "including irrelevant nodes with score zero. Copy IDs exactly.",
                    batch, {**query_context, "node_ids": sorted(expected)}, score_rows, state=state, question=question,
                    response_schema=fixed_rows_schema("nodes", [object_schema({
                        "id": {"const": nid}, "direct_support": {"type": "number", "minimum": 0, "maximum": 5},
                    }) for nid in sorted(expected)]))
            except VerificationInputTooLarge:
                if len(batch) == 1:
                    raise
                middle = len(batch) // 2
                batches[:0] = [batch[:middle], batch[middle:]]
                continue
            result.prizes.update(scores)
            for node in batch:
                state.verification_cache[digest(["node", query_key, node.node_id, node.version])] = scores[node.node_id]
        anchors = sorted(candidates, key=lambda n: (-result.prizes[n.node_id], n.node_id))[:self.config.anchor_count]
        pending = []
        for node in candidates:
            options = sorted((a for a in anchors if a.node_id != node.node_id), key=lambda a: (
                -(a.node_id in node.source_refs or node.node_id in a.source_refs),
                -(bool(node.asset_id) and a.asset_id == node.asset_id),
                -result.prizes[a.node_id], a.node_id))[:self.config.anchors_per_candidate]
            if not options:
                continue
            key = digest(["ecv", query_key, node.node_id, node.version,
                          [(a.node_id, a.version) for a in options]])
            if key in state.verification_cache:
                pending.append((node, options, key, state.verification_cache[key]))
            else:
                pending.append((node, options, key, None))
        groups = [pending[start:start + self.config.verifier_batch_size]
                  for start in range(0, len(pending), self.config.verifier_batch_size)]
        while groups:
            group = groups.pop(0)
            missing = [item for item in group if item[3] is None]
            answers = {node.node_id: cached for node, _, _, cached in group if cached is not None}
            if missing:
                eligible = {n.node_id: {a.node_id for a in options} for n, options, _, _ in missing}
                shown = {n.node_id: n for n, _, _, _ in missing}
                shown.update({a.node_id: a for _, options, _, _ in missing for a in options})
                # Include original textual supports, even when not among scoring anchors.
                for n in list(shown.values()):
                    shown.update({ref: by_id[ref] for ref in n.requires if ref in by_id})

                def ecv_rows(row: dict) -> list[dict]:
                    rows = validate_candidate_rows(row.get("candidates"),
                        expected_aliases=set(eligible), eligible_anchors=eligible)
                    for r in rows:
                        support = r.get("support_ids", [])
                        if not isinstance(support, list) or not set(support) <= set(shown):
                            raise ValueError("ECV supports must reference displayed evidence")
                        claim = r.get("claim")
                        if claim is not None:
                            if not isinstance(claim, dict) or not str(claim.get("description", "")).strip():
                                raise ValueError("invalid query claim")
                            if type(claim.get("requires_text")) is not bool:
                                raise ValueError("claim requires_text must be boolean")
                            fact = Fact(**claim["fact"])
                            if (claim["requires_text"] or fact.basis == "textual") and not any(shown[nid].kind == "text" for nid in support):
                                raise ValueError("identity claim lacks original text")
                    return rows

                claim_schema = object_schema({
                    "description": {"type": "string", "minLength": 1},
                    "requires_text": {"type": "boolean"},
                    "fact": object_schema({"subject": {"type": "string", "minLength": 1},
                        "predicate": {"type": "string", "minLength": 1},
                        "value": {"type": "string", "minLength": 1},
                        "basis": {"enum": ["visual", "textual"]}}),
                })
                ecv_schema = fixed_rows_schema("candidates", [object_schema({
                    "id": {"const": nid}, "direct_support": {"const": 0},
                    "best_anchor_id": {"enum": [None, *sorted(eligible[nid])]},
                    "incremental_support": {"type": "number", "minimum": 0, "maximum": 5},
                    "role": {"enum": sorted(ALL_ROLES)},
                    "support_ids": {"type": "array", "items": {"enum": sorted(shown)}, "maxItems": len(shown)},
                    "claim": {"anyOf": [{"type": "null"}, claim_schema]},
                }) for nid in sorted(eligible)])
                try:
                    rows = self.call("ecv",
                        "Verify candidate incremental support relative to ONLY the eligible anchors. "
                        "Roles: new_fact, clarification, corroboration, redundant, conflict, irrelevant. "
                        "Same image, crop_of, proximity, or similar appearance alone gives NO connection benefit. "
                        "Credit only a verified identity binding, reference resolution, attribute completion, "
                        "or other factual evidence relation. Return all necessary evidence in support_ids. "
                        "Identity and ownership require original textual support. If jointly established, "
                        "include one reusable query interpretation in claim, otherwise null. Output "
                        "{candidates:[{id,direct_support:0,best_anchor_id:ID|null,incremental_support:0..5,"
                        "role,support_ids:[ID],claim:{description,requires_text:bool,"
                        "fact:{subject,predicate,value,basis:'visual'|'textual'}}|null}]}. "
                        "Use the same entity IDs and fact tuple for repeated interpretations; "
                        "the fact tuple identifies the conclusion independently of its support set. "
                        "Return exactly one row for EACH ID in candidate_ids, in that order. "
                        "Copy those IDs verbatim, including image and text IDs. "
                        "An anchor supplies context and needs no row unless also listed in candidate_ids.",
                        list(shown.values()), {**query_context, "candidate_ids": sorted(eligible),
                            "eligible_anchors": {nid: sorted(values) for nid, values in eligible.items()}},
                        ecv_rows, state=state, question=question, response_schema=ecv_schema)
                except VerificationInputTooLarge:
                    if len(group) == 1:
                        raise
                    middle = len(group) // 2
                    groups[:0] = [group[:middle], group[middle:]]
                    continue
                answers.update({str(r["id"]): r for r in rows})
                for node, _, key, _ in missing:
                    state.verification_cache[key] = answers[node.node_id]
            for node, _, _, _ in group:
                row = answers[node.node_id]
                anchor = row.get("best_anchor_id")
                if row.get("role") not in POSITIVE_ROLES or not anchor:
                    continue
                support = frozenset({node.node_id, str(anchor), *row.get("support_ids", [])})
                if not support <= by_id.keys():
                    continue  # Cached rows cannot outlive their supporting candidates.
                incremental = float(row["incremental_support"]) / 5
                result.edges.append(Edge(node.node_id, str(anchor), 0.99 * incremental, row["role"], support))
                if row.get("claim"):
                    description = row["claim"]["description"]
                    fact = Fact(**row["claim"]["fact"])
                    key = digest([fact.subject, fact.predicate, fact.value])
                    claim_id = "claim:" + digest([key, sorted(support)])[:24]
                    result.claims.append(Claim(claim_id, description, support,
                                               node.node_id, incremental, key))
                    result.requires[claim_id] = support
        return result

    def gaps(self, question: dict, nodes: list[Candidate], claims: list[Claim], state: QueryState) -> list[Gap]:
        def parse(row: dict) -> list[Gap]:
            values = row.get("gaps")
            if not isinstance(values, list) or len(values) > 6:
                raise ValueError("gaps must be a list of at most six items")
            result = []
            for value in values:
                kind = value["kind"]
                if kind not in {"target_missing", "attribute_missing", "ambiguous_binding"}:
                    raise ValueError("unknown evidence gap")
                ids = tuple(value.get("node_ids", []))
                if not set(ids) <= {n.node_id for n in nodes}:
                    raise ValueError("gap references an unselected node")
                if not str(value.get("description", "")).strip():
                    raise ValueError("gap needs a description")
                result.append(Gap(kind, value["description"], ids))
            return result
        return self.call("check_gaps",
            "Check whether the selected evidence actually supports all requested facts. "
            "Report only target_missing, attribute_missing, ambiguous_binding. Conflicting identity "
            "or attributes count as ambiguous_binding. Do not invent a new gap merely to gather more "
            "information. Output {gaps:[{kind,description,node_ids:[selected IDs]}]}; use [] when sufficient.",
            nodes, {"question": question_text(question), "instruction": question.get("instruction", ""),
                "tools": question.get("tools", []), "claims": [c.description for c in claims]}, parse, state=state, question=question)
