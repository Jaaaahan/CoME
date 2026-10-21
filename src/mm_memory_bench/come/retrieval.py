from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..methods.backends import MultiModalEmbedder
from ..methods.media import question_text, uniformly_sample_video
from ..methods.vector_index import FaissFlatIPIndex, VectorIndex
from .store import CropCache, SourceCatalog, atomic_json
from .types import Candidate, CoMEConfig, PROTOCOL, digest


def deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    groups: dict[str, Candidate] = {}
    for node in candidates:
        key = node.fact_key or node.node_id
        previous = groups.get(key)
        if previous is None:
            groups[key] = node
        else:
            if previous.node_id == node.node_id and previous.version != node.version:
                groups[key] = node
                continue
            groups[key] = replace(previous,
                source_refs=tuple(sorted(set(previous.source_refs) | set(node.source_refs))),
                requires=previous.requires | node.requires)
    return list(groups.values())


class HybridRetriever:
    """Three existing FAISS text/image routes merged by reciprocal rank."""

    def __init__(self, embedder: MultiModalEmbedder, catalog: SourceCatalog,
                 crops: CropCache, config: CoMEConfig, *, root: Path | None = None,
                 index_factory: Callable[[int], VectorIndex] = FaissFlatIPIndex) -> None:
        self.embedder, self.catalog, self.crops, self.config = embedder, catalog, crops, config
        self.root, self.index_factory = root, index_factory
        self.nodes: dict[str, Candidate] = {}
        self.routes: dict[str, tuple[VectorIndex, list[str]]] = {}
        self.signatures: dict[str, str] = {}
        self.last_scores: dict[str, dict[str, float]] = {}

    def rebuild(self, observations: Iterable[Candidate]) -> None:
        import json

        previous_nodes = self.nodes
        self.nodes = {n.node_id: n for n in deduplicate([*self.catalog.nodes.values(), *observations])}
        for kind in ("observation", "text", "image"):
            nodes = sorted((n for n in self.nodes.values() if n.kind == kind), key=lambda n: n.node_id)
            signature = digest([PROTOCOL, getattr(self.embedder, "model_identity", type(self.embedder).__name__),
                                self.embedder.dimension, [asdict(n) for n in nodes], self.config.image_max_edge])
            if self.signatures.get(kind) == signature:
                continue
            index = self.index_factory(self.embedder.dimension)
            metadata_path = self.root / f"{kind}.json" if self.root else None
            index_path = self.root / f"{kind}-{signature}.faiss" if self.root else None
            if metadata_path and metadata_path.exists() and index_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("signature") == signature:
                    index.load(index_path)
                    ids = metadata["ids"]
                    if index.size != len(ids) or any(nid not in self.nodes for nid in ids):
                        raise ValueError("retrieval index/source snapshot mismatch")
                    self.routes[kind] = index, ids
                    self.signatures[kind] = signature
                    continue
            units, ids = [], []
            encode_nodes = nodes
            if kind in self.routes:
                previous_index, previous_ids = self.routes[kind]
                # An incremental rebuild may add observations. Reuse vectors for
                # unchanged nodes; replacements/removals need a fresh index.
                if all(nid in self.nodes and previous_nodes[nid] == self.nodes[nid]
                       for nid in previous_ids):
                    index, ids = previous_index, list(previous_ids)
                    encoded_ids = set(previous_ids)
                    encode_nodes = [node for node in nodes if node.node_id not in encoded_ids]
            for node in encode_nodes:
                if kind == "image":
                    asset = self.catalog.assets[node.asset_id]
                    if asset["type"] == "video":
                        images = uniformly_sample_video(asset["path"], asset["video_frames"],
                                                        max_edge=self.config.image_max_edge)
                    else:
                        images = [str(self.crops.image(node.asset_id))]
                    units.extend({"text": "", "image": image} for image in images)
                    ids.extend([node.node_id] * len(images))
                else:
                    units.append({"text": node.text})
                    ids.append(node.node_id)
            for start in range(0, len(units), self.config.retrieval_batch_size):
                batch = units[start:start + self.config.retrieval_batch_size]
                vectors = self.embedder.encode_units(batch)
                if len(vectors) != len(batch):
                    raise ValueError("embedder returned incomplete batch")
                index.add(vectors)
            if index_path:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = index_path.with_suffix(".tmp")
                index.save(temporary)
                os.replace(temporary, index_path)
                atomic_json(metadata_path, {"signature": signature, "ids": ids})
            self.routes[kind] = index, ids
            self.signatures[kind] = signature

    def retrieve(self, question: dict, *, expansion: int = 1) -> list[Candidate]:
        text = question_text(question)
        query_units = [{"text": text}]
        for part in question.get("prompt", []):
            if part.get("type") == "image" and part.get("path"):
                query_units.append({"text": text, "image": part["path"]})
            elif part.get("type") == "video" and part.get("path"):
                query_units.extend({"text": text, "image": frame} for frame in
                    uniformly_sample_video(part["path"], 8, max_edge=self.config.image_max_edge))
        batches = []
        for start in range(0, len(query_units), self.config.retrieval_batch_size):
            batch = query_units[start:start + self.config.retrieval_batch_size]
            vectors = self.embedder.encode_units(batch)
            if len(vectors) != len(batch):
                raise ValueError("embedder returned incomplete query batch")
            batches.append(vectors)
        query_vectors = np.concatenate(batches, axis=0)
        ranked: dict[str, list[str]] = {}
        fused: dict[str, float] = {}
        self.last_scores = {}
        for kind, (index, ids) in self.routes.items():
            if not ids:
                ranked[kind] = []
                continue
            # Overfetch frame rows before source-level deduplication.
            scores, rows = index.search(query_vectors, min(len(ids), self.config.source_top_k * expansion * 8))
            best: dict[str, float] = {}
            for values, positions in zip(scores, rows):
                for score, row in zip(values, positions):
                    if int(row) >= 0:
                        nid = ids[int(row)]
                        best[nid] = max(best.get(nid, -float("inf")), float(score))
            ranked[kind] = sorted(best, key=lambda nid: (-best[nid], nid))[:self.config.source_top_k * expansion]
            for rank, nid in enumerate(ranked[kind], 1):
                fused[nid] = fused.get(nid, 0.0) + 1 / (60 + rank)
                self.last_scores.setdefault(nid, {})[kind] = best[nid]
        limit = self.config.candidate_limit * expansion
        # Reserve a third for each nonempty route, then fill unused capacity.
        selected: list[str] = []
        for kind in ("observation", "text", "image"):
            selected.extend(ranked.get(kind, [])[:limit // 3])
        selected = list(dict.fromkeys(selected))
        for nid in sorted(fused, key=lambda nid: (-fused[nid], nid)):
            if len(selected) >= limit:
                break
            if nid not in selected:
                selected.append(nid)
        return [self.nodes[nid] for nid in selected]
