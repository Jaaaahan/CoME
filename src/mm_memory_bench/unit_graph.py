"""Memory/observation identities over the original offline relation graph.

Vector reconstruction is offline only. Online code adapts retrieved unit IDs
to the existing candidate-pool algorithm and preserves independent unit evidence.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .methods.vector_index import normalized_float32
from .retrieval_units import (OBSERVATION_REPRESENTATION, UNIT_GRAPH_PROTOCOL,
                              unit_identity, unit_metadata)


def reconstruct_unit_vectors(index_path: Path, state: Mapping[str, Any]) -> tuple[list[dict], np.ndarray]:
    """One canonical vector per memory and one fused vector per observation."""
    import faiss

    source = faiss.read_index(str(index_path))
    units = state["embedding_units"]
    if source.ntotal != len(units):
        raise ValueError("unit index/checkpoint row mismatch")
    groups: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row_id, unit in enumerate(units):
        representation = unit.get("representation")
        if representation and representation != OBSERVATION_REPRESENTATION:
            if unit.get("observation_id"):
                raise ValueError("legacy observation encoding; rebuild checkpoint")
            continue
        node = unit_metadata(unit)
        uid = node["unit_id"]
        if node["kind"] == "observation" and (uid in groups or not unit.get("image")):
            raise ValueError("observation must have one multimodal retrieval unit")
        groups.setdefault(uid, []).append(row_id)
        metadata[uid] = node
    nodes = [metadata[uid] for uid in sorted(groups)]
    if not nodes:
        raise ValueError("cannot build an empty unit graph")
    for node in nodes:
        if unit_identity(node["memory_id"]) not in metadata:
            raise ValueError("observation has no parent memory unit")
    vectors = []
    for node in nodes:
        rows = np.asarray([source.reconstruct(i) for i in groups[node["unit_id"]]], dtype=np.float32)
        vectors.append(normalized_float32(rows).mean(axis=0))
    return nodes, normalized_float32(np.asarray(vectors, dtype=np.float32))


class StoredUnitGraph:
    """Read-only unit metadata and precomputed adjacency; no online vector index."""

    def __init__(self, descriptors: Sequence[Mapping[str, Any]],
                 adjacency: Mapping[str, Sequence[tuple[str, float, str]]], *,
                 observation_contract: Mapping[str, Any], semantic_k: int) -> None:
        self.adjacency = adjacency
        self.semantic_k = semantic_k
        self.contexts: dict[str, dict[str, Any]] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        for descriptor in descriptors:
            if (descriptor.get("protocol") != UNIT_GRAPH_PROTOCOL or
                    descriptor.get("observations") != observation_contract):
                raise ValueError("unit graph observation contract mismatch; rebuild relations")
            cid = str(descriptor["context_id"])
            if cid in self.contexts:
                raise ValueError("duplicate unit graph context")
            nodes = {node["unit_id"]: node for node in descriptor["units"]}
            if len(nodes) != len(descriptor["units"]) or self.nodes.keys() & nodes.keys():
                raise ValueError("duplicate unit graph identity")
            for uid, node in nodes.items():
                if (uid != unit_identity(node["memory_id"], node.get("observation_id")) or
                        unit_identity(node["memory_id"]) not in nodes):
                    raise ValueError("invalid unit graph identity or parent")
                if any(other not in nodes for other, _, _ in adjacency.get(uid, [])):
                    raise ValueError("unit graph edge leaves its context")
            self.nodes.update(nodes)
            self.contexts[cid] = {"nodes": nodes,
                                  "checkpoint_index_sha256": descriptor["checkpoint_index_sha256"]}
        if not self.contexts:
            raise ValueError("missing offline unit graphs; rebuild relations")

    def retrieval_prior(self, prior: Mapping[str, Any], visible_memory_ids: set[str] | None = None
                        ) -> tuple[dict[str, Any], set[str]]:
        """Expose actual hit nodes to the unchanged top-L candidate algorithm."""
        cid = str(prior.get("context_id", ""))
        if cid not in self.contexts:
            raise ValueError("source prior has no matching unit graph context")
        context = self.contexts[cid]
        if prior.get("checkpoint_index_sha256") != context["checkpoint_index_sha256"]:
            raise ValueError("source prior and unit graph use different checkpoint indexes")
        nodes = context["nodes"]
        visible = {node["memory_id"] for node in nodes.values()}
        if visible_memory_ids is not None:
            visible &= set(visible_memory_ids)
        if "visible_memory_ids" in prior:
            visible &= set(map(str, prior["visible_memory_ids"]))
        visible_units = {uid for uid, node in nodes.items() if node["memory_id"] in visible}
        if "unit_hits" not in prior:
            raise ValueError("source prior lacks retrieved unit identities; regenerate priors")
        scores: dict[str, float] = {}
        for hit in prior["unit_hits"]:
            uid = hit["unit_id"]
            if uid not in nodes or any(hit.get(key) != value for key, value in nodes[uid].items()):
                raise ValueError("retrieved unit does not match graph metadata")
            score = float(hit["score"])
            if not math.isfinite(score):
                raise ValueError("invalid retrieved unit score")
            if uid in visible_units:
                scores[uid] = max(scores.get(uid, -math.inf), score)
        ranked = sorted(scores, key=lambda uid: (-scores[uid], uid))
        return {"retrieval_ids": ranked, "retrieval_scores": [scores[uid] for uid in ranked]}, visible_units

    def project(self, selected: Sequence[str], seeds: Sequence[str]) -> dict[str, Any]:
        """Keep selected units independent; parent nodes are provenance only."""
        candidate_ids = list(selected)
        selected_set = set(selected)
        graph_nodes = selected_set | {unit_identity(self.nodes[uid]["memory_id"]) for uid in selected}
        edges = []
        for uid in sorted(selected_set):
            for other, score, relation in self.adjacency.get(uid, []):
                if other in selected_set and uid < other and relation != "belongs_to":
                    edges.append({"source": uid, "target": other, "relation": relation, "similarity": score})
        for uid in sorted(graph_nodes):
            node = self.nodes[uid]
            if node["kind"] == "observation":
                edges.append({"source": uid, "target": unit_identity(node["memory_id"]),
                              "relation": "belongs_to"})
        return {"candidate_ids": candidate_ids, "unit_graph": {
            "protocol": UNIT_GRAPH_PROTOCOL, "seed_unit_ids": list(seeds),
            "candidate_unit_ids": list(selected),
            "nodes": [self.nodes[uid] for uid in sorted(graph_nodes)], "edges": edges,
            "semantic_k": self.semantic_k, "expansion_hops": 1}}
