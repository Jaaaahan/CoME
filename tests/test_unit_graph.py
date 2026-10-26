"""Real FAISS regression checks for the original graph rules with unit nodes."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from mm_memory_bench.methods.vector_index import FaissFlatIPIndex
from mm_memory_bench.retrieval_units import OBSERVATION_REPRESENTATION, unit_identity, unit_metadata
from scripts.build_graphmemix_relation_store import offline_unit_graph
from scripts.graphmemix_core import build_candidate_graph, candidate_pool, load_candidate_graph


def archive(tmp_path, *, entries=None, max_k=1, explicit=None):
    entries = entries or [
        ("a", None, [0., 1., 0.]),
        ("a", "red", [1., 0., 0.]),
        ("a", "other", [0., 0., 1.]),
        ("b", None, [0., 0., 1.]),
        ("b", "red", [.99, .1, 0.]),
        ("c", None, [0., .99, .1]),
    ]
    units = [{"memory_id": mid, **({"observation_id": oid, "observation_version": 1,
              "representation": OBSERVATION_REPRESENTATION, "image": "source.png"}
             if oid else {})} for mid, oid, _ in entries]
    index = FaissFlatIPIndex(3)
    index.add(np.asarray([v for _, _, v in entries], dtype=np.float32))
    checkpoint = tmp_path / "checkpoint.faiss"
    index.save(checkpoint)
    contract = {"encoding_protocol": "fixture"}
    state = {"context_id": "ctx", "embedding_units": units, "config": {"observations": contract}}
    descriptor = offline_unit_graph(checkpoint, state, explicit=explicit or {},
                                    max_k=max_k, explicit_similarity=.99)
    for unit in units:
        unit["checkpoint_index_sha256"] = descriptor["checkpoint_index_sha256"]
    path = tmp_path / "relations.jsonl"
    path.write_text(json.dumps(descriptor) + "\n")
    return path, contract, descriptor, units


def prior(*units, visible=None):
    return {"context_id": "ctx", "retrieval_ids": list(dict.fromkeys(u["memory_id"] for u in units)),
            "checkpoint_index_sha256": units[0]["checkpoint_index_sha256"],
            "unit_hits": [{**unit_metadata(unit), "score": 1. - i * .01} for i, unit in enumerate(units)],
            **({"visible_memory_ids": visible} if visible is not None else {})}


def expand(graph, request, *, source_top_l=1, candidate_limit=2, visible=None):
    return build_candidate_graph(request, graph, source_top_l=source_top_l,
                                  candidate_limit=candidate_limit, visible_memory_ids=visible)


def test_offline_knn_builds_unit_edges_and_online_only_traverses_them(tmp_path, monkeypatch):
    import faiss

    searches = []
    original = faiss.IndexFlatIP.search

    def search(index, vectors, k, **kwargs):
        searches.append(len(vectors))
        return original(index, vectors, k, **kwargs)

    monkeypatch.setattr(faiss.IndexFlatIP, "search", search)
    path, contract, descriptor, units = archive(tmp_path)
    assert searches == [6]  # Same offline all-node kNN as the original graph.
    assert any(e.get("semantic_similarity") is not None for e in descriptor["edges"])
    monkeypatch.setattr(faiss.IndexFlatIP, "search", lambda *a, **k: pytest.fail("online kNN"))
    monkeypatch.setattr(faiss, "read_index", lambda *a, **k: pytest.fail("online index loading"))
    graph = load_candidate_graph(path, "full", 1, observation_contract=contract)
    result = expand(graph, prior(units[1]))
    assert result["candidate_ids"] == [unit_identity("a", "red"), unit_identity("b", "red")]
    assert result["unit_graph"]["candidate_unit_ids"] == [unit_identity("a", "red"), unit_identity("b", "red")]
    trace = result["unit_graph"]
    assert trace["seed_unit_ids"] == [unit_identity("a", "red")]
    assert {tuple(edge[k] for k in ("source", "target")) for edge in trace["edges"]
            if edge["relation"] == "belongs_to"} == {
                (unit_identity("a", "red"), unit_identity("a")),
                (unit_identity("b", "red"), unit_identity("b"))}
    assert expand(graph, prior(units[0]))["candidate_ids"] == [unit_identity("a"), unit_identity("c")]


def test_top_l_and_candidate_limit_use_original_algorithm_without_memory_quotas(tmp_path):
    path, contract, _, units = archive(tmp_path, max_k=3)
    graph = load_candidate_graph(path, "full", 3, observation_contract=contract)
    request = prior(units[1], units[2])  # Two observations of the SAME memory.
    before = deepcopy(request)
    result = expand(graph, request, source_top_l=2, candidate_limit=4)
    roots = [unit_identity("a", "red"), unit_identity("a", "other")]
    expected = candidate_pool({"retrieval_ids": roots}, graph.adjacency, source_top_l=2, candidate_limit=4)
    assert result["unit_graph"]["seed_unit_ids"] == roots
    assert result["unit_graph"]["candidate_unit_ids"] == expected
    assert len(expected) == 4
    limited = expand(graph, request, source_top_l=2, candidate_limit=2)
    assert limited["unit_graph"]["candidate_unit_ids"] == roots
    assert limited["candidate_ids"] == roots  # Same-parent observations remain independent evidence.
    assert request == before


def test_mutual_knn_and_rank_filter_are_preserved(tmp_path):
    entries = [("a", None, [0., 0., 1.]), ("a", "red", [1., 0., 0.]),
               ("b", None, [0., 1., 0.]), ("b", "red", [.98, .2, 0.]),
               ("c", None, [.97, .24, 0.])]
    path, contract, descriptor, units = archive(tmp_path, entries=entries, max_k=2)
    a, b = unit_identity("a", "red"), unit_identity("b", "red")
    edge = next(e for e in descriptor["edges"] if e["left"] == a and e["right"] == b)
    assert (edge["semantic_rank_left"], edge["semantic_rank_right"]) == (1, 2)
    strict = load_candidate_graph(path, "full", 1, observation_contract=contract)
    loose = load_candidate_graph(path, "full", 2, observation_contract=contract)
    assert b not in {other for other, _, _ in strict.adjacency[a]}
    assert b in {other for other, _, _ in loose.adjacency[a]}
    assert expand(strict, prior(units[1]), candidate_limit=3)["candidate_ids"] == [a, unit_identity("a")]


def test_visibility_filters_stored_neighbors_without_recomputing_or_refilling(tmp_path):
    path, contract, _, units = archive(tmp_path)
    graph = load_candidate_graph(path, "full", 1, observation_contract=contract)
    result = expand(graph, prior(units[1], visible=["a", "c"]), candidate_limit=4, visible={"a", "b"})
    assert result["candidate_ids"] == [unit_identity("a", "red"), unit_identity("a")]
    assert {n["memory_id"] for n in result["unit_graph"]["nodes"]} == {"a"}


def test_explicit_edges_and_one_hop_rule_are_preserved(tmp_path):
    explicit = {("a", "c"): {"left": "a", "right": "c", "explicit_relations": ["same_round"]}}
    path, contract, _, units = archive(tmp_path, explicit=explicit)
    graph = load_candidate_graph(path, "explicit", 1, observation_contract=contract)
    assert expand(graph, prior(units[0]), candidate_limit=4)["candidate_ids"] == [
        unit_identity("a"), unit_identity("a", "other"), unit_identity("a", "red"), unit_identity("c")]
    # The observation can reach its parent, but not the parent's neighbor in one hop.
    assert expand(graph, prior(units[1]), candidate_limit=4)["candidate_ids"] == [unit_identity("a", "red"), unit_identity("a")]
    graph = load_candidate_graph(path, "none", 1, observation_contract=contract)
    assert expand(graph, prior(units[1]), candidate_limit=4)["unit_graph"]["candidate_unit_ids"] == [unit_identity("a", "red")]


def test_stale_graphs_contracts_and_lost_hit_identity_are_rejected(tmp_path):
    path, contract, descriptor, units = archive(tmp_path)
    with pytest.raises(ValueError, match="contract"):
        load_candidate_graph(path, "full", 1, observation_contract={"encoding_protocol": "old"})
    graph = load_candidate_graph(path, "full", 1, observation_contract=contract)
    bad = prior(units[1]); bad["checkpoint_index_sha256"] = "different"
    with pytest.raises(ValueError, match="different checkpoint"):
        expand(graph, bad)
    bad = prior(units[1]); bad.pop("unit_hits")
    with pytest.raises(ValueError, match="unit identities"):
        expand(graph, bad)
    bad = prior(units[1]); bad["context_id"] = "another-context"
    with pytest.raises(ValueError, match="context"):
        expand(graph, bad)
    bad = prior(units[1]); bad["unit_hits"][0]["memory_id"] = "b"
    with pytest.raises(ValueError, match="metadata"):
        expand(graph, bad)
    descriptor["record_type"] = "unit_index"  # Rejected online-search artifact from the previous revision.
    path.write_text(json.dumps(descriptor) + "\n")
    with pytest.raises(ValueError, match="missing offline"):
        load_candidate_graph(path, "full", 1, observation_contract=contract)
