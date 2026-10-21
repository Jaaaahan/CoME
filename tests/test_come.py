from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from mm_memory_bench.come.legacy_models import CoMEModels
from mm_memory_bench.come.evaluate import summarize
from mm_memory_bench.come.packet import PacketBuilder
from mm_memory_bench.come.legacy_pipeline import GraphMemixPipeline
from mm_memory_bench.come.pipeline import ObservationPreprocessor
from mm_memory_bench.come.retrieval import HybridRetriever, deduplicate
from mm_memory_bench.memory_scope import visible_ids
from mm_memory_bench.come.selection import closure, select_evidence
from mm_memory_bench.come.store import CropCache, ObservationStore, SourceCatalog
from mm_memory_bench.come.types import (Candidate, Claim, CoMEConfig, Edge, Fact, Gap,
                                      Observation, QUERY_PROTOCOL, QueryState, Verification)
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.backends import estimate_multimodal_input_tokens


class MemoryIndex:
    """Test implementation of the real VectorIndex protocol, including persistence."""
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def size(self):
        return len(self.vectors)

    def add(self, vectors):
        vectors = np.asarray(vectors)
        self.vectors = np.concatenate([self.vectors, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)])

    def search(self, queries, top_k):
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        scores = queries @ self.vectors.T
        rows = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        return np.take_along_axis(scores, rows, axis=1), rows

    def save(self, path):
        with open(path, "wb") as handle:
            np.save(handle, self.vectors)

    def load(self, path):
        with open(path, "rb") as handle:
            self.vectors = np.load(handle)


class Embedder:
    dimension = 3
    model_identity = "test-embedder-v1"

    def __init__(self):
        self.units = []

    def encode_units(self, units):
        self.units.extend(units)
        return np.array([[1, 1 if "cup" in u.get("text", "") else 0,
                          1 if u.get("image") else 0] for u in units], dtype=np.float32)


@pytest.fixture
def catalog(tmp_path):
    image = tmp_path / "photo.png"
    Image.new("RGB", (100, 80), "red").save(image)
    return SourceCatalog([
        {"memory_id": "m1", "source_id": "photo-1", "sequence": 0,
         "content": [{"type": "image", "asset_id": "a1", "path": str(image)},
                     {"type": "text", "text": "The person on the left is me. A cup."}]},
        {"memory_id": "m2", "sequence": 1,
         "content": [{"type": "text", "text": "A later, unrelated conversation."}]},
    ])


def observation(catalog, **kwargs):
    image = catalog.image_node("a1", "m1")
    return Observation(**{**dict(id="o1", memory_id="m1", asset_id="a1",
        regions=((0, 0, 20, 20),), description="A red cup", source_refs=(image.node_id,),
        version=1, fact=Fact("a1:cup1", "color", "red"),
        asset_version=catalog.assets["a1"]["sha256"]), **kwargs})


def test_observation_offsets_versions_and_textual_sources(tmp_path, catalog):
    text = next(n for n in catalog.nodes.values() if n.kind == "text" and n.memory_id == "m1")
    source = catalog.memories["m1"]["content"][text.location["part_index"]]["text"]
    assert source[text.location["start"]:text.location["end"]] == text.text
    store = ObservationStore(tmp_path / "observations.jsonl", catalog)
    first = observation(catalog)
    assert store.put(first)
    assert not store.put(first)
    with pytest.raises(ValueError, match="version"):
        store.put(replace(first, description="different"))
    changed = replace(first, version=2, description="A bright red cup")
    store.put(changed)
    assert ObservationStore(store.path, catalog).latest["o1"] == changed
    identity = replace(first, id="identity", fact=Fact("a1:person1", "identity", "user", "textual"))
    with pytest.raises(ValueError, match="textual"):
        catalog.validate(identity)
    identity = replace(identity, source_refs=(*identity.source_refs, text.node_id))
    assert catalog.candidate(identity).requires == {text.node_id}
    with pytest.raises(ValueError, match="bounds"):
        catalog.validate(replace(first, regions=((0, 0, 101, 80),)))


def test_crop_union_cache_and_source_replacement(tmp_path, catalog):
    cache = CropCache(tmp_path / "crops", catalog)
    assert not cache.root.exists()
    regions = ((1, 2, 10, 20), (60, 30, 90, 70))
    path = cache.image("a1", regions)
    assert Image.open(path).size == (89, 68)
    assert cache.image("a1", regions) == path
    Image.new("RGB", (100, 80), "blue").save(catalog.assets["a1"]["path"])
    with pytest.raises(ValueError, match="changed"):
        cache.image("a1", regions)


@pytest.mark.parametrize("size", [(1000, 2), (2, 1000), (1000, 1), (1, 1000), (1000, 5), (62, 39)])
def test_thin_crop_model_canvas_preserves_pixels_and_source_region(tmp_path, size):
    source = tmp_path / "source.png"
    image = Image.new("RGB", size, (23, 45, 67))
    image.putpixel((0, 0), (255, 0, 0))
    image.save(source)
    catalog = SourceCatalog([{"memory_id": "m", "content": [
        {"type": "image", "asset_id": "a", "path": str(source)}]}])
    cache = CropCache(tmp_path / "crops", catalog)
    region = ((0, 0, *size),)
    original = cache.image("a", region)
    prepared, info = cache.model_image("a", region)
    with Image.open(original) as original_image, Image.open(prepared) as prepared_image:
        assert original_image.size == size
        assert max(prepared_image.size) / min(prepared_image.size) <= 200
        if max(size) / min(size) > 200:
            assert prepared != original
            left, top, right, bottom = info["padding_ltrb"]
            restored = prepared_image.crop((left, top, prepared_image.width - right,
                                           prepared_image.height - bottom))
            assert restored.tobytes() == original_image.tobytes()
            assert prepared_image.getpixel((0, 0)) == (127, 127, 127)
        else:
            assert prepared == original and not info
    assert cache.model_image("a", region) == (prepared, info)
    node = catalog.candidate(Observation("thin", "m", "a", region, "A thin visible region",
        (catalog.image_node("a", "m").node_id,), 1, Fact("a:region1", "color", "blue"),
        catalog.assets["a"]["sha256"]))
    content = PacketBuilder(catalog, cache, CoMEConfig()).node_content([node, node])
    assert sum(p["type"] == "image_url" for p in content) == 1
    notes = [p for p in content if p["type"] == "text" and "visual_input_padding" in p["text"]]
    assert len(notes) == bool(info)
    assert node.regions == region


def test_duplicate_facts_preserve_values_and_supports(catalog):
    original = catalog.candidate(observation(catalog))
    duplicate = replace(original, node_id="duplicate", requires=frozenset({"text1"}))
    conflict = catalog.candidate(observation(catalog, id="blue", fact=Fact("a1:cup1", "color", "blue")))
    result = deduplicate([original, duplicate, conflict])
    assert len(result) == 2
    assert result[0].requires == {"text1"}
    assert result[1].node_id == "blue"


def test_three_routes_and_index_resume(tmp_path, catalog):
    embedder = Embedder()
    cache = CropCache(tmp_path / "crops", catalog)
    config = CoMEConfig(candidate_limit=3, proposal_k=3)
    retriever = HybridRetriever(embedder, catalog, cache, config,
                                root=tmp_path / "indices", index_factory=MemoryIndex)
    obs = catalog.candidate(observation(catalog))
    retriever.rebuild([obs])
    assert all(not u.get("image") for u in embedder.units if u.get("text") == obs.text)
    query = {"prompt": [{"type": "text", "text": "cup"}]}
    assert {n.kind for n in retriever.retrieve(query)} == {"observation", "text", "image"}
    new_embedder = Embedder()
    resumed = HybridRetriever(new_embedder, catalog, cache, config,
                              root=tmp_path / "indices", index_factory=MemoryIndex)
    resumed.rebuild([obs])
    assert not new_embedder.units
    assert [n.node_id for n in resumed.retrieve(query)] == [n.node_id for n in retriever.retrieve(query)]


def test_query_media_obeys_embedding_batch_limit(tmp_path, catalog, monkeypatch):
    image = catalog.assets["a1"]["path"]
    monkeypatch.setattr("mm_memory_bench.come.retrieval.uniformly_sample_video",
                        lambda *args, **kwargs: [image] * 8)
    query = {"prompt": [{"type": "text", "text": "cup"},
                        {"type": "image", "path": image},
                        {"type": "video", "path": "test.mp4"}]}
    cache = CropCache(tmp_path / "crops", catalog)
    observations = [catalog.candidate(observation(catalog))]
    baseline = HybridRetriever(Embedder(), catalog, cache, CoMEConfig(), index_factory=MemoryIndex)
    baseline.rebuild(observations)

    class LimitedEmbedder(Embedder):
        def encode_units(self, units):
            assert len(units) <= 1
            return super().encode_units(units)

    limited = HybridRetriever(LimitedEmbedder(), catalog, cache,
                              CoMEConfig(retrieval_batch_size=1), index_factory=MemoryIndex)
    limited.rebuild(observations)
    limited.embedder.units.clear()
    assert limited.retrieve(query) == baseline.retrieve(query)
    assert len(limited.embedder.units) == 10


def test_incremental_retrieval_update_preserves_base_checkpoint(tmp_path, catalog):
    pytest.importorskip("faiss")
    from mm_memory_bench.methods.vector_index import FaissFlatIPIndex
    root = tmp_path / "indices"
    embedder = Embedder()
    cache = CropCache(tmp_path / "crops", catalog)
    config = CoMEConfig()
    base = catalog.candidate(observation(catalog))
    retriever = HybridRetriever(embedder, catalog, cache, config, root=root, index_factory=FaissFlatIPIndex)
    retriever.rebuild([base])
    checkpoint = {p.name: p.read_bytes() for p in root.iterdir()}
    added = catalog.candidate(observation(catalog, id="o2", description="A cup handle",
                              fact=Fact("a1:cup1", "has", "handle")))
    embedder.units.clear()
    retriever.root = None  # Exercise an in-memory index update without persisting it.
    retriever.rebuild([base, added])
    assert embedder.units == [{"text": added.text}]
    assert retriever.routes["observation"][0].size == 2
    assert {p.name: p.read_bytes() for p in root.iterdir()} == checkpoint
    # A new isolated question loads the ingestion index with no embedding work.
    next_embedder = Embedder()
    next_query = HybridRetriever(next_embedder, catalog, cache, config, root=root, index_factory=FaissFlatIPIndex)
    next_query.rebuild([base])
    assert next_embedder.units == []
    assert next_query.routes["observation"][0].size == 1
    assert added.node_id not in next_query.nodes
    # A changed or removed node must not leave its old vector in the index.
    changed = replace(base, text="A blue cup", version="updated")
    embedder.units.clear()
    retriever.rebuild([changed])
    assert embedder.units == [{"text": changed.text}]
    assert retriever.routes["observation"][0].size == 1


def node(nid, *, requires=()):
    return Candidate(nid, "text", "m", nid, requires=frozenset(requires), fact_key=nid)


def test_budget_withdraws_identity_claim_preserves_visual_fact():
    nodes = {n.node_id: n for n in [node("person"), node("holding"), node("identity_text")]}
    claim = Claim("q1", "User holds cup", frozenset(nodes), "holding", .9, "user-holds")
    verification = Verification(prizes={"person": .6, "holding": .9, "identity_text": .1}, claims=[claim])
    costs = {"person": 1, "holding": 1, "identity_text": 20}
    result = select_evidence(nodes, verification,
        lambda selected, claims: sum(costs[nid] for nid in selected), 2,
        CoMEConfig(proposal_k=1))
    assert set(result.node_ids) == {"person", "holding"}
    assert not result.claims


def test_shared_support_is_retained_and_charged_once():
    nodes = {n.node_id: n for n in [node("a", requires=["bridge"]), node("b", requires=["bridge"]), node("bridge")]}
    verification = Verification(prizes={"a": .9, "b": .8, "bridge": .0})
    result = select_evidence(nodes, verification, lambda selected, claims: len(selected), 3, CoMEConfig())
    assert set(result.node_ids) == set(nodes)
    assert result.tokens == 3
    assert result.diagnostics["dependency_closed"]


def test_dependency_cycle_missing_support_and_empty_pool():
    nodes = {n.node_id: n for n in [node("a", requires=["b"]), node("b", requires=["a"]), node("bad", requires=["absent"])]}
    v = Verification(prizes={"a": .9, "b": .8, "bad": 1})
    result = select_evidence(nodes, v, lambda ids, claims: len(ids), 1, CoMEConfig())
    assert not result.node_ids
    assert result.diagnostics["invalid_dependencies"] == ["bad"]
    assert closure(["a"], {"a": frozenset({"b"}), "b": frozenset({"a"})}) == {"a", "b"}
    assert not select_evidence({}, Verification(), lambda ids, claims: 10, 10, CoMEConfig()).node_ids


def test_same_image_edges_do_not_create_reward():
    nodes = {nid: node(nid) for nid in ("a", "b")}
    v = Verification(prizes={"a": .9, "b": .1}, edges=[Edge("a", "b", .99, "crop_of")])
    result = select_evidence(nodes, v, lambda ids, claims: len(ids), 2, CoMEConfig())
    assert result.node_ids == ["a"]


def test_packet_counts_shared_crop_once_but_different_crops_separately(tmp_path, catalog):
    config = CoMEConfig()
    packet = PacketBuilder(catalog, CropCache(tmp_path / "crops", catalog), config)
    a = catalog.candidate(observation(catalog))
    b = replace(a, node_id="o2", text="A cup handle")
    c = replace(a, node_id="o3", regions=((40, 40, 60, 60),))
    content = packet.node_content([a, b, c, catalog.image_node("a1", "m1")])
    assert sum(p["type"] == "image_url" for p in content) == 3


class ScriptedModel:
    def __init__(self, *, gaps=0, empty_observations=False):
        self.config = GenerationConfig(max_output_tokens=1000)
        self.remaining_gaps = gaps
        self.empty_observations = empty_observations
        self.tasks = []
        self.inspections = 0
        self.reader_messages = []

    def count_input_tokens(self, messages, tools=None):
        return estimate_multimodal_input_tokens(messages, tools)

    def complete(self, messages, *, tools=None, response_format=None):
        if messages[0]["role"] != "system":
            self.reader_messages.append(messages)
            return "The observed cup is red."
        task = messages[0]["content"].split("task=", 1)[1].split(".", 1)[0]
        self.tasks.append(task)
        payload = json.loads(messages[1]["content"][0]["text"])
        if task == "observe":
            if payload.get("focus"):
                self.inspections += 1
            value = "red" if not payload.get("focus") else f"detail-{self.inspections}"
            return json.dumps({"coordinate_system": "normalized_1000", "observations": [] if self.empty_observations else [{
                "regions": [[0, 0, 200, 250]], "description": f"A cup with {value}", "source_refs": [],
                "fact": {"subject": "a1:cup1", "predicate": "color" if value == "red" else "detail",
                         "value": value, "basis": "visual"}}]})
        if task == "ground_observations":
            return json.dumps({"checks": [{"id": nid, "visible_supported": True,
                "requires_text": False, "text_supported": False} for nid in payload["proposed_ids"]]})
        if task == "node_verify":
            return json.dumps({"nodes": [{"id": nid, "direct_support": 4} for nid in payload["node_ids"]]})
        if task == "ecv":
            return json.dumps({"candidates": [{"id": nid, "direct_support": 0, "best_anchor_id": None,
                "incremental_support": 0, "role": "irrelevant", "support_ids": [], "claim": None}
                for nid in payload.get("candidate_ids", payload["eligible_anchors"])]})
        if task == "check_gaps":
            if self.remaining_gaps:
                self.remaining_gaps -= 1
                return json.dumps({"gaps": [{"kind": "attribute_missing", "description": f"Need detail {self.remaining_gaps}", "node_ids": []}]})
            return '{"gaps": []}'
        raise AssertionError(task)


def pipeline(tmp_path, catalog, model, **config):
    return GraphMemixPipeline(catalog, tmp_path / "state", Embedder(), model, model,
        GenerationConfig(), CoMEConfig(**config), index_factory=MemoryIndex)


def test_observation_local_ids_are_namespaced_and_relations_reused(tmp_path, catalog):
    class LocalEntities(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                row = json.loads(raw)
                cup = row["observations"][0]
                cup["fact"]["subject"] = "cup1"
                row["observations"].append({**cup, "description": "A person holds the cup",
                    "fact": {"subject": "person1", "predicate": "holds", "value": "cup1", "basis": "visual"}})
                return json.dumps(row)
            return raw
    second = {"memory_id": "m3", "content": [{"type": "image", "asset_id": "a2",
        "path": catalog.assets["a1"]["path"]}]}
    catalog = SourceCatalog([*catalog.memories.values(), second])
    method = pipeline(tmp_path, catalog, LocalEntities())
    method.ingest()
    facts = {(o.fact.subject, o.fact.predicate, o.fact.value) for o in method.base_store.latest.values()}
    assert facts == {("a1:cup1", "color", "red"), ("a1:person1", "holds", "a1:cup1"),
                     ("a2:cup1", "color", "red"), ("a2:person1", "holds", "a2:cup1")}


@pytest.mark.parametrize("repair", [False, True])
def test_foreign_entity_namespace_is_rejected_with_actionable_retry(tmp_path, catalog, repair):
    class ForeignEntity(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." not in messages[0]["content"]:
                return raw
            row = json.loads(raw)
            retried = len(messages) > 2
            if retried:
                assert messages[-2]["role"] == "assistant"
                assert "foreign:cup1" in messages[-2]["content"]
                assert "'a1:'" in messages[-1]["content"]
            row["observations"][0]["fact"]["subject"] = "cup1" if repair and retried else "foreign:cup1"
            return json.dumps(row)
    method = pipeline(tmp_path, catalog, ForeignEntity())
    if repair:
        method.ingest()
        assert next(iter(method.base_store.latest.values())).fact.subject == "a1:cup1"
    else:
        method.ingest()
        assert not method.base_store.latest
        assert method.models.last_observation_report["status"] == "rejected_all"
    diagnostics = [json.loads(line) for line in (method.root / "model_validation_errors.jsonl").read_text().splitlines()]
    assert len(diagnostics) == (1 if repair else 3)
    assert all(row["task"] == "observe" and "foreign:cup1" in row["response"] for row in diagnostics)


def test_observation_retry_constrains_fact_value_after_array_response(tmp_path, catalog):
    class ArrayValueUntilConstrained(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.observe_formats = []

        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." not in messages[0]["content"]:
                return raw
            response_format = kwargs["response_format"]
            self.observe_formats.append(response_format["type"])
            if response_format["type"] == "json_schema":
                fact = response_format["json_schema"]["schema"]["properties"]["observations"]["items"]["properties"]["fact"]
                assert fact["properties"]["value"] == {"type": "string", "minLength": 1}
                return raw
            row = json.loads(raw)
            row["observations"][0]["fact"]["value"] = [0, 0, 200, 250]
            return json.dumps(row)

    model = ArrayValueUntilConstrained()
    service = ObservationPreprocessor(list(catalog.memories.values()), tmp_path / "offline", model)
    rows = service.ingest()
    assert model.observe_formats == ["json_object", "json_schema"]
    assert len(rows) == 2
    assert len(service.store.latest) == 1


@pytest.mark.parametrize("invalid_field,invalid_value", [
    ("fact.value", "[0, 0, 200, 250]"),
    ("fact.value", ""),
    ("description", [0, 0, 200, 250]),
    ("description", ""),
])
def test_observation_retries_nontext_fields_without_coercion(tmp_path, catalog, invalid_field, invalid_value):
    class NontextUntilConstrained(ScriptedModel):
        def __init__(self):
            super().__init__()
            self.observe_formats = []

        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." not in messages[0]["content"]:
                return raw
            response_format = kwargs["response_format"]
            self.observe_formats.append(response_format["type"])
            if response_format["type"] == "json_schema":
                properties = response_format["json_schema"]["schema"]["properties"]["observations"]["items"]["properties"]
                assert properties["description"] == {"type": "string", "minLength": 1}
                assert properties["fact"]["properties"]["value"] == properties["description"]
                return raw
            row = json.loads(raw)
            observation = row["observations"][0]
            if invalid_field == "fact.value":
                observation["fact"]["value"] = invalid_value
            else:
                observation["description"] = invalid_value
            return json.dumps(row)

    model = NontextUntilConstrained()
    service = ObservationPreprocessor(list(catalog.memories.values()), tmp_path / "offline", model)
    service.ingest()
    assert model.observe_formats == ["json_object", "json_schema"]
    assert len(service.store.latest) == 1


@pytest.mark.parametrize("count", [9, 10, 21])
def test_overlong_observation_response_preserves_bounded_valid_facts(tmp_path, catalog, count):
    class Overlong(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                row = json.loads(raw)
                good = row["observations"][0]
                row["observations"] = [{**good, "fact": {**good["fact"],
                    "subject": f"cup{i}"}} for i in range(count)]
                return json.dumps(row)
            return raw

    model = Overlong()
    service = ObservationPreprocessor(list(catalog.memories.values()), tmp_path / "offline", model)
    service.ingest()
    assert len(service.store.latest) == 8
    assert service.models.last_observation_report["status"] == "validated_partial"
    assert service.models.last_observation_report["errors"]
    assert model.tasks.count("ground_observations") == 1


@pytest.mark.parametrize("supported", [True, False])
def test_truncated_observation_response_only_recovers_complete_grounded_facts(tmp_path, catalog, supported):
    class Truncated(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                good = json.loads(raw)["observations"][0]
                return ('{"coordinate_system":"normalized_1000","observations":[' +
                        json.dumps(good) + ', {"fact":{"value":"' + '0' * 5000)
            if "task=ground_observations." in messages[0]["content"]:
                row = json.loads(raw)
                for check in row["checks"]:
                    check["visible_supported"] = supported
                return json.dumps(row)
            return raw

    model = Truncated()
    service = ObservationPreprocessor(list(catalog.memories.values()), tmp_path / "offline", model)
    if supported:
        service.ingest()
        assert len(service.store.latest) == 1
        assert next(iter(service.store.latest.values())).fact.value == "red"
        assert service.models.last_observation_report["status"] == "validated_partial"
    else:
        with pytest.raises(ValueError, match="incomplete for 1 images"):
            service.ingest()
        assert not service.store.latest
        assert not service.ingestion["completed_images"]
    assert model.tasks.count("ground_observations") == 1


def test_observation_repair_schema_uses_configured_count_limit(tmp_path, catalog):
    from mm_memory_bench.come.types import PreprocessConfig

    class CheckSchema(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                fmt = kwargs["response_format"]
                if fmt["type"] == "json_schema":
                    assert fmt["json_schema"]["schema"]["properties"]["observations"]["maxItems"] == 3
                    return raw
                row = json.loads(raw)
                row["observations"][0]["fact"]["value"] = []
                return json.dumps(row)
            return raw

    service = ObservationPreprocessor(list(catalog.memories.values()), tmp_path / "offline",
                                      CheckSchema(), PreprocessConfig(observations_per_image=3))
    service.ingest()
    assert len(service.store.latest) == 1


@pytest.mark.parametrize("repair", [True, False])
def test_grounding_cannot_claim_original_text_when_none_was_supplied(tmp_path, catalog, repair):
    class ConfusesPixelsWithText(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=ground_observations." not in messages[0]["content"]:
                return raw
            if repair and len(messages) > 2:
                assert "original text" in messages[-1]["content"]
                return raw
            row = json.loads(raw)
            for check in row["checks"]:
                check.update(visible_supported=True, requires_text=True, text_supported=True)
            return json.dumps(row)

    memory = {**catalog.memories["m1"], "content": [
        part for part in catalog.memories["m1"]["content"] if part["type"] == "image"]}
    model = ConfusesPixelsWithText()
    service = ObservationPreprocessor([memory], tmp_path / "offline", model)
    if repair:
        service.ingest()
        assert len(service.store.latest) == 1
        assert model.tasks.count("ground_observations") == 2
    else:
        with pytest.raises(ValueError, match="incomplete for 1 images"):
            service.ingest()
        assert not service.store.latest and not service.ingestion["completed_images"]
        assert service.models.last_observation_report["status"] == "grounding_failed"


def test_grounded_partial_observations_resume_and_complete(tmp_path, catalog):
    class PersistentEmptyValue(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." not in messages[0]["content"]:
                return raw
            row = json.loads(raw)
            invalid = dict(row["observations"][0])
            invalid["description"] = "The sky is blue."
            invalid["fact"] = {"subject": "sky1", "predicate": "is located at",
                               "value": "", "basis": "visual"}
            row["observations"].append(invalid)
            return json.dumps(row)

    root = tmp_path / "offline"
    memories = list(catalog.memories.values())
    service = ObservationPreprocessor(memories, root, PersistentEmptyValue())
    image = next(node for node in service.catalog.nodes.values() if node.kind == "image")
    accepted = service.models.observe(image, service.store)
    assert len(accepted) == 1
    assert service.models.last_observation_report["status"] == "validated_partial"
    service.manifest.write_text(json.dumps(service.ingestion))
    service.store.put(accepted[0])

    resumed = ObservationPreprocessor(memories, root, PersistentEmptyValue())
    resumed.ingest()
    assert len(resumed.store.latest) == 1
    assert image.node_id in resumed.ingestion["completed_images"]
    outcome = resumed.ingestion["image_outcomes"][image.node_id]
    assert outcome["status"] == "validated_partial"
    assert outcome["accepted"] == 1 and outcome["errors"]


@pytest.mark.parametrize("blank", ["", " "])
def test_empty_value_embedded_in_predicate_becomes_text(tmp_path, catalog, blank):
    class EmbeddedProperty(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." not in messages[0]["content"]:
                return raw
            row = json.loads(raw)
            row["observations"][0]["fact"].update(predicate="is_red", value=blank)
            return json.dumps(row)

    service = ObservationPreprocessor(list(catalog.memories.values()),
                                      tmp_path / "offline", EmbeddedProperty())
    service.ingest()
    fact = next(iter(service.store.latest.values())).fact
    assert (fact.predicate, fact.value) == ("is", "red")
    assert service.models.last_observation_report["status"] == "validated"


def test_failed_image_is_recorded_without_stopping_other_images(tmp_path, catalog):
    class OneImageFails(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." not in messages[0]["content"]:
                return raw
            row = json.loads(raw)
            asset = json.loads(messages[1]["content"][0]["text"])["asset_id"]
            row["observations"][0]["fact"]["subject"] = "cup1"
            if asset == "a1":
                row["observations"][0]["fact"]["value"] = ""
            return json.dumps(row)

    second = {"memory_id": "m3", "content": [{"type": "image", "asset_id": "a2",
        "path": catalog.assets["a1"]["path"]}]}
    service = ObservationPreprocessor([*catalog.memories.values(), second],
                                      tmp_path / "offline", OneImageFails())
    with pytest.raises(ValueError, match="incomplete for 1 images"):
        service.ingest()
    images = {node.asset_id: node.node_id for node in service.catalog.nodes.values()
              if node.kind == "image"}
    assert images["a1"] not in service.ingestion["completed_images"]
    assert images["a2"] in service.ingestion["completed_images"]
    assert service.ingestion["image_outcomes"][images["a1"]]["errors"]
    assert any(row.asset_id == "a2" for row in service.store.latest.values())


@pytest.mark.parametrize("size,box,expected", [
    ((750, 1000), [738, 530, 888, 912], (553, 530, 666, 912)),
    ((1479, 1218), [100, 200, 900, 800], (147, 243, 1332, 975)),
    ((3, 2), [1, 1, 2, 2], (0, 0, 1, 1)),
    ((875, 500), [0, 0, 1000, 1000], (0, 0, 875, 500)),
])
def test_normalized_boxes_convert_to_original_pixels(tmp_path, size, box, expected):
    path = tmp_path / "source.png"
    Image.new("RGB", size).save(path)
    catalog = SourceCatalog([{"memory_id": "m1", "content": [
        {"type": "image", "asset_id": "a1", "path": str(path)}]}])
    class Boxes(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                row = json.loads(raw)
                row["observations"][0]["regions"] = [box]
                return json.dumps(row)
            return raw
    method = pipeline(tmp_path, catalog, Boxes())
    method.ingest()
    assert next(iter(method.base_store.latest.values())).regions == (expected,)


@pytest.mark.parametrize("coordinate_system,box", [(None, [0, 0, 20, 20]),
    ("original_pixels", [0, 0, 20, 20]), ("normalized_1000", [0, 0, 1001, 1000])])
def test_ambiguous_or_invalid_extraction_coordinates_are_rejected(tmp_path, catalog, coordinate_system, box):
    class BadCoordinates(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                row = json.loads(raw)
                row["coordinate_system"] = coordinate_system
                row["observations"][0]["regions"] = [box]
                return json.dumps(row)
            return raw
    method = pipeline(tmp_path, catalog, BadCoordinates())
    method.ingest()
    assert not method.base_store.latest
    assert method.models.last_observation_report["status"] == "rejected_all"


@pytest.mark.parametrize("isolate_queries", [True, False])
def test_inspection_reads_originals_without_creating_observations(tmp_path, catalog, monkeypatch, isolate_queries):
    import base64
    import io

    second_image = tmp_path / "second.png"
    Image.new("RGB", (120, 90), "blue").save(second_image)
    catalog = SourceCatalog([*catalog.memories.values(), {"memory_id": "m3", "content": [
        {"type": "image", "asset_id": "a2", "path": str(second_image)}]}])

    class RecordingModel(ScriptedModel):
        def __init__(self):
            super().__init__(gaps=2)
            self.verified = []

        def complete(self, messages, **kwargs):
            if messages[0]["role"] == "system" and "task=node_verify." in messages[0]["content"]:
                self.verified.extend(json.loads(messages[1]["content"][0]["text"])["node_ids"])
            raw = super().complete(messages, **kwargs)
            if messages[0]["role"] == "system" and "task=observe." in messages[0]["content"]:
                row = json.loads(raw)
                row["observations"][0]["fact"]["subject"] = "cup1"
                return json.dumps(row)
            return raw

    model = RecordingModel()
    method = pipeline(tmp_path, catalog, model, candidate_limit=3, proposal_k=3,
                      inspection_images_per_round=1, isolate_queries=isolate_queries)
    method.ingest()
    initial = dict(method.base_store.latest)
    base_bytes = (method.root / "observations.jsonl").read_bytes()
    # Old query-learned records must not enter the frozen ingestion snapshot,
    # even when a caller retains the legacy isolate_queries=False setting.
    learned = ObservationStore(method.root / "learned.jsonl", catalog)
    learned.put(observation(catalog, id="legacy-query-observation",
                            fact=Fact("a1:cup1", "shape", "round")))
    learned_bytes = learned.path.read_bytes()
    method._query_components(set(catalog.memories))
    index_bytes = {str(p.relative_to(method.root)): p.read_bytes()
                   for p in (method.root / "indices").rglob("*") if p.is_file()}

    def retrieve(retriever, question, *, expansion=1):
        assert "legacy-query-observation" not in retriever.nodes
        # Start with crops and text. Original images appear only during expansion.
        initial_nodes = [n for n in retriever.nodes.values() if n.kind == "observation"][:1]
        initial_nodes.extend(n for n in retriever.nodes.values() if n.kind == "text")
        if expansion == 1:
            return initial_nodes
        # Put images last to exercise prioritization under the candidate cap.
        return [*initial_nodes, *(n for n in retriever.nodes.values() if n.kind == "image")]

    monkeypatch.setattr(HybridRetriever, "retrieve", retrieve)
    def unexpected_observe(*args, **kwargs):
        pytest.fail("Query inspection must not extract or ground new observations")
    monkeypatch.setattr(CoMEModels, "observe", unexpected_observe)
    result = method.answer({"question_id": "q1", "prompt": [{"type": "text", "text": "cup details"}]})
    metadata = result["metadata"]
    assert metadata["new_observations"] == 0
    assert metadata["query_protocol"] == QUERY_PROTOCOL
    assert len(metadata["rounds"]) == 3
    assert metadata["stop_reason"] == "complete"
    originals = [catalog.image_node(aid, mid).node_id for aid, mid in [("a1", "m1"), ("a2", "m3")]]
    assert [r["inspected_images"] for r in metadata["rounds"]] == [[], [originals[0]], [originals[1]]]
    assert set(originals) <= set(model.verified)
    assert set(originals) <= {n["node_id"] for n in metadata["selected_nodes"]}
    assert all(not n["regions"] for n in metadata["selected_nodes"] if n["kind"] == "image")
    image_sizes = []
    for part in model.reader_messages[-1][0]["content"]:
        if part["type"] == "image_url":
            with Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))) as im:
                image_sizes.append(im.size)
    assert {(100, 80), (120, 90)} <= set(image_sizes)
    assert method.base_store.latest == initial
    assert (method.root / "observations.jsonl").read_bytes() == base_bytes
    assert learned.path.read_bytes() == learned_bytes
    assert {str(p.relative_to(method.root)): p.read_bytes()
            for p in (method.root / "indices").rglob("*") if p.is_file()} == index_bytes
    assert model.tasks.count("observe") == 2  # Only the two ingestion calls.
    assert model.tasks.count("ground_observations") == 2
    for path in (method.root / "queries").glob("*/observations.json"):
        assert json.loads(path.read_text())["observations"] == []
    assert metadata["reader_input_tokens"] <= method.generation.max_model_len - method.generation.max_output_tokens
    assert metadata["selection"]["dependency_closed"]
    # A second question must start from the same observation snapshot.
    second = method.answer({"question_id": "q2", "prompt": [{"type": "text", "text": "cup"}]})
    assert not second["metadata"]["new_observations"]
    assert second["metadata"]["rounds"][0]["inspected_images"] == []


def test_invalid_observation_rows_are_rejected_without_losing_valid_rows(tmp_path, catalog):
    class Mixed(ScriptedModel):
        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            if "task=observe." in messages[0]["content"]:
                row = json.loads(raw)
                good = row["observations"][0]
                row["observations"] += [
                    {**good, "regions": [[10, 500, 1000, 500]]},
                    {**good, "description": "The person is the user", "fact": {
                        "subject": "person1", "predicate": "identity", "value": "user", "basis": "textual"}},
                    {**good, "source_refs": ["missing-source"]},
                ]
                return json.dumps(row)
            return raw
    model = Mixed()
    method = pipeline(tmp_path, catalog, model)
    summary = method.ingest()
    assert summary["observations"] == 1
    assert summary["quality"]["rejected_observations"] == 3
    assert model.tasks.count("observe") == 3
    assert model.tasks.count("ground_observations") == 1
    assert next(iter(method.base_store.latest.values())).fact.value == "red"
    resumed_model = Mixed()
    resumed = pipeline(tmp_path, catalog, resumed_model)
    assert resumed.ingest()["observations"] == 1
    assert resumed_model.tasks == []


@pytest.mark.parametrize("stage", ["observe", "ground_observations", "node_verify", "check_gaps"])
def test_invalid_model_output_preserves_originals_and_unresolved_state(tmp_path, catalog, stage):
    class Invalid(ScriptedModel):
        def complete(self, messages, **kwargs):
            if messages[0]["role"] == "system" and f"task={stage}." in messages[0]["content"]:
                return 'not valid JSON'
            return super().complete(messages, **kwargs)
    method = pipeline(tmp_path, catalog, Invalid())
    method.ingest()
    result = method.answer({"question_id": "q", "prompt": [{"type": "text", "text": "cup"}]})
    if stage in {"observe", "ground_observations"}:
        assert not method.base_store.latest
        assert any(n["kind"] == "image" for n in result["metadata"]["selected_nodes"])
        assert result["metadata"]["ingestion_quality"]["images_without_observations"] == 1
    else:
        assert result["metadata"]["stop_reason"] == "invalid_model_output"
        assert result["metadata"]["unresolved"]
        assert not result["metadata"]["claims"]


def test_transport_failure_still_stops_ingestion_without_marking_image_complete(tmp_path, catalog):
    class Unavailable(ScriptedModel):
        def complete(self, messages, **kwargs):
            raise OSError("model service unavailable")
    method = pipeline(tmp_path, catalog, Unavailable())
    with pytest.raises(OSError, match="unavailable"):
        method.ingest()
    assert not method.ingestion["completed_images"]


def test_generation_schema_requires_all_ids_and_eligible_ecv_anchors(tmp_path, catalog):
    jsonschema = pytest.importorskip("jsonschema")
    import copy

    class Constrained(ScriptedModel):
        checked = set()

        def complete(self, messages, **kwargs):
            raw = super().complete(messages, **kwargs)
            response_format = kwargs.get("response_format") or {}
            if response_format.get("type") != "json_schema":
                return raw
            spec = response_format["json_schema"]
            validator = jsonschema.Draft202012Validator(spec["schema"])
            data = json.loads(raw)
            validator.validate(data)
            field = next(iter(data))
            bad = copy.deepcopy(data)
            bad[field][0]["id"] = "invented_candidate_1"
            with pytest.raises(jsonschema.ValidationError):
                validator.validate(bad)
            bad = copy.deepcopy(data)
            bad[field].pop()
            with pytest.raises(jsonschema.ValidationError):
                validator.validate(bad)
            bad = copy.deepcopy(data)
            bad[field].append(bad[field][0])
            with pytest.raises(jsonschema.ValidationError):
                validator.validate(bad)
            if spec["name"] == "ecv":
                bad = copy.deepcopy(data)
                bad[field][0]["best_anchor_id"] = "unshown_anchor"
                with pytest.raises(jsonschema.ValidationError):
                    validator.validate(bad)
            self.checked.add(spec["name"])
            return raw

    model = Constrained()
    method = pipeline(tmp_path, catalog, model)
    method.ingest()
    method.answer({"question_id": "q1", "prompt": [{"type": "text", "text": "cup"}]})
    assert model.checked == {"ground_observations", "node_verify", "ecv"}


def test_original_image_fallback_and_visible_scope(tmp_path, catalog):
    model = ScriptedModel(empty_observations=True)
    method = pipeline(tmp_path, catalog, model)
    method.ingest()
    result = method.answer({"question_id": "q", "prompt": [{"type": "text", "text": "cup"}]}, visible_memory_ids={"m1"})
    assert result["retrieved_memory_ids"] == ["m1"]
    assert any(n["kind"] == "image" for n in result["metadata"]["selected_nodes"])
    assert any(p["type"] == "image_url" for p in model.reader_messages[-1][0]["content"])
    assert "later, unrelated" not in json.dumps(model.reader_messages[-1])


def test_exhausted_work_budget_keeps_unknowns(tmp_path, catalog):
    model = ScriptedModel()
    method = pipeline(tmp_path, catalog, model, max_work_calls=1)
    method.ingest()
    result = method.answer({"question_id": "q", "prompt": [{"type": "text", "text": "cup"}]})
    assert result["metadata"]["stop_reason"] == "work_budget"
    assert result["metadata"]["unresolved"]
    assert result["metadata"]["work_calls"] == 1


def test_missing_target_and_ambiguous_binding_expand_sources(tmp_path, catalog):
    method = pipeline(tmp_path, catalog, ScriptedModel())
    method.ingest()
    cat, store, packet, models, retriever = method._query_components({"m1", "m2"})
    for kind in ("target_missing", "ambiguous_binding"):
        delta = method._inspect({"prompt": [{"type": "text", "text": "cup"}]},
            [Gap(kind, kind)], QueryState(), cat, retriever, {}, 0)
        assert {n.kind for n in delta.candidates} >= {"text", "image"}


def test_inspection_resolves_crop_to_original_without_repeating_or_leaking_scope(tmp_path, catalog, monkeypatch):
    method = pipeline(tmp_path, catalog, ScriptedModel())
    method.ingest()
    cat, store, packet, models, retriever = method._query_components({"m1"})
    crop = store.candidates()[0]
    image = cat.image_node(crop.asset_id, crop.memory_id)
    # The full original ranks outside expansion, but the gap names its crop.
    monkeypatch.setattr(retriever, "retrieve", lambda *args, **kwargs: [crop])
    state = QueryState()
    candidates = {crop.node_id: crop}
    delta = method._inspect({}, [Gap("attribute_missing", "Need the surrounding pixels", (crop.node_id,))],
                            state, cat, retriever, candidates, 0)
    assert delta.candidates == [image]
    assert delta.image_ids == [image.node_id]
    assert all(n.memory_id == "m1" for n in delta.candidates)
    # Changing the gap wording cannot cause another retrieval of the same original.
    again = method._inspect({}, [Gap("attribute_missing", "Still missing", (crop.node_id,))],
                            state, cat, retriever, candidates, 1)
    assert not again.has_progress


def test_inspection_stops_when_originals_are_already_available(tmp_path, catalog):
    model = ScriptedModel(gaps=10)
    method = pipeline(tmp_path, catalog, model)
    method.ingest()
    result = method.answer({"question_id": "q", "prompt": [{"type": "text", "text": "cup"}]})
    assert result["metadata"]["stop_reason"] == "no_progress"
    assert result["metadata"]["unresolved"]
    assert result["metadata"]["new_observations"] == 0
    assert model.tasks.count("observe") == 1


def test_scope_modes_reject_future_and_unknown_sources(catalog):
    memories = list(catalog.memories.values())
    assert visible_ids({"memory_scope": {"mode": "prefix", "max_sequence": 0}}, memories) == {"m1"}
    assert visible_ids({"memory_scope": {"mode": "ids", "memory_ids": ["m2"]}}, memories) == {"m2"}
    with pytest.raises(ValueError, match="unavailable"):
        visible_ids({"memory_scope": {"mode": "ids", "memory_ids": ["absent"]}}, memories)


def test_real_faiss_three_routes_and_checkpoint(tmp_path, catalog):
    pytest.importorskip("faiss")
    from mm_memory_bench.methods.vector_index import FaissFlatIPIndex

    kwargs = dict(embedder=Embedder(), catalog=catalog, crops=CropCache(tmp_path / "crops", catalog),
        config=CoMEConfig(candidate_limit=3, proposal_k=3), root=tmp_path / "indices",
        index_factory=FaissFlatIPIndex)
    observations = [catalog.candidate(observation(catalog))]
    first = HybridRetriever(**kwargs)
    first.rebuild(observations)
    query = {"prompt": [{"type": "text", "text": "cup"}]}
    expected = first.retrieve(query)
    assert {n.kind for n in expected} == {"text", "observation", "image"}
    resumed = HybridRetriever(**kwargs)
    resumed.rebuild(observations)
    assert resumed.retrieve(query) == expected


class IdentityModel(ScriptedModel):
    def complete(self, messages, **kwargs):
        if messages[0]["role"] == "system" and "task=ecv." in messages[0]["content"]:
            payload = json.loads(messages[1]["content"][0]["text"])
            descriptions = [json.loads(p["text"]) for p in messages[1]["content"][1:]
                            if p["type"] == "text" and p["text"].startswith('{"node_id"')]
            text_id = next(n["node_id"] for n in descriptions if n["kind"] == "text")
            rows = []
            for nid, options in payload["eligible_anchors"].items():
                rows.append({"id": nid, "direct_support": 0, "best_anchor_id": options[0],
                    "incremental_support": 4, "role": "clarification", "support_ids": [text_id],
                    "claim": {"description": "The user holds the cup", "requires_text": True,
                        "fact": {"subject": "user", "predicate": "holding", "value": "a1:cup1", "basis": "textual"}}})
            return json.dumps({"candidates": rows})
        return super().complete(messages, **kwargs)


def test_ecv_identity_supports_remain_directed_and_rewards_deduplicate(tmp_path, catalog):
    config = CoMEConfig()
    packet = PacketBuilder(catalog, CropCache(tmp_path / "crops", catalog), config)
    models = CoMEModels(IdentityModel(), catalog, packet, config)
    candidates = [catalog.candidate(observation(catalog)),
                  next(n for n in catalog.nodes.values() if n.kind == "text" and n.memory_id == "m1")]
    result = models.verify({"prompt": [{"type": "text", "text": "Who holds the cup?"}]}, candidates, QueryState())
    assert len(result.claims) == 2
    assert len({c.fact_key for c in result.claims}) == 1
    for claim in result.claims:
        assert result.requires[claim.id] == {n.node_id for n in candidates}
    selected = select_evidence({n.node_id: n for n in candidates}, result,
        lambda ids, claims: len(ids), 10, config)
    assert len(selected.claims) == 1


def test_grounding_rejects_identity_without_original_text(tmp_path, catalog):
    class UngroundedIdentity(ScriptedModel):
        def complete(self, messages, **kwargs):
            if messages[0]["role"] == "system" and "task=ground_observations." in messages[0]["content"]:
                payload = json.loads(messages[1]["content"][0]["text"])
                return json.dumps({"checks": [{"id": nid, "visible_supported": True,
                    "requires_text": True, "text_supported": False} for nid in payload["proposed_ids"]]})
            return super().complete(messages, **kwargs)
    method = pipeline(tmp_path, catalog, UngroundedIdentity())
    method.ingest()
    assert not method.base_store.latest


def test_query_images_are_sent_to_node_verifier(tmp_path, catalog):
    class Capture(ScriptedModel):
        def complete(self, messages, **kwargs):
            if messages[0]["role"] == "system" and "task=node_verify." in messages[0]["content"]:
                assert any(p["type"] == "image_url" for p in messages[1]["content"])
            return super().complete(messages, **kwargs)
    method = pipeline(tmp_path, catalog, Capture())
    cat, store, packet, models, retriever = method._query_components({"m1"})
    text = next(n for n in cat.nodes.values() if n.kind == "text")
    question = {"prompt": [{"type": "text", "text": "What is in this image?"},
                           {"type": "image", "path": catalog.assets["a1"]["path"]}]}
    assert models.verify(question, [text], QueryState()).prizes[text.node_id] > 0


def test_oversized_verifier_batches_split_before_calling_model(tmp_path, catalog):
    class SmallContext(ScriptedModel):
        def count_input_tokens(self, messages, tools=None):
            if messages[0]["role"] == "system" and "task=node_verify." in messages[0]["content"]:
                payload = json.loads(messages[1]["content"][0]["text"])
                if len(payload["node_ids"]) > 2:
                    return 32700
            return super().count_input_tokens(messages, tools)
    model = SmallContext()
    method = pipeline(tmp_path, catalog, model)
    models = method._query_components({"m1"})[3]
    result = models.verify({"prompt": [{"type": "text", "text": "cup"}]},
                          [node("a"), node("b"), node("c")], QueryState())
    assert set(result.prizes) == {"a", "b", "c"}
    assert model.tasks.count("node_verify") == 2


def test_retention_uses_independent_annotations_not_self_reported_closure():
    predictions = [{"question_id": "q1", "metadata": {
        "selected_nodes": [{"node_id": "a"}], "requires": {}, "unresolved": [],
        "work_calls": 1, "reader_input_tokens": 20}}]
    report = summarize(predictions, [{"question_id": "q1", "required_supports": [["a", "b"], ["a"]]}])
    assert report["structural_dependency_closure_rate"] == 1
    assert report["complete_evidence_retention_rate"] == .5


def test_release_runner_stages_and_resume_with_real_faiss(tmp_path, catalog, monkeypatch):
    pytest.importorskip("faiss")
    from mm_memory_bench.come import legacy_runner as runner
    from pathlib import Path

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    def write_rows(name, rows):
        (bundle / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (bundle / "manifest.json").write_text('{}')
    write_rows("contexts.jsonl", [{"context_id": "ctx"}])
    write_rows("assets.jsonl", [{"asset_id": "a1", "path": catalog.assets["a1"]["path"]}])
    write_rows("memories.jsonl", [{"context_id": "ctx", "memory_id": "m1", "sequence": 0,
        "content": [{"type": "image", "asset_id": "a1"}, {"type": "text", "text": "A cup"}]}])
    write_rows("questions.jsonl", [{"context_id": "ctx", "question_id": "q1", "memory_scope": {"mode": "all"},
        "prompt": [{"type": "text", "text": "What color is the cup?"}],
        "answer": {"text": "PRIVATE_GOLD_MUST_NOT_BE_EXPOSED"}}])
    backends = []
    def backend(config):
        model = ScriptedModel()
        model.config = config
        backends.append(model)
        return model
    monkeypatch.setattr(runner, "GMEQwen2VLEmbedder", lambda name: Embedder())
    monkeypatch.setattr(runner, "OpenAICompatibleQwenVL", backend)
    output = tmp_path / "predictions.jsonl"
    legacy_config = tmp_path / "legacy.json"
    legacy_config.write_text(json.dumps({"benchmark_suite": {"max_model_len": 32768, "reader_max_output_tokens": 1000, "verifier_max_output_tokens": 8192, "embedding_model": "fixture"}}))
    argv = ["runner", "--config", str(legacy_config),
        "--bundle", str(bundle), "--artifact-root", str(tmp_path / "artifacts"),
        "--output", str(output), "--model", "fixture", "--base-url", "http://localhost/v1"]
    for stage in ("checkpoint", "observations", "reader", "reader"):
        monkeypatch.setattr("sys.argv", [*argv, "--stage", stage])
        assert runner.main() == 0
    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(results) == 1
    assert results[0]["metadata"]["protocol"] == "graphmemix-come-1"
    assert results[0]["metadata"]["query_protocol"] == QUERY_PROTOCOL
    assert "PRIVATE_GOLD" not in json.dumps([m.reader_messages for m in backends])
    # The ingestion artifacts remain reusable, but old query-policy predictions
    # must not be silently resumed into the current output.
    manifest = output.with_suffix(".contract.json")
    old_contract = json.loads(manifest.read_text())
    old_contract.pop("query_protocol")
    from mm_memory_bench.come.types import digest
    old_contract["contract"] = digest({k: v for k, v in old_contract.items() if k != "contract"})
    manifest.write_text(json.dumps(old_contract))
    with pytest.raises(ValueError, match="incompatible prediction resume contract"):
        runner.main()
