from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mm_memory_bench.methods import ConcreteMemixMethod, FaissFlatIPIndex, GenerationConfig
from mm_memory_bench.methods import concrete_memix
from mm_memory_bench.methods.memory_backend import backend_contract, backend_hash
from scripts.run_memix_answers_from_retrieval import hydrate_reader


class FixtureEmbedder:
    dimension = 2

    def encode_units(self, units):
        return np.asarray([
            [1.0, 0.0] if "apple" in str(unit.get("text", "")) else [0.0, 1.0]
            for unit in units
        ], dtype=np.float32)


class RecordingAnswer:
    def __init__(self):
        self.messages = []

    def complete(self, messages, *, tools=None, response_format=None):
        self.messages.append(messages)
        return "apple"


def make_method(tmp_path, **kwargs):
    return ConcreteMemixMethod(
        GenerationConfig(top_k=1), backend="standalone",
        memix_repo=tmp_path / "does-not-exist", embedder=FixtureEmbedder(),
        index=FaissFlatIPIndex(2), checkpoint_dir=tmp_path / "checkpoint",
        ingest_batch_size=1, source_top_k=10, **kwargs,
    )


def test_real_faiss_checkpoint_resume_and_reader_without_memix(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("standalone must never load external Memix code")
    monkeypatch.setattr(concrete_memix, "_load_memix_core", forbidden)
    answer = RecordingAnswer()
    method = make_method(tmp_path, answer_model=answer)
    method.begin_context({"context_id": "fixture"})
    for mid, text in (("m1", "apple receipt"), ("m2", "train ticket")):
        method.ingest({"memory_id": mid, "source_id": mid, "kind": "email",
                       "content": [{"type": "text", "text": text}]})
    question = {"question_id": "q1", "prompt": [{"type": "text", "text": "apple"}]}
    result = method.answer(question)
    assert result["retrieved_memory_ids"] == ["m1"]
    assert result["method"] == "standalone-dense-v1"
    method.end_context()
    state_path = next((tmp_path / "checkpoint").glob("contexts/*/state.json"))
    state = json.loads(state_path.read_text())
    assert state["config"]["format"] == "graphmemix-standalone-checkpoint-1"
    assert state["config"]["core_sha256"] == backend_hash("standalone")

    resumed = make_method(tmp_path, answer_model=answer)
    resumed.begin_context({"context_id": "fixture"})
    assert resumed.index.size == 2
    assert resumed.answer(question)["retrieved_memory_ids"] == ["m1"]
    resumed.end_context()

    reader = hydrate_reader(
        checkpoint_state=state_path, memix_repo=tmp_path / "missing",
        backend="standalone", generation=GenerationConfig(top_k=1),
        answer_model=answer, source_top_k=10, public_source_ids={}, public_asset_paths={},
    )
    assert reader._generate_answer(question, ["m1"]) == "apple"
    packet = json.dumps(answer.messages[-1])
    assert "apple receipt" in packet
    assert "train ticket" not in packet


def test_standalone_rejects_memix_options_and_incompatible_checkpoints(tmp_path):
    with pytest.raises(ValueError, match="planner/verifier"):
        make_method(tmp_path, use_llm_planner=True)
    method = make_method(tmp_path)
    method.begin_context({"context_id": "fixture"})
    method.ingest({"memory_id": "m1", "kind": "email", "content": [{"type": "text", "text": "apple"}]})
    method.end_context()
    path = next((tmp_path / "checkpoint").glob("contexts/*/state.json"))
    state = json.loads(path.read_text())
    state["config"]["core_sha256"] = "unrelated-backend"
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="incompatible"):
        make_method(tmp_path).begin_context({"context_id": "fixture"})
    assert "memix_core_sha256" not in backend_contract("standalone")
    with pytest.raises(FileNotFoundError):
        backend_hash("memix", tmp_path / "missing")


def test_multiview_scores_pool_by_memory_id(tmp_path):
    method = make_method(tmp_path)
    method._embedding_units = [
        {"memory_id": "m1"}, {"memory_id": "m1"}, {"memory_id": "m2"},
    ]
    method.index.add(np.asarray([[1, 0], [.8, .2], [0, 1]], dtype=np.float32))
    result = method._source_detail({"prompt": [{"type": "text", "text": "apple"}]})
    assert result["retrieval_ids"] == ["m1", "m2"]
    assert result["retrieval_scores"] == pytest.approx([1.0, 0.0])
