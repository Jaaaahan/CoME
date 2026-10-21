"""Managed-pool environment must not turn the local embedder into a DDP worker."""
import os

import pytest


@pytest.mark.parametrize("world_size", ["1", "4"])
def test_gme_remains_local_with_inherited_worker_environment(tmp_path, monkeypatch, world_size):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mm_memory_bench.methods.gme import GMEQwen2VLEmbedder

    config = transformers.BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
    )
    transformers.BertModel(config).save_pretrained(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden_process_group(*args, **kwargs):
        pytest.fail("single-process GME must not initialize torch.distributed")

    monkeypatch.setattr(torch.distributed, "init_process_group", forbidden_process_group)
    embedder = GMEQwen2VLEmbedder(str(tmp_path), dimension=16)
    assert next(embedder.model.parameters()).device.type == "cpu"
    assert os.environ["WORLD_SIZE"] == world_size
    assert os.environ["RANK"] == "0"
    assert "LOCAL_RANK" not in os.environ
    assert not torch.distributed.is_initialized()
