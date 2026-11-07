"""Shard migration must preserve stores and exactly reproduce the full snapshot."""
import json
import sys

from mm_memory_bench.come import runner
from mm_memory_bench.come.fields import ObservationSnapshot, raw_memories, write_snapshot
from mm_memory_bench.core import BundleWriter, iter_jsonl
from mm_memory_bench.reader import BundleReader


def test_split_resume_and_merge(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    config = tmp_path / 'config.json'
    config.write_text('{}')
    with BundleWriter(bundle, {'benchmark': 'test'}) as writer:
        for c in range(4):
            writer.add_context({'context_id': f'c{c}', 'benchmark': 'test'})
            for m in range(8):
                writer.add_memory({'context_id': f'c{c}', 'memory_id': f'c{c}m{m}',
                    'kind': 'dialogue_message', 'sequence': m,
                    'content': [{'type': 'text', 'text': f'memory {m}'}]})
    base = ['runner', '--config', str(config), '--bundle', str(bundle),
            '--artifact-root', str(tmp_path / 'extraction'), '--model', 'Qwen/Qwen3-VL-8B-Instruct',
            '--base-url', 'http://127.0.0.1:1/v1', '--workers', '4']
    monkeypatch.setattr(sys, 'argv', base + ['--output', str(tmp_path / 'original.jsonl')])
    assert runner.main() == 0
    memories = raw_memories(BundleReader(bundle))
    rows = []
    for i in range(4):
        output = tmp_path / f'worker-{i}.jsonl'
        monkeypatch.setattr(sys, 'argv', base + ['--output', str(output), '--worker-id', str(i),
                                               '--context-workers', '4'])
        assert runner.main() == 0
        snapshot = ObservationSnapshot(output, memories[i::4])
        rows.extend(iter_jsonl(output))
        assert runner.main() == 0
    output = tmp_path / 'merged.jsonl'
    write_snapshot(output, rows, extraction_contract=snapshot.contract['extraction_contract'])
    ObservationSnapshot(output, memories)
    assert output.read_bytes() == (tmp_path / 'original.jsonl').read_bytes()
