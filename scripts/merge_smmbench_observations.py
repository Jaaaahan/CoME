"""Validate and merge disjoint observations without changing their contracts."""
import argparse
from pathlib import Path
from mm_memory_bench.come.fields import ObservationSnapshot, raw_memories, write_snapshot
from mm_memory_bench.core import iter_jsonl
from mm_memory_bench.reader import BundleReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
memories = raw_memories(BundleReader(args.root / 'data/smmbench'))
rows = []
contract = None
for worker in range(4):
    path = args.root / f'artifacts/observations/smmbench-worker-{worker}.jsonl'
    snapshot = ObservationSnapshot(path, memories[worker::4])
    current = snapshot.contract['extraction_contract']
    if contract is not None and current != contract:
        raise ValueError('Shard extraction contracts differ')
    contract = current
    rows.extend(iter_jsonl(path))
output = args.root / 'artifacts/observations/smmbench.jsonl'
write_snapshot(output, rows, extraction_contract=contract)
ObservationSnapshot(output, memories)
print(f'Merged and validated {len(rows)} memories', flush=True)
