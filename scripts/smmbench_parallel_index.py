"""Use one GME process per allocated GPU, with disjoint context writers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--embedding-model', required=True)
args = parser.parse_args()
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(devices) != 4:
    raise ValueError(f'Expected four allocated GPUs, got {devices}')
children, logs = [], []
try:
    for i, device in enumerate(devices):
        command = [sys.executable, str(Path(__file__).with_name('build_memix_checkpoints.py')),
                   '--bundle', str(args.root / 'data/smmbench'),
                   '--observations', str(args.root / 'artifacts/observations/smmbench.jsonl'),
                   '--checkpoint-root', str(args.root / 'artifacts/checkpoints/smmbench'),
                   '--embedding-model', args.embedding_model, '--backend', 'standalone',
                   '--memory-view', 'derived', '--batch-size', '8', '--checkpoint-interval', '500',
                   '--context-shards', '4', '--context-shard-index', str(i)]
        log = (args.root / f'logs/index-{os.environ["SLURM_JOB_ID"]}-gpu-{i}.log').open('w')
        logs.append(log)
        children.append(subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': device},
                                         stdout=log, stderr=subprocess.STDOUT))
        print(f'Started context shard {i}/4 on allocated GPU {device}', flush=True)
    while True:
        statuses = [child.poll() for child in children]
        if any(status not in (None, 0) for status in statuses):
            raise RuntimeError(f'Index shard failed: {statuses}; see per-GPU logs')
        if all(status == 0 for status in statuses):
            break
        time.sleep(5)
finally:
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    for log in logs:
        log.close()
print('All four context shards completed', flush=True)
