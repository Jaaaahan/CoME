"""Check complete memory, graph and question coverage before sealing a build."""
import argparse
import json
import os
from pathlib import Path
from mm_memory_bench.core import iter_jsonl, validate_bundle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root
bundle = root / 'data/smmbench'
report = validate_bundle(bundle, check_assets=True)
assert report['missing_asset_count'] == 0
expected = {r['memory_id'] for r in iter_jsonl(bundle / 'memories.jsonl')}
states = [json.loads(p.read_text()) for p in (root / 'artifacts/checkpoints/smmbench').glob('contexts/*/state.json')]
assert len(states) == report['counts']['contexts']
processed = [mid for state in states for mid in state['processed_memory_ids']]
assert len(processed) == len(expected) and set(processed) == expected
observed = [r['memory_id'] for r in iter_jsonl(root / 'artifacts/observations/smmbench.jsonl')]
assert len(observed) == len(expected) and set(observed) == expected
summary = json.loads((root / 'artifacts/relations/smmbench.summary.json').read_text())
assert summary['memories'] == len(expected) and summary['unit_graphs'] == len(states)
questions = {r['question_id'] for r in iter_jsonl(bundle / 'questions.jsonl')}
priors = [r['question_id'] for r in iter_jsonl(root / 'artifacts/source_priors/smmbench.jsonl')]
assert len(priors) == len(questions) and set(priors) == questions
report.update(status='complete', model='Qwen/Qwen3-VL-8B-Instruct',
              job_id=os.environ.get('SLURM_JOB_ID'), graph_edges=summary['edges'])
(root / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
