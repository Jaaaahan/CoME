"""Replay the earliest unsaved caption without changing benchmark artifacts."""
import argparse
import json
import os
from pathlib import Path

from mm_memory_bench.captions import CAPTION_PROMPT
from mm_memory_bench.core import resolve_asset_path
from mm_memory_bench.methods.backends import OpenAICompatibleQwenVL
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.concrete_memix import _reader_data_url
from mm_memory_bench.reader import BundleReader

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repetition-penalty', type=float, default=1.0)
p.add_argument('--budgets', type=int, nargs='+', default=[1000, 4000])
args = p.parse_args()
root = Path(os.environ['DATA_ROOT'])
experiment = Path(os.environ['EXPERIMENT_ROOT'])
bundle = root / 'data/unified/h2hmem_main'
assets = BundleReader(bundle).assets
sidecar = experiment / 'artifacts/captions/h2hmem.jsonl'
existing = {json.loads(line)['asset_id'] for line in sidecar.read_text().splitlines()}
aid = next(aid for aid in assets if aid not in existing)
messages = [{'role': 'user', 'content': [
    {'type': 'text', 'text': CAPTION_PROMPT},
    {'type': 'image_url', 'image_url': {'url': _reader_data_url(str(resolve_asset_path(bundle, assets[aid])), 2500)}}]}]
client = OpenAICompatibleQwenVL(GenerationConfig(model='Qwen/Qwen3-VL-8B-Instruct',
    base_url=os.environ['BASE_URL'], max_model_len=32768,
    max_output_tokens=1000, reasoning_effort='none'), token_count_mode='remote')
count = client.count_input_tokens(messages)
results = []
for budget in args.budgets:
    payload = client.completion_payload(messages, min(budget, 32768 - count))
    payload['seed'] = 0
    if args.repetition_penalty != 1.0:
        payload['repetition_penalty'] = args.repetition_penalty
    response = client._post('/chat/completions', payload)
    result = {'asset_id': aid, 'budget': budget, 'repetition_penalty': args.repetition_penalty, 'response': response}
    results.append(result)
    (experiment / f'validation/caption-diagnosis-penalty-{args.repetition_penalty}.json').write_text(json.dumps(results, indent=2) + '\n')
    choice = response['choices'][0]
    print(json.dumps({'asset_id': aid, 'budget': budget, 'repetition_penalty': args.repetition_penalty,
                      'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage'),
                      'tail': choice['message'].get('content', '')[-240:]}), flush=True)
    if choice.get('finish_reason') == 'stop':
        break
