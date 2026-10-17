"""Freeze a question-independent Qwen caption for all 1,300 H2H assets."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from mm_memory_bench.captions import CAPTION_PROMPT, CAPTION_PROMPT_VERSION
from mm_memory_bench.core import resolve_asset_path
from mm_memory_bench.methods.backends import OpenAICompatibleQwenVL
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.concrete_memix import _reader_data_url
from mm_memory_bench.reader import BundleReader

# Existing successful captions used the unchanged default decoding path.
# The v2 policy adds explicitly recorded anti-repetition retries on truncation.
COMPATIBLE_RUNNER_SHA256 = {'bda97458cc75027458c26aac02a0ecac32631128fcf13f64b755e3ce0f2b292f'}
RETRY_POLICY = {'max_attempts': 3, 'trigger': 'finish_reason=length',
                'repetition_penalties': [1.0, 1.1, 1.2],
                'other_errors': 'retry without changing decoding'}


class CaptionTruncated(RuntimeError):
    pass


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--concurrency', type=int, default=2)
    args = p.parse_args()
    root = Path(os.environ['DATA_ROOT'])
    experiment = Path(os.environ['EXPERIMENT_ROOT'])
    suite = Path(os.environ['SUITE_ROOT'])
    bundle = root / 'data/unified/h2hmem_main'
    reader = BundleReader(bundle)
    assets = reader.assets
    assert len(assets) == 1300
    paths = {aid: resolve_asset_path(bundle, asset) for aid, asset in assets.items()}
    hashes = {aid: sha(path) for aid, path in paths.items()}
    output = experiment / 'artifacts/captions/h2hmem.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    model = 'Qwen/Qwen3-VL-8B-Instruct'
    contract = {'protocol': 'cuhk-h2h-frozen-caption-2', 'model': model,
                'model_reference': json.loads((suite / 'qwen-weights-reference.json').read_text()),
                'prompt_version': CAPTION_PROMPT_VERSION, 'prompt': CAPTION_PROMPT,
                'prompt_sha256': hashlib.sha256(CAPTION_PROMPT.encode()).hexdigest(),
                'temperature': 0, 'seed': 0, 'max_output_tokens': 1000,
                'max_model_len': 32768, 'image_max_edge': 2500,
                'asset_sha256': hashes, 'runner_sha256': sha(Path(__file__)),
                'retry_policy': RETRY_POLICY,
                'input_scope': 'image pixels and fixed caption prompt only; no questions or answers'}
    with (output.parent / 'h2hmem.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp = output.with_suffix('.contract.json')
        if cp.exists():
            previous = json.loads(cp.read_text())
            if previous != contract:
                old_runner = previous.get('runner_sha256')
                legacy = {k: v for k, v in contract.items() if k != 'retry_policy'}
                legacy.update(protocol='cuhk-h2h-frozen-caption-1', runner_sha256=old_runner)
                assert old_runner in COMPATIBLE_RUNNER_SHA256 and previous == legacy, 'caption contract changed'
                archive = output.with_suffix(f'.contract.{old_runner}.json')
                if not archive.exists():
                    archive.write_bytes(cp.read_bytes())
                temporary = cp.with_suffix('.tmp')
                temporary.write_text(json.dumps(contract, indent=2) + '\n')
                temporary.replace(cp)
                print(json.dumps({'event': 'caption_runner_migrated', 'previous_sha256': old_runner,
                                  'new_sha256': contract['runner_sha256'],
                                  'default_generation_parameters_unchanged': True,
                                  'new_retry_policy': RETRY_POLICY}), flush=True)
        else:
            cp.write_text(json.dumps(contract, indent=2) + '\n')
        records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()] if output.exists() else []
        existing = {r['asset_id']: r for r in records}
        assert len(existing) == len(records)
        for aid, row in existing.items():
            assert aid in assets and row['image_sha256'] == hashes[aid]
            assert row['caption'].strip() and row['model'] == model
            assert row.get('repetition_penalty', 1.0) in RETRY_POLICY['repetition_penalties']
        client = OpenAICompatibleQwenVL(GenerationConfig(model=model, base_url=args.base_url,
            max_model_len=32768, max_output_tokens=1000, reasoning_effort='none'), token_count_mode='remote')
        thread_state = threading.local()
        def checked_transport(route, payload):
            if route == '/chat/completions':
                payload = {**payload, 'seed': 0}
                if thread_state.repetition_penalty != 1.0:
                    payload['repetition_penalty'] = thread_state.repetition_penalty
            result = client._post(route, payload)
            if route == '/chat/completions' and result['choices'][0].get('finish_reason') != 'stop':
                reason = result['choices'][0].get('finish_reason')
                error = CaptionTruncated if reason == 'length' else RuntimeError
                raise error('caption did not terminate normally: ' + json.dumps({
                    'finish_reason': reason,
                    'usage': result.get('usage')}))
            return result
        client._transport = checked_transport
        def caption(aid):
            penalty_index = 0
            messages = [{'role': 'user', 'content': [
                {'type': 'text', 'text': CAPTION_PROMPT},
                {'type': 'image_url', 'image_url': {'url': _reader_data_url(str(paths[aid]), 2500)}},
            ]}]
            for attempt in range(3):
                thread_state.repetition_penalty = RETRY_POLICY['repetition_penalties'][penalty_index]
                try:
                    text = client.complete(messages).strip()
                    assert text, 'empty caption'
                    return {'asset_id': aid, 'caption': text, 'model': model,
                            'image_sha256': hashes[aid], 'input_tokens': client.last_input_tokens,
                            'output_tokens': client.last_completion_tokens,
                            'finish_reason': 'stop', 'attempts': attempt + 1,
                            'repetition_penalty': thread_state.repetition_penalty}
                except Exception as exc:
                    if attempt == 2:
                        raise
                    if isinstance(exc, CaptionTruncated):
                        penalty_index = min(penalty_index + 1, 2)
                    time.sleep(2 * (attempt + 1))
        missing = [aid for aid in assets if aid not in existing]
        print(json.dumps({'event': 'caption_start', 'existing': len(existing), 'remaining': len(missing)}), flush=True)
        failures = []
        with output.open('a') as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(caption, aid): aid for aid in missing}
            for future in concurrent.futures.as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    failure = {'asset_id': futures[future], 'error': f'{type(exc).__name__}: {exc}'}
                    failures.append(failure)
                    output.with_suffix('.failures.json').write_text(json.dumps(failures, indent=2) + '\n')
                    print(json.dumps({'event': 'caption_error', **failure}), flush=True)
                    continue
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                existing[row['asset_id']] = row
                if len(existing) % 50 == 0:
                    print(json.dumps({'event': 'caption_progress', 'done': len(existing), 'total': 1300}), flush=True)
        output.with_suffix('.failures.json').write_text(json.dumps(failures, indent=2) + '\n')
        if failures:
            raise RuntimeError(f'{len(failures)} caption assets failed; {len(existing)}/1300 successes safely saved')
        assert set(existing) == set(assets)
        assert len(BundleReader(bundle, caption_sidecar=output).asset_captions) == 1300
        summary = {'complete': True, 'assets': 1300, 'sidecar_sha256': sha(output),
                   'contract_sha256': sha(cp), 'shared_by': ['graphmemix', 'dense']}
        output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'event': 'H2H_CAPTIONS_COMPLETE', **summary}), flush=True)


if __name__ == '__main__':
    main()
