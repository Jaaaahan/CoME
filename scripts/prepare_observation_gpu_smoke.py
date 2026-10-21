#!/usr/bin/env python3
"""Freeze a small, evidence-preserving real-data fixture for the Slurm smoke job.

This deliberately samples with gold evidence to exercise the pipeline; results
are integration diagnostics, never benchmark accuracy estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from mm_memory_bench.core import BundleWriter, iter_jsonl, resolve_asset_path, validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'purpose': 'real-model integration smoke, gold-assisted fixture, not benchmark accuracy',
              'datasets': {}}
    config = json.loads((root / 'configs/release/graphmemix.json').read_text())
    config['benchmark_suite']['embedding_model'] = str(args.data_root / 'models/gme-Qwen2-VL-2B-Instruct')
    for dataset in ('mem_gallery', 'memeye'):
        source = args.data_root / 'data/unified' / dataset
        tables = {name: list(iter_jsonl(source / f'{name}.jsonl'))
                  for name in ('contexts', 'memories', 'assets', 'questions')}
        questions = [q for q in tables['questions']
                     if q.get('variant', {}).get('rotation_index', 0) == 0][:3]
        context_ids = {q['context_id'] for q in questions}
        evidence = {e['memory_id'] for q in questions for e in q.get('evidence', [])}
        context_memories = [m for m in tables['memories'] if m['context_id'] in context_ids]
        # Real evidence plus earlier distractors, retaining original IDs and order.
        selected = evidence | {m['memory_id'] for m in context_memories[:14 if dataset == 'mem_gallery' else 8]}
        memories = [m for m in context_memories if m['memory_id'] in selected]
        assets = {p['asset_id'] for row in memories + questions
                  for p in row.get('content', row.get('prompt', [])) if 'asset_id' in p}
        target = output / 'data' / dataset
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest['smoke_test'] = {'source_bundle': str(source), 'gold_assisted_selection': True}
        with BundleWriter(target, manifest) as writer:
            for row in tables['contexts']:
                if row['context_id'] in context_ids:
                    writer.add_context(row)
            for row in memories:
                writer.add_memory(row)
            for row in tables['assets']:
                if row['asset_id'] in assets:
                    writer.add_asset({**row, 'path': str(resolve_asset_path(source, row))})
            for row in questions:
                writer.add_question(row)
        validation = validate_bundle(target, check_assets=True)
        assert validation['missing_asset_count'] == 0, validation
        report['datasets'][dataset] = {'validation': validation,
            'image_parts': sum(p['type'] == 'image' for m in memories for p in m['content']),
            'question_ids': [q['question_id'] for q in questions]}
        config['benchmark_suite']['datasets'][dataset]['expected_questions'] = len(questions)
    frozen = output / 'source'
    for name in ('src', 'scripts', 'configs'):
        shutil.copytree(root / name, frozen / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    helper = frozen / 'cluster/cuhk-suite'
    helper.mkdir(parents=True)
    for name in ('token_proxy.py', 'check_server.py', 'process_lifecycle.sh'):
        shutil.copy2(root / 'cluster/cuhk-suite' / name, helper / name)
    # Record text and image counts at the actual HTTP boundary; no base64 blobs.
    proxy = helper / 'token_proxy.py'
    value = proxy.read_text().replace('import argparse,base64,', 'import os,time,argparse,base64,')
    value = value.replace('   else:self.forward(body)', '''   else:
    if self.path.endswith('/chat/completions') and os.environ.get('COME_SMOKE_TRACE'):
     payload=json.loads(body)
     messages=[]
     for message in payload.get('messages',[]):
      parts=message.get('content',[])
      if isinstance(parts,str):parts=[{'type':'text','text':parts}]
      messages.append({'role':message['role'],'text':'\\n'.join(p.get('text','') for p in parts if p.get('type')=='text'),'image_count':sum(p.get('type')=='image_url' for p in parts)})
     with open(os.environ['COME_SMOKE_TRACE'],'a') as handle:
      handle.write(json.dumps({'time':time.time(),'messages':messages,'response_format':payload.get('response_format')})+'\\n')
    self.forward(body)''')
    proxy.write_text(value)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (output / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')
    hashes = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(frozen.rglob('*')) if p.is_file()}
    (output / 'source-manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (output / 'logs').mkdir()
    print(json.dumps({'output': str(output), **report}, indent=2))


if __name__ == '__main__':
    main()
