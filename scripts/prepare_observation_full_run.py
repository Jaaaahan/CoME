#!/usr/bin/env python3
"""Freeze benchmark inputs and current source for a full Slurm run."""
import argparse
import json
import shutil
from pathlib import Path

from mm_memory_bench.come.store import file_digest
from mm_memory_bench.core import iter_jsonl, resolve_asset_path, validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=['atm', 'mem_gallery', 'h2hmem'],
                        default=['atm', 'mem_gallery'])
    parser.add_argument('--gpus', type=int, choices=[3, 4], default=4)
    parser.add_argument('--caption-sidecar', type=Path,
                        help='Completed H2HMem caption sidecar; required for h2hmem.')
    args = parser.parse_args()
    if 'h2hmem' in args.datasets and args.caption_sidecar is None:
        parser.error('--caption-sidecar is required for h2hmem')
    if args.gpus == 3 and args.datasets != ['h2hmem']:
        parser.error('the three-GPU launcher supports --datasets h2hmem')
    project = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads((project / 'configs/release/graphmemix.json').read_text())
    config['benchmark_suite']['embedding_model'] = str(args.data_root.resolve() / 'models/gme-Qwen2-VL-2B-Instruct')
    (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = {'full_dataset': True, 'question_filter': None, 'backend': 'standalone',
              'memory_view': 'derived', 'extraction_workers': 2 if args.gpus == 3 else 4,
              'gpus_per_job': args.gpus, 'datasets': {}}
    for dataset in args.datasets:
        bundle = config['benchmark_suite']['datasets'][dataset]['bundle']
        source = args.data_root / 'data/unified' / ('h2hmem_main' if dataset == 'h2hmem' else bundle)
        target = root / 'data' / bundle
        target.mkdir(parents=True)
        for name in ('manifest.json', 'contexts.jsonl', 'memories.jsonl', 'questions.jsonl'):
            shutil.copy2(source / name, target / name)
        with (target / 'assets.jsonl').open('w') as handle:
            for asset in iter_jsonl(source / 'assets.jsonl'):
                asset['path'] = str(resolve_asset_path(source, asset))
                handle.write(json.dumps(asset, ensure_ascii=False) + '\n')
        validation = validate_bundle(target, check_assets=True)
        assert validation['missing_asset_count'] == 0, validation
        assert validation['counts']['questions'] == config['benchmark_suite']['datasets'][dataset]['expected_questions']
        report['datasets'][dataset] = validation
        for folder in ('logs', 'validation', 'metrics'):
            (root / dataset / folder).mkdir(parents=True)
        if dataset == 'h2hmem':
            allowlist = args.data_root / 'data/derived/h2hmem/evidence_supported_v1/question_ids.txt'
            allowed = set(allowlist.read_text().splitlines())
            questions = {row['question_id'] for row in iter_jsonl(target / 'questions.jsonl')}
            assert len(allowed) == 1982 and questions == allowed
            # The reporting slice must retain the complete available dialogue memory.
            full = args.data_root / 'data/unified/h2hmem'
            for name in ('contexts.jsonl', 'memories.jsonl'):
                assert file_digest(target / name) == file_digest(full / name), name
            shutil.copy2(allowlist, target / 'question_ids.txt')
            report['question_filter'] = 'mmmb-h2hmem-evidence-supported-1.0'
            captions = list(iter_jsonl(args.caption_sidecar))
            assets = {row['asset_id']: row for row in iter_jsonl(target / 'assets.jsonl')}
            assert len(captions) == len(assets) == len({row['asset_id'] for row in captions}) == 1300
            for row in captions:
                assert row['caption'].strip() and row['model'] == 'Qwen/Qwen3-VL-8B-Instruct'
                assert row['image_sha256'] == file_digest(Path(assets[row['asset_id']]['path']))
            caption_root = root / dataset / 'artifacts/captions'
            caption_root.mkdir(parents=True)
            shutil.copy2(args.caption_sidecar, caption_root / 'h2hmem.jsonl')
            report['captions'] = {'source': str(args.caption_sidecar.resolve()), 'assets': len(captions),
                                  'sha256': file_digest(caption_root / 'h2hmem.jsonl')}
    frozen = root / 'source'
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    for name in ('src', 'scripts', 'configs'):
        shutil.copytree(project / name, frozen / name, ignore=ignore)
    suite = frozen / 'cluster/cuhk-suite'
    suite.mkdir(parents=True)
    for name in ('token_proxy.py', 'check_server.py', 'process_lifecycle.sh', 'watch_server.py',
                 'qwen_judge.py', 'memeye_evaluator.py'):
        shutil.copy2(project / 'cluster/cuhk-suite' / name, suite / name)
    shutil.copytree(project / 'cluster/cuhk-suite/atm_evaluator', suite / 'atm_evaluator', ignore=ignore)
    if 'h2hmem' in args.datasets:
        shutil.copytree(project / 'cluster/cuhk-suite/h2h_evaluator', suite / 'h2h_evaluator', ignore=ignore)
    for name, folder in [('source-manifest.json', frozen), ('data-manifest.json', root / 'data')]:
        hashes = {str(p.relative_to(folder)): file_digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        (root / name).write_text(json.dumps(hashes, indent=2) + '\n')
    (root / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'root': str(root), **report}, indent=2))


if __name__ == '__main__':
    main()
