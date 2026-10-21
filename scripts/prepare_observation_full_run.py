#!/usr/bin/env python3
"""Freeze the complete ATM and Mem-Gallery inputs and current source for Slurm."""
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
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads((project / 'configs/release/graphmemix.json').read_text())
    config['benchmark_suite']['embedding_model'] = str(args.data_root.resolve() / 'models/gme-Qwen2-VL-2B-Instruct')
    (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    report = {'full_dataset': True, 'question_filter': None, 'backend': 'standalone',
              'memory_view': 'derived', 'extraction_workers': 4, 'gpus_per_job': 4, 'datasets': {}}
    for dataset in ('atm', 'mem_gallery'):
        bundle = config['benchmark_suite']['datasets'][dataset]['bundle']
        source = args.data_root / 'data/unified' / bundle
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
    for name, folder in [('source-manifest.json', frozen), ('data-manifest.json', root / 'data')]:
        hashes = {str(p.relative_to(folder)): file_digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        (root / name).write_text(json.dumps(hashes, indent=2) + '\n')
    (root / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'root': str(root), **report}, indent=2))


if __name__ == '__main__':
    main()
