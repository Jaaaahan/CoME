#!/usr/bin/env python3
"""SwanLab telemetry for full-data ENM generation and separately scored trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_enm_local_search import read_plan
from run_enm_sweep import evaluation_set, FIXED, write_json

PROXIES = ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')


def identity(plan_path, row):
    plan = read_plan(plan_path)
    ids, dataset = evaluation_set(Path(plan['settings']['bundle']))
    directory = Path(plan['settings']['output_root']) / row['id']
    return dict(id=hashlib.sha256(str(directory).encode()).hexdigest()[:8],
        name=f"{Path(plan['settings']['bundle']).name}-{plan['settings']['reader'].get('search_decision_mode', 'grounded')}-{row['id']}",
        config={**row['parameters'], **FIXED, 'tuning_questions': len(ids),
            'evaluation_scope': 'full_dataset', 'score_role': 'tuning_set_accuracy',
            'questions_sha256': dataset['questions_sha256'], 'judge_model': plan['settings']['judge']['model'],
            'search_decision_mode': plan['settings']['reader'].get('search_decision_mode', 'grounded')},
        directory=str(directory))


def start(payload, project):
    import swanlab
    directory = Path(payload['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    run = swanlab.init(project=project, workspace='Yuhan_Jia', experiment_name=payload['name'],
        id=payload['id'], resume='allow', config=payload['config'], mode='online',
        logdir=str(directory / 'swanlog'))
    write_json(directory / 'swanlab.json', dict(id=payload['id'], url=run.url, project=project))
    return run


def main():
    for name in PROXIES:
        os.environ.pop(name, None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'sync', 'register'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--base-url')
    parser.add_argument('--project', default='AgentMemory')
    args = parser.parse_args()
    if args.command == 'sync':
        payload = json.load(sys.stdin)
    else:
        plan = read_plan(args.plan)
        payload = identity(args.plan, plan['trials'][args.index])
    run = start(payload, args.project)
    try:
        if args.command == 'sync':
            run.log(payload['metrics'], step=1000000)
        elif args.command == 'register':
            run.log({'progress/total_questions': payload['config']['tuning_questions']}, step=0)
        else:
            command = [sys.executable, '-u', str(Path(__file__).with_name('run_enm_local_search.py')),
                'generate', '--plan', str(args.plan), '--index', str(args.index), '--base-url', args.base_url]
            process = subprocess.Popen(command)
            last = -1
            while True:
                path = Path(payload['directory']) / 'predictions.jsonl'
                count = sum(1 for line in path.open() if line.strip()) if path.exists() else 0
                if count != last:
                    run.log({'progress/generated_questions': count,
                        'progress/fraction': count / payload['config']['tuning_questions']}, step=count)
                    last = count
                status = process.poll()
                if status is not None:
                    if status:
                        raise subprocess.CalledProcessError(status, command)
                    break
                time.sleep(60)
    except BaseException as exc:
        run.finish(state='crashed', error=type(exc).__name__)
        raise
    else:
        run.finish()


if __name__ == '__main__':
    main()
