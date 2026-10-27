#!/usr/bin/env python3
"""Check immutable full-run artifacts and compute native metrics plus local judging."""
import argparse
from collections import defaultdict
import json
import statistics
import sys
from pathlib import Path

from mm_memory_bench.come.store import atomic_json, file_digest
from mm_memory_bench.core import iter_jsonl
from mm_memory_bench.judge import judge_predictions
from evaluate_atm_retrieval import memory_sessions, metrics_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['atm', 'mem_gallery', 'h2hmem'], required=True)
    parser.add_argument('--base-url')
    parser.add_argument('--seal-only', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    run = root / args.dataset
    artifacts = run / 'artifacts/observations-v1'
    seal = run / 'validation/offline-hashes.json'
    hashes = {str(p.relative_to(run)): file_digest(p) for p in sorted(artifacts.rglob('*')) if p.is_file()}
    if args.seal_only:
        # Never replace the original seal when resuming a query run.
        if seal.exists():
            assert json.loads(seal.read_text()) == hashes, 'offline artifacts changed after sealing'
        else:
            atomic_json(seal, hashes)
        print(json.dumps({'event': 'offline_sealed', 'files': len(hashes)}))
        return
    assert json.loads(seal.read_text()) == hashes, 'query changed offline artifacts'
    bundle = root / 'data' / ('atm_bench' if args.dataset == 'atm' else args.dataset)
    questions = {r['question_id']: r for r in iter_jsonl(bundle / 'questions.jsonl')}
    expected = {'atm': 1044, 'mem_gallery': 1711, 'h2hmem': 1982}[args.dataset]
    assert len(questions) == expected
    question_ids_path = bundle / 'question_ids.txt' if args.dataset == 'h2hmem' else None
    if question_ids_path:
        assert set(question_ids_path.read_text().splitlines()) == set(questions)
    graph = run / 'runs/observations-v1/qwen3vl8b' / args.dataset
    for path in [graph / 'node_scores.jsonl', graph / 'ecv_scores.jsonl',
                 graph / 'selection/original_ecv_edge_p1.jsonl', graph / 'pruning/root0.12_node0.jsonl',
                 graph / 'reader/predictions.jsonl']:
        values = list(iter_jsonl(path))
        assert len(values) == expected and {r['question_id'] for r in values} == set(questions), path
    prediction_path = graph / 'reader/predictions.jsonl'
    predictions = {r['question_id']: r for r in iter_jsonl(prediction_path)}
    assert all(p['prediction'].strip() and p.get('metadata', {}).get('status') != 'error' for p in predictions.values())
    assert all(p['metadata']['query_added_observations'] == 0 for p in predictions.values())
    suite = Path(__file__).resolve().parents[1] / 'cluster/cuhk-suite'
    sys.path[:0] = [str(suite), str(suite / 'atm_evaluator')]
    import memeye_evaluator as native
    from memqa.utils.evaluator.evaluate_qa import deterministic_accuracy, list_jaccard_score
    from qwen_judge import QwenJudge
    if args.dataset == 'h2hmem':
        sys.path.insert(0, str(suite / 'h2h_evaluator'))
        from Lexical_metrics import EnglishTextProcessor, EvaluationMetricsCalculator
        provenance = json.loads((suite / 'h2h_evaluator/provenance.json').read_text())
        assert file_digest(suite / 'h2h_evaluator/Lexical_metrics.py') == provenance['sha256']
        calculator = EvaluationMetricsCalculator(EnglishTextProcessor())
    scores = defaultdict(list)
    for qid, question in questions.items():
        answer, truth = predictions[qid]['prediction'], str(question['answer']['text'])
        if args.dataset == 'atm':
            kind = question['task']['subcategory']
            prompt = '\n'.join(p.get('text', '') for p in question['prompt'] if p['type'] == 'text')
            if kind == 'list_recall':
                scores['list_jaccard'].append(list_jaccard_score(truth, answer))
            else:
                key = 'number_date_accuracy' if kind == 'number' else 'open_end_deterministic_match_diagnostic'
                scores[key].append(float(deterministic_accuracy(truth, answer, prompt)))
        elif args.dataset == 'h2hmem':
            for key, value in calculator.calculate_single_pair(answer, truth).items():
                scores[key].append(value)
        else:
            scores['exact_match'].append(float(native.score_open(answer, truth)[0]))
            scores['token_f1'].append(native.f1_score(answer, truth))
            scores['bleu1'].append(native.bleu_score(answer, truth, weights=(1., 0., 0., 0.)))
    observations = list(iter_jsonl(artifacts / 'observations' / (args.dataset + '.jsonl')))
    usage = defaultdict(int)
    for path in (artifacts / 'extraction' / args.dataset).rglob('ingestion.json'):
        for key, value in json.loads(path.read_text())['usage'].items():
            usage[key] += value
    report = {'protocol': 'graphmemix-observation-full-local-1', 'dataset': args.dataset,
              'questions': expected, 'backend': 'standalone', 'memory_view': 'derived',
              'offline_files_unchanged': len(hashes), 'observations': sum(len(r['come']['observations']) for r in observations),
              'extraction_usage': dict(usage), 'predictions_sha256': file_digest(prediction_path),
              'retrieval': metrics_for(questions.values(), predictions,
                  max(p['metadata']['retrieval_metadata']['original_k'] for p in predictions.values()),
                  memory_sessions(bundle / 'memories.jsonl')),
              'native_metrics': {key: {'value': statistics.mean(values), 'questions': len(values)} for key, values in scores.items()},
              'judge_model': QwenJudge.model, 'official_gpt5_mini_judge_run': False}
    if args.dataset == 'h2hmem':
        report['reporting_track'] = 'mmmb-h2hmem-evidence-supported-1.0'
        report['allowlist_sha256'] = file_digest(question_ids_path)
        report['native_evaluator'] = provenance
    atomic_json(run / 'metrics/native.json', report)
    if args.base_url:
        judged = judge_predictions(QwenJudge(args.base_url), bundle, prediction_path,
                                   run / 'metrics/local-qwen-judgments.jsonl', concurrency=args.concurrency,
                                   question_ids_path=question_ids_path)
        assert not judged['failed_judgments'], judged
        report['local_qwen_judge'] = judged
    atomic_json(run / 'metrics/summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
