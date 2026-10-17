"""Score the 1,982-question H2H reporting track with the upstream lexical evaluator."""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import nltk

from evaluate_atm_retrieval import memory_sessions, metrics_for

suite = Path(__file__).parent
sys.path.insert(0, str(suite / 'h2h_evaluator'))
from Lexical_metrics import EnglishTextProcessor, EvaluationMetricsCalculator


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(os.environ['DATA_ROOT'])
    experiment = Path(os.environ['EXPERIMENT_ROOT'])
    bundle = root / 'data/unified/h2hmem_main'
    allowed_path = root / 'data/derived/h2hmem/evidence_supported_v1/question_ids.txt'
    allowed = set(allowed_path.read_text().splitlines())
    questions = {q['question_id']: q for q in rows(bundle / 'questions.jsonl')}
    assert len(questions) == 1982 and set(questions) == allowed
    graph = experiment / 'runs/qwen3vl8b/h2hmem'
    for filename in ('node_scores.jsonl', 'ecv_scores.jsonl', 'pruning/root0.12_node0.jsonl'):
        records = rows(graph / filename)
        assert len(records) == 1982 and {r['question_id'] for r in records} == allowed
        assert not any(r.get('error') or r.get('status') == 'error' for r in records)
    calc = EvaluationMetricsCalculator(EnglishTextProcessor())
    provenance = json.loads((suite / 'h2h_evaluator/provenance.json').read_text())
    assert digest(suite / 'h2h_evaluator/Lexical_metrics.py') == provenance['sha256']
    preparation = json.loads((allowed_path.parent / 'preparation.json').read_text())
    stopwords_sha = digest(Path(nltk.data.find('corpora/stopwords/english')))
    assert stopwords_sha == preparation['stopwords_sha256'], 'stopword resource changed'
    report = {'protocol': 'cuhk-h2h-native-1', 'reporting_questions': 1982,
              'official_full_release_questions': 2236, 'allowlist_sha256': digest(allowed_path),
              'metric': 'macro mean of per-question stopword-filtered token-set Recall',
              'evaluator': provenance, 'nltk': nltk.__version__, 'stopwords_sha256': stopwords_sha,
              'backend': 'standalone replacement (not the unavailable original memix-core)',
              'memory_view': 'derived', 'config_sha256': digest(suite / 'h2h-config.json'),
              'caption_sidecar_sha256': digest(experiment / 'artifacts/captions/h2hmem.jsonl'),
              'variants': {}}
    sessions = memory_sessions(bundle / 'memories.jsonl')
    for method, path in [('graphmemix', graph / 'reader/predictions.jsonl'),
                         ('dense', experiment / 'runs/dense/h2hmem/predictions.jsonl')]:
        predictions = rows(path)
        by_id = {r['question_id']: r for r in predictions}
        assert len(predictions) == len(by_id) == 1982 and set(by_id) == allowed
        assert all(r.get('prediction', '').strip() and r.get('metadata', {}).get('status') != 'error' for r in predictions)
        scores = []
        groups = collections.defaultdict(list)
        for qid, q in questions.items():
            score = calc.calculate_single_pair(by_id[qid]['prediction'], str(q['answer']['text']))
            scores.append(score)
            groups[q['task']['subcategory']].append(score['recall'])
        report['variants'][method] = {
            'complete': True, 'count': len(scores), 'predictions_sha256': digest(path),
            'native_percent': {k: 100 * statistics.mean(s[k] for s in scores) for k in scores[0]},
            'recall_by_category': {k: {'questions': len(v), 'percent': 100 * statistics.mean(v)} for k, v in groups.items()},
            'retrieval_diagnostic': metrics_for(questions.values(), by_id, 10, sessions)}
    output = experiment / 'metrics/h2hmem.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
