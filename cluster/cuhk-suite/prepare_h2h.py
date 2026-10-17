"""Validate the pinned release and make the paper's reporting view, keeping all memory."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from mm_memory_bench.core import validate_bundle
from mm_memory_bench.reader import BundleReader


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_frozen(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f'existing frozen input differs: {path}'
    else:
        path.write_bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--check-env', action='store_true')
    args = p.parse_args()
    suite = Path(__file__).parent
    root = args.data_root.resolve()
    entries = json.loads((suite / 'h2h-data-manifest.json').read_text())
    for entry in entries:
        path = root / entry['path']
        data = path.read_bytes()
        assert len(data) == entry['size'], path
        if 'lfs' in entry:
            actual = hashlib.sha256(data).hexdigest()
            expected = entry['lfs']['oid']
        else:
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            expected = entry['oid']
        assert actual == expected, path
    bundle = root / 'data/unified/h2hmem'
    validation = validate_bundle(bundle, check_assets=True)
    assert validation['counts'] == {'assets': 1300, 'contexts': 25, 'memories': 7078, 'questions': 2236}
    assert validation['missing_asset_count'] == 0
    derived = root / 'data/derived/h2hmem/evidence_supported_v1'
    subprocess.run([sys.executable, str(Path(os.environ['PROJECT_ROOT']) / 'scripts/build_h2h_evidence_track.py'),
                    '--bundle', str(bundle), '--output-dir', str(derived)], check=True)
    summary = json.loads((derived / 'summary.json').read_text())
    assert summary['selected_questions'] == 1982
    assert summary['excluded_by_reason'] == {'missing_dyadic_dialogue3_session6': 8, 'session0_question_container': 246}
    allowed = set((derived / 'question_ids.txt').read_text().splitlines())
    assert len(allowed) == 1982
    # Same directory depth preserves every original relative asset path.
    view = root / 'data/unified/h2hmem_main'
    view.mkdir(parents=True, exist_ok=True)
    for name in ('contexts', 'memories', 'assets'):
        write_frozen(view / f'{name}.jsonl', (bundle / f'{name}.jsonl').read_bytes())
    selected = [line for line in (bundle / 'questions.jsonl').read_bytes().splitlines()
                if json.loads(line)['question_id'] in allowed]
    write_frozen(view / 'questions.jsonl', b'\n'.join(selected) + b'\n')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['counts']['questions'] = 1982
    manifest['reporting_track'] = {'protocol': summary['protocol'], 'full_release_questions': 2236,
                                  'allowlist_sha256': digest(derived / 'question_ids.txt'),
                                  'memory_scope': 'complete available dialogue; unchanged from release'}
    write_frozen(view / 'manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    validate_bundle(view, check_assets=True)
    batches = list(BundleReader(view).iter_context_batches())
    assert len(batches) == 25 and sum(len(b.memories) for b in batches) == 7078
    assert sum(len(b.questions) for b in batches) == 1982
    memory_assets = {p['asset_id'] for b in batches for m in b.memories for p in m['content'] if 'asset_id' in p}
    assert len(memory_assets) == 1262
    report = {'status': 'ready', 'hf_revision': '1cff9951fa3deb404c049e7e507f65c38b4751dd',
              'verified_files': len(entries), 'verified_bytes': sum(e['size'] for e in entries),
              'full_release': validation, 'reporting_track': summary,
              'memory_assets': len(memory_assets), 'caption_assets': 1300, 'view': str(view),
              'view_sha256': {p.name: digest(p) for p in sorted(view.iterdir())},
              'categories': dict(collections.Counter(q['task']['subcategory'] for b in batches for q in b.questions))}
    if args.check_env:
        import nltk
        for item in ('tokenizers/punkt', 'tokenizers/punkt_tab/english', 'corpora/stopwords'):
            nltk.data.find(item)
        sys.path.insert(0, str(suite / 'h2h_evaluator'))
        from Lexical_metrics import EnglishTextProcessor, EvaluationMetricsCalculator
        calc = EvaluationMetricsCalculator(EnglishTextProcessor())
        # Actual upstream set recall, including stopword filtering.
        assert calc.calculate_single_pair('The red car.', 'A red car and a blue bike.')['recall'] == 0.5
        report['native_metric_probe'] = 'passed'
        report['nltk'] = nltk.__version__
        report['stopwords_sha256'] = digest(Path(nltk.data.find('corpora/stopwords/english')))
    (derived / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
