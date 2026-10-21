#!/usr/bin/env python3
"""Audit real GPU smoke artifacts, including HTTP prompts and immutable indexes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import faiss

from mm_memory_bench.come.fields import ObservationSnapshot
from mm_memory_bench.core import iter_jsonl
from mm_memory_bench.reader import BundleReader
from build_graphmemix_relation_store import canonical_unit_rows


def load(path: Path):
    return json.loads(path.read_text())


def rows(path: Path):
    return list(iter_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    requests = rows(root / 'requests.jsonl')
    report = {'allocation': load(root / 'allocation.json'), 'datasets': {},
              'limitations': 'Gold-assisted small fixture; standalone backend; no official judge or full-benchmark accuracy measurement.'}
    all_observations = set()
    known_questions = []
    for dataset in ('mem_gallery', 'memeye'):
        bundle = root / 'data' / dataset
        questions = {q['question_id']: q for q in rows(bundle / 'questions.jsonl')}
        known_questions += [p['text'] for q in questions.values() for p in q['prompt'] if p['type'] == 'text']
        memories = {m['memory_id']: m for m in rows(bundle / 'memories.jsonl')}
        artifacts = root / 'artifacts/observations-v1'
        run = root / 'runs/observations-v1/qwen3vl8b' / dataset
        sidecar = artifacts / 'observations' / (dataset + '.jsonl')
        snapshot = ObservationSnapshot.from_bundle(sidecar, BundleReader(bundle))
        fields = {r['memory_id']: r['come'] for r in rows(sidecar)}
        obs = {o['id']: o for field in fields.values() for o in field['observations']}
        assert obs, f'{dataset}: real extraction produced no observations'
        all_observations.update(obs)
        before = load(root / (dataset + '-offline-hashes.json'))
        offline_paths = [sidecar, sidecar.with_suffix('.manifest.json')]
        offline_paths += [p for folder in ('checkpoints', 'extraction')
                         for p in (artifacts / folder / dataset).rglob('*') if p.is_file()]
        assert set(before) == {str(p.relative_to(root)) for p in offline_paths}, 'query added/deleted offline files'
        for name, expected in before.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, f'query mutated {name}'
        query_start = (root / (dataset + '-offline-hashes.json')).stat().st_mtime
        query_end = (root / 'logs' / (dataset + '-query.log')).stat().st_mtime
        query_requests = [r for r in requests if query_start <= r['time'] <= query_end]
        assert query_requests and all('CoME task=' not in r['messages'][0]['text'] for r in query_requests)
        checkpoint = artifacts / 'checkpoints' / dataset
        units, observation_units = 0, 0
        for path in checkpoint.glob('contexts/*/state.json'):
            state = load(path)
            snapshot.require_contract(state['config']['observations'])
            assert not state['media_failures'], state['media_failures']
            index = faiss.read_index(str(path.parent / state['index_file']))
            assert index.ntotal == len(state['embedding_units'])
            vectors = state['embedding_units']
            canonical = {i for indexes in canonical_unit_rows(vectors).values() for i in indexes}
            for i, unit in enumerate(vectors):
                assert unit['memory_id'] in memories
                if unit.get('observation_id'):
                    observation_units += 1
                    assert unit['representation'] == 'come_observation_text'
                    assert unit['observation_id'] in obs
                    assert obs[unit['observation_id']]['memory_id'] == unit['memory_id']
                    assert i not in canonical, 'observation used as canonical graph vector'
            units += index.ntotal
        assert observation_units == len(obs), (observation_units, len(obs))
        relations = rows(artifacts / 'relation_store' / (dataset + '.jsonl'))
        assert relations, 'relation store did not exercise graph edges'
        assert all(e['left'] in memories and e['right'] in memories for e in relations)
        stages = {
            'priors': artifacts / 'source_priors' / (dataset + '.jsonl'),
            'node': run / 'node_scores.jsonl', 'ecv': run / 'ecv_scores.jsonl',
            'select': run / 'selection/original_ecv_edge_p1.jsonl',
            'prune': run / 'pruning/root0.12_node0.jsonl',
            'reader': run / 'reader/predictions.jsonl',
        }
        stage_reports = {}
        by_stage = {}
        for stage, path in stages.items():
            values = rows(path)
            by_stage[stage] = {v['question_id']: v for v in values}
            assert len(values) == len(questions) and set(by_stage[stage]) == set(questions), stage
            shown = set()
            for value in values:
                metadata = value.get('metadata', {})
                contract = (metadata.get('reader_run_contract') if stage == 'reader' else
                            metadata if stage in ('select', 'prune') else value)
                snapshot.require_contract(contract.get('observations'))
                ids = value.get('retrieval_ids', value.get('retrieved_memory_ids', value.get('candidate_ids', [])))
                assert len(ids) == len(set(ids)) and set(ids) <= memories.keys(), (stage, ids)
                for mid, oids in value.get('observation_ids', metadata.get('observation_ids', {})).items():
                    assert mid in memories and set(oids) <= obs.keys()
                    assert all(obs[oid]['memory_id'] == mid for oid in oids)
                    shown.update(oids)
                if stage == 'ecv':
                    assert value['returned_candidates'] == value['eligible_edge_candidates']
                    for edge in value['edge_scores'].values():
                        assert edge['left'] in memories and edge['right'] in memories
                if stage == 'priors':
                    assert value['observation_hits'], 'no observation embedding hit in real retrieval'
                    for mid, hits in value['observation_hits'].items():
                        assert mid in ids
                        assert all(obs[h['observation_id']]['memory_id'] == mid for h in hits)
                if stage == 'reader':
                    assert value['prediction'].strip() and metadata.get('status') != 'error'
                    assert metadata['query_added_observations'] == 0
            stage_reports[stage] = {'questions': len(values), 'shown_observations': len(shown)}
        predictions = []
        for qid, value in by_stage['reader'].items():
            selected = by_stage['prune'][qid]['retrieved_memory_ids']
            assert value['retrieved_memory_ids'] == selected
            gold = {e['memory_id'] for e in questions[qid].get('evidence', [])}
            predictions.append({'question_id': qid, 'question': questions[qid]['prompt'],
                'prediction': value['prediction'], 'reference': questions[qid]['answer'],
                'selected_memories': len(selected), 'selected_gold_memories': len(gold & set(selected)),
                'gold_memories': len(gold)})
        ingestions = list((artifacts / 'extraction' / dataset).glob('contexts/*/ingestion.json'))
        assert len(ingestions) == 1, 'smoke fixture should contain one context per dataset'
        ingestion = load(ingestions[0])
        report['datasets'][dataset] = {'memories': len(memories), 'observations': len(obs),
            'observed_images': len(ingestion['completed_images']), 'extraction_usage': ingestion['usage'],
            'grounding_outcomes': ingestion['image_outcomes'], 'embedding_units': units,
            'observation_embedding_units': observation_units, 'graph_edges': len(relations),
            'offline_files_unchanged': len(before), 'stages': stage_reports,
            'predictions': predictions,
            'observation_examples': list(obs.values())[:3]}
    traces = {'observe': [], 'ground': [], 'node': [], 'ecv': [], 'reader': []}
    for request in requests:
        system = request['messages'][0]['text']
        text = '\n'.join(m['text'] for m in request['messages'])
        stage = ('observe' if 'CoME task=observe.' in system else
                 'ground' if 'CoME task=ground_observations.' in system else
                 'node' if 'evidence candidates' in system else
                 'ecv' if 'evidence-chain verifier' in system else
                 'reader' if 'Use only the following Memix evidence packet.' in text else None)
        if stage is None:
            continue  # server warmup
        found = set(re.findall(r'obs:[a-f0-9]{24}', text))
        if stage in ('observe', 'ground'):
            assert not any(q in text for q in known_questions), 'evaluation question reached offline extraction'
        else:
            assert found <= all_observations, f'unknown observations reached {stage}'
        traces[stage].append({'time': request['time'], 'observation_ids': sorted(found),
                              'images': sum(m['image_count'] for m in request['messages'])})
    for stage, values in traces.items():
        assert values, f'no real HTTP requests for {stage}'
        if stage in ('node', 'ecv', 'reader'):
            assert any(v['observation_ids'] for v in values), f'observations absent from real {stage} input'
    report['http_trace'] = {stage: {'requests': len(values),
        'with_observations': sum(bool(v['observation_ids']) for v in values),
        'image_count': sum(v['images'] for v in values)} for stage, values in traces.items()}
    report['status'] = 'structural_checks_passed'
    report['semantic_correctness'] = 'Not established by this audit; requires comparison with original sources.'
    (root / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(root / 'audit.json'),
                      'http_trace': report['http_trace']}, indent=2))


if __name__ == '__main__':
    main()
