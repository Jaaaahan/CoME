"""Behavioral checks for offline CoME units through the GraphMemix path."""
from __future__ import annotations

from copy import deepcopy
import base64
import importlib
import inspect
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

from mm_memory_bench.come.fields import (ObservationSnapshot, export_field, manifest_path,
    raw_memories, write_snapshot)
from mm_memory_bench.come.models import ObservationExtractor
from mm_memory_bench.come.pipeline import ObservationPreprocessor
from mm_memory_bench.come.store import SourceCatalog, file_digest
from mm_memory_bench.come.types import Fact, Observation
from mm_memory_bench.harness import _resolved_memory
from mm_memory_bench.memory_scope import visible_ids
from mm_memory_bench.methods import ConcreteMemixMethod, GenerationConfig, FaissFlatIPIndex
from mm_memory_bench.observation_evidence import render_observations
from mm_memory_bench.reader import BundleReader
from mm_memory_bench.retrieval_units import unit_identity
from mm_memory_bench.unit_evidence import UnitEvidenceCatalog, selected_evidence_ids
from scripts.build_graphmemix_relation_store import canonical_unit_rows, reconstruct_vectors
from scripts.graphmemix_core import candidate_pool, memory_snippet, source_score_map, missing_atomic_floor
from scripts.run_memix_answers_from_retrieval import (hydrate_reader, generate_text_only_answer,
                                                    generate_action_aware_answer)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, values):
    path.write_text(''.join(json.dumps(row) + '\n' for row in values))


class Embedder:
    dimension = 2
    model_identity = 'integration-embedding-1'

    def __init__(self, *args):
        self.seen = []

    def encode_units(self, units):
        self.seen.extend(deepcopy(units))
        return np.asarray([[1., 0.] if 'vermilion' in u.get('text', '') else [0., 1.]
                           for u in units], dtype=np.float32)


class Model:
    messages = []
    last_input_tokens = 100
    last_completion_tokens = 10
    last_cached_input_tokens = 0
    last_preflight_tokens = 100
    last_preflight_source = 'fixture'
    last_tokenize_error = None
    last_finish_reason = 'stop'

    def __init__(self, config=None, *args, **kwargs):
        self.config = config or GenerationConfig()

    def complete(self, messages, **kwargs):
        self.messages.append(deepcopy(messages))
        system = messages[0].get('content', '') if messages[0]['role'] == 'system' else ''
        if 'task=observe.' in system:
            payload = json.loads(messages[1]['content'][0]['text'])
            assert 'focus' not in payload and 'question' not in payload
            return json.dumps({'coordinate_system': 'normalized_1000', 'observations': [{
                'description': 'A vermilion cup.', 'regions': [[100, 200, 700, 800]],
                'source_refs': [], 'fact': {'subject': 'cup1', 'predicate': 'color',
                                           'value': 'vermilion', 'basis': 'visual'}}]})
        if 'task=ground_observations.' in system:
            payload = json.loads(messages[1]['content'][0]['text'])
            return json.dumps({'checks': [{'id': oid, 'visible_supported': True,
                'requires_text': False, 'text_supported': False} for oid in payload['proposed_ids']]})
        if 'evidence candidates' in system:
            payload = json.loads(messages[1]['content'])
            return json.dumps({'selected': [{'id': c['id'], 'score': 5} for c in payload['candidates']]})
        if 'evidence-chain verifier' in system:
            payload = json.loads(messages[1]['content'])
            return json.dumps({'candidates': [{'id': c['id'], 'direct_support': 0,
                'best_anchor_id': c['eligible_anchor_relations'][0]['anchor_id'],
                'incremental_support': 3, 'role': 'new_fact'} for c in payload['candidates']]})
        return 'The cup is vermilion.'


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'bundle'
    root.mkdir()
    image = root / 'photo.png'
    Image.new('RGB', (200, 80), 'red').save(image)
    (root / 'manifest.json').write_text('{}')
    write_rows(root / 'assets.jsonl', [{'asset_id': 'a1', 'path': str(image)}])
    write_rows(root / 'contexts.jsonl', [{'context_id': 'ctx'}])
    write_rows(root / 'memories.jsonl', [
        {'context_id': 'ctx', 'memory_id': 'm0', 'session_id': 's', 'sequence': 0,
         'timestamp': '2025-04-28',
         'source_id': 'source0', 'content': [{'type': 'image', 'asset_id': 'a1'},
         {'type': 'text', 'text': 'This is my cup.'}],
         'metadata': {'derived': {'caption': 'A cup on a table.'}, 'answer': 'PRIVATE'}},
        *[{'context_id': 'ctx', 'memory_id': f'm{i}', 'sequence': i, 'session_id': 's',
           'source_id': f'source{i}', 'content': [{'type': 'text', 'text': f'Unrelated entry {i}.'}]}
          for i in range(1, 12)]])
    write_rows(root / 'questions.jsonl', [{'context_id': 'ctx', 'question_id': 'q1',
        'memory_scope': {'mode': 'all'}, 'prompt': [{'type': 'text', 'text': 'Which cup is vermilion?'}],
        'answer': {'text': 'PRIVATE'}, 'evidence': [{'memory_id': 'm0'}]}])
    Model.messages = []
    return root


def make_snapshot(bundle, tmp_path, *, many=False, textual=False, count=None):
    reader = BundleReader(bundle)
    memories = raw_memories(reader)
    catalog = SourceCatalog(memories)
    asset = catalog.assets['a1']
    refs = [catalog.image_node('a1', 'm0').node_id]
    if textual:
        refs += [n.node_id for n in catalog.nodes.values() if n.kind == 'text' and n.memory_id == 'm0']
    observations = [Observation(f'o{i:03}', 'm0', 'a1', ((20, 16, 140, 64),),
        f'A vermilion cup {i}.', tuple(refs), 1,
        Fact(f'a1:cup{i}', 'color', 'vermilion', 'textual' if textual else 'visual'), asset['sha256'])
        for i in range(count if count is not None else 100 if many else 1)]
    data = [{'context_id': 'ctx', 'memory_id': m['memory_id'],
             'come': export_field(m, catalog, observations)} for m in memories]
    path = tmp_path / 'observations.jsonl'
    write_snapshot(path, data, extraction_contract={'model': 'fixture'})
    return reader, path, ObservationSnapshot.from_bundle(path, reader)


def invoke(monkeypatch, module, *args):
    monkeypatch.setattr(sys, 'argv', [module.__name__, *map(str, args)])
    return module.main()


def test_offline_runner_does_not_read_questions_or_load_embeddings(bundle, tmp_path, monkeypatch):
    from mm_memory_bench.come import runner
    (bundle / 'questions.jsonl').unlink()
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'come': {}, 'observations': {'max_output_tokens': 512}}))
    monkeypatch.setattr(runner, 'OpenAICompatibleQwenVL', Model)
    monkeypatch.setattr(BundleReader, 'iter_context_batches', lambda *a, **kw: pytest.fail('read questions'))
    output = tmp_path / 'observations.jsonl'
    args = ['--config', config, '--bundle', bundle, '--artifact-root', tmp_path / 'extract',
            '--output', output, '--model', 'extractor', '--base-url', 'http://unused']
    assert invoke(monkeypatch, runner, *args) == 0
    assert len(Model.messages) == 2
    obs = rows(output)[0]['come']['observations'][0]
    assert obs['regions'] == [[20, 16, 140, 64]]
    assert obs['coordinate_system'] == 'oriented_pixel_xyxy'
    assert len(rows(output)) == 12  # Includes contexts/memories with no questions.
    assert rows(output)[1]['come']['observations'] == []
    before = file_digest(output)
    config.write_text(json.dumps({'come': {}, 'observations': {'max_output_tokens': 512},
                                 'benchmark_suite': {'reader_k': 999, 'embedding_model': 'different'}}))
    write_rows(bundle / 'questions.jsonl', [{'question_id': 'new questions must not affect extraction'}])
    assert invoke(monkeypatch, runner, *args) == 0
    assert file_digest(output) == before and len(Model.messages) == 2
    # A tightened grounding prompt must invalidate earlier extraction results,
    # even though the source bytes and output schema have not changed.
    manifest = manifest_path(output)
    metadata = json.loads(manifest.read_text())
    metadata['extraction_contract']['model_protocol'] = 'previous-grounding-protocol'
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='incompatible observation extraction contract'):
        invoke(monkeypatch, runner, *args)
    assert len(Model.messages) == 2
    assert set(inspect.signature(ObservationExtractor.observe).parameters) == {'self', 'image', 'store'}
    assert not hasattr(ObservationPreprocessor, 'answer')


def test_parallel_extraction_keeps_memory_local_sources_and_resumes_failed_shard(bundle, tmp_path, monkeypatch):
    from mm_memory_bench.come import runner
    memories = rows(bundle / 'memories.jsonl')
    memories[1]['content'] = deepcopy(memories[0]['content'])
    write_rows(bundle / 'memories.jsonl', memories)
    (bundle / 'questions.jsonl').unlink()
    config = tmp_path / 'parallel-config.json'
    config.write_text(json.dumps({'come': {}, 'observations': {'max_output_tokens': 512}}))
    monkeypatch.setattr(runner, 'OpenAICompatibleQwenVL', Model)
    output = tmp_path / 'parallel.jsonl'
    args = ['--config', config, '--bundle', bundle, '--artifact-root', tmp_path / 'parallel-extract',
            '--output', output, '--model', 'extractor', '--base-url', 'http://unused', '--workers', 4]
    ingest = ObservationPreprocessor.ingest
    def fail_one(service):
        if 'm0' in service.catalog.memories:
            raise RuntimeError('interrupted shard')
        return ingest(service)
    with monkeypatch.context() as patch:
        patch.setattr(ObservationPreprocessor, 'ingest', fail_one)
        with pytest.raises(RuntimeError, match='interrupted shard'):
            invoke(monkeypatch, runner, *args)
    assert not output.exists() and not manifest_path(output).exists()
    assert len(Model.messages) == 2  # The other image shard completed and persisted.
    assert invoke(monkeypatch, runner, *args) == 0
    assert len(Model.messages) == 4  # Successful shards were not extracted twice.
    data = rows(output)
    assert {row['memory_id'] for row in data} == {m['memory_id'] for m in memories}
    assert sum(len(row['come']['observations']) for row in data) == 2
    for row in data:
        for obs in row['come']['observations']:
            assert all(ref['memory_id'] == row['memory_id'] for ref in obs['source_refs'])
    ObservationSnapshot.from_bundle(output, BundleReader(bundle))
    assert invoke(monkeypatch, runner, *args) == 0 and len(Model.messages) == 4
    serial = tmp_path / 'serial.jsonl'
    assert invoke(monkeypatch, runner, '--config', config, '--bundle', bundle,
                  '--artifact-root', tmp_path / 'serial-extract', '--output', serial,
                  '--model', 'extractor', '--base-url', 'http://unused') == 0
    assert output.read_bytes() == serial.read_bytes()


@pytest.mark.parametrize('change, error', [
    ('incomplete', 'incomplete'), ('text', 'digest'), ('asset', 'source changed'),
    ('version', 'incompatible'), ('coordinate', 'coordinate'), ('ref', 'memory-local'),
])
def test_sidecar_rejects_stale_incomplete_or_unresolvable_sources(bundle, tmp_path, change, error):
    reader, path, snapshot = make_snapshot(bundle, tmp_path)
    data = rows(path)
    if change == 'asset':
        Image.new('RGB', (200, 80), 'blue').save(bundle / 'photo.png')
    elif change == 'incomplete':
        meta = json.loads(manifest_path(path).read_text()); meta['complete'] = False
        manifest_path(path).write_text(json.dumps(meta))
    elif change == 'text':
        data[0]['come']['observations'][0]['description'] = 'modified'
        write_rows(path, data)
    else:
        if change == 'version':
            data[0]['come']['schema_version'] = 2
        elif change == 'coordinate':
            data[0]['come']['observations'][0]['coordinate_system'] = 'normalized_1000'
        else:
            data[0]['come']['observations'][0]['source_refs'][0]['memory_id'] = 'm1'
        write_snapshot(path, data, extraction_contract={'model': 'fixture'})
    with pytest.raises(ValueError, match=error):
        ObservationSnapshot.from_bundle(path, reader)


def test_grounded_text_is_atomic_and_total_snippet_budget_is_preserved(bundle, tmp_path):
    reader, path, snapshot = make_snapshot(bundle, tmp_path, textual=True)
    memory = snapshot.attach(rows(bundle / 'memories.jsonl')[0])
    text, ids = render_observations(memory, max_chars=1200)
    assert 'This is my cup.' in text and 'o000' in text and ids == ['o000']
    assert len(memory_snippet(memory, 420, include_observations=True)) <= 420
    assert 'vermilion' in memory_snippet(memory, 420, include_observations=True)
    assert render_observations(memory, max_chars=30) == ('', [])
    missing = deepcopy(memory)
    missing['metadata']['come']['observations'][0]['source_refs'][1]['text'] = ''
    assert render_observations(missing, max_chars=1200) == ('', [])
    long = deepcopy(memory)
    long['metadata']['come']['observations'][0]['description'] *= 300
    assert render_observations(long, max_chars=1200) == ('', [])
    # Caller mutation cannot alter the shared read-only snapshot.
    memory['metadata']['come']['observations'].clear()
    assert snapshot.attach(rows(bundle / 'memories.jsonl')[0])['metadata']['come']['observations']


def test_real_index_preserves_observation_hits_and_separate_parent_vectors(bundle, tmp_path):
    reader, path, snapshot = make_snapshot(bundle, tmp_path, many=True)
    method = ConcreteMemixMethod(GenerationConfig(top_k=1), backend='standalone',
        answer_model=Model(), embedder=Embedder(), index=FaissFlatIPIndex(2),
        checkpoint_dir=tmp_path / 'checkpoint', observation_contract=snapshot.contract,
        source_top_k=10, ingest_batch_size=1)
    method.begin_context({'context_id': 'ctx'})
    base = ConcreteMemixMethod(GenerationConfig(top_k=1), backend='standalone',
        answer_model=Model(), embedder=Embedder(), index=FaissFlatIPIndex(2), source_top_k=10)
    base.begin_context({'context_id': 'ctx'})
    for memory in rows(bundle / 'memories.jsonl'):
        resolved = _resolved_memory(reader, memory, memory_view='derived')
        original_record, original_units = base._adapt_memory(resolved)
        enhanced = _resolved_memory(reader, memory, memory_view='derived', observations=snapshot)
        record, units = method._adapt_memory(enhanced)
        assert record['text'] == original_record['text']
        assert units[:len(original_units)] == original_units
        method.ingest(enhanced)
    detail = method._source_detail({'prompt': [{'type': 'text', 'text': 'vermilion'}]})
    assert detail['retrieval_ids'][0] == 'm0'
    assert len(detail['retrieval_ids']) == len(set(detail['retrieval_ids'])) == 10
    assert detail['retrieval_scores'][0] == pytest.approx(1.)
    assert len(detail['observation_hits']['m0']) == 100
    assert detail['unit_hits'][0]['kind'] == 'observation'
    assert detail['unit_hits'][0]['observation_id'] == 'o000'
    observed = [unit for unit in method.embedder.seen if unit.get('observation_id')]
    assert all(unit['image'] == str(bundle / 'photo.png') and
               unit['image_region'] == [20, 16, 140, 64] for unit in observed)
    visible = method._source_detail({'prompt': [{'type': 'text', 'text': 'vermilion'}],
                                     'visible_memory_ids': ['m1', 'm2']})
    assert set(visible['retrieval_ids']) == {'m1', 'm2'} and not visible['observation_hits']
    assert all(len(ids) == 1 for ids in canonical_unit_rows(method._embedding_units).values())
    method.end_context()
    state_path = next((tmp_path / 'checkpoint').glob('contexts/*/state.json'))
    state = json.loads(state_path.read_text())
    _, vectors = reconstruct_vectors(state_path.parent / state['index_file'], state)
    assert np.allclose(vectors, [[0., 1.]] * 12)  # Parent vectors remain separate from observation vectors.
    before = file_digest(path), file_digest(state_path), file_digest(state_path.parent / state['index_file'])
    hydrated = hydrate_reader(checkpoint_state=state_path, memix_repo=tmp_path / 'unused',
        backend='standalone', generation=GenerationConfig(top_k=1), answer_model=Model(),
        source_top_k=10, public_source_ids={}, public_asset_paths={}, observations=snapshot)
    q = {'prompt': [{'type': 'text', 'text': 'Which cup?'}]}
    hydrated._generate_answer(q, ['m0'])
    generate_text_only_answer(hydrated, q, ['m0'])
    generate_action_aware_answer(hydrated, q, ['m0'], {'m0': 'text'}, low_image_max_edge=448)
    assert all('vermilion' in json.dumps(m) for m in Model.messages[-3:])
    assert before == (file_digest(path), file_digest(state_path), file_digest(state_path.parent / state['index_file']))
    assert not any('task=observe' in json.dumps(m) for m in Model.messages)
    with pytest.raises(ValueError, match='snapshot'):
        hydrate_reader(checkpoint_state=state_path, memix_repo=tmp_path, backend='standalone',
                       generation=GenerationConfig(), answer_model=Model(), source_top_k=10,
                       public_source_ids={}, public_asset_paths={})


def test_visibility_limits_graph_expansion(bundle):
    memories = rows(bundle / 'memories.jsonl')
    adjacency = {'m0': [('m11', 1., 'future'), ('m1', .9, 'visible')]}
    for scope in [{'mode': 'prefix', 'max_sequence': 1}, {'mode': 'ids', 'memory_ids': ['m0', 'm1']}]:
        visible = visible_ids({'context_id': 'ctx', 'memory_scope': scope}, memories)
        assert candidate_pool({'retrieval_ids': ['m0']}, adjacency, source_top_l=1,
                              candidate_limit=2, visible_memory_ids=visible) == ['m0', 'm1']
    assert candidate_pool({'retrieval_ids': ['m0']}, adjacency, source_top_l=1,
                          candidate_limit=1) == ['m0']


def test_unit_evidence_preserves_same_parent_identity_scope_and_scores(bundle, tmp_path):
    _, _, snapshot = make_snapshot(bundle, tmp_path, count=3)
    catalog = UnitEvidenceCatalog({m['memory_id']: snapshot.attach(m) for m in rows(bundle / 'memories.jsonl')})
    parent, first, second = unit_identity('m0'), unit_identity('m0', 'o000'), unit_identity('m0', 'o001')
    ids = [second, parent, first]
    selected = catalog.selection(ids)
    assert selected_evidence_ids(selected, unit_level=True) == ids
    assert selected['retrieved_memory_ids'] == ['m0']  # Benchmark reporting only.
    assert catalog.records[parent]['metadata']['come']['observations'] == []
    assert [o['id'] for o in catalog.records[first]['metadata']['come']['observations']] == ['o000']
    snippet = memory_snippet(catalog.records[first], include_observations=True)
    assert 'o000' in snippet and 'o001' not in snippet and 'This is my cup.' not in snippet
    for invalid in ([first, first], [unit_identity('unknown')]):
        with pytest.raises(ValueError):
            catalog.selection(invalid)
    with pytest.raises(ValueError, match='scope'):
        catalog.require([first], {'context_id': 'ctx', 'memory_scope': {'mode': 'ids', 'memory_ids': ['m1']}})
    with pytest.raises(ValueError, match='lacks unit IDs'):
        selected_evidence_ids({'retrieved_memory_ids': ['m0']}, unit_level=True)
    prior = {'retrieval_ids': ['m0'], 'retrieval_scores': [.9],
             'unit_hits': [{'unit_id': first, 'score': .9}, {'unit_id': second, 'score': .4}]}
    assert source_score_map(prior, unit_level=True) == {first: .9, second: .4}
    assert missing_atomic_floor(prior, unit_level=True) == pytest.approx(.39)
    assert missing_atomic_floor(prior) == pytest.approx(.89)


def test_source_context_uses_validated_raw_text_and_separate_bounded_budget(bundle, tmp_path):
    from mm_memory_bench.observation_context import SOURCE_CONTEXT_TEXT_CHARS
    data = rows(bundle / 'memories.jsonl')
    original = 'This is the final ring/card screen. ' + 'Additional source detail. ' * 80
    data[0]['content'][1]['text'] = original
    write_rows(bundle / 'memories.jsonl', data)
    reader, _, snapshot = make_snapshot(bundle, tmp_path)
    # Derived views contain extra caption text. Context must be reconstructed
    # from the validated raw sources rather than copied from that view.
    derived = _resolved_memory(reader, data[0], memory_view='derived', observations=snapshot)
    catalog = UnitEvidenceCatalog({'m0': derived})
    unit = catalog.records[unit_identity('m0', 'o000')]
    context = unit['source_context']
    assert context['original_text'] == original[:SOURCE_CONTEXT_TEXT_CHARS]
    assert context['original_text_truncated'] is True
    assert context['recorded_timestamp'] == '2025-04-28'
    assert 'A cup on a table.' not in json.dumps(context) and 'PRIVATE' not in json.dumps(context)
    assert 'vermilion' in memory_snippet(unit, include_observations=True)
    assert len(memory_snippet(unit, include_observations=True)) <= 420
    context['original_text'] = 'mutated'
    assert snapshot.attach(data[0])['observation_source_context']['original_text'] != 'mutated'


@pytest.mark.parametrize('mode', ['normal', 'text', 'actions'])
def test_reader_receives_only_selected_units_and_their_crops(bundle, tmp_path, mode):
    reader, _, snapshot = make_snapshot(bundle, tmp_path, count=3)
    method = ConcreteMemixMethod(GenerationConfig(top_k=2), backend='standalone',
        answer_model=Model(), embedder=Embedder(), index=FaissFlatIPIndex(2),
        checkpoint_dir=tmp_path / 'checkpoint', observation_contract=snapshot.contract)
    method.begin_context({'context_id': 'ctx'})
    for memory in rows(bundle / 'memories.jsonl'):
        method.ingest(_resolved_memory(reader, memory, memory_view='derived', observations=snapshot))
    method.end_context()
    state_path = next((tmp_path / 'checkpoint').glob('contexts/*/state.json'))
    # Checkpoint paths may refer to another machine; asset rebasing must also
    # apply to observation crops, independently of the parent memory view.
    state = json.loads(state_path.read_text())
    for unit in state['embedding_units']:
        if unit.get('observation_id'):
            unit['image'] = '/unavailable/old-machine.png'
    state_path.write_text(json.dumps(state))
    hydrated = hydrate_reader(checkpoint_state=state_path, memix_repo=tmp_path / 'unused',
        backend='standalone', generation=GenerationConfig(top_k=2), answer_model=Model(),
        source_top_k=10, public_source_ids={}, public_asset_paths={'a1': str(bundle / 'photo.png')},
        observations=snapshot)
    ids = [unit_identity('m0', 'o001'), unit_identity('m0', 'o000')]
    question = {'prompt': [{'type': 'text', 'text': 'Which cup?'}]}
    if mode == 'normal':
        hydrated._generate_answer(question, ids)
    elif mode == 'text':
        generate_text_only_answer(hydrated, question, ids)
    else:
        generate_action_aware_answer(hydrated, question, ids, {ids[0]: 'low', ids[1]: 'high'},
                                     low_image_max_edge=40)
    content = Model.messages[-1][0]['content']
    blocks = content if isinstance(content, list) else [{'type': 'text', 'text': content}]
    text = '\n'.join(p['text'] for p in blocks if p['type'] == 'text')
    assert text.index(f'unit_id={ids[0]}') < text.index(f'unit_id={ids[1]}')
    assert text.count('Evidence ') == 2 and 'A vermilion cup 1.' in text and 'A vermilion cup 0.' in text
    assert 'This is my cup.' in text
    assert 'recorded_timestamp' in text and '2025-04-28' in text and 'session_id' in text
    assert 'o002' not in text and 'A cup on a table.' not in text and 'PRIVATE' not in text
    sizes = []
    for part in blocks:
        if part['type'] == 'image_url':
            with Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',', 1)[1]))) as pixels:
                sizes.append(pixels.size)
    assert sizes == ([] if mode == 'text' else [(40, 16), (120, 48)] if mode == 'actions' else [(120, 48)] * 2)
    parent = unit_identity('m0')
    hydrated._generate_answer(question, [parent, ids[0]])
    text = json.dumps(Model.messages[-1])
    assert 'This is my cup.' in text and 'o001' in text
    assert 'o000' not in text and 'o002' not in text  # No auto-added sibling observations.


@pytest.mark.parametrize('observation_only,final_k', [(False, 3), (True, 2), (False, 20)])
def test_cli_pipeline_fields_reach_actual_messages_and_caches_are_read_only(bundle, tmp_path, monkeypatch, observation_only, final_k):
    observation_count = 3 if observation_only else 24 if final_k == 20 else 1
    reader, sidecar, snapshot = make_snapshot(bundle, tmp_path, count=observation_count)
    build = importlib.import_module('scripts.build_memix_checkpoints')
    priors = importlib.import_module('scripts.run_memix_main_track_retrieval')
    relations = importlib.import_module('scripts.build_graphmemix_relation_store')
    node = importlib.import_module('scripts.score_graphmemix_verifier')
    ecv = importlib.import_module('scripts.score_graphmemix_ecv')
    select = importlib.import_module('scripts.evaluate_graphmemix_ecv')
    prune = importlib.import_module('scripts.evaluate_graphmemix_exact_pruning')
    answer = importlib.import_module('scripts.run_memix_answers_from_retrieval')
    monkeypatch.setattr(build, 'GMEQwen2VLEmbedder', Embedder)
    monkeypatch.setattr(priors, 'GMEQwen2VLEmbedder', Embedder)
    monkeypatch.setattr(node, 'OpenAICompatibleQwenVL', Model)
    monkeypatch.setattr(ecv, 'ChatClient', Model)
    monkeypatch.setattr(answer, 'OpenAICompatibleQwenVL', Model)
    monkeypatch.setattr(ObservationExtractor, 'observe', lambda *a, **k: pytest.fail('query extracted observations'))
    monkeypatch.setattr(ObservationExtractor, 'call', lambda *a, **k: pytest.fail('query grounded observations'))
    monkeypatch.setattr(SourceCatalog, '__init__', lambda *a, **k: pytest.fail('query rebuilt old CoME nodes'))
    checkpoint = tmp_path / 'checkpoint'
    model_path = tmp_path / 'embedding'
    model_path.mkdir(); (model_path / 'config.json').write_text('{}')
    common = ['--bundle', bundle, '--observations', sidecar]
    assert invoke(monkeypatch, build, *common, '--checkpoint-root', checkpoint,
                  '--embedding-model', model_path, '--backend', 'standalone', '--memory-view', 'derived') == 0
    state = next(checkpoint.glob('contexts/*/state.json'))
    before = {p: file_digest(p) for p in [sidecar, manifest_path(sidecar), state, *checkpoint.rglob('*.faiss')]}
    prior_path, graph, nodes, edges = [tmp_path / f'{n}.jsonl' for n in ['priors', 'relations', 'node', 'ecv']]
    assert invoke(monkeypatch, priors, *common, '--checkpoint-root', checkpoint,
        '--embedding-model', model_path, '--backend', 'standalone', '--processes', 1,
        '--source-top-k', 10, '--output', tmp_path / 'retrieval.jsonl', '--source-priors', prior_path) == 0
    assert rows(prior_path)[0]['retrieval_ids'][0] == 'm0'
    assert rows(prior_path)[0]['observations'] == snapshot.contract
    assert rows(prior_path)[0]['unit_hits'][0]['observation_id'] == 'o000'
    searched_nodes = []
    original_neighbors = relations.mutual_neighbors
    def track_neighbors(vectors, k):
        searched_nodes.append(len(vectors))
        return original_neighbors(vectors, k)
    monkeypatch.setattr(relations, 'mutual_neighbors', track_neighbors)
    assert invoke(monkeypatch, relations, *common, '--checkpoint-root', checkpoint, '--output', graph) == 0
    graphs = [row for row in rows(graph) if row.get('record_type') == 'unit_graph']
    assert searched_nodes == [12 + observation_count]
    assert len(graphs) == 1 and len(graphs[0]['units']) == searched_nodes[0]
    assert any(edge.get('semantic_similarity') is not None for edge in graphs[0]['edges'])
    before[graph] = file_digest(graph)
    node_args = [*common, '--source-priors', prior_path, '--relation-store', graph, '--graph-mode', 'full',
                 '--source-top-l', 24 if final_k == 20 else 2,
                 '--candidate-limit', 48 if final_k == 20 else 2 if observation_only else 4,
                 '--output', nodes, '--all-questions', '--concurrency', 1]
    assert invoke(monkeypatch, node, *node_args) == 0
    assert rows(nodes)[0]['observations'] == snapshot.contract
    assert rows(nodes)[0]['unit_graph']['seed_unit_ids'][0] == '["m0","o000"]'
    assert rows(nodes)[0]['candidate_ids'] == rows(nodes)[0]['unit_graph']['candidate_unit_ids']
    if observation_only:
        assert rows(nodes)[0]['candidate_ids'] == [unit_identity('m0', 'o000'), unit_identity('m0', 'o001')]
        assert 'o002' not in json.dumps(Model.messages[-1])
    assert 'o000' in json.dumps(Model.messages[-1]) and 'vermilion' in json.dumps(Model.messages[-1])
    node_payload = json.loads(Model.messages[-1][1]['content'])
    unit_context = next(c['source_context'] for c in node_payload['candidates']
                        if c.get('unit_id') == unit_identity('m0', 'o000'))
    assert unit_context['recorded_timestamp'] == '2025-04-28'
    assert unit_context['session_id'] == 's'
    assert unit_context['original_text'] == 'This is my cup.'
    assert invoke(monkeypatch, ecv, *common, '--candidate-cache', nodes, '--relation-store', graph,
                  '--output', edges, '--anchor-mode', 'atomic', '--anchor-count', 2, '--edge-only', '--concurrency', 1) == 0
    assert rows(edges)[0]['observations'] == snapshot.contract
    assert 'o000' in json.dumps(Model.messages[-1])
    if not observation_only:  # Observation-only candidates have no explicit edges to verify.
        edge_payload = json.loads(Model.messages[-1][1]['content'])
        if final_k == 3:
            assert any(c.get('source_context') == unit_context for c in edge_payload['candidates'])
        assert any(c.get('source_context') == unit_context for c in edge_payload['anchor_evidence'])
    calibration = Path(__file__).resolve().parents[1] / 'configs/release/calibration/shared_dev.json'
    assert invoke(monkeypatch, select, *common, '--ecv-scores', edges, '--original-verifier', nodes,
        '--source-priors', prior_path, '--relation-store', graph, '--calibration', calibration,
        '--output-dir', tmp_path / 'selection', '--edge-power', 1, '--k', final_k) == 0
    fixed = tmp_path / 'selection/original_ecv_edge_p1.jsonl'
    assert invoke(monkeypatch, prune, *common, '--scores', nodes, '--ecv-scores', edges,
        '--source-priors', prior_path, '--retrieval', fixed, '--calibration', calibration,
        '--output-dir', tmp_path / 'pruning', '--root-cost', .12) == 0
    final = tmp_path / 'pruning/root0.12_node0.jsonl'
    predictions = tmp_path / 'predictions.jsonl'
    answer_args = [*common, '--retrieval', final, '--checkpoint-root', checkpoint, '--backend', 'standalone',
                   '--output', predictions, '--reader-media-mode', 'text', '--allow-variable-retrieval',
                   '--concurrency', 1, '--resume']
    assert invoke(monkeypatch, answer, *answer_args) == 0
    result = rows(predictions)[0]
    assert result['metadata']['query_added_observations'] == 0
    assert result['metadata']['observation_ids'][unit_identity('m0', 'o000')] == ['o000']
    assert result['metadata']['retrieved_unit_ids'] == rows(final)[0]['retrieved_unit_ids']
    assert set(rows(final)[0]['retrieved_unit_ids']) <= set(rows(fixed)[0]['retrieved_unit_ids'])
    if final_k == 20:
        assert len(result['metadata']['retrieved_unit_ids']) == 20
        assert 'R@20' in json.loads((tmp_path / 'pruning/summary.json').read_text())['variants']['root0.12_node0']
    if observation_only:
        expected = [unit_identity('m0', 'o000'), unit_identity('m0', 'o001')]
        assert rows(nodes)[0]['candidate_ids'] == rows(edges)[0]['candidate_ids'] == expected
        assert rows(fixed)[0]['retrieved_unit_ids'] == rows(final)[0]['retrieved_unit_ids'] == expected
        assert result['metadata']['retrieved_unit_ids'] == expected
        assert result['retrieved_memory_ids'] == ['m0']  # Source-level benchmark reporting only.
        assert 'o001' in json.dumps(Model.messages[-1]) and 'o002' not in json.dumps(Model.messages[-1])
        assert 'This is my cup.' in json.dumps(Model.messages[-1])
        assert '2025-04-28' in json.dumps(Model.messages[-1])
    assert 'o000' in json.dumps(Model.messages[-1]) and 'PRIVATE' not in json.dumps(Model.messages)
    calls = len(Model.messages)
    assert invoke(monkeypatch, answer, *answer_args) == 0
    assert len(Model.messages) == calls
    assert before == {p: file_digest(p) for p in before}
    with monkeypatch.context() as patch:
        patch.setattr(node, 'rendering_contract', lambda: {'protocol': 'changed'})
        with pytest.raises(RuntimeError, match='incompatible'):
            invoke(monkeypatch, node, *node_args)
    with monkeypatch.context() as patch:
        patch.setattr(answer, 'rendering_contract', lambda: {'protocol': 'changed'})
        with pytest.raises(ValueError, match='rendering'):
            invoke(monkeypatch, answer, *answer_args)
    assert len(Model.messages) == calls
    # A new valid snapshot cannot be mixed with old priors, checkpoints or predictions.
    data = rows(sidecar); data[0]['come']['observations'][0]['description'] += ' Changed.'
    write_snapshot(sidecar, data, extraction_contract={'model': 'fixture'})
    with pytest.raises(ValueError, match='different observation'):
        invoke(monkeypatch, node, *node_args)
    with pytest.raises(ValueError, match='different observation'):
        invoke(monkeypatch, answer, *answer_args)


def test_exif_rotated_source_is_converted_once_and_crop_uses_oriented_pixels(tmp_path):
    from mm_memory_bench.come.store import CropCache
    image = Image.new('RGB', (200, 80), 'red')
    exif = image.getexif(); exif[274] = 6
    path = tmp_path / 'rotated.jpg'; image.save(path, exif=exif)
    memories = [{'context_id': 'ctx', 'memory_id': 'm0',
                 'content': [{'type': 'image', 'asset_id': 'a1', 'path': str(path)}]}]
    service = ObservationPreprocessor(memories, tmp_path / 'extract', Model())
    data = service.ingest()
    obs = data[0]['come']['observations'][0]
    assert obs['image_size'] == [80, 200]
    assert obs['regions'] == ((8, 40, 56, 160),)
    sidecar = tmp_path / 'observations.jsonl'
    write_snapshot(sidecar, data, extraction_contract=service.contract)
    restored = ObservationSnapshot(sidecar, memories).attach(memories[0])
    region = restored['metadata']['come']['observations'][0]['regions']
    crop = CropCache(tmp_path / 'crops', service.catalog).image('a1', region)
    with Image.open(crop) as pixels:
        assert pixels.size == (48, 120)


def test_valid_empty_observation_field_preserves_baseline_inputs_and_messages(bundle, tmp_path):
    reader, path, snapshot = make_snapshot(bundle, tmp_path)
    data = rows(path)
    for row in data:
        row['come']['observations'] = []
    write_snapshot(path, data, extraction_contract={'model': 'fixture'})
    snapshot = ObservationSnapshot.from_bundle(path, reader)
    raw = rows(bundle / 'memories.jsonl')[0]
    plain = _resolved_memory(reader, raw, memory_view='derived')
    enhanced = _resolved_memory(reader, raw, memory_view='derived', observations=snapshot)
    base = ConcreteMemixMethod(GenerationConfig(top_k=1), backend='standalone',
        answer_model=Model(), embedder=Embedder(), index=FaissFlatIPIndex(2), source_top_k=10)
    extra = ConcreteMemixMethod(GenerationConfig(top_k=1), backend='standalone',
        answer_model=Model(), embedder=Embedder(), index=FaissFlatIPIndex(2), source_top_k=10,
        observation_contract=snapshot.contract)
    for method in [base, extra]:
        method.begin_context({'context_id': 'ctx'})
    baseline_record, baseline_units = base._adapt_memory(plain)
    extra_record, extra_units = extra._adapt_memory(enhanced)
    assert baseline_units == extra_units
    assert base._reader_text(baseline_record) == extra._reader_text(extra_record)
    assert memory_snippet(raw) == memory_snippet(snapshot.attach(raw), include_observations=True)
    # Disabled metadata ingress still drops private and unvalidated extension data.
    forged = deepcopy(raw); forged['metadata']['come'] = {'question': 'PRIVATE'}
    assert _resolved_memory(reader, forged, memory_view='derived') == plain
    base.ingest(plain); extra.ingest(enhanced)
    base._flush_pending(); extra._flush_pending()
    question = {'prompt': [{'type': 'text', 'text': 'cup'}]}
    a, b = base._source_detail(question), extra._source_detail(question)
    assert a['retrieval_ids'] == b['retrieval_ids'] and a['retrieval_scores'] == b['retrieval_scores']
    base._generate_answer(question, ['m0']); extra._generate_answer(question, ['m0'])
    assert Model.messages[-1] == Model.messages[-2]


def test_multiview_retrieval_does_not_stop_when_only_one_view_has_enough_memories(tmp_path):
    # View 1's first 80 rows all belong to m0. The union with view 2 contains
    # ten memories, but an unseen m1 at .9 must beat view 2's .5 candidates.
    class Views:
        dimension = 2
        def encode_units(self, units):
            return np.eye(2, dtype=np.float32)
    class ScoredIndex:
        def reset(self):
            pass
        def search(self, queries, top_k):
            values = np.asarray([[1.] * 100 + [.9] + [.1] * 9,
                                 [.0] * 100 + [.0] + [.5] * 9], dtype=np.float32)
            order = np.argsort(-values, axis=1, kind='stable')[:, :top_k]
            return np.take_along_axis(values, order, axis=1), order
    method = ConcreteMemixMethod(GenerationConfig(top_k=1), backend='standalone',
        answer_model=Model(), embedder=Views(), index=ScoredIndex(), source_top_k=10,
        observation_contract={'fixture': True})
    method._embedding_units = ([{'memory_id': 'm0'}] * 100 + [{'memory_id': f'm{i}'} for i in range(1, 11)])
    detail = method._source_detail({'prompt': [{'type': 'text', 'text': 'two views'}]})
    assert detail['retrieval_ids'][:2] == ['m0', 'm1']
    assert detail['retrieval_scores'][:2] == pytest.approx([1., .9])


def test_release_stages_resume_and_disabled_config_match_legacy(tmp_path, monkeypatch):
    from scripts import run_graphmemix_release as release
    root = Path(__file__).resolve().parents[1]
    enhanced = json.loads((root / 'configs/release/graphmemix.json').read_text())
    legacy = json.loads((root / 'configs/release/graphmemix_legacy.json').read_text())
    # Independent units now have a larger final evidence budget than the
    # frozen memory-level baseline. Other retrieval settings remain comparable.
    assert enhanced['benchmark_suite']['reader_k'] == 20
    assert legacy['benchmark_suite']['reader_k'] == 10
    assert {k: v for k, v in enhanced['benchmark_suite'].items() if k not in {'judge', 'reader_k'}} == {
        k: v for k, v in legacy['benchmark_suite'].items() if k not in {'judge', 'reader_k'}}
    commands = []
    monkeypatch.setattr(release.Runner, 'run', lambda self, command, **kwargs: commands.append(command))
    common = ['benchmark', '--dataset', 'atm', '--backbone', 'qwen3vl8b', '--to-stage', 'reader', '--dry-run']
    assert invoke(monkeypatch, release, *common) == 0
    assert commands[0][1:3] == ['-m', 'mm_memory_bench.come.runner']
    assert commands[1][1].endswith('build_memix_checkpoints.py')
    assert len(commands) == 9
    assert all('--observations' in command for command in commands[1:])
    assert commands[-1][1].endswith('run_memix_answers_from_retrieval.py')
    assert '--memory-view' in commands[1] and 'derived' in commands[1]
    selection_command = next(c for c in commands if c[1].endswith('evaluate_graphmemix_ecv.py'))
    assert selection_command[selection_command.index('--k') + 1] == '20'
    extraction_command = deepcopy(commands[0])
    commands.clear()
    assert invoke(monkeypatch, release, 'benchmark', '--dataset', 'atm', '--backbone', 'gemma4_12b',
                  '--model', 'different-reader', '--to-stage', 'observations', '--dry-run') == 0
    assert commands == [extraction_command]  # Reader changes neither extraction inputs nor its path.
    commands.clear()
    assert invoke(monkeypatch, release, '--config', root / 'configs/release/graphmemix_legacy.json', *common) == 0
    baseline_commands = deepcopy(commands)
    commands.clear()
    enhanced['observations']['enabled'] = False
    enhanced['benchmark_suite']['reader_k'] = legacy['benchmark_suite']['reader_k']
    disabled = tmp_path / 'disabled.json'; disabled.write_text(json.dumps(enhanced))
    assert invoke(monkeypatch, release, '--config', disabled, *common) == 0
    assert commands == baseline_commands
    commands.clear()
    # Starting after extraction fails early on a missing snapshot, without
    # dispatching another preprocessor or writing alternate checkpoint data.
    with pytest.raises(ValueError, match='incomplete'):
        invoke(monkeypatch, release, 'benchmark', '--dataset', 'atm', '--backbone', 'qwen3vl8b',
               '--from-stage', 'checkpoint', '--to-stage', 'checkpoint',
               '--artifact-root', tmp_path / 'artifacts', '--data-root', tmp_path / 'data')
    assert not commands
