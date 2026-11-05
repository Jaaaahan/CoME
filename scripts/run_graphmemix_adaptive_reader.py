#!/usr/bin/env python3
"""Read up to N units adaptively from a frozen GraphMemix observation checkpoint.

Requires only offline observations, checkpoint, priors and relations. Does not
consume node-verifier scores, ECV scores, forest selection or pruning outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mm_memory_bench import adaptive_reader
from mm_memory_bench.adaptive_reader import (PROTOCOL, AdaptiveDecision, PacketCapture,
    candidate_relations, run_adaptive_reader)
from mm_memory_bench.come.fields import ObservationSnapshot
from mm_memory_bench.core import resolve_asset_path
from mm_memory_bench.harness import _resolved_question
from mm_memory_bench.memory_scope import visible_ids
from mm_memory_bench.methods import GMEQwen2VLEmbedder
from mm_memory_bench.methods.base import GenerationConfig
from mm_memory_bench.methods.media import question_text
from mm_memory_bench.observation_evidence import rendering_contract
from mm_memory_bench.reader import BundleReader
from mm_memory_bench.unit_graph import reconstruct_unit_vectors
from graphmemix_core import build_candidate_graph, file_sha256, load_candidate_graph
from run_memix_answers_from_retrieval import (SinglePassQwenVL, checkpoint_states, hydrate_reader, read_jsonl)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'observations', 'checkpoint-root', 'source-priors', 'relation-store', 'embedding-model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--backend', choices=['standalone', 'memix'], default='standalone')
    p.add_argument('--memix-repo', type=Path, default=Path('vendor/memix-core'))
    p.add_argument('--initial-limit', type=int, default=48)
    p.add_argument('--source-top-l', type=int, default=24)
    p.add_argument('--candidate-limit', type=int, default=48)
    p.add_argument('--batch-size', type=int, choices=[1, 2], default=2)
    p.add_argument('--query-table-size', type=int, default=20)
    p.add_argument('--query-threshold', type=float, default=.95)
    p.add_argument('--max-rounds', type=int, default=10)
    p.add_argument('--max-llm-calls', type=int, default=10)
    p.add_argument('--semantic-k', type=int, default=8)
    p.add_argument('--image-max-edge', type=int, default=1280)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--max-output-tokens', type=int, default=1000)
    p.add_argument('--reader-ocr-budget-chars', type=int, default=320)
    p.add_argument('--decision-retries', type=int, default=3)
    p.add_argument('--search-decision-mode', choices=['compact', 'grounded'], default='compact')
    p.add_argument('--question-ids', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not (0 < args.source_top_l <= args.candidate_limit and 0 < args.initial_limit <= args.candidate_limit):
        p.error('require 0 < source-top-l <= candidate-limit and 0 < initial-limit <= candidate-limit')
    if min(args.semantic_k, args.image_max_edge, args.decision_retries) < 1:
        p.error('semantic-k, image-max-edge and decision-retries must be positive')
    if not (1 <= args.max_rounds <= 10 and 1 <= args.query_table_size <= 20 and args.max_llm_calls > 0):
        p.error('require 1..10 rounds, 1..20 units per query table and positive max-llm-calls')
    if not -1 <= args.query_threshold <= 1:
        p.error('query threshold must be in [-1, 1]')
    if args.output.exists() and not args.resume:
        p.error('output exists; use --resume or a new output path')
    reader = BundleReader(args.bundle)
    snapshot = ObservationSnapshot.from_bundle(args.observations, reader)
    memories = read_jsonl(args.bundle/'memories.jsonl', 'memory_id')
    questions = read_jsonl(args.bundle/'questions.jsonl', 'question_id')
    priors = read_jsonl(args.source_priors, 'question_id')
    graph = load_candidate_graph(args.relation_store, 'full', args.semantic_k,
                                 observation_contract=snapshot.contract)
    descriptors = {r['context_id']: r for line in args.relation_store.read_text().splitlines()
                   if line.strip() and (r := json.loads(line)).get('record_type') == 'unit_graph'}
    states = checkpoint_states(args.checkpoint_root)
    wanted = set(questions)
    if args.question_ids:
        wanted = set(args.question_ids.read_text().split())
        if not wanted or not wanted <= questions.keys():
            raise ValueError('invalid question allowlist')
    if not wanted <= priors.keys():
        raise ValueError('missing question priors')
    generation = GenerationConfig(model=args.model, base_url=args.base_url, reasoning_effort='none', top_k=args.initial_limit,
        max_model_len=args.max_model_len, max_output_tokens=args.max_output_tokens)
    embedder = GMEQwen2VLEmbedder(str(args.embedding_model))
    model = SinglePassQwenVL(generation)
    questions_digest = file_sha256(args.bundle/'questions.jsonl')
    contract = {'protocol': PROTOCOL, 'observations': snapshot.contract,
        'implementation_sha256': file_sha256(Path(adaptive_reader.__file__)),
        'runner_sha256': file_sha256(Path(__file__)),
        'questions_sha256': questions_digest, 'memories_sha256': file_sha256(args.bundle/'memories.jsonl'),
        'assets_sha256': file_sha256(args.bundle/'assets.jsonl'),
        'priors_sha256': file_sha256(args.source_priors), 'relations_sha256': file_sha256(args.relation_store),
        'embedding_identity': embedder.model_identity, 'model': args.model,
        'query_encoding': 'text-only-question_text-and-generated-query-same-encode_units',
        'rendering': rendering_contract(), 'initial_limit': args.initial_limit,
        'source_top_l': args.source_top_l, 'candidate_limit': args.candidate_limit,
        'batch_size': args.batch_size, 'query_threshold': args.query_threshold,
        'max_rounds': args.max_rounds, 'query_table_size': args.query_table_size,
        'max_units': adaptive_reader.MAX_READ_UNITS,
        'query_gate': 'proposed-next-batch-mean-strictly-greater-than-active-next-batch-mean',
        'short_batch_mean': 'available-units-only',
        'max_llm_calls': args.max_llm_calls,
        'final_answer_mode': 'canonical-come-packet-and-tools-unmodified',
        'stopping_policy': ('compact-action-and-query-v1' if args.search_decision_mode == 'compact' else
                            'read-evidence-supported-requirements-and-explicit-coverage-v1'),
        'search_decision_mode': args.search_decision_mode,
        'remaining_candidate_count_in_prompt': args.search_decision_mode != 'compact',
        'semantic_k': args.semantic_k, 'image_max_edge': args.image_max_edge,
        'reader_ocr_budget_chars': args.reader_ocr_budget_chars, 'max_model_len': args.max_model_len,
        'max_output_tokens': args.max_output_tokens, 'decision_retries': args.decision_retries}
    completed = read_jsonl(args.output, 'question_id') if args.output.exists() else {}
    if not completed.keys() <= wanted:
        raise ValueError('existing output has questions outside this run')
    assets = {str(k): str(resolve_asset_path(args.bundle, v)) for k,v in reader.assets.items()}
    sources = {mid: str(m['source_id']) for mid,m in memories.items() if m.get('source_id') is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cid in sorted({questions[qid]['context_id'] for qid in wanted}):
        path = states[cid]
        state = json.loads(path.read_text())
        snapshot.require_contract(state['config'].get('observations'))
        # Content identity hashes actual weights/config; directory manifests also
        # include machine-local download receipts and are not portable.
        if state['config'].get('observation_index', {}).get('embedding_model_identity') != embedder.model_identity:
            raise ValueError('query embedder differs from checkpoint embedding model')
        index_path = path.parent/state['index_file']
        index_hash, state_hash = file_sha256(index_path), file_sha256(path)
        if graph.contexts[cid]['checkpoint_index_sha256'] != index_hash:
            raise ValueError('graph and checkpoint indexes differ')
        local_contract = {**contract, 'checkpoint_index_sha256': index_hash, 'checkpoint_state_sha256': state_hash}
        nodes, matrix = reconstruct_unit_vectors(index_path, state)
        vectors = {node['unit_id']: vector for node,vector in zip(nodes,matrix)}
        method = hydrate_reader(checkpoint_state=path, memix_repo=args.memix_repo, backend=args.backend,
            generation=generation, answer_model=PacketCapture(), source_top_k=args.candidate_limit,
            public_source_ids=sources, public_asset_paths=assets, observations=snapshot,
            reader_ocr_budget_chars=args.reader_ocr_budget_chars)
        for qid in sorted(q for q in wanted if questions[q]['context_id'] == cid):
            prior = priors[qid]
            snapshot.require_contract(prior.get('observations'))
            if any(prior.get(k) != v for k,v in {
                'questions_sha256': questions_digest, 'checkpoint_index_sha256': index_hash,
                'checkpoint_state_sha256': state_hash}.items()):
                raise ValueError('stale or incompatible source priors')
            if qid in completed:
                if completed[qid].get('metadata', {}).get('adaptive_run_contract') != local_contract:
                    raise ValueError('incompatible adaptive output; use a new path')
                continue
            started = time.perf_counter()
            question = _resolved_question(reader, questions[qid])
            candidate = build_candidate_graph(prior, graph, source_top_l=args.source_top_l,
                candidate_limit=args.candidate_limit, visible_memory_ids=visible_ids(questions[qid], memories.values()))
            ids = candidate['candidate_ids']
            relations = candidate_relations(ids, graph.nodes, memories, descriptors[cid]['edges'], semantic_k=args.semantic_k)
            decision = AdaptiveDecision(method, model, question, relations,
                image_max_edge=args.image_max_edge, retries=args.decision_retries, max_calls=args.max_llm_calls,
                search_decision_mode=args.search_decision_mode)
            def decide_step(visited, remaining, trace, order, controller):
                value = decision(visited, remaining, trace, order, controller)
                print(json.dumps({'event': 'adaptive_step', 'question_id': qid,
                    'visited_units': len(visited), 'remaining_units': remaining,
                    'controller': controller, 'decision': value}, ensure_ascii=False), flush=True)
                return value
            result = run_adaptive_reader(question_text(question), ids, vectors,
                encode=lambda text: embedder.encode_units([{'text': text}])[0],
                decide=decide_step, limit=args.initial_limit, batch_size=args.batch_size,
                query_threshold=args.query_threshold, max_rounds=args.max_rounds,
                query_table_size=args.query_table_size)
            visited = result.pop('retrieved_unit_ids')
            prediction = result.pop('prediction')
            answer_status = result.pop('answer_status')
            row = {'question_id': qid, 'context_id': cid, 'prediction': prediction,
                'retrieved_memory_ids': list(dict.fromkeys(graph.nodes[u]['memory_id'] for u in visited)),
                'metadata': {**result, 'retrieved_unit_ids': visited, 'candidate_ids': ids,
                    'adaptive_run_contract': local_contract, 'observations': snapshot.contract,
                    'llm_calls': decision.calls, 'search_calls': decision.search_calls,
                    'answer_calls': decision.answer_calls, 'visited_units': len(visited),
                    'elapsed_seconds': time.perf_counter()-started, 'status': answer_status}}
            if answer_status != 'ok':
                row['metadata']['error_type'] = 'InvalidFinalAnswer'
            with args.output.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            completed[qid] = row
            print(json.dumps({'done': len(completed), 'total': len(wanted), 'question_id': qid,
                              'visited_units': len(visited), 'llm_calls': decision.calls}), flush=True)
        method.end_context()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
