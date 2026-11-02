import json

import numpy as np
import pytest

from mm_memory_bench.adaptive_reader import (AdaptiveDecision, SYSTEM, candidate_relations,
    parse_decision, parse_search_decision, run_adaptive_reader)

VECTORS = {'a': [1., 0.], 'b': [.8, .6], 'c': [0., 1.]}


def search_response(*, stop=False, next_query='q'):
    row = dict(supported=[{'requirement':'Identify the requested objects', 'evidence':['U00']}],
               missing=[] if stop else ['Find the remaining requested objects'],
               coverage={'complete':stop, 'reason':'Read evidence covers the request' if stop else 'More matches may exist',
                         'evidence':['U00'] if stop else []}, action='stop' if stop else 'continue')
    row.update({'stop_reason':'sufficient_evidence'} if stop else {'next_query':next_query})
    return json.dumps(row)


def run_single_reader(*args, **kwargs):
    """Keep single-unit cases as regression coverage for the configurable step size."""
    kwargs.setdefault('batch_size', 1)
    return run_adaptive_reader(*args, **kwargs)


def test_query_switch_then_cached_original_query_can_be_accepted_again():
    calls = []
    def decide(visited, remaining, trace, order, control):
        calls.append((visited, remaining))
        return ({'action': 'stop', 'answer': 'done'} if not remaining else
                {'action': 'continue', 'next_query': 'vertical' if len(visited) == 1 else 'original'})
    result = run_single_reader('original', list(VECTORS), VECTORS,
        encode=lambda q: [0, 1] if q == 'vertical' else [1, 0], decide=decide)
    assert result['initial_order'] == ['a', 'b', 'c']
    assert result['retrieved_unit_ids'] == ['a', 'c', 'b']
    assert [r['query_transition']['action'] for r in result['trace'][:-1]] == ['accept', 'accept']
    assert result['trace'][1]['query_transition']['cache_match'] == 'exact'
    assert calls == [(('a',), 2), (('a', 'c'), 1), (('a', 'c', 'b'), 0)]
    assert result['stop_reason'] == 'exhausted'


def test_reject_non_improving_query_and_cache_without_consuming_its_best_unit():
    def decide(visited, remaining, trace, order, control):
        return {'action': 'stop', 'answer': 'done'} if not remaining else {'action': 'continue', 'next_query': 'negative'}
    result = run_single_reader('original', list(VECTORS), VECTORS,
        encode=lambda q: [1, 0] if q == 'original' else [-1, -.1], decide=decide)
    assert result['trace'][0]['query_transition']['proposed_next_units'] == ['c']
    assert result['trace'][0]['query_transition']['action'] == 'reject'
    assert result['retrieved_unit_ids'] == ['a', 'b', 'c']
    assert result['query_rejection_count'] == 2 and result['query_encoding_calls'] == 2
    assert result['trace'][1]['query_transition']['cache_match'] == 'exact'
    assert all(t['controller']['active_query_id'] == 0 for t in result['trace'])


def test_rejected_query_preserves_last_accepted_direction_not_original_order():
    def vec(degrees):
        radians = np.deg2rad(degrees)
        return [np.cos(radians), np.sin(radians)]
    vectors = {u: vec(angle) for u, angle in [('a',0), ('original_next',5), ('c',90), ('d',80), ('e',75)]}
    def decide(v, r, t, o, control):
        if len(v) == 3:
            return {'action':'stop', 'answer':'done'}
        return {'action':'continue', 'next_query':'vertical' if len(v)==1 else 'worse'}
    result = run_single_reader('original', list(vectors), vectors,
        encode=lambda q: vec({'original':0, 'vertical':90, 'worse':50}[q]), decide=decide)
    assert result['retrieved_unit_ids'] == ['a', 'c', 'd']
    rejected = result['trace'][1]['query_transition']
    assert rejected['action'] == 'reject' and rejected['previous_query_id'] == rejected['active_query_id'] == 1
    assert rejected['proposed_next_units'] == ['e'] and rejected['previous_next_units'] == ['d']
    assert rejected['previous_mean_score'] == pytest.approx(np.cos(np.deg2rad(10)))
    # This query beats the ORIGINAL score, but loses against the active direction.
    assert result['initial_scores']['e'] < rejected['proposed_mean_score'] < rejected['previous_mean_score']
    assert result['trace'][2]['controller']['active_query'] == 'vertical'


@pytest.mark.parametrize('new_score,expected_action,second_unit', [
    (.7, 'reject', 'b'), (.8, 'reject', 'b'), (.9, 'accept', 'c')])
def test_compare_each_querys_next_unread_evidence(new_score, expected_action, second_unit):
    def decide(v, r, t, o, c):
        return {'action':'stop','answer':'done'} if len(v)==2 else {'action':'continue','next_query':'new'}
    result = run_single_reader('original', list(VECTORS), VECTORS,
        encode=lambda q: [1,0] if q=='original' else [-np.sqrt(1-new_score**2), new_score],
        decide=decide)
    gate = result['trace'][0]['query_transition']
    # c improves over its score of zero under the old query in all three cases.
    # The actual baseline is the old list's next unread b, whose score is 0.8.
    assert gate['proposed_next_units']==['c'] and gate['previous_next_units']==['b']
    assert gate['proposed_mean_score'] > result['initial_scores']['c']
    assert gate['previous_mean_score']==.8 and gate['action']==expected_action
    assert result['retrieved_unit_ids']==['a', second_unit]


def test_equal_scores_reject_distinct_query_without_a_margin():
    def decide(v, r, t, o, c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'reflected'}
    r = run_single_reader('original', ['a','b'], {'a':[1,0], 'b':[0,1]},
        encode=lambda q: [1,1] if q=='original' else [-1,1], decide=decide)
    gate = r['trace'][0]['query_transition']
    assert gate['proposed_mean_score'] == gate['previous_mean_score'] and gate['action']=='reject'


def test_any_strict_score_increase_accepts_without_an_added_margin():
    def vec(degrees):
        radians = np.deg2rad(degrees)
        return [np.cos(radians), np.sin(radians)]
    def decide(v, r, t, o, c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'slightly_better'}
    r = run_single_reader('original', ['a','b'], {'a':vec(0), 'b':vec(10)},
        encode=lambda q: vec(0 if q=='original' else .01), decide=decide, query_threshold=1)
    gate = r['trace'][0]['query_transition']
    assert 0 < gate['proposed_mean_score'] - gate['previous_mean_score'] < .001
    assert gate['action']=='accept'


def test_accepted_direction_continues_without_per_unit_original_score_gate():
    def decide(v, r, t, o, c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'vertical'}
    r = run_single_reader('original', list(VECTORS), VECTORS,
        encode=lambda q: [1,0] if q=='original' else [0,1], decide=decide)
    assert r['retrieved_unit_ids']==['a','c','b']
    assert r['trace'][1]['query_transition']['action']=='reuse'
    assert r['trace'][2]['controller']['active_query']=='vertical'
    assert r['trace'][2]['selections'][0]['selected_query_score'] < r['initial_scores']['b']


def test_early_stop_and_limit_exclude_unvisited_units():
    def decide(visited, remaining, trace, order, control):
        assert order == ('a', 'b') and visited == ('a',) and remaining == 1
        return {'action': 'stop', 'answer': 'enough'}
    result = run_single_reader('q', list(VECTORS), VECTORS,
        encode=lambda q: [1, 0], decide=decide, limit=2, batch_size=1)
    assert result['retrieved_unit_ids'] == ['a'] and result['stop_reason'] == 'model_stop'


@pytest.mark.parametrize('value', [[], {'action': 'continue', 'next_query': ''},
    {'action': 'stop', 'answer': ''}, {'action': 'something'}])
def test_invalid_decisions(value):
    with pytest.raises(ValueError):
        parse_decision(json.dumps(value), exhausted=False)
    with pytest.raises(ValueError):
        parse_decision('{"action":"continue","next_query":"x"}', exhausted=True)


def test_invalid_vectors_and_candidates():
    for ids, vectors in [([], {}), (['a','a'], VECTORS), (['a'], {'a':[0,0]}), (['a'], {'a':[float('nan'),1]})]:
        with pytest.raises(ValueError):
            run_single_reader('q', ids, vectors, encode=lambda q:[1,0], decide=None)


def test_relations_preserve_semantics_when_explicit_edge_also_present():
    nodes = {u: {'memory_id': m} for u,m in [('a','m1'),('b','m1'),('c','m2'),('outside','m3')]}
    memories = {'m1': {'session_id':'s'}, 'm2': {'session_id':'s'}, 'm3': {'session_id':'s'}}
    edges = [{'left':'a','right':'b','semantic_similarity':.7,'explicit_relations':['belongs_to']},
             {'left':'a','right':'outside','semantic_similarity':.9},
             {'left':'a','right':'c','semantic_similarity':.6,'semantic_rank_left':9,'semantic_rank_right':1}]
    r = candidate_relations(['a','b','c'], nodes, memories, edges)
    assert r['a'] == {'belongs_to_memory':'m1','semantic_neighbors':[{'unit_id':'b','similarity':.7}],
                      'same_session_units':['b','c']}


def test_decision_retries_on_exhaustion_preserving_multimodal_history():
    class Reader:
        answer_model = object()
        def _generate_answer(self, question, ids, **kw):
            return self.answer_model.complete([{'role':'user','content':[
                {'type':'text','text':uid} for uid in ids] + [{'type':'image_url','image_url':{'url':'data:image/png;base64,abc'}}]}])
    class Model:
        calls = []
        def complete(self, messages, **kw):
            self.calls.append(json.loads(json.dumps(messages)))
            return '' if len(self.calls)==1 else 'done'
    reader, model = Reader(), Model()
    original = reader.answer_model
    relations = {u: {'belongs_to_memory':'m','semantic_neighbors':[], 'same_session_units':[]} for u in ['a','b']}
    decision = AdaptiveDecision(reader, model, {}, relations)
    assert decision(('a','b'), 0, (), ('a','b')) == {'action':'stop','answer':'done'}
    assert reader.answer_model is original and decision.calls == 2
    assert model.calls[0][0]['content'][0]['text'] == 'a'
    assert model.calls[0][0]['content'][1]['text'] == 'b'
    assert model.calls[0][0]['content'][2]['type'] == 'image_url'


def test_query_table_semantic_and_exact_reuse_and_threshold():
    from mm_memory_bench.adaptive_reader import QueryTable
    encoded = []
    def encode(q):
        encoded.append(q)
        return {'first':[1,0], 'paraphrase':[.999,.01], 'different':[0,1]}[q]
    table = QueryTable(encode, np.eye(2), ['a','b'], threshold=.95)
    first, mode, _ = table.resolve('first')
    assert mode == 'new'
    again, mode, _ = table.resolve('paraphrase')
    assert again is first and mode == 'semantic'
    assert table.resolve(' PARAPHRASE ')[1] == 'exact'
    assert encoded == ['first','paraphrase']
    assert table.resolve('different')[1] == 'new' and len(table.entries) == 2


def test_repeated_query_reads_one_at_a_time_up_to_ten_rounds_without_blocking():
    vectors = {str(i): [1, .01*i] for i in range(48)}
    encoded = []
    def encode(q):
        encoded.append(q)
        return [1,0]
    def decide(visited, remaining, trace, order, control):
        if control['force_answer']:
            return {'action':'stop','answer':'insufficient evidence'}
        return {'action':'continue','next_query':'question'}
    r = run_single_reader('question', list(vectors), vectors, encode=encode, decide=decide)
    assert r['stop_reason'] == 'round_budget'
    assert r['retrieved_unit_ids'] == [str(i) for i in range(10)]
    assert len(r['trace']) == 10 and all(len(t['unit_ids'])==1 for t in r['trace'])
    assert encoded == ['question']
    assert len(r['query_table']) == 1 and r['query_table'][0]['reads'] == 10
    assert len(r['query_table'][0]['ranking']) == 20
    assert r['trace'][-1]['controller']['force_answer']


def test_semantic_cache_reuses_direction_and_reaches_fifth_evidence():
    vectors = {str(i): [1, .01*i] for i in range(30)}
    encoded=[]
    def encode(q):
        encoded.append(q)
        return [1,0] if q=='question' else [.999,.01]
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'enough'} if len(v)==5 else {'action':'continue','next_query':'paraphrase'}
    r=run_single_reader('question',list(vectors),vectors,encode=encode,decide=decide)
    assert r['retrieved_unit_ids']==['0','1','2','3','4'] and r['stop_reason']=='model_stop'
    assert encoded==['question','paraphrase'] and len(r['query_table'])==1
    assert r['trace'][0]['query_transition']['cache_match']=='semantic'
    assert r['trace'][0]['query_transition']['action']=='reuse'


def test_new_query_table_has_twenty_unread_units_and_no_global_rereads():
    ids=[str(i) for i in range(48)]
    vectors={u:[1,.1*i] for i,u in enumerate(ids)}
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'vertical'}
    result=run_single_reader('q',ids,vectors,encode=lambda q:[1,0] if q=='q' else [0,1], decide=decide)
    assert len(result['retrieved_unit_ids'])==len(set(result['retrieved_unit_ids']))==10
    # The old list wins twice; the cached new direction wins once the old list's
    # next unread score drops below its own best score. No new encoding is needed.
    assert result['retrieved_unit_ids'][:4]==['0','1','2','47']
    assert [t['query_transition']['action'] for t in result['trace'][:3]]==['reject','reject','accept']
    assert result['trace'][2]['query_transition']['cache_match']=='exact'
    assert result['query_encoding_calls']==2
    assert all(len(e['ranking'])==20 and len(e['scores'])==20 for e in result['query_table'])
    assert '0' not in result['query_table'][1]['ranking']
    assert result['stop_reason']=='round_budget'


def test_round_budget_and_short_query_table_exhaustion():
    ids = [str(i) for i in range(48)]
    vectors = {u:[1,.02*i] for i,u in enumerate(ids)}
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'q'}
    result = run_single_reader('q',ids,vectors,encode=lambda q:[1,0],decide=decide,max_rounds=2)
    assert result['stop_reason']=='round_budget' and len(result['retrieved_unit_ids'])==2
    result = run_single_reader('q',ids,vectors,encode=lambda q:[1,0],decide=decide,query_table_size=3)
    assert result['stop_reason']=='query_exhausted' and len(result['retrieved_unit_ids'])==3
    for params in [{'batch_size':4}, {'max_rounds':11}, {'query_table_size':21}]:
        with pytest.raises(ValueError):
            run_single_reader('q',ids,vectors,encode=lambda q:[1,0],decide=decide,**params)


def test_actual_api_budget_includes_format_retries():
    class Reader:
        answer_model=None
        def _generate_answer(self,*args,**kw):
            self.answer_model.complete([{'role':'user','content':'evidence'}])
    class Model:
        calls=0
        def complete(self,messages,**kw):
            self.calls+=1
            if self.calls==1: return '{}'
            assert messages == [{'role':'user','content':'evidence'}]
            assert kw == {}
            return 'unknown'
    model=Model()
    decision=AdaptiveDecision(Reader(),model,{}, {'a':{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]}},max_calls=2)
    control=dict(force_answer=False,stop_reason=None,new_unit_ids=['a'])
    assert decision(('a',),10,(),('a',),control)['action']=='stop'
    assert model.calls==2 and control['stop_reason']=='budget_exhausted' and control['stop_detail']=='call_budget'
    with pytest.raises(ValueError,match='API call budget'):
        decision(('a',),10,(),('a',),control)
    assert model.calls==2


@pytest.mark.parametrize('second_score,action,selected', [
    (.4, 'reject', ['b','d']), (.6, 'reject', ['b','d']), (.8, 'accept', ['c','e'])])
def test_two_unit_gate_compares_means_not_best_score(second_score, action, selected):
    vectors = {'a':[1,0,0], 'a2':[1,0,0], 'b':[.8,0,.6], 'd':[.8,0,-.6],
               'c':[0,1,0], 'e':[0,second_score,np.sqrt(1-second_score**2)]}
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if len(v)==4 else {'action':'continue','next_query':'new'}
    result=run_adaptive_reader('q',list(vectors),vectors,
        encode=lambda q:[1,0,0] if q=='q' else [0,1,0],decide=decide)
    gate=result['trace'][0]['query_transition']
    assert gate['proposed_next_units']==['c','e'] and gate['previous_next_units']==['b','d']
    assert gate['proposed_scores'][0] > max(gate['previous_scores'])
    assert gate['proposed_mean_score']==pytest.approx((1+second_score)/2)
    assert gate['previous_mean_score']==.8 and gate['action']==action
    assert result['retrieved_unit_ids']==['a','a2',*selected]
    assert all(len(t['unit_ids'])==2 for t in result['trace'])


def test_two_unit_batches_short_final_batch_and_cached_mean_use_remaining_only():
    vectors={str(i):[1,.1*i] for i in range(5)}
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'q'}
    result=run_adaptive_reader('q',list(vectors),vectors,encode=lambda q:[1,0],decide=decide)
    assert [t['unit_ids'] for t in result['trace']]==[['0','1'],['2','3'],['4']]
    gate=result['trace'][1]['query_transition']
    assert gate['proposed_next_units']==gate['previous_next_units']==['4']
    assert gate['proposed_mean_score']==gate['previous_mean_score']==result['initial_scores']['4']
    assert result['query_encoding_calls']==1 and result['stop_reason']=='exhausted'


def test_two_unit_semantic_cache_can_continue_beyond_four_evidence_units():
    vectors={str(i):[1,.01*i] for i in range(30)}
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if len(v)==6 else {'action':'continue','next_query':'paraphrase'}
    result=run_adaptive_reader('q',list(vectors),vectors,
        encode=lambda q:[1,0] if q=='q' else [.999,.01],decide=decide)
    assert result['retrieved_unit_ids']==[str(i) for i in range(6)]
    assert result['query_encoding_calls']==2 and len(result['query_table'])==1
    assert result['trace'][0]['query_transition']['cache_match']=='semantic'
    assert result['trace'][1]['query_transition']['cache_match']=='exact'


def test_two_unit_round_budget_is_ten_rounds_and_twenty_unique_units():
    ids=[str(i) for i in range(48)]
    matrix=np.eye(48)
    def encode(q):
        pair=int(q)
        return matrix[2*pair]+matrix[2*pair+1]
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':str(c['round'])}
    result=run_adaptive_reader('0',ids,dict(zip(ids,matrix)),encode=encode,decide=decide)
    assert result['retrieved_unit_ids']==ids[:20]
    assert len(result['trace'])==10 and result['stop_reason']=='round_budget'
    assert all(len(t['unit_ids'])==2 for t in result['trace'])
    assert all(len(e['ranking'])<=20 for e in result['query_table'])


def test_two_unit_repeated_query_stops_at_twenty_cached_units():
    ids=[str(i) for i in range(48)]
    def decide(v,r,t,o,c):
        return {'action':'stop','answer':'done'} if c['force_answer'] else {'action':'continue','next_query':'q'}
    result=run_adaptive_reader('q',ids,{u:[1,0] for u in ids},encode=lambda q:[1,0],decide=decide)
    assert result['retrieved_unit_ids']==ids[:20]
    assert len(result['trace'])==10 and result['stop_reason']=='round_budget'


@pytest.mark.parametrize('raw', ['', '   ', None])
def test_final_call_invalid_decision_records_unanswered_without_extra_call(raw):
    class Reader:
        answer_model = None
        def _generate_answer(self, *args, **kwargs):
            self.answer_model.complete([{'role':'user','content':'evidence'}])
    class Model:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            return raw
    model = Model()
    decision = AdaptiveDecision(Reader(), model, {},
        {'a':{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]}}, max_calls=1)
    control = dict(force_answer=False, stop_reason=None, new_unit_ids=['a'])
    result = decision(('a',), 10, (), ('a',), control)
    assert result['action'] == 'stop' and 'no valid final answer' in result['answer'].lower()
    assert model.calls == 1 and control['stop_reason'] == 'budget_exhausted' and control['stop_detail'] == 'call_budget'
    assert control['answer_fallback']['reason'] == 'invalid_final_answer'
    assert control['invalid_answers'][0]['raw'] == raw


def test_invalid_final_decision_does_not_abort_reader_or_next_question():
    class Reader:
        answer_model = None
        def _generate_answer(self, *args, **kwargs):
            self.answer_model.complete([{'role':'user','content':'evidence'}])
    class Model:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            return search_response() if messages[0]['role']=='system' else ''
    ids=[str(i) for i in range(48)]
    relations={u:{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]} for u in ids}
    model=Model()
    for _ in range(2):
        decision=AdaptiveDecision(Reader(),model,{},relations)
        result=run_adaptive_reader('q',ids,{u:[1,0] for u in ids},encode=lambda q:[1,0],decide=decision)
        assert result['answer_status']=='invalid_final_answer'
        assert len(result['retrieved_unit_ids'])==20 and len(result['trace'])==10
        assert decision.calls==10 and result['stop_reason']=='budget_exhausted'
        assert result['stop_detail']=='round_budget'
        assert result['trace'][-1]['controller']['answer_fallback']['reason']=='invalid_final_answer'
    assert model.calls==20



def test_final_stage_uses_original_question_and_all_twenty_images_not_search_history():
    original = {'prompt':[{'type':'text','text':'ORIGINAL QUESTION: count the red objects.'}]}
    class Reader:
        answer_model = None
        def _generate_answer(self, question, ids, **kwargs):
            assert question is original
            self.answer_model.complete([{'role':'user','content':
                [{'type':'text','text':question['prompt'][0]['text']}] +
                [{'type':'image_url','image_url':{'url':'data:image/png;base64,'+uid}} for uid in ids]}])
    class Model:
        packets=[]
        def complete(self, messages, **kwargs):
            self.packets.append(json.loads(json.dumps(messages)))
            if messages[0]['role'] == 'user':
                assert kwargs == {}
                assert len(messages)==1
                assert messages[0]['content'][0]['text']=='ORIGINAL QUESTION: count the red objects.'
                assert 'intermediate-search-only' not in json.dumps(messages)
                assert len([p for p in messages[0]['content'] if p['type']=='image_url'])==20
                return 'There are three red objects.'
            return search_response(next_query='intermediate-search-only')
    ids=[str(i) for i in range(48)]
    relations={u:{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]} for u in ids}
    model=Model()
    decision=AdaptiveDecision(Reader(),model,original,relations)
    result=run_adaptive_reader('original',ids,{u:[1,0] for u in ids},encode=lambda q:[1,0],decide=decision)
    assert result['prediction']=='There are three red objects.' and result['answer_status']=='ok'
    assert decision.calls==10 and decision.search_calls==9 and decision.answer_calls==1
    assert len(result['retrieved_unit_ids'])==20 and len(result['trace'])==10
    assert result['trace'][-1]['controller']['stage']=='final_answer'


def test_early_search_stop_preserves_assessment_and_uses_canonical_final_answer():
    class Reader:
        answer_model=None
        def _generate_answer(self,*args,**kwargs):
            self.answer_model.complete([{'role':'user','content':'original question plus evidence'}])
    class Model:
        calls=0
        def complete(self,messages,**kwargs):
            self.calls+=1
            if self.calls==1:
                assert messages[0]['content']==SYSTEM
                return search_response(stop=True)
            assert messages==[{'role':'user','content':'original question plus evidence'}]
            assert kwargs=={}
            return 'final answer'
    ids=list(VECTORS)
    relations={u:{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]} for u in ids}
    decision=AdaptiveDecision(Reader(),Model(),{},relations)
    result=run_adaptive_reader('q',ids,VECTORS,encode=lambda q:[1,0],decide=decision)
    assert result['prediction']=='final answer' and len(result['retrieved_unit_ids'])==2
    assert result['stop_reason']=='sufficient_evidence' and decision.search_calls==decision.answer_calls==1
    controller=result['trace'][0]['controller']
    assert controller['search_decision']==json.loads(search_response(stop=True))
    assert controller['search_decision_raw']==search_response(stop=True)


@pytest.mark.parametrize('change,error', [
    ({'supported':[]}, 'sufficient_evidence'),
    ({'supported':[{'requirement':'Match', 'evidence':['U02']}]}, 'read_units'),
    ({'supported':[{'requirement':'Match', 'evidence':[]}]}, 'read_units'),
    ({'missing':['Other matching photos may remain']}, 'missing'),
    ({'coverage':{'complete':False, 'reason':'Only one match found', 'evidence':[]}}, 'sufficient_evidence'),
    ({'coverage':{'complete':True, 'reason':'All found', 'evidence':['U02']}}, 'read_units'),
    ({'coverage':{'complete':True, 'reason':'', 'evidence':['U00']}}, 'coverage'),
    ({'coverage':{'complete':True, 'reason':'All found', 'evidence':[]}}, 'read_units'),
    ({'coverage':None}, 'coverage'),
    ({'stop_reason':'budget_exhausted'}, 'stop_reason'),
])
def test_sufficiency_rejects_ungrounded_or_incomplete_assessments(change, error):
    row=json.loads(search_response(stop=True))
    row.update(change)
    with pytest.raises(ValueError, match=error):
        parse_search_decision(json.dumps(row), read_aliases=['U00','U01'])


def test_cannot_find_more_keeps_missing_requirements_and_requires_explanation():
    row=json.loads(search_response())
    row.update(action='stop', stop_reason='cannot_find_more')
    with pytest.raises(ValueError, match='stop_explanation'):
        parse_search_decision(json.dumps(row), read_aliases=['U00'])
    row['stop_explanation']='Available query directions do not address the missing location.'
    parsed=parse_search_decision(json.dumps(row), read_aliases=['U00'])
    assert parsed['missing'] and not parsed['coverage']['complete']
    assert parsed['stop_reason']=='cannot_find_more'


@pytest.mark.parametrize('invalid_kind', ['unread', 'missing', 'list_incomplete'])
def test_invalid_sufficiency_is_repaired_and_reads_more_evidence(invalid_kind):
    invalid=json.loads(search_response(stop=True))
    if invalid_kind=='unread':
        invalid['supported'][0]['evidence']=['U04']
    elif invalid_kind=='missing':
        invalid['missing']=['Find other matching photos']
    else:
        invalid['coverage']={'complete':False, 'reason':'One match does not establish all photos', 'evidence':[]}
    class Reader:
        answer_model=None
        def _generate_answer(self, question, ids, **kwargs):
            self.answer_model.complete([{'role':'user','content':list(ids)}])
    class Model:
        calls=0
        def complete(self, messages, **kwargs):
            self.calls+=1
            if self.calls==1:
                context=json.loads(messages[-1]['content'])
                assert context['read_units']==[
                    {'alias':'U00','unit_id':'0','evidence_number':1},
                    {'alias':'U01','unit_id':'1','evidence_number':2}]
                assert 'U04' in context['unit_aliases']  # Known but unread: cannot cite it.
                return json.dumps(invalid)
            if self.calls==2:
                assert 'Invalid retrieval decision' in messages[-1]['content']
                return search_response()
            if self.calls==3:
                context=json.loads(messages[-1]['content'])
                assert context['previous_assessment']==json.loads(search_response())
                return search_response(stop=True)
            assert messages==[{'role':'user','content':['0','1','2','3']}]
            return 'The matching photos are ...'
    ids=[str(i) for i in range(6)]
    relations={u:{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]} for u in ids}
    decision=AdaptiveDecision(Reader(),Model(),{},relations)
    result=run_adaptive_reader('q',ids,{u:[1,0] for u in ids},encode=lambda q:[1,0],decide=decision)
    assert result['retrieved_unit_ids']==ids[:4]
    assert result['stop_reason']=='sufficient_evidence'
    assert result['trace'][0]['controller']['invalid_decisions'][0]['raw']==json.dumps(invalid)
    assert result['trace'][0]['controller']['search_decision']['missing']
    assert decision.calls==4


@pytest.mark.parametrize('reason', ['cannot_find_more', 'invalid_decision_retries_exhausted', 'exhausted'])
def test_incomplete_search_and_retry_exhaustion_never_become_sufficient(reason):
    class Reader:
        answer_model=None
        def _generate_answer(self, *args, **kwargs):
            self.answer_model.complete([{'role':'user','content':'original question and evidence'}])
    class Model:
        def complete(self, messages, **kwargs):
            if messages[0]['role']=='user':
                return 'Insufficient information.'
            if reason=='invalid_decision_retries_exhausted':
                return '{}'
            row=json.loads(search_response())
            row.update(action='stop', stop_reason='cannot_find_more', stop_explanation='No useful remaining search direction.')
            return json.dumps(row)
    ids=list(VECTORS)[:1 if reason=='exhausted' else 3]
    relations={u:{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]} for u in ids}
    decision=AdaptiveDecision(Reader(),Model(),{},relations,retries=2)
    result=run_adaptive_reader('q',ids,VECTORS,encode=lambda q:[1,0],decide=decision)
    assert result['stop_reason']==('budget_exhausted' if reason=='invalid_decision_retries_exhausted' else 'cannot_find_more')
    controller=result['trace'][-1]['controller']
    if reason=='cannot_find_more':
        assert controller['search_decision']['missing']
        assert controller['search_decision']['stop_explanation']
    else:
        assert result['stop_detail']==reason
        if reason=='invalid_decision_retries_exhausted':
            assert len(controller['invalid_decisions'])==2
    assert decision.calls<=decision.max_calls


@pytest.mark.parametrize('tool_mode', [None, 'plan'])
def test_final_request_equals_real_come_renderer_including_tools_and_images(tmp_path, tool_mode):
    from PIL import Image
    from mm_memory_bench.methods.concrete_memix import ConcreteMemixMethod
    from mm_memory_bench.adaptive_reader import PacketCapture
    path=tmp_path/'evidence.png'
    Image.new('RGB', (4,4), 'red').save(path)
    reader=ConcreteMemixMethod.__new__(ConcreteMemixMethod)
    reader.drop_reader_ocr=False
    reader.reader_ocr_budget_chars=0
    reader.observation_contract=None
    reader._record_by_id={'a':{'text':'A red object', 'source_id':'photo-1', 'modality':'image',
                              'content':[{'type':'image', 'path':str(path)}]}}
    question={'instruction':'Explain your answer with evidence.',
              'prompt':[{'type':'text','text':'What is shown?'}, {'type':'image','path':str(path)}],
              'tools':[{'type':'function','function':{'name':'find_photo','parameters':{'type':'object'}}}],
              'tool_mode':tool_mode}
    baseline=PacketCapture()
    reader.answer_model=baseline
    reader._generate_answer(question, ['a'], image_max_edge=1280)
    class Model:
        def complete(self, messages, **kwargs):
            assert messages==baseline.messages
            assert kwargs==baseline.kwargs
            assert kwargs['tools']==(None if tool_mode=='plan' else question['tools'])
            return 'It is red.\n\nEvidence: photo-1 shows a red object.'
    decision=AdaptiveDecision(reader,Model(),question,
        {'a':{'belongs_to_memory':'m','same_session_units':[], 'semantic_neighbors':[]}})
    result=decision(('a',),0,(),('a',))
    assert result['answer']=='It is red.\n\nEvidence: photo-1 shows a red object.'
    assert reader.answer_model is baseline
