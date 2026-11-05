"""Compact retrieval decisions preserve the canonical final-answer request."""
import json
import pytest
from mm_memory_bench.adaptive_reader import AdaptiveDecision


@pytest.mark.parametrize('search_reply', ['{"action":"stop"}', '{"action":"continue","next_query":"find the other photos"}'])
def test_compact_search_keeps_final_packet_and_tools_unchanged(search_reply):
    packet=[{'role':'system','content':'canonical answer system'}, {'role':'user','content':'original question and evidence'}]
    tools=[{'type':'function','function':{'name':'answer','parameters':{'type':'object'}}}]
    class Reader:
        answer_model=None
        def _generate_answer(self,*args,**kwargs):
            self.answer_model.complete(packet,tools=tools)
    class Model:
        calls=[]
        def complete(self,messages,**kwargs):
            self.calls.append((messages,kwargs))
            if messages==packet:
                assert kwargs=={'tools':tools}
                return 'Canonical final answer'
            assert 'supported' not in messages[0]['content']
            assert 'coverage' not in messages[0]['content']
            control=json.loads(messages[-1]['content'])
            assert 'previous_assessment' not in control
            return search_reply
    model=Model()
    relations={'a':{'belongs_to_memory':'m','same_session_units':[],'semantic_neighbors':[]}}
    decision=AdaptiveDecision(Reader(),model,{},relations,search_decision_mode='compact')
    control=dict(force_answer=False,stop_reason=None,new_unit_ids=['a'])
    result=decision(('a',),2,(),('a',),control)
    if result['action']=='continue':
        assert result==json.loads(search_reply)
        result=decision(('a',),0,(),('a',),dict(force_answer=True,stop_reason='exhausted',new_unit_ids=[]))
    assert result=={'action':'stop','answer':'Canonical final answer'}
    assert decision.search_calls==decision.answer_calls==1
    assert control['search_decision']==json.loads(search_reply)


def test_compact_invalid_query_retries_with_short_instruction():
    class Reader:
        answer_model=None
        def _generate_answer(self,*args,**kwargs):
            self.answer_model.complete([{'role':'user','content':'canonical packet'}])
    class Model:
        calls=0
        def complete(self,messages,**kwargs):
            self.calls+=1
            if self.calls==1: return '{"action":"continue","next_query":" "}'
            assert 'supported/missing/coverage' not in messages[-1]['content']
            return '{"action":"continue","next_query":"blue box"}'
    decision=AdaptiveDecision(Reader(),Model(),{}, {'a':{'belongs_to_memory':'m','same_session_units':[],'semantic_neighbors':[]}},search_decision_mode='compact')
    result=decision(('a',),2,(),('a',),dict(force_answer=False,stop_reason=None,new_unit_ids=['a']))
    assert result=={'action':'continue','next_query':'blue box'}
    assert decision.search_calls==2 and decision.answer_calls==0


def test_compact_hides_remaining_counts_but_keeps_read_to_unread_graph_links():
    class Reader:
        answer_model=None
        def _generate_answer(self,*args,**kwargs):
            self.answer_model.complete([{'role':'user','content':'Evidence 1: already read a'}])
    class Model:
        def complete(self,messages,**kwargs):
            control=json.loads(messages[-1]['content'])
            assert 'remaining_count' not in control
            assert 'unread_candidates' not in json.dumps(control['controller'])
            # Read anchors are still visible, without supplying unread evidence content.
            assert control['relations']['U00']['semantic_neighbors']==[{'unit':'U01','similarity':0.9}]
            assert control['read_units']==[{'alias':'U00','unit_id':'a','evidence_number':1}]
            return '{"action":"continue","next_query":"find the missing object"}'
    relations={'a':{'belongs_to_memory':'m','same_session_units':['b'],
                       'semantic_neighbors':[{'unit_id':'b','similarity':0.9}]}}
    decision=AdaptiveDecision(Reader(),Model(),{},relations,search_decision_mode='compact')
    controller=dict(force_answer=False,stop_reason=None,new_unit_ids=['a'],
                    query_table=[dict(query_id=0,query='q',reads=1,cache_hits=0,unread_candidates=1)])
    decision(('a',),1,(),('a','b'),controller)
    assert controller['query_table'][0]['unread_candidates']==1  # Internal telemetry is intact.


def test_query_switch_and_cached_query_reuse_never_retrieve_read_evidence():
    from mm_memory_bench.adaptive_reader import run_adaptive_reader
    ids=list('abcdefgh')
    vectors={uid:vector for uid,vector in zip(ids,[(1,0),(.99,.1),(.8,.6),(.7,.7),(.6,.8),(.1,.99),(0,1),(-.1,.99)])}
    queries={'q0':[1,0],'q1':[0,1],'q0-alias':[.999,.001],'q2':[-1,0]}
    proposals=iter(['q1','q0','q0-alias','q2'])
    seen=set()
    def decide(visited,remaining,trace,order,control):
        assert len(visited)==len(set(visited))
        assert not (set(control['new_unit_ids']) & seen)
        seen.update(control['new_unit_ids'])
        transition=control['query_transition']
        if transition:
            previously_read={uid for t in trace for uid in t['unit_ids']}
            assert not (set(transition['proposed_next_units']) & previously_read)
            assert not (set(transition['previous_next_units']) & previously_read)
        if control['force_answer']:return {'action':'stop','answer':'done'}
        return {'action':'continue','next_query':next(proposals)}
    result=run_adaptive_reader('q0',ids,vectors,encode=queries.__getitem__,decide=decide,
                               limit=8,batch_size=2,query_table_size=8)
    assert len(result['retrieved_unit_ids'])==len(seen)==8
    assert result['trace'][1]['controller']['cache_match']=='new'
    assert result['trace'][2]['controller']['cache_match']=='exact'
    assert result['trace'][3]['controller']['cache_match']=='semantic'
