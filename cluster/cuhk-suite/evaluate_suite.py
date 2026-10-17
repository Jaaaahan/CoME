"""Evaluate completed full-data runs; never interpret failed answers as scores."""
from pathlib import Path
import argparse,collections,hashlib,json,statistics,sys

suite=Path(__file__).parent
sys.path.insert(0,str(suite/'atm_evaluator'))
from memqa.utils.evaluator.evaluate_qa import deterministic_accuracy,list_jaccard_score
import memeye_evaluator as native
from evaluate_atm_retrieval import metrics_for

root=Path('/share/workspace3/yhjia/graphmemix')
experiment=root/'experiments/standalone-4090-20261015'
parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['atm','memeye','mem_gallery']);args=parser.parse_args()
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def mean(values):return statistics.mean(values) if values else None
report={'protocol':'graphmemix-standalone-cuhk-full-1','backbone':'Qwen/Qwen3-VL-8B-Instruct','memory_view':'derived (published configuration; resolved media counts in input-validation reports)','official_llm_judge_run':False,'memeye_evaluator_commit':'0358e70d714980980dfd8c87903384db474f0b16','memeye_evaluator_sha256':hashlib.sha256((suite/'memeye_evaluator.py').read_bytes()).hexdigest(),'atm_evaluator':json.loads((suite/'atm_evaluator/provenance.json').read_text()),'datasets':{}}
for dataset,bundle,expected in [('atm','atm_bench',1044),('memeye','memeye',1855),('mem_gallery','mem_gallery',1711)]:
 if args.dataset and dataset!=args.dataset:continue
 questions={r['question_id']:r for r in rows(root/'data/unified'/bundle/'questions.jsonl')}
 assert len(questions)==expected
 record={'questions':expected,'variants':{}}
 graph=experiment/'runs/qwen3vl8b'/dataset
 split_by_id={r['question_id']:r['split'] for r in rows(graph/'node_scores.jsonl')}
 record['published_calibration_split_counts']=dict(collections.Counter(split_by_id.values()))
 for stage in ('node_scores.jsonl','ecv_scores.jsonl'):
  stage_rows=rows(graph/stage)
  assert len(stage_rows)==expected and {r['question_id'] for r in stage_rows}==set(questions)
  assert not any(r.get('error') or r.get('status')=='error' for r in stage_rows),(dataset,stage)
 ecv=rows(graph/'ecv_scores.jsonl');pruned=rows(graph/'pruning/root0.12_node0.jsonl')
 record['graph_diagnostics']={'relation_store':json.loads((experiment/'artifacts/relation_store'/f'{dataset}.summary.json').read_text()),'ecv_eligible_candidates':sum(r.get('eligible_edge_candidates',0) for r in ecv),'selected_edges':sum(r['metadata']['chosen_edges'] for r in pruned),'mean_selected_memories':mean([len(r['retrieved_memory_ids']) for r in pruned])}
 for name,path in [('graphmemix',graph/'reader/predictions.jsonl'),('dense',experiment/'runs/dense'/dataset/'predictions.jsonl')]:
  predictions=rows(path);by_id={r['question_id']:r for r in predictions}
  assert len(predictions)==len(by_id)==expected and set(by_id)==set(questions)
  assert all(r.get('prediction','').strip() and r.get('metadata',{}).get('status')!='error' for r in predictions),(dataset,name,'failed predictions')
  groups=collections.defaultdict(list)
  heldout_groups=collections.defaultdict(list)
  for qid,q in questions.items():
   answer=by_id[qid]['prediction'];truth=str(q['answer']['text']);prompt='\n'.join(p.get('text','') for p in q['prompt'] if p['type']=='text')
   def add(key,value):
    groups[key].append(value)
    if split_by_id[qid]=='heldout':heldout_groups[key].append(value)
   if dataset=='atm':
    em=float(deterministic_accuracy(truth,answer,prompt));add('all_deterministic_match_diagnostic',em)
    kind=q['task']['subcategory']
    if kind=='number':add('number_date_accuracy',em)
    elif kind=='list_recall':add('list_jaccard',list_jaccard_score(truth,answer))
    else:add('open_end_deterministic_match_diagnostic',em)
   elif dataset=='memeye' and q['subset']=='mcq':
    valid={str(c['choice_id']).upper() for c in q['choices']}
    add('mcq_accuracy_four_rotations',float(native.extract_choice(answer,valid)==truth.upper()))
   else:
    prefix='open_' if dataset=='memeye' else ''
    add(prefix+'exact_match',float(native.score_open(answer,truth)[0]))
    add(prefix+'token_f1',native.f1_score(answer,truth))
    add(prefix+'bleu1',native.bleu_score(answer,truth,weights=(1.,0.,0.,0.)))
  heldout_questions=[q for qid,q in questions.items() if split_by_id[qid]=='heldout']
  record['variants'][name]={'completed':len(predictions),'retrieval':metrics_for(questions.values(),by_id,10),'heldout_retrieval':metrics_for(heldout_questions,by_id,10),'answer_metrics':{key:{'value':mean(values),'questions':len(values)} for key,values in groups.items()},'heldout_answer_metrics':{key:{'value':mean(values),'questions':len(values)} for key,values in heldout_groups.items()},'predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 report['datasets'][dataset]=record
output=experiment/'metrics'/(f'{args.dataset}.json' if args.dataset else 'suite-results.json')
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
