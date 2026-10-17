from pathlib import Path
import collections, hashlib, json, os, socket, time
import torch
from mm_memory_bench.core import validate_bundle
from mm_memory_bench.reader import BundleReader
from mm_memory_bench.harness import _resolved_memory

root=Path('/share/workspace3/yhjia/graphmemix')
suite=Path(__file__).parent
experiment=root/'experiments/standalone-4090-20261015'
report={'job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'bundles':{},'model_files':[]}
assert torch.cuda.is_available()
report['gpu']=torch.cuda.get_device_name(0)
assert '4090' in report['gpu'], report['gpu']
x=torch.arange(256,dtype=torch.float32,device='cuda').reshape(16,16)
torch.testing.assert_close((x@x.T).cpu(),x.cpu()@x.cpu().T)
report['cuda_matmul']='passed'
for model,entries in [
 ('gme-Qwen2-VL-2B-Instruct',[{'file':r['path'],'expected':r['lfs']['oid']} for r in json.loads((root/'models/gme-Qwen2-VL-2B-Instruct/weights-manifest.json').read_text())]),
 ('Qwen3-VL-8B-Instruct',json.loads((suite/'qwen-weights-reference.json').read_text())['files']),
]:
 for entry in entries:
  with (root/'models'/model/entry['file']).open('rb') as handle: actual=hashlib.file_digest(handle,'sha256').hexdigest()
  assert actual==entry['expected'], (model,entry['file'])
  report['model_files'].append({'model':model,'file':entry['file'],'sha256':actual})
print('MODEL_HASHES_VERIFIED',flush=True)
entries=json.loads((suite/'additional-data-manifest.json').read_text())+json.loads((suite.parent/'atm-media-manifest.json').read_text())
for n,entry in enumerate(entries,1):
 target=root/entry['path'] if entry['path'].startswith('data/') else root/'data'/entry['path']
 assert target.is_file() and target.stat().st_size==entry['size'],str(target)
 if entry.get('sha256'):
  with target.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
  assert actual==entry['sha256'],str(target)
 if n%1000==0:print(json.dumps({'verified_files':n,'total':len(entries)}),flush=True)
report['verified_data_files']=len(entries)
for name,expected in [('atm_bench',1044),('memeye',1855),('mem_gallery',1711)]:
 bundle=root/'data/unified'/name
 validation=validate_bundle(bundle,check_assets=True)
 reader=BundleReader(bundle)
 memories=[json.loads(l) for l in (bundle/'memories.jsonl').read_text().splitlines()]
 questions=[json.loads(l) for l in (bundle/'questions.jsonl').read_text().splitlines()]
 assert len(questions)==expected and len({q['question_id'] for q in questions})==expected
 resolved=collections.Counter(part['type'] for m in memories for part in _resolved_memory(reader,m,memory_view='derived')['content'])
 report['bundles'][name]={'questions':len(questions),'memories':len(memories),'validation':validation,'resolved_memory_content_types':dict(resolved),'question_content_types':dict(collections.Counter(p['type'] for q in questions for p in q['prompt']))}
output=experiment/'validation'/f'inputs-{os.environ["SLURM_JOB_ID"]}.json'
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
