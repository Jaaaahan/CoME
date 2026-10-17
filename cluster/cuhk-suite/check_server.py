from pathlib import Path
import argparse,base64,hashlib,io,json,os,time,urllib.request
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

p=argparse.ArgumentParser();p.add_argument('--base-url',required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--startup-timeout-seconds',type=float,default=3600);a=p.parse_args()
start=time.monotonic()
next_report=start+60
while time.monotonic()-start<a.startup_timeout_seconds:
 os.kill(a.pid,0)
 try:
  with urllib.request.urlopen(a.base_url+'/models',timeout=10) as r:models=json.load(r)
  break
 except Exception:
  if time.monotonic()>=next_report:
   print(json.dumps({'event':'waiting_for_model_service','elapsed_seconds':round(time.monotonic()-start),'timeout_seconds':a.startup_timeout_seconds}),flush=True)
   next_report=time.monotonic()+60
  time.sleep(10)
else:raise RuntimeError('vLLM startup timed out')
assert any(x['id']=='Qwen/Qwen3-VL-8B-Instruct' for x in models['data']),models
buf=io.BytesIO();Image.new('RGB',(56,56),'red').save(buf,format='PNG')
image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
messages=[{'role':'user','content':[{'type':'text','text':'What color are these images? Reply with one word.'}]+[{'type':'image_url','image_url':{'url':image}} for _ in range(12)]}]
def post(url,payload):
 req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
payload={'model':'Qwen/Qwen3-VL-8B-Instruct','messages':messages}
tokens=post(a.base_url.removesuffix('/v1')+'/tokenize',{**payload,'add_generation_prompt':True})
answer=post(a.base_url+'/chat/completions',{**payload,'max_tokens':16,'temperature':0})
assert 'red' in answer['choices'][0]['message']['content'].lower(),answer
assert tokens['count']==answer['usage']['prompt_tokens'],('Multimodal token preflight disagrees with generation',tokens['count'],answer['usage']['prompt_tokens'])
report={'models':models,'twelve_image_tokenize':tokens['count'],'twelve_image_answer':answer,'startup_and_smoke_seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID']}
report['multimodal_token_counts_match']=True
# The benchmark verifier uses constrained JSON, which initializes additional
# server thread pools that an unconstrained image smoke test never exercises.
def check_json(index):
 result=post(a.base_url+'/chat/completions',{
  'model':'Qwen/Qwen3-VL-8B-Instruct',
  'messages':[{'role':'user','content':'Return a JSON object with key status and value ok.'}],
  'response_format':{'type':'json_object'},'temperature':0,'max_tokens':64,
 })
 assert json.loads(result['choices'][0]['message']['content'])=={'status':'ok'},result
 return result
with ThreadPoolExecutor(max_workers=2) as pool:
 report['concurrent_json_checks']=list(pool.map(check_json,range(2)))
report['concurrent_json_checks_passed']=True
report['startup_and_smoke_seconds']=time.monotonic()-start
report['token_proxy_sha256']=hashlib.sha256((Path(__file__).parent/'token_proxy.py').read_bytes()).hexdigest()
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
