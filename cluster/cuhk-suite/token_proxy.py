"""Local vLLM proxy with processor-based Qwen3-VL multimodal token counts.

vLLM 0.11.2 /tokenize counts image placeholders, not expanded image tokens.
Only /tokenize is replaced. Generation payloads are forwarded unchanged.
"""
import argparse,base64,hashlib,io,json,threading,urllib.error,urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from PIL import Image
from transformers import AutoProcessor

p=argparse.ArgumentParser();p.add_argument('--model-path',required=True);p.add_argument('--upstream',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
processor=AutoProcessor.from_pretrained(a.model_path,local_files_only=True)
lock=threading.Lock();cache=OrderedDict()
def token_count(payload):
 key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
 with lock:
  if key in cache:return cache[key]
  converted=[]
  for message in payload['messages']:
   content=message['content']
   if isinstance(content,str):content=[{'type':'text','text':content}]
   parts=[]
   for part in content:
    if part['type']=='text':parts.append(part)
    elif part['type']=='image_url':
     url=part['image_url']['url']
     if not url.startswith('data:image/'):raise ValueError('Expected an embedded image')
     with Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))) as image:
      parts.append({'type':'image','image':image.convert('RGB')})
    else:raise ValueError('Unsupported content type: '+part['type'])
   converted.append({'role':message['role'],'content':parts})
  kwargs={'tokenize':True,'add_generation_prompt':payload.get('add_generation_prompt',True),'return_dict':True,'return_tensors':'pt'}
  if payload.get('tools'):kwargs['tools']=payload['tools']
  inputs=processor.apply_chat_template(converted,**kwargs)
  count=int(inputs['input_ids'].shape[-1])
  cache[key]=count
  if len(cache)>64:cache.popitem(last=False)
  return count

class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def respond(self,status,body):
  self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def forward(self,body=None):
  request=urllib.request.Request(a.upstream+self.path,data=body,headers={'Content-Type':'application/json'},method=self.command)
  try:
   with urllib.request.urlopen(request,timeout=1200) as response:self.respond(response.status,response.read())
  except urllib.error.HTTPError as error:self.respond(error.code,error.read())
 def do_GET(self):
  try:self.forward()
  except Exception as error:self.respond(503,json.dumps({'error':str(error)}).encode())
 def do_POST(self):
  try:
   body=self.rfile.read(int(self.headers['Content-Length']))
   if self.path=='/tokenize':
    self.respond(200,json.dumps({'count':token_count(json.loads(body)),'count_source':'qwen3vl_processor_cpu'}).encode())
   else:self.forward(body)
  except Exception as error:self.respond(500,json.dumps({'error':str(error)}).encode())

print(json.dumps({'event':'TOKEN_PROXY_READY','port':a.port,'processor':a.model_path}),flush=True)
ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
