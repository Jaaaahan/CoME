"""Replay the failing two-image request through the real server and proxy."""
import argparse, base64, io, json, urllib.error, urllib.request
from pathlib import Path
from PIL import Image
from qwen_image_padding import POLICY
p=argparse.ArgumentParser()
p.add_argument('--upstream', required=True)
p.add_argument('--proxy', required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
def post(url, body):
    request=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response:
        return json.load(response)
buf=io.BytesIO();Image.new('RGB',(1280,6),(80,100,120)).save(buf,format='JPEG')
url='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()
payload={'model':'Qwen/Qwen3-VL-8B-Instruct','messages':[{'role':'user','content':[
    {'type':'text','text':'Briefly describe these images.'},
    {'type':'image_url','image_url':{'url':url}},
    {'type':'image_url','image_url':{'url':url}}]}],'max_tokens':16,'temperature':0}
try:
    post(a.upstream+'/v1/chat/completions',payload)
except urllib.error.HTTPError as exc:
    error=exc.read().decode()
    assert exc.code==400 and 'Qwen3VLProcessor' in error, error
else:
    raise AssertionError('Expected the unpadded regression request to reproduce HTTP 400')
tokenized=post(a.proxy+'/tokenize',payload)
assert tokenized['count'] > 0
result=post(a.proxy+'/v1/chat/completions',payload)
assert result['choices'][0]['message'].get('content'), result
report={'status':'passed','policy':POLICY,'original_size':[1280,6],
        'unpadded_status':400,'padded_status':200,'token_count':tokenized['count']}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
