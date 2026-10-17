"""File staging only; run on CUHK storage node."""
from pathlib import Path
import concurrent.futures, json, os, time, urllib.request, urllib.parse

ROOT=Path(os.environ.get('GRAPHMEMIX_STAGING_ROOT','/share/workspace3/yhjia/graphmemix'))
REVISION='1cff9951fa3deb404c049e7e507f65c38b4751dd'
entries=json.loads((Path(__file__).parent/'additional-data-manifest.json').read_text())
entries.sort(key=lambda row:(not row['path'].startswith('data/unified/'),row['path']))
opener=urllib.request.build_opener()
def fetch(entry):
    target=ROOT/entry['path']
    if target.exists() and target.stat().st_size==entry['size']:
        return 'existing'
    target.parent.mkdir(parents=True,exist_ok=True)
    url='https://huggingface.co/datasets/oking0197/graphmemix-benchmarks/resolve/'+REVISION+'/'+urllib.parse.quote(entry['path'])
    for attempt in range(5):
        try:
            with opener.open(url,timeout=90) as response, target.with_suffix(target.suffix+'.download').open('wb') as output:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    output.write(chunk)
            partial=target.with_suffix(target.suffix+'.download')
            assert partial.stat().st_size==entry['size'],str(target)
            partial.replace(target)
            return 'downloaded'
        except Exception:
            if attempt==4:raise
            time.sleep(2*(attempt+1))
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
    for n,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,entry) for entry in entries]),1):
        result=future.result()
        if n%100==0 or n==len(entries):
            print(json.dumps({'done':n,'total':len(entries),'seconds':round(time.monotonic()-start)}),flush=True)
print('DATA_DOWNLOAD_COMPLETE',flush=True)
