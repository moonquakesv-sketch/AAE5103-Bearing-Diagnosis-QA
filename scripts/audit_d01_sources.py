"""D01 official-page/MAT audit only; no slicing, model fitting or evaluation."""
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json, sys, platform, subprocess, time
import numpy as np
import scipy, scipy.io, pywt, pandas, yaml, torch, torchvision, sklearn
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/d01'
BASE = 'https://engineering.case.edu/bearingdatacenter/'
PAGES = ['download-data-file','normal-baseline-data','12k-drive-end-bearing-fault-data',
         'apparatus-and-procedures','fault-specifications','bearing-information']
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]; self.href=None; self.label=''
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.href=dict(attrs).get('href'); self.label=''
    def handle_data(self,data):
        if self.href: self.label+=data
    def handle_endtag(self,tag):
        if tag=='a' and self.href:
            self.links.append((self.label.strip(),self.href)); self.href=None

def sha(b): return hashlib.sha256(b).hexdigest()
def fetch(url,path):
    if path.exists():
        body=path.read_bytes()
        return body, {'source_url':url,'resolved_url':url,'http_status':None,
            'retrieval':'reused D01 download from interrupted audit', 'bytes':len(body),'sha256':sha(body)}
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'AAE5103 academic metadata audit'}),timeout=60) as r:
                body=r.read(); meta={'source_url':url,'resolved_url':r.url,'http_status':r.status,
                    'content_type':r.headers.get('Content-Type'),'bytes':len(body),'sha256':sha(body)}
            break
        except Exception:
            if attempt==2: raise
            time.sleep(2)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(body)
    return body,meta

def main():
    report={'checked_at':datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
      'scope':'D01 metadata/identity audit; no classification or test scoring', 'pages':{},'records':[]}
    links={}
    for name in PAGES:
        body,meta=fetch(BASE+name,OUT/'sources'/f'{name}.html')
        report['pages'][name]=meta
        parser=Links();parser.feed(body.decode());links[name]=dict(parser.links)
    for label,prefix,page in [('normal','Normal','normal-baseline-data'),
        ('inner_race','IR007','12k-drive-end-bearing-fault-data'),
        ('outer_race','OR007@6','12k-drive-end-bearing-fault-data'),
        ('rolling_element','B007','12k-drive-end-bearing-fault-data')]:
        for hp in range(4):
            name=f'{prefix}_{hp}'; url=urljoin(BASE,links[page][name])
            filename=Path(urlparse(url).path).name; number=int(filename.split('.')[0])
            path=ROOT/'data/raw/cwru'/filename
            body,meta=fetch(url,path)
            data=scipy.io.loadmat(path); key=f'X{number:03d}_DE_time'
            if key not in data: raise ValueError(f'{filename}: expected {key}, got {list(data)}')
            arrays={}
            for k,v in data.items():
                if k.startswith('__'): continue
                a=np.asarray(v)
                arrays[k]={'shape':list(a.shape),'dtype':str(a.dtype),
                    'sha256_float64':sha(np.asarray(a,dtype='<f8').ravel().tobytes())}
                if a.size==1: arrays[k]['scalar']=float(a.ravel()[0])
            v=np.asarray(data[key]).ravel()
            row={'official_name':name,'true_label':label,'load_hp':hp,'file':filename,
              'record_id':f'cwru_{number:03d}','channel':key,'raw_path':str(path.relative_to(ROOT)),
              'n_samples':v.size,'finite':bool(np.isfinite(v).all()),'arrays':arrays,**meta}
            report['records'].append(row)
            print(name,filename,key,v.size,'keys',','.join(arrays),flush=True)
    all_arrays={}
    for r in report['records']:
        for k,a in r['arrays'].items():
            if k.endswith('_time'):
                all_arrays.setdefault(a['sha256_float64'],[]).append(f"{r['file']}:{k}")
    report['duplicate_arrays']=[v for v in all_arrays.values() if len(v)>1]
    selected=[r['arrays'][r['channel']]['sha256_float64'] for r in report['records']]
    report['selected_channel_hashes_unique']=len(set(selected))==len(selected)
    report['environment']={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
       'versions':{m.__name__:m.__version__ for m in (np,scipy,pywt,pandas,yaml,torch,torchvision,sklearn)},
       'mps_available':torch.backends.mps.is_available(),'mps_built':torch.backends.mps.is_built()}
    check=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
    report['environment']['pip_check']={'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr}
    (OUT/'source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('duplicates',report['duplicate_arrays']);print('selected hashes unique',report['selected_channel_hashes_unique'])
    print('wrote runs/d01/source_audit.json')
if __name__=='__main__': main()
