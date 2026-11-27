#!/usr/bin/env python3
import sys,os,json,copy,hashlib,time
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path('/root/pairUAV_img/hf_upload_staging'); START=int(sys.argv[1]); GID=f'{START:04d}_{START+4:04d}'
PAYLOAD=BASE/'payload'; REPORTS=BASE/'reports'/GID; REPORTS.mkdir(parents=True,exist_ok=True)
SOURCE=Path('/root/pairUAV_img/processed_paired_json_uniform_tilt'); KROOT=Path('/root/pairUAV_img/processed_kml_uniform_tilt'); IROOT=Path('/root/pairUAV_img/processed_images_uniform_tilt')
def sha_path(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return p.read_bytes()
def stage(rel,data):
 d=PAYLOAD/rel; d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists(): assert d.read_bytes()==data,d
 else:d.write_bytes(data)
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
old=json.loads((SOURCE/f'group_{GID}_validation.json').read_text()); assert old['status']=='validated' and old['full_image_hash_check'] is True and old['pairs']==5000
bindings=old['validated_artifact_sha256']; checked=0
for path,expected in bindings.items(): assert sha_path(Path(path))==expected,path; checked+=1
group=json.loads((SOURCE/f'group_{GID}.json').read_text()); assert group['status']=='complete' and group['pairs']==5000
manifest_by_path={}; ids=[]; pair_ids=set()
for n in range(START,START+5):
 tour=f'{n:04d}'; jp=SOURCE/f'{tour}_group_{GID}.jsonl'; rows=[json.loads(x) for x in jp.read_text().splitlines()]; assert len(rows)==1000
 receipt=json.loads((IROOT/f'{tour}_redesigned/render_receipt.json').read_text())['images']
 original=json.loads((SOURCE/f'{tour}_group_{GID}.json').read_text()); assert rows==original['records']
 rewritten=[]; refs=set()
 for row in rows:
  assert row['tour_id']==tour and row['pair_id'] not in pair_ids; pair_ids.add(row['pair_id']); q=copy.deepcopy(row)
  for suffix in ('a','b'):
   image=f'{row[suffix]:04d}'; refs.add(image)
   ip=IROOT/f'{tour}_redesigned/{image}.jpg'; kp=KROOT/tour/f'{image}.kml'
   # These hashes are independently tied to the validation receipt and report bindings.
   ib=read(ip); kb=read(kp); assert sha_path(ip)==receipt[image]['sha256']; assert sha_path(kp)==receipt[image]['source_sha256']
   for rel,data in ((f'images/{tour}/{image}.jpg',ib),(f'kml/{tour}/{image}.kml',kb)):
    stage(rel,data); manifest_by_path.setdefault(rel,{'path':rel,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
   q[f'image_{suffix}']=f'images/{tour}/{image}.jpg'; q[f'kml_{suffix}']=f'kml/{tour}/{image}.kml'
  rewritten.append(q)
 assert len(refs)==150
 data=(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rewritten)).encode(); rel=f'pairs/{tour}.jsonl'; stage(rel,data); manifest_by_path.setdefault(rel,{'path':rel,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
manifest=list(manifest_by_path.values()); assert len(manifest)==1505
write(REPORTS/'manifest.json',manifest); summary={'status':'validated_and_staged','group':GID,'tours':[f'{n:04d}' for n in range(START,START+5)],'pairs':5000,'images':750,'kml':750,'files':1505,'bytes':sum(x['size'] for x in manifest),'validated_bindings':checked,'unique_pair_ids':len(pair_ids)}; write(REPORTS/'summary.json',summary); write(REPORTS/'READY.json',{'status':'ready','files':1505,'bytes':summary['bytes']}); print(json.dumps(summary),flush=True)
