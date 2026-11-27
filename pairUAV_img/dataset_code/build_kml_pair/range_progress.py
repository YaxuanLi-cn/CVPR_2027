#!/usr/bin/env python3
"""Read manifest progress without confusing exported images with validated pairs."""
import argparse,json,time
from pathlib import Path
from group_batch_pipeline import load

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=160);p.add_argument('--end',type=int,default=239)
 p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json_uniform_tilt'));p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images_uniform_tilt'));a=p.parse_args()
 state=load(a.json_root/f'range_{a.start:04d}_{a.end:04d}_progress.json',{})
 complete=[g for g,v in state.get('groups',{}).items() if v.get('status')=='complete']
 gid=state.get('active_group');active=load(a.json_root/f'group_{gid}_progress.json',{}) if gid else {}
 rows=[]
 for t in active.get('tours',[]):
  im=a.image_root/f'{t}_redesigned';ids=set(load(im/'render_receipt.json',{'images':{}}).get('images',{}))
  for p in (im/'earth_pro_export/runs').glob('*/manifest.json'):
   for e in load(p,{}).get('images',[]):ids.add(Path(e['source']).stem)
  ad=a.json_root/t/'visual_audit_redesigned'
  summary=load(ad/'summary.json',{}) or load(ad/'partial_summary.json',{})
  rows.append({'tour':t,'recorded_images':len(ids),'audited_images':summary.get('images',summary.get('verified_images',0)),'visual_pass':summary.get('accepted_pairs',summary.get('accepted_edges',0))})
 print(json.dumps({'range':[a.start,a.end],'status':state.get('status'),'completed_tours':len(complete)*5,'completed_pairs':len(complete)*5000,'active_group':gid,'stage':active.get('status'),'tour':active.get('tour'),'tours':rows,'error':active.get('error',state.get('error')),'counts_are_manifest_based':True},ensure_ascii=False))
if __name__=='__main__':main()
