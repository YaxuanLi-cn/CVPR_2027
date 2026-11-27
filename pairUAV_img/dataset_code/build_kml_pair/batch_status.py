#!/usr/bin/env python3
"""Read-only progress: verified receipts, committed exports, and completed groups."""
import argparse,json,time
from pathlib import Path

def load(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}

def main():
 p=argparse.ArgumentParser();p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'));p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images_relaxed_batch'));a=p.parse_args()
 state=load(a.json_root/'group_batch_manifest.json');complete=[g for g in state.get('groups',[]) if g['status']=='complete'];active=[g for g in state.get('groups',[]) if g['status']!='complete']
 print(json.dumps({'completed_groups':len(complete),'completed_tours':sum(len(g['tours']) for g in complete),'configuration':state.get('configuration'),'groups':state.get('groups')},indent=2))
 for g in active:
  for tour in g['tours']:
   idir=a.image_root/f'{tour}_redesigned';receipt=load(idir/'render_receipt.json').get('images',{});exports={}
   for m in (idir/'earth_pro_export/runs').glob('*/manifest.json'):
    for e in load(m).get('images',[]):exports[Path(e['source']).stem]=e
   if receipt or exports:
    newest=max((Path(e['output']).stat().st_mtime for e in exports.values() if Path(e['output']).exists()),default=0)
    print(json.dumps({'tour':tour,'receipt_entries':len(receipt),'manifest_committed_unique_images':len(set(receipt)|set(exports)),'latest_export_age_seconds':round(time.time()-newest) if newest else None,'note':'Counts from manifests; use validate_group.py for full hash and pair verification.'}))
if __name__=='__main__':main()
