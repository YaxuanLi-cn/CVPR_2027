#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from PIL import Image
from common import AXES,eligible

def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images'));p.add_argument('--image-dir',type=Path);p.add_argument('--require-images',action='store_true');a=p.parse_args();tour=str(a.tour).zfill(4);d=json.loads((a.json_root/f'{tour}.json').read_text());r=d.get('records',[]);assert len(r)>1000,f'need >1000 pairs, got {len(r)}'; bins=len(d.get('distribution',{}).get('x_m',[])) or 12; hist={k:[0]*bins for k in AXES}; tilt_range=d.get('tilt_difference_range_deg',[-60,60]); ranges={k:(tilt_range if k=='tilt_deg' else v) for k,v in AXES.items()}; seen=set()
 if d.get('status')=='complete_visual_screen':
  t=d.get('visual_screen',{}).get('thresholds',{}); assert t and d['visual_screen']['selected_pairs']==len(r), 'visual screening metadata missing'
  for q in r:
   assert q.get('inliers',0)>=t['min_inliers'] and q.get('inlier_ratio',0)>=t['min_inlier_ratio'] and q.get('hull_fraction',0)>=t['min_hull_fraction'] and q.get('grid_cells',0)>=t['min_grid_cells'], f"pair lacks required visual evidence: {q.get('pair_id')}"
   pair=tuple(sorted((q['a'],q['b'])));assert pair not in seen, f'duplicate unordered pair: {pair}';seen.add(pair)
 for q in r:
  x=q['delta'];assert eligible(x),f'out of range: {x}'
  for k,(lo,hi) in AXES.items():
   if k=='tilt_deg': lo,hi=tilt_range
   hist[k][max(0,min(bins-1,int((x[k]-lo)/(hi-lo)*bins)))]+=1
 expected=len(r)//bins
 for k,v in hist.items(): assert len(set(v))==1 and v[0]*bins==len(r),f'{k} not uniform: {v}'
 if a.require_images:
  n=d['images_rendered'];
  for i in range(n):
   with Image.open((a.image_dir or (a.image_root/tour))/f'{i:04d}.jpg') as im: assert im.size==(3686,2073);im.verify()
 print(json.dumps({'tour':tour,'images':d.get('images_rendered'), 'pairs':len(r),'bins':hist,'uniform_per_bin':expected,'declared_tilt_range_deg':tilt_range},indent=2));print('PASS')
if __name__=='__main__':main()
