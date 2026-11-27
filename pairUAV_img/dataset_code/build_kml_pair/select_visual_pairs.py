#!/usr/bin/env python3
"""Select image-verified pairs with exact marginal bins and the widest feasible tilt interval.

The image screen is generic feature geometry, not semantic building recognition.
"""
from __future__ import annotations
import argparse,json,math,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import coo_matrix
from common import AXES,save_json,sha

def hist_bin(v,lo,hi,n):return max(0,min(n-1,int((v-lo)/(hi-lo)*n)))
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--tour',required=True);ap.add_argument('--count',type=int,default=1200);ap.add_argument('--bins',type=int,default=12);ap.add_argument('--tilt-limit',type=float,help='Test one symmetric tilt interval instead of searching');ap.add_argument('--minimum-tilt-limit',type=float,default=45.0);ap.add_argument('--step',type=float,default=1.0);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--max-pairs-per-image',type=int);ap.add_argument('--audit-dir',type=Path);ap.add_argument('--output-json',type=Path);ap.add_argument('--image-dir',type=Path)
 ap.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml'));ap.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images'));ap.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'))
 a=ap.parse_args();tour=a.tour.zfill(4)
 if a.count<=1000 or a.count%a.bins:ap.error('count must be >1000 and divisible by bins')
 audit=a.audit_dir or (a.json_root/tour/'visual_audit');summary=json.loads((audit/'summary.json').read_text());
 if summary['status']!='visual_screen_only':raise RuntimeError('Expected completed actual-image visual audit')
 edges=[json.loads(line) for line in (audit/'matches.jsonl').open() if json.loads(line).get('passed')]
 if not edges:raise RuntimeError('No visually screened candidates')
 d=json.loads((a.kml_root/tour/'views.json').read_text());keys=list(AXES);quota=a.count//a.bins
 def solve(limit):
  valid=[e for e in edges if abs(e['delta']['tilt_deg'])<=limit+1e-9]
  if len(valid)<a.count:return None,valid
  variables=[(j,o) for j in range(len(valid)) for o in (1,-1)]
  nrows=len(keys)*a.bins+len(valid)+(len(d['views']) if a.max_pairs_per_image is not None else 0);rows=[];cols=[];vals=[]
  offsets={k:i*a.bins for i,k in enumerate(keys)};pair_start=len(keys)*a.bins;degree_start=pair_start+len(valid)
  for c,(j,o) in enumerate(variables):
   e=valid[j]
   for k in keys:
    lo,hi=AXES[k]
    if k=='tilt_deg':lo,hi=-limit,limit
    b=hist_bin(o*e['delta'][k],lo,hi,a.bins)
    rows.append(offsets[k]+b);cols.append(c);vals.append(1.)
   rows.append(pair_start+j);cols.append(c);vals.append(1.)
   if a.max_pairs_per_image is not None:
    for idx in (e['a'],e['b']):rows.append(degree_start+idx);cols.append(c);vals.append(1.)
  mat=coo_matrix((vals,(rows,cols)),shape=(nrows,len(variables))).tocsc()
  lb=[quota]*(len(keys)*a.bins)+[0]*len(valid);ub=[quota]*(len(keys)*a.bins)+[1]*len(valid)
  if a.max_pairs_per_image is not None:
   lb += [0]*len(d['views']); ub += [a.max_pairs_per_image]*len(d['views'])
  cost=np.zeros(len(variables))
  res=milp(cost,integrality=np.ones(len(variables)),bounds=Bounds(0,1),constraints=LinearConstraint(mat,lb,ub),options={'time_limit':120,'mip_rel_gap':0.0,'presolve':True})
  if res.x is None:
   print('MILP status',res.status,res.message,flush=True)
   return None,valid
  chosen=[]
  for c in np.flatnonzero(res.x>.5):
   j,o=variables[c];e=valid[j];q=dict(e);q['a'],q['b']=(e['a'],e['b']) if o==1 else (e['b'],e['a']);q['delta']={k:o*v for k,v in e['delta'].items()};chosen.append(q)
  return chosen,valid
 max_abs=max(abs(e['delta']['tilt_deg']) for e in edges)
 # Search from the widest data-supported symmetric range downward. The first exact feasible solve is the widest at step resolution.
 steps=math.floor(max_abs/a.step)
 chosen=None;limit=None;attempts=[]
 if a.tilt_limit is not None:
  candidates=[round(a.tilt_limit,6)]
 else:
  top=max(max_abs,a.minimum_tilt_limit)
  candidates=[round(i*a.step,6) for i in range(math.floor(top/a.step),-1,-1) if i*a.step+1e-9>=a.minimum_tilt_limit]
  if not candidates: candidates=[round(a.minimum_tilt_limit,6)]
 for candidate in candidates:
  if candidate<=0:continue
  result,valid=solve(candidate);attempts.append({'limit_deg':candidate,'candidates':len(valid),'feasible':result is not None})
  print(f'tilt [-{candidate:.2f},{candidate:.2f}] candidates={len(valid)} feasible={result is not None}',flush=True)
  if result is not None:chosen=result;limit=candidate;break
 if chosen is None:raise RuntimeError(f'No >1000 exact balanced visually-screened pairs at {a.bins} bins with tilt half-range >= {a.minimum_tilt_limit}; tried {len(attempts)} tilt ranges')
 # Verify exact marginal balance and endpoint support before publishing.
 ranges={k:(-limit,limit) if k=='tilt_deg' else v for k,v in AXES.items()};hist={k:[0]*a.bins for k in keys}
 for e in chosen:
  for k,(lo,hi) in ranges.items():hist[k][hist_bin(e['delta'][k],lo,hi,a.bins)]+=1
 if any(v!=[quota]*a.bins for v in hist.values()):raise RuntimeError(f'Histogram verification failed: {hist}')
 selected_tilts=[e['delta']['tilt_deg'] for e in chosen]
 old=a.output_json or (a.json_root/f'{tour}.json');backup=a.json_root/f'{tour}_unfiltered_legacy.json'
 if a.output_json is None and old.exists() and not backup.exists():shutil.copy2(old,backup)
 records=[];out=a.image_dir or (a.image_root/tour)
 for idx,e in enumerate(chosen):
  q={k:e[k] for k in ('a','b','delta','inliers','matches','inlier_ratio','hull_fraction','grid_cells','model')};q.update(pair_id=f'{tour}_visual_{idx:06d}',tour_id=tour,kml_a=str(a.kml_root/tour/f'{e["a"]:04d}.kml'),kml_b=str(a.kml_root/tour/f'{e["b"]:04d}.kml'),image_a=str(out/f'{e["a"]:04d}.jpg'),image_b=str(out/f'{e["b"]:04d}.jpg'),anchor=d['anchor'],seed=d['seed_sha256'],visual_screen='SIFT mutual ratio + USAC_MAGSAC; see visual_audit/summary.json');records.append(q)
 result={'schema_version':'3.0','status':'complete_visual_screen','tour_id':tour,'seed_sha256':d['seed_sha256'],'images_rendered':len(d['views']),'pairs':len(records),'records':records,'distribution':hist,'bins':a.bins,'bin_semantics':'equal-width marginal bins; exact equal count per bin','visual_screen':{'method':summary['schema_version'],'thresholds':summary['thresholds'],'accepted_candidate_edges':len(edges),'selected_pairs':len(records),'audit_summary':str(audit/'summary.json'),'limitations':summary['limitations']},'tilt_difference_range_deg':[-limit,limit],'tilt_selected_minmax_deg':[min(selected_tilts),max(selected_tilts)],'tilt_search':{'minimum_allowed_symmetric_limit_deg':a.minimum_tilt_limit,'widest_candidate_supported_symmetric_limit_deg':max_abs,'step_deg':a.step,'attempts':attempts,'selection':'first feasible exact balanced MILP while searching from widest candidate-supported range downward'},'uniformity_limitations':['Uniform marginal bins are not a uniform joint distribution.',f'{a.bins} bins are marginal bins only; the five-way joint distribution is not constrained to be uniform.','Image feature geometry is not semantic building identity or proof of uniquely observable metric ENU pose.']}
 save_json(old,result);old.with_suffix('.jsonl').write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in records)+'\n')
 print(json.dumps({'tour':tour,'pairs':len(records),'bins':a.bins,'uniform_per_bin':quota,'tilt_range_deg':result['tilt_difference_range_deg'],'selected_tilt_minmax_deg':result['tilt_selected_minmax_deg'],'json':str(old),'legacy_backup':str(backup) if backup.exists() else None},indent=2))
if __name__=='__main__':main()
