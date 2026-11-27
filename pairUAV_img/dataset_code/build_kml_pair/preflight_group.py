#!/usr/bin/env python3
"""Generate deterministic KML views and preflight group capacity before rendering."""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from common import AXES,sha,source_files
from solver_utils import solve_feasibility
from pathlib import Path

def migrate_unrendered_design(d, out, tour, cmd, image_root, kml_root, images):
 # Migrate legacy assumed terrain only before any image has been rendered.
 # Keep the old design until a complete replacement has been generated.
 idir=image_root/f'{tour}_redesigned'
 has_images=any(idir.rglob('*.jpg')) if idir.exists() else False
 if not d['anchor'].get('terrain_calibration') and not has_images:
  with tempfile.TemporaryDirectory(prefix='terrain_design_',dir=kml_root) as tmp:
   revised=list(cmd);revised[revised.index('--kml-root')+1]=tmp
   subprocess.run(list(map(str,revised)),check=True)
   replacement=Path(tmp)/tour
   checked=json.loads((replacement/'views.json').read_text())
   if len(checked['views'])!=images or len(list(replacement.glob('*.kml')))!=images:
    raise ValueError(f'{tour}: incomplete calibrated replacement')
   backup=kml_root/'.legacy_terrain'/tour
   backup.parent.mkdir(exist_ok=True)
   if backup.exists():raise FileExistsError(f'Unresolved terrain migration: {backup}')
   out.rename(backup)
   try:replacement.rename(out)
   except BaseException:
    backup.rename(out)
    raise
   d=checked
   print(f'{tour}: calibrated unrendered design; prior design retained at {backup}',flush=True)
 return d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tours',nargs='+',required=True);ap.add_argument('--seed',type=int,default=49);ap.add_argument('--images',type=int,default=150);ap.add_argument('--per-tour',type=int,default=1000);ap.add_argument('--bins',type=int,default=5);ap.add_argument('--max-pairs-per-image',type=int,default=150);ap.add_argument('--tilt-limit',type=float,default=45);ap.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'));ap.add_argument('--kml-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--audit-root',type=Path);ap.add_argument('--audited-tours',nargs='*',default=[]);ap.add_argument('--partial-tours',nargs='*',default=[]);ap.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images_relaxed_batch'));ap.add_argument('--no-repair',action='store_true');a=ap.parse_args()
 if a.bins<5 or a.per_tour<1000 or a.per_tour*len(a.tours)%a.bins:ap.error('use at least 5 bins and 1000 pairs per tour')
 a.tours=[str(t).zfill(4) for t in a.tours];a.audited_tours=[str(t).zfill(4) for t in a.audited_tours]
 if len(set(a.tours))!=len(a.tours):ap.error('Duplicate tours')
 if a.audited_tours and (a.audit_root is None or not set(a.audited_tours)<=set(a.tours)):ap.error('Audited tours require audit-root and must belong to group')
 if a.partial_tours and (a.audit_root is None or a.image_root is None or not set(a.partial_tours)<=set(a.tours)):ap.error('Partial tours need audit/image roots and group membership')
 from repair_capacity import recover_commit
 recover_commit(a.kml_root,a.tours)
 b=Path(__file__).resolve().parent;py=b/'.venv/bin/python';rows=[]
 for t in a.tours:
  tour=str(t).zfill(4);out=a.kml_root/tour
  cmd=[str(py),str(b/'design_views.py'),'--tour',tour,'--seed',str(a.seed+int(tour)),'--images',str(a.images),'--input-root',a.input_root,'--kml-root',a.kml_root,'--camera-height',500,'--position-radius',15,'--tilt-min',0,'--tilt-max',45,'--paired-tilt-groups','--tilt-group-count',15,'--edge-fraction','.8','--hfov',90]
  if not (out/'views.json').exists(): subprocess.run(list(map(str,cmd)),check=True)
  d=json.loads((out/'views.json').read_text());
  d=migrate_unrendered_design(d,out,tour,cmd,a.image_root,a.kml_root,a.images)
  if len(d['views'])!=a.images or d['seed']!=a.seed+int(tour):raise ValueError(f'{tour}: existing design does not match requested image count/seed')
  if d['anchor']['source_sha256']!=sha(source_files(a.input_root,tour)[0]):raise ValueError(f'{tour}: source first-key changed')
  edges=[e for e in d['candidate_edges'] if abs(e['delta']['tilt_deg'])<=a.tilt_limit+1e-9]; rows.append({'tour':tour,'images':len(d['views']),'geometric_candidates':len(edges),'candidate_edges':len(d['candidate_edges']),'seed_sha256':d['seed_sha256']})
# Solve the same group balance using geometry edges only. This is a preflight
 # gate; image audit still runs after rendering.
 ds={r['tour']:json.loads((a.kml_root/r['tour']/'views.json').read_text()) for r in rows}; all_edges=[]
 for r in rows:
  candidates=ds[r['tour']]['candidate_edges']
  if r['tour'] in a.audited_tours:
   ad=a.audit_root/r['tour']/'visual_audit_redesigned'
   summary=json.loads((ad/'summary.json').read_text())
   if summary['views_sha256']!=sha(a.kml_root/r['tour']/'views.json'):raise ValueError('Stale visual audit')
   candidates=[]
   for line in (ad/'matches.jsonl').open():
    e=json.loads(line)
    if e['passed']:candidates.append({k:e[k] for k in ('a','b','delta')})
   r['visual_candidates']=len(candidates)
  elif r['tour'] in a.partial_tours:
   from partial_visual_audit import read_cache
   import audit_visual_pairs as visual
   from group_batch_pipeline import THRESHOLDS
   visual.THRESHOLDS.update(THRESHOLDS)
   paths=[a.image_root/f"{r['tour']}_redesigned"/(v['id']+'.jpg') for v in ds[r['tour']]['views']]
   checked=read_cache(a.audit_root/r['tour']/'visual_audit_redesigned',a.kml_root/r['tour']/'views.json',paths,visual.THRESHOLDS)
   if not checked:raise ValueError('No valid partial audit records')
   candidates=[e for e in candidates if (e['a'],e['b']) not in checked or checked[(e['a'],e['b'])]['passed']]
   r['partial_checked_edges']=len(checked);r['optimistic_remaining_edges']=len(candidates)
  for e in candidates:
   if abs(e['delta']['tilt_deg'])<=a.tilt_limit+1e-9:
    q=dict(e);q['tour']=r['tour'];all_edges.append(q)
 T=len(rows); B=a.bins; axes=list(AXES); vars=[(j,o) for j in range(len(all_edges)) for o in (1,-1)]; n=len(vars); hist=len(axes)*B; tour0=hist; edge0=tour0+T; deg0=edge0+len(all_edges); offsets={r['tour']:sum(len(ds[z['tour']]['views']) for z in rows[:i]) for i,r in enumerate(rows)}; rr=[];cc=[];vv=[]
 for c,(j,o) in enumerate(vars):
  e=all_edges[j]; ti=[z['tour'] for z in rows].index(e['tour'])
  for ai,k in enumerate(axes):
   lo,hi=AXES[k]
   if k=='tilt_deg':lo,hi=-a.tilt_limit,a.tilt_limit
   b=max(0,min(B-1,int((o*e['delta'][k]-lo)/(hi-lo)*B)));rr.append(ai*B+b);cc.append(c);vv.append(1.)
  rr.append(tour0+ti);cc.append(c);vv.append(1.);rr.append(edge0+j);cc.append(c);vv.append(1.)
  for idx in (e['a'],e['b']):rr.append(deg0+offsets[e['tour']]+idx);cc.append(c);vv.append(1.)
 nviews=sum(len(ds[r['tour']]['views']) for r in rows); A=coo_matrix((vv,(rr,cc)),shape=(deg0+nviews,n)).tocsc();q=(a.per_tour*T)//B; lb=np.r_[[q]*(hist),[a.per_tour]*T,[0]*len(all_edges),[1 if ds[r['tour']].get('image_tilt_constraint') else 0 for r in rows for _ in ds[r['tour']]['views']]];ub=np.r_[[q]*(hist),[a.per_tour]*T,[1]*len(all_edges),[a.max_pairs_per_image]*nviews]
 res,feasible,attempts=solve_feasibility(A,lb,ub,120)
 result={'status':'geometry_preflight','geometry_feasible':feasible,'audited_tours':a.audited_tours,'partial_tours':a.partial_tours,'remaining_tours_use_optimistic_geometry':len(a.audited_tours)<len(a.tours),'solver_status':res.message,'solver_attempts':attempts,'feasibility_status':'feasible' if feasible else ('infeasible' if res.status==2 else 'unknown'),'tours':[str(x).zfill(4) for x in a.tours],'per_tour':a.per_tour,'bins':a.bins,'tilt_limit_deg':a.tilt_limit,'rows':rows,'group_target_per_bin':q,'render_recommendation':'render only when geometry_feasible=true'}
 from capacity_diagnostics import capacity_diagnostics
 result['capacity_diagnostics']=capacity_diagnostics(all_edges,a.tours,a.per_tour,a.bins,a.tilt_limit)
 if not feasible and res.status==2 and not a.no_repair:
  from repair_capacity import repair
  repaired=repair(a,result)
  if repaired is not None:result=repaired
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
