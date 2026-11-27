#!/usr/bin/env python3
"""Compute exact max equally binned marginal subset from cached visual-pair audit."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import coo_matrix
from common import AXES,save_json

def solve(edges,nviews,bins,tilt_limit,time_limit,max_pairs_per_image):
 valid=[e for e in edges if abs(e['delta']['tilt_deg'])<=tilt_limit+1e-9]
 pairs=[(j,o) for j in range(len(valid)) for o in (1,-1)]; n=len(pairs)
 axes=list(AXES); histrows=len(axes)*bins; pairstart=histrows; degstart=pairstart+len(valid); rows=[];cols=[];vals=[]
 for c,(j,o) in enumerate(pairs):
  e=valid[j]
  for ai,k in enumerate(axes):
   lo,hi=AXES[k]
   if k=='tilt_deg':lo,hi=-tilt_limit,tilt_limit
   v=o*e['delta'][k]; b=max(0,min(bins-1,int((v-lo)/(hi-lo)*bins)))
   rows.append(ai*bins+b);cols.append(c);vals.append(1.)
  rows.append(pairstart+j);cols.append(c);vals.append(1.)
  for image in (e['a'],e['b']):rows.append(degstart+image);cols.append(c);vals.append(1.)
 # integer q subtracts from every histogram bin; maximizing q maximizes five-axis balanced set
 qcol=n
 for r in range(histrows):rows.append(r);cols.append(qcol);vals.append(-1.)
 A=coo_matrix((vals,(rows,cols)),shape=(histrows+len(valid)+nviews,n+1)).tocsc()
 lb=np.zeros(A.shape[0]);ub=np.zeros(A.shape[0]);ub[pairstart:degstart]=1;ub[degstart:]=max_pairs_per_image
 objective=np.zeros(n+1);objective[qcol]=-1
 result=milp(objective,integrality=np.ones(n+1),bounds=Bounds(np.zeros(n+1),np.r_[np.ones(n),max_pairs_per_image]),constraints=LinearConstraint(A,lb,ub),options={'time_limit':time_limit,'mip_rel_gap':.001,'presolve':True})
 q=int(round(result.x[qcol])) if result.x is not None and result.status==0 else None
 return {'bins':bins,'tilt_limit_deg':tilt_limit,'candidate_edges':len(valid),'max_equal_per_bin':q,'max_pairs':q*bins if q is not None else None,'solver_status':result.message,'optimal':result.status==0}

def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--bins',type=int,nargs='+',default=[12,10,8,6,4,3,2]);p.add_argument('--max-pairs-per-image',type=int,default=22500);p.add_argument('--tilt-limits',type=float,nargs='+',default=[39.619,36,32,28,24,20,18,16,14,12]);p.add_argument('--time-limit',type=int,default=60);p.add_argument('--output-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));a=p.parse_args();tour=a.tour.zfill(4);audit=a.output_root/tour/'visual_audit';edges=[json.loads(x) for x in (audit/'matches.jsonl').open() if json.loads(x).get('passed')];views=json.loads((Path('/root/pairUAV_img/processed_kml')/tour/'views.json').read_text())['views'];rows=[]
 for b in a.bins:
  for D in a.tilt_limits:
   r=solve(edges,len(views),b,D,a.time_limit,a.max_pairs_per_image);rows.append(r);print(json.dumps(r),flush=True)
 result={'tour_id':tour,'visual_edge_count':len(edges),'method':'MILP maximizes equal count shared by every marginal bin of all five delta variables; unordered pair uniqueness and per-image degree<=150','results':rows,'interpretation':'A time-limited/nonoptimal result is unknown, not infeasible. An optimal max_pairs below 1001 proves the requested count infeasible for that tilt interval and cached image candidate graph.'}
 save_json(a.output_root/tour/'visual_audit'/'capacity.json',result)
if __name__=='__main__':main()
