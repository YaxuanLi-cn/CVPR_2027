#!/usr/bin/env python3
"""For one cached image-audit graph, maximize the common exact marginal-bin quota."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import coo_matrix
from common import AXES,save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--views-json',type=Path,required=True);p.add_argument('--bins',type=int,default=12);p.add_argument('--tilt-limit',type=float,default=39.6);p.add_argument('--max-pairs-per-image',type=int,default=150);p.add_argument('--time-limit',type=int,default=300);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 es=[json.loads(l) for l in (a.audit_dir/'matches.jsonl').open() if json.loads(l).get('passed') and abs(json.loads(l)['delta']['tilt_deg'])<=a.tilt_limit+1e-9];views=json.load(open(a.views_json))['views'];keys=list(AXES);nb=a.bins;hist_n=len(keys)*nb;pair0=hist_n;deg0=pair0+len(es);rows=[];cols=[];vals=[];vars=[(j,o) for j in range(len(es)) for o in (1,-1)];n=len(vars)
 for c,(j,o) in enumerate(vars):
  e=es[j]
  for kidx,k in enumerate(keys):
   lo,hi=AXES[k];
   if k=='tilt_deg':lo,hi=-a.tilt_limit,a.tilt_limit
   v=e['delta'][k]*o;bi=max(0,min(nb-1,int((v-lo)/(hi-lo)*nb)));rows.append(kidx*nb+bi);cols.append(c);vals.append(1.)
  rows.append(pair0+j);cols.append(c);vals.append(1.)
  for ix in (e['a'],e['b']):rows.append(deg0+ix);cols.append(c);vals.append(1.)
 qcol=n
 for r in range(hist_n):rows.append(r);cols.append(qcol);vals.append(-1.)
 A=coo_matrix((vals,(rows,cols)),shape=(hist_n+len(es)+len(views),n+1)).tocsc();lb=np.zeros(A.shape[0]);ub=np.zeros(A.shape[0]);ub[pair0:deg0]=1;ub[deg0:]=a.max_pairs_per_image;c=np.zeros(n+1);c[qcol]=-1
 res=milp(c,integrality=np.ones(n+1),bounds=Bounds(np.zeros(n+1),np.r_[np.ones(n),a.max_pairs_per_image]),constraints=LinearConstraint(A,lb,ub),options={'time_limit':a.time_limit,'mip_rel_gap':.005,'presolve':True});q=int(round(res.x[qcol])) if res.x is not None and res.status==0 else None;result={'bins':nb,'tilt_limit_deg':a.tilt_limit,'candidate_edges':len(es),'max_equal_per_bin':q,'max_pairs':q*nb if q is not None else None,'optimal':res.status==0,'solver_status':res.message};print(json.dumps(result,indent=2));save_json(a.out,result)
if __name__=='__main__':main()
