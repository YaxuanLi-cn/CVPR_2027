#!/usr/bin/python3
"""Integer constrained marginal balancing with unordered-pair uniqueness."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from common import AXES,bin_ids,eligible,delta

def candidates(views):
 out=[]
 for a in range(len(views)):
  for b in range(a+1,len(views)):
   d=delta(views[a],views[b])
   if not eligible(d):continue
   ga,gb=views[a].get('visible_grid'),views[b].get('visible_grid')
   unknown=ga is None or gb is None or len(ga)<=1 or len(gb)<=1
   if unknown:
    common=None;overlap=None
   else:
    ma=np.asarray(ga,dtype=bool);mb=np.asarray(gb,dtype=bool)
    common=int((ma&mb).sum());overlap=common/max(1,min(ma.sum(),mb.sum()))
    if common<40 or overlap<.35:continue
   if abs(d['heading_deg'])<6 and abs(d['tilt_deg'])<4 and np.linalg.norm([d['x_m'],d['y_m'],d['z_m']])<3:continue
   out.append({'a':a,'b':b,'delta':d,'predicted_overlap':float(overlap) if overlap is not None else None,'common_grid_points':common})
 return out

def select(edges,views,count=1200,bins=12,seed=0,time_limit=600):
 if count%bins:raise ValueError('pair count must be divisible by bins for exact equal bin counts')
 n=len(edges);quota=count//bins
 if n<count:raise ValueError(f'Only {n} distinct unordered eligible pairs; need {count}')
 rows=[];cols=[];vals=[];nr=len(AXES)*bins;lb=[quota]*nr;ub=[quota]*nr
 degree_row=nr+n;lb.extend([0]*n);ub.extend([1]*n);lb.extend([0]*len(views));ub.extend([150]*len(views))
 for j,e in enumerate(edges):
  for orientation in (0,1):
   c=2*j+orientation;d=e['delta'] if not orientation else {k:-v for k,v in e['delta'].items()}
   for axis,q in enumerate(bin_ids(d,bins)):rows.append(axis*bins+q);cols.append(c);vals.append(1.)
   for rr in (nr+j,degree_row+e['a'],degree_row+e['b']):rows.append(rr);cols.append(c);vals.append(1.)
 A=coo_matrix((vals,(rows,cols)),shape=(len(lb),2*n)).tocsc();A.indices=A.indices.astype(np.int32);A.indptr=A.indptr.astype(np.int32)
 rng=np.random.default_rng(seed);cost=rng.uniform(0,.01,size=2*n)
 for j,e in enumerate(edges):cost[2*j:2*j+2]+=1-(e.get('predicted_overlap') if e.get('predicted_overlap') is not None else 1)
 result=milp(cost,integrality=np.ones(2*n),bounds=Bounds(0,1),constraints=LinearConstraint(A,np.array(lb),np.array(ub)),options={'time_limit':time_limit,'mip_rel_gap':.25,'presolve':True,'mip_heuristic_effort':.4})
 if result.x is None:raise RuntimeError(f'No feasible uniform pair subset: {result.message}')
 chosen=np.flatnonzero(result.x>.5); counts=np.zeros((len(AXES),bins),int);pairs=[]
 for c in chosen:
  e=edges[c//2];a,b=(e['a'],e['b']) if c%2==0 else (e['b'],e['a']);d=delta(views[a],views[b]);pairs.append(dict(e,a=a,b=b,delta=d))
  for k,q in enumerate(bin_ids(d,bins)):counts[k,q]+=1
 if len(pairs)!=count or not np.all(counts==quota):raise RuntimeError('Solver result failed exact integer validation')
 return pairs,dict(zip(AXES,counts.tolist())),str(result.message)
