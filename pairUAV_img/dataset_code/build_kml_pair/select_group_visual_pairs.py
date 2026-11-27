#!/usr/bin/env python3
"""Select visually screened pairs jointly across a group of tours.

Each tour receives exactly --per-tour pairs (therefore at least 1000 when the
approved default is used); the aggregate group has exact equal-width marginal
histograms. This lets one tour compensate another tour's sparse bins.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from common import AXES, save_json
from solver_utils import solve_feasibility

def hbin(v, lo, hi, bins):
    return max(0, min(bins - 1, int((v - lo) / (hi - lo) * bins)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tours', nargs='+', required=True)
    ap.add_argument('--per-tour', type=int, default=1000)
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--tilt-limit', type=float, default=45)
    ap.add_argument('--kml-root', type=Path, required=True)
    ap.add_argument('--audit-root', type=Path, required=True)
    ap.add_argument('--image-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--time-limit', type=int, default=600)
    ap.add_argument('--max-pairs-per-image', type=int, default=150)
    a = ap.parse_args()
    tours = [str(t).zfill(4) for t in a.tours]
    if a.per_tour < 1000 or a.per_tour * len(tours) % a.bins:
        ap.error('--per-tour must be >=1000 and group total divisible by --bins')
    edges=[]; views_by={}; summaries={}; edge_counts={}
    for tour in tours:
        audit=a.audit_root/tour/'visual_audit_redesigned'
        if not (audit/'matches.jsonl').exists():
            raise FileNotFoundError(audit/'matches.jsonl')
        rows=[]
        for line in (audit/'matches.jsonl').open():
            edge=json.loads(line)
            if edge.get('passed'):
                rows.append({k:v for k,v in edge.items() if k not in ('points_a','points_b')})
        rows=[e for e in rows if abs(e['delta']['tilt_deg']) <= a.tilt_limit+1e-9]
        d=json.loads((a.kml_root/tour/'views.json').read_text()); views_by[tour]=d
        summaries[tour]=json.loads((audit/'summary.json').read_text())
        edge_counts[tour]=len(rows)
        for e in rows:
            q=dict(e);q['tour']=tour;edges.append(q)
    axes=list(AXES); T=len(tours); B=a.bins; total=len(edges)
    # Two orientations per unordered visual edge.
    variables=[(j,o) for j in range(total) for o in (1,-1)]
    n=len(variables); hist_rows=len(axes)*B; tour_rows=hist_rows; edge0=tour_rows+T; degree0=edge0+total
    rows=[];cols=[];vals=[]
    tour_index={t:i for i,t in enumerate(tours)}
    for c,(j,o) in enumerate(variables):
        e=edges[j]; ti=tour_index[e['tour']]
        for ai,k in enumerate(axes):
            lo,hi=AXES[k]
            if k=='tilt_deg': lo,hi=-a.tilt_limit,a.tilt_limit
            rows.append(ai*B+hbin(o*e['delta'][k],lo,hi,B)); cols.append(c); vals.append(1.)
        rows.append(tour_rows+ti); cols.append(c); vals.append(1.)
        rows.append(edge0+j); cols.append(c); vals.append(1.)
        for idx in (e['a'],e['b']):
            rows.append(degree0+sum(len(views_by[t]['views']) for t in tours[:ti])+idx); cols.append(c); vals.append(1.)
    nviews=sum(len(views_by[t]['views']) for t in tours)
    A=coo_matrix((vals,(rows,cols)),shape=(degree0+nviews,n)).tocsc()
    group_quota=(a.per_tour*T)//B
    min_degree=[1 if views_by[t].get('image_tilt_constraint') else 0 for t in tours for _ in views_by[t]['views']]
    lb=np.r_[[group_quota]*(hist_rows),[a.per_tour]*T,[0]*total,min_degree]
    ub=np.r_[[group_quota]*(hist_rows),[a.per_tour]*T,[1]*total,[a.max_pairs_per_image]*nviews]
    res,feasible,attempts=solve_feasibility(A,lb,ub,a.time_limit)
    if not feasible:
        result={'status':'infeasible' if res.status==2 else 'unknown','tours':tours,'per_tour':a.per_tour,'bins':B,'tilt_limit_deg':a.tilt_limit,'edge_counts':edge_counts,'solver_status':res.message,'solver_attempts':attempts}
        from capacity_diagnostics import capacity_diagnostics
        result['capacity_diagnostics']=capacity_diagnostics(edges,tours,a.per_tour,B,a.tilt_limit)
        save_json(a.output_root/f'group_{tours[0]}_{tours[-1]}_capacity.json',result);print(json.dumps(result,indent=2));raise RuntimeError(res.message)
    chosen=[]
    for c in np.flatnonzero(res.x>.5):
        j,o=variables[c];e=edges[j];q=dict(e);q['a'],q['b']=(e['a'],e['b']) if o==1 else (e['b'],e['a']);q['delta']={k:o*v for k,v in e['delta'].items()};chosen.append(q)
    out=a.output_root;out.mkdir(parents=True,exist_ok=True); group_records=[]
    for tour in tours:
        rr=[e for e in chosen if e['tour']==tour]
        td=views_by[tour]; audit=a.audit_root/tour/'visual_audit_redesigned'; image_dir=a.image_root/f'{tour}_redesigned'
        hist={k:[0]*B for k in axes}
        records=[]
        for i,e in enumerate(rr):
            for k in axes:
                lo,hi=AXES[k]
                if k=='tilt_deg':lo,hi=-a.tilt_limit,a.tilt_limit
                hist[k][hbin(e['delta'][k],lo,hi,B)]+=1
            records.append({'pair_id':f'{tour}_group_{tours[0]}_{tours[-1]}_{i:06d}','tour_id':tour,'a':e['a'],'b':e['b'],'delta':e['delta'],'inliers':e['inliers'],'matches':e['matches'],'inlier_ratio':e['inlier_ratio'],'hull_fraction':e['hull_fraction'],'grid_cells':e['grid_cells'],'model':e['model'],'kml_a':str(a.kml_root/tour/f'{e["a"]:04d}.kml'),'kml_b':str(a.kml_root/tour/f'{e["b"]:04d}.kml'),'image_a':str(image_dir/f'{e["a"]:04d}.jpg'),'image_b':str(image_dir/f'{e["b"]:04d}.jpg'),'anchor':td['anchor'],'seed':td['seed_sha256'],'visual_screen':'SIFT mutual ratio + USAC_MAGSAC; group-level marginal balancing'})
        result={'schema_version':'group-visual-1.0','status':'complete_group_visual_screen','tour_id':tour,'group_id':f'{tours[0]}_{tours[-1]}','group_tours':tours,'images_rendered':len(td['views']),'pairs':len(records),'records':records,'distribution':hist,'bins':B,'bin_semantics':'equal-width marginal bins; exact per-tour count is not required, aggregate group count is exact','tilt_difference_range_deg':[-a.tilt_limit,a.tilt_limit],'visual_screen':{'thresholds':summaries[tour]['thresholds'],'accepted_candidate_edges':edge_counts[tour],'selected_pairs':len(records),'audit_summary':str(audit/'summary.json')}}
        save_json(out/f'{tour}_group_{tours[0]}_{tours[-1]}.json',result)
        (out/f'{tour}_group_{tours[0]}_{tours[-1]}.jsonl').write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in records)+'\n')
        group_records.extend(records)
    group_hist={k:[0]*B for k in axes}
    for e in chosen:
        for k in axes:
            lo,hi=AXES[k]
            if k=='tilt_deg':lo,hi=-a.tilt_limit,a.tilt_limit
            group_hist[k][hbin(e['delta'][k],lo,hi,B)]+=1
    report={'status':'complete','group_id':f'{tours[0]}_{tours[-1]}','tours':tours,'pairs':len(chosen),'per_tour':{t:sum(e['tour']==t for e in chosen) for t in tours},'bins':B,'group_distribution':group_hist,'target_per_bin':group_quota,'tilt_difference_range_deg':[-a.tilt_limit,a.tilt_limit],'edge_counts':edge_counts,'visual_thresholds':{t:summaries[t]['thresholds'] for t in tours}}
    save_json(out/f'group_{tours[0]}_{tours[-1]}.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
