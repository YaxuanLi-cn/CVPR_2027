#!/usr/bin/env python3
"""Does supp/trackD/sim_nav_seeds.py reproduce the archived run (outputs/sim/nav_log.pkl, noise seed 0) on one site?"""
import sys, json, numpy as np, pandas as pd
d = sys.argv[1] if len(sys.argv) > 1 else '/root/rasr-pairuav/outputs/supp/trackD/sim_verify_seed0'
A = pd.read_pickle('/root/rasr-pairuav/outputs/sim/nav_log.pkl'); N = pd.read_pickle(f'{d}/nav_log.pkl'); t = N.tour.unique(); A = A[A.tour.isin(t) & A.pol.isin(N.pol.unique())]
M = A.merge(N, on=['tour', 'ep', 'pol', 'step'], suffixes=('_arch', '_new')); print('rows archived', len(A), 'new', len(N), 'merged', len(M)); res = {}
for p in ('s1', 's2', 's2p'):
    m = M[M.pol == p]; res[p] = {}
    for k in sorted(m.step.unique()):
        q = m[m.step == k]; dd = (q.pos_err_new - q.pos_err_arch).abs()
        res[p][int(k)] = dict(max_abs_diff_m=float(dd.max()), median_abs_diff_m=float(dd.median()), identical_within_1mm=float((dd < 1e-3).mean()), sr1_arch=float(100 * (q.pos_err_arch < 1).mean()), sr1_new=float(100 * (q.pos_err_new < 1).mean()), med_arch=float(q.pos_err_arch.median()), med_new=float(q.pos_err_new.median()))
        print(p, 'step', k, {kk: round(v, 4) for kk, v in res[p][int(k)].items()})
    q = m[m.step < m.step.max()]; print(p, 'pred_dist max abs diff', float((q.pred_dist_new - q.pred_dist_arch).abs().max()), 'median', float((q.pred_dist_new - q.pred_dist_arch).abs().median()))
json.dump(res, open(f'{d}/verify.json', 'w'), indent=1)
