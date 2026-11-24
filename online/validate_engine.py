#!/usr/bin/env python3
"""Check that the online engine reproduces the archived batch predictions on dataset test pairs."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/online'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from engine import Engine, SGD
from fusion_lib import add_targets, wrap
n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
P = add_targets(pd.read_pickle(f'{SGD}/pairs.pkl')); A = pd.read_pickle(f'{SGD}/stage2/s2_pred.pkl'); A = A[A.split == 'test']
T = P.merge(A.drop(columns=['split', 'tour']), on='manifest_index').sample(n, random_state=7).reset_index(drop=True)
eng = Engine(); R = eng.predict([(f'/root/pairUAV-5D-temp/{a}', f'/root/pairUAV-5D-temp/{b}') for a, b in zip(T.image_a, T.image_b)]); eng.close()
R.to_pickle('/root/rasr-pairuav/outputs/online_models/validate_pred.pkl'); print('timing', {k: round(v / n, 3) for k, v in R.attrs['timing'].items()})
for st, pre in (('s1', 's1_'), ('s2', 'p_')):
    print(f'--- {st}: MAE online | archived | mean |online - archived| | median')
    for k in ('heading', 'tilt', 'dist', 'z', 'lx', 'ly'):
        y = T['dist' if k == 'dist' else k].values; o = R[f'{st}_{k}'].values; a = T[pre + k].values; d = (lambda u: np.abs(wrap(u))) if k == 'heading' else np.abs
        print(f'{k:8s} {d(o - y).mean():.4f} | {d(a - y).mean():.4f} | {d(o - a).mean():.4f} | {np.median(d(o - a)):.4f}')
ha = T.heading_a.values; print('abs heading err (deg) median: s1', np.median(np.abs(wrap(R.s1_absh - ha))).round(2), 's2', np.median(np.abs(wrap(R.s2_absh - ha))).round(2))
for st in ('s1', 's2'): print(st, 'world MAE x/y/z', np.abs(R[f'{st}_x'] - T.x).mean().round(4), np.abs(R[f'{st}_y'] - T.y).mean().round(4), np.abs(R[f'{st}_zw'] - T.z).mean().round(4))
