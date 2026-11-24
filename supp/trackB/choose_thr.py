#!/usr/bin/env python3
"""Inlier threshold per (matcher, variant), chosen on the calibration pairs: largest share of pairs with translation-direction error < 5 deg
(failed pairs are misses; ties -> lower median). Prints `matcher thr[,thr]` lines = the thresholds needed on the test / real sets."""
import sys, os, json, numpy as np, pandas as pd
ROOT='/root/rasr-pairuav'; sys.path[:0]=[f'{ROOT}/supp',f'{ROOT}/superglue_fusion',f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
from eval_pairuav import pick, unit, dir_err, NAMES, HAS_TTA, GRID
from fusion_lib import add_targets
from rb_recon import subset
def choose(m, tta, Tc, idc, G=None):
    G = pd.read_pickle(f'{ROOT}/outputs/supp/trackB/geom/{m}_train.pkl') if G is None else G; thrs = sorted(set(G.thr.unique()) & set(GRID)); cal = {t: unit(pick(G, idc, t, tta)) for t in thrs}; h = cal[thrs[0]].have.values
    de = {t: dir_err(cal[t], Tc)[h] for t in thrs}; frac = {t: float((de[t] < 5).mean()) for t in thrs}; med = {t: float(np.median(de[t])) for t in thrs}
    return max(thrs, key=lambda t: (frac[t], -med[t])), frac, med, int(h.sum())
if __name__ == '__main__':
    Tc = add_targets(subset(300, split='train')); idc = Tc.manifest_index.values; out = {}
    for m in NAMES:
        if not os.path.exists(f'{ROOT}/outputs/supp/trackB/geom/{m}_train.pkl'): continue
        ts = sorted({choose(m, tta, Tc, idc)[0] for tta in ((False, True) if HAS_TTA[m] else (False,))}); out[m] = ts; print(m, ','.join(str(t) for t in ts))
    json.dump(out, open(f'{ROOT}/outputs/supp/trackB/geom/chosen_thr.json', 'w'))
