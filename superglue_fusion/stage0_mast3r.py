#!/usr/bin/env python3
"""Stage 0: MASt3R-feature-only LightGBM (tour-grouped OOF). Control experiment + candidate predictions for fusion."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
if __name__ == '__main__':
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r()
    D = P.merge(M.drop(columns='split'), on='manifest_index'); assert len(D) == 60000
    feats = [c for c in M.columns if c not in ('manifest_index', 'split')]
    tr, te = D[D.split == 'train'].reset_index(drop=True), D[D.split == 'test'].reset_index(drop=True)
    Xtr, Xte, g = tr[feats].values, te[feats].values, tr.tour.values
    out_tr, out_te = tr[['manifest_index']].copy(), te[['manifest_index']].copy()
    targets = {'x': 'l1', 'y': 'l1', 'z': 'l1', 'tilt': 'l1', 'hsin': 'l2', 'hcos': 'l2', 'logd': 'l1', 'tilt_a': 'l1', 'tilt_b': 'l1', 'lx': 'l1', 'ly': 'l1', 'ux': 'l2', 'uy': 'l2', 'uz': 'l2'}
    def run(t, obj):
        return t, oof_fit(Xtr, tr[t].values, g, Xte, dict(objective=obj, n_jobs=1))
    for t, (o, p) in Parallel(n_jobs=3)(delayed(run)(t, obj) for t, obj in targets.items()):
        out_tr['m_' + t] = o; out_te['m_' + t] = p
    for o in (out_tr, out_te):
        o['m_heading'] = np.rad2deg(np.arctan2(o.m_hsin, o.m_hcos)); o['m_hconf'] = np.hypot(o.m_hsin, o.m_hcos); o['m_dist'] = np.exp(o.m_logd)
    pd.concat([out_tr, out_te]).to_pickle(f'{SG}/stage0.pkl')
    res = {'train_oof': report(tr, out_tr.m_x, out_tr.m_y, out_tr.m_z, out_tr.m_tilt, out_tr.m_heading, 'stage0 OOF '),
           'test': report(te, out_te.m_x, out_te.m_y, out_te.m_z, out_te.m_tilt, out_te.m_heading, 'stage0 TEST')}
    for nm, o, d in (('oof', out_tr, tr), ('test', out_te, te)):
        res[f'{nm}_aux'] = dict(MAE_dist_direct=float(np.mean(np.abs(o.m_dist - d.dist))), MAE_tilt_a=float(np.mean(np.abs(o.m_tilt_a - d.tilt_a))), MAE_tilt_b=float(np.mean(np.abs(o.m_tilt_b - d.tilt_b))),
                                MAE_lx=float(np.mean(np.abs(o.m_lx - d.lx))), MAE_ly=float(np.mean(np.abs(o.m_ly - d.ly))), MAE_uz=float(np.mean(np.abs(o.m_uz - d.uz))))
        print(nm, res[f'{nm}_aux'])
    json.dump(res, open(f'{SG}/stage0_metrics.json', 'w'), indent=2)
