#!/usr/bin/env python3
"""Track C: summarise the per-image predictions written by sens.py.  Usage: sens_combine.py <tag> <out.json>
When both networks were run for the tag, the ensemble (sum of von Mises natural parameters) is added."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/supp/trackC')
from common import wrap, SG, OUT
from nets import combine
tag, outf = sys.argv[1], sys.argv[2]; O = f'{OUT}/trackC'; R_ = np.load(f'{O}/sens_{tag}_raw.npz', allow_pickle=True); J = json.load(open(f'{O}/sens_{tag}_raw.json'))
B_ = np.load(f'{O}/sens_{tag}_bev.npz', allow_pickle=True) if os.path.exists(f'{O}/sens_{tag}_bev.npz') else None; gt = R_['gt']
if B_ is not None: assert (B_['image'] == R_['image']).all()
def circ_mean(d): d = np.deg2rad(d); return float(np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean())))
def summ(h, k, h0, k0):
    e = np.abs(wrap(h - gt)); sh = wrap(h - h0)
    return dict(mae=float(e.mean()), median=float(np.median(e)), gt10=float((e > 10).mean()), gt45=float((e > 45).mean()), gt90=float((e > 90).mean()), shift_signed_mean=circ_mean(sh), shift_abs_mean=float(np.abs(sh).mean()), shift_abs_median=float(np.median(np.abs(sh))),
                shift_gt10=float((np.abs(sh) > 10).mean()), stays_within_30deg=float((np.abs(sh) < 30).mean()), flips_180_within_30deg=float((np.abs(wrap(sh - 180)) < 30).mean()), kappa_ratio=float(np.median(k) / np.median(k0)))
get = lambda Z, nm: (Z[f'{nm}|h'].astype(np.float64), Z[f'{nm}|k'].astype(np.float64))
R = {'n_images': int(len(gt)), 'n_buildings': int(len(set(R_['tour']))), 'seed': 0, 'subset': J['subset'], 'raw_weights': J['raw_weights'], 'nadir_network': B_ is not None, 'tilt_for_nadir_network': 'coarse-geometry estimate of the image in its first test pair', 'conditions': {}}
c_raw = get(R_, 'clean'); c_bev = get(B_, 'clean') if B_ is not None else None; c_ens = combine(c_raw, c_bev) if B_ is not None else None
arch = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image').ah_pred.reindex(R_['image']).values
R['check_vs_archived'] = dict(raw_view_mean_abs_diff=float(np.abs(wrap(c_raw[0] - arch)).mean()) if J['raw_weights'] == 'archived' else None, nadir_mean_abs_diff=float(np.abs(wrap(c_bev[0] - B_['archived_bev'])).mean()) if B_ is not None else None)
for nm in J['conds']:
    r = {'raw-view': summ(*get(R_, nm), *c_raw)}
    if B_ is not None: r['nadir-rectified'] = summ(*get(B_, nm), *c_bev); r['ensemble'] = summ(*combine(get(R_, nm), get(B_, nm)), *c_ens)
    R['conditions'][nm] = r; print(f"{nm:66s} " + ' | '.join(f"{k}: MAE {v['mae']:6.2f} med {v['median']:5.2f} >45 {100*v['gt45']:4.1f}% shift {v['shift_abs_mean']:6.2f} flip {100*v['flips_180_within_30deg']:4.1f}% kappa x{v['kappa_ratio']:.2f}" for k, v in r.items()))
if 'zoom' in J: R['zoom_raw_view'] = {nm: summ(*get(R_, nm), *c_raw) for nm in J['zoom']}; print({k: (round(v['mae'], 2), round(v['median'], 2)) for k, v in R['zoom_raw_view'].items()})
if 'mechanism_raw_view_single_view' in J: R['mechanism_raw_view_single_view'] = J['mechanism_raw_view_single_view']
json.dump(R, open(f'{O}/{outf}', 'w'), indent=1); print('wrote', outf)
