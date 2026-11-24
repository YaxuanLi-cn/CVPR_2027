#!/usr/bin/env python3
"""Where does the plane-induced part of the warp help? Paired difference in endpoint success between the full refinement
(plane-induced warp) and the compute-matched control with rotation-only alignment, by altitude change, tilt, baseline and relief.
Same 3,000 test pairs and fusion protocol as eval_ctrl.py. Output: outputs/supp/trackA/plane_vs_rot.json"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from common import *
from fusion_lib import add_targets
O = f'{OUT}/trackA'; P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); ids = np.load(f'{O}/ids_small_test.npy'); T = P[P.manifest_index.isin(ids)].sort_values('manifest_index').reset_index(drop=True); gc = gt_cam(T)
def vec(d, pre='p_', al=0.5):
    d = d.set_index('manifest_index').reindex(T.manifest_index); lx, ly, z, dist = (d[pre + k].values.astype(float) for k in ('lx', 'ly', 'z', 'dist')); nn = np.sqrt(lx**2 + ly**2 + z**2).clip(1e-6); sc = (dist / nn) ** al
    return dict(x=lx * sc, y=ly * sc, z=z * sc, tilt=d[pre + 'tilt'].values, heading=d[pre + 'heading'].values)
F = pd.read_pickle(f'{O}/full_pred.pkl'); Rr = pd.read_pickle(f'{O}/rot_tiles_pred.pkl'); ef = errors(vec(F), gc); er = errors(vec(Rr), gc); e1 = errors(vec(F, 's1_', 0.75), gc)
Pl = pd.read_pickle(f'{O}/planarity_pairs.pkl').set_index('manifest_index').reindex(T.manifest_index)
T['tilt_max'] = np.maximum(T.tilt_a, T.tilt_b); T['absz'] = np.abs(T.z); T['relief'] = Pl.relief.values; T['absdh'] = np.abs(T.heading)
res = {}
def table(col, edges, labels):
    out = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = ((T[col] >= lo) & (T[col] < hi)).values
        if m.sum() < 30: continue
        bt = boot_indices(T.tour.values[m], 1000, 0); a, b, c = (ef['epe'][m] < 1).astype(float), (er['epe'][m] < 1).astype(float), (e1['epe'][m] < 1).astype(float); da, db = (ef['d'][m] <= 1).astype(float), (er['d'][m] <= 1).astype(float)
        out.append(dict(bin=lab, n=int(m.sum()), coarse=float(c.mean()), rot=float(b.mean()), plane=float(a.mean()), diff=float((a - b).mean()), ci=ci(lambda i: (a[i] - b[i]).mean(), bt), diff_d=float((da - db).mean()), ci_d=ci(lambda i: (da[i] - db[i]).mean(), bt)))
    return out
res['by_abs_dz_m'] = table('absz', [0, 5, 15, 25, 100], ['0-5', '5-15', '15-25', '>=25'])
res['by_tilt_max'] = table('tilt_max', [0, 15, 25, 35, 46], ['0-15', '15-25', '25-35', '35-45'])
res['by_baseline_m'] = table('dist', [0, 10, 20, 30, 100], ['0-10', '10-20', '20-30', '>=30'])
res['by_relief'] = table('relief', [0, 0.05, 0.10, 0.20, 10], ['<5%', '5-10%', '10-20%', '>=20%'])
res['by_abs_dheading'] = table('absdh', [0, 45, 90, 135, 181], ['0-45', '45-90', '90-135', '135-180'])
json.dump(res, open(f'{O}/plane_vs_rot.json', 'w'), indent=1)
for k, rows in res.items():
    print(k)
    for r in rows: print(f"  {r['bin']:8s} n={r['n']:4d} coarse {100*r['coarse']:5.1f} rot {100*r['rot']:5.1f} plane {100*r['plane']:5.1f} | plane-rot SR_EPE {100*r['diff']:+5.1f} [{100*r['ci'][0]:+.1f}, {100*r['ci'][1]:+.1f}] | SR_d {100*r['diff_d']:+5.1f} [{100*r['ci_d'][0]:+.1f}, {100*r['ci_d'][1]:+.1f}]")
