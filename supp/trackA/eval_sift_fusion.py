#!/usr/bin/env python3
"""Is DreamNav complementary to the classical SIFT pipeline?  On the 3,000 test pairs of the control subset (fusions fitted on its 3,000
training pairs): the SIFT + essential matrix + flight-height baseline alone, and the two fusion stages with and without the SIFT pose
and scale cues as additional candidates (stage2_fusion_ctrl.py --sift).  Camera-frame metrics from supp/common.py.
Output: outputs/supp/trackA/sift_fusion.json"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from common import *
from fusion_lib import add_targets
O = f'{OUT}/trackA'; AL = 0.5
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); ids = np.load(f'{O}/ids_small_test.npy'); idt = np.load(f'{O}/ids_ctrl_train.npy')
T = P[P.manifest_index.isin(ids)].sort_values('manifest_index').reset_index(drop=True); Tr = P[P.manifest_index.isin(idt)].sort_values('manifest_index').reset_index(drop=True); gc = gt_cam(T); n = len(T)
boots = boot_indices(T.tour.values, 1000, 0)
def vec(lx, ly, z, dist, tilt, heading, al=AL):
    lx, ly, z, dist = (np.asarray(v, float) for v in (lx, ly, z, dist)); nn = np.sqrt(lx**2 + ly**2 + z**2).clip(1e-6); sc = (dist / nn) ** al
    return dict(x=lx * sc, y=ly * sc, z=z * sc, tilt=np.asarray(tilt, float), heading=np.asarray(heading, float))
def load(tag):
    f = f'{O}/{tag}_pred.pkl'; return pd.read_pickle(f).set_index('manifest_index').reindex(T.manifest_index) if os.path.exists(f) else None
G = pd.read_pickle(f'{O}/geom_sift.pkl').set_index('manifest_index')
def sift_pred(Tx, H=None):
    g = G.reindex(Tx.manifest_index); ok = (g.sf_ok.fillna(0).values == 1); h = ((g.sf_hgta_p75 + g.sf_hgtb_p75) / 2).values.astype(float); ok &= np.isfinite(h) & (h > 1e-6)
    if H is None: H = float(np.median((Tx.dist.values * h)[ok]))
    d = np.where(ok, H / np.where(ok, h, 1.0), 0.0); z = lambda c: np.where(ok, g[c].values.astype(float), 0.0)
    return dict(x=d * z('sf_vx'), y=d * z('sf_vy'), z=d * z('sf_vz'), tilt=z('sf_dtilt'), heading=z('sf_dh')), ok, H
_, _, H = sift_pred(Tr); ps, oks, _ = sift_pred(T, H)
ROWS = [('SIFT + E + flight-height prior', ps)]
F = load('full'); ROWS.append(('Coarse stage', vec(F.s1_lx, F.s1_ly, F.s1_z, F.s1_dist, F.s1_tilt, F.s1_heading, 0.75)))
for tag, lab in (('sift_coarse', 'Coarse stage + SIFT candidates'), ('full', 'Refined stage'), ('sift_full', 'Refined stage + SIFT candidates')):
    d = load(tag)
    if d is None: print('missing', tag); continue
    ROWS.append((lab, vec(d.p_lx, d.p_ly, d.p_z, d.p_dist, d.p_tilt, d.p_heading)))
E = {lab: errors(p, gc) for lab, p in ROWS}; res = dict(n=n, n_train=int(len(Tr)), H=H, sift_fail=float(100 * (1 - oks.mean())), rows={}, paired={})
for lab, e in E.items():
    r = dict(d_mae=float(e['d'].mean()), sr_d=float(100 * (e['d'] <= 1).mean()), epe_mean=float(e['epe'].mean()), epe_med=float(np.median(e['epe'])), sr_epe=float(100 * (e['epe'] < 1).mean()), sr_epe2=float(100 * (e['epe'] < 2).mean()),
             theta_mae=float(e['theta'].mean()), psi_mae=float(e['psi'].mean()), dir_med=float(np.median(e['dir'])))
    r['ci_sr_d'] = [float(v) for v in np.percentile([100 * (e['d'][b] <= 1).mean() for b in boots], [2.5, 97.5])]; r['ci_sr_epe'] = [float(v) for v in np.percentile([100 * (e['epe'][b] < 1).mean() for b in boots], [2.5, 97.5])]; res['rows'][lab] = r
def paired(a, b):
    out = {}
    for k, f in (('sr_d', lambda e: (e['d'] <= 1).astype(float) * 100), ('sr_epe', lambda e: (e['epe'] < 1).astype(float) * 100), ('d_mae', lambda e: e['d']), ('epe_mean', lambda e: e['epe'])):
        x = f(E[a]) - f(E[b]); out[k] = dict(delta=float(x.mean()), ci=[float(v) for v in np.percentile([x[i].mean() for i in boots], [2.5, 97.5])])
    return out
for a, b in (('Refined stage + SIFT candidates', 'Refined stage'), ('Refined stage + SIFT candidates', 'SIFT + E + flight-height prior'), ('Coarse stage + SIFT candidates', 'Coarse stage'), ('Coarse stage + SIFT candidates', 'SIFT + E + flight-height prior'),
             ('Refined stage', 'Coarse stage + SIFT candidates'), ('Refined stage', 'SIFT + E + flight-height prior')):
    if a in E and b in E: res['paired'][f'{a} - {b}'] = paired(a, b)
json.dump(res, open(f'{O}/sift_fusion.json', 'w'), indent=1)
for lab, r in res['rows'].items(): print(f'{lab:34s} d {r["d_mae"]:.2f} SRd {r["sr_d"]:.1f} SRe {r["sr_epe"]:.1f} EPE med {r["epe_med"]:.2f} theta {r["theta_mae"]:.3f} dir {r["dir_med"]:.2f}')
for k, v in res['paired'].items(): print(f'{k}: SRd {v["sr_d"]["delta"]:+.1f} [{v["sr_d"]["ci"][0]:+.1f},{v["sr_d"]["ci"][1]:+.1f}]  SRe {v["sr_epe"]["delta"]:+.1f} [{v["sr_epe"]["ci"][0]:+.1f},{v["sr_epe"]["ci"][1]:+.1f}]')
