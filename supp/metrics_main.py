#!/usr/bin/env python3
"""Supplementary Table: every method of Table 2 under explicit metric definitions (supp/common.py), on the same 6,364 test pairs.

For each method: range error d (Table 2's column), 3-D endpoint error, success rates under the range and the endpoint
definition, the 5-DoF joint success, the error left after a perfect yaw alignment (no compass can do better), and
tour-bootstrap confidence intervals. DreamNav is additionally reported in the camera frame / with a true heading.
Output: outputs/supp/metrics/main.json, main.md
"""
import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav')
from common import *
from scipy.spatial import cKDTree
O = f'{OUT}/metrics'; os.makedirs(O, exist_ok=True)
T = load_test(); n = len(T); gw = gt_world(T); gc = gt_cam(T); boots = boot_indices(T.tour.values, 1000, 0); print('test pairs', n, 'tours', T.tour.nunique())

M = {}      # name -> dict(pred=..., frame='world'|'cam', extra)
def add(name, pred, frame='world', **kw): M[name] = dict(pred=pred, frame=frame, **kw)

# ---- external baselines (per-row predictions of the runs reported in Table 2), aligned on the labels
FILES = {'pi0.5': 'openpi/pairuav_v2/results/pi05/test_predictions.npz', 'pi0-FAST': 'openpi/pairuav_v2/results/pi0_fast/test_predictions.npz',
         'ViNT': 'visualnav-transformer/pairuav5d_v2/runs/vint_native/test_predictions.npz', 'NoMaD': 'visualnav-transformer/pairuav5d_v2/runs/nomad_native/test_predictions.npz',
         'DINO-WM': 'gaoyuezhou-dino_wm/outputs_pairuav5d/test_predictions.npz', 'Visual Foresight': 'visualnav-transformer/outputs_pairuav5d_visual_foresight_v2/test_predictions.npz',
         'DINOv3 ViT-7B': 'dinov3/outputs_xyz7b_mast3r_split/test_predictions.npz', 'Sample4Geo': 'sample4geo/outputs_xyz_mast3r_split_rerun/train_test_predictions.npz'}
q = T[['x', 'y', 'z', 'tilt']].values
for name, f in FILES.items():
    d = np.load(f'{BL}/{f}'); p, t = d['pred_xyz_tilt_heading'], d['true_xyz_tilt_heading']; dist, ix = cKDTree(t[:, :4]).query(q)
    assert dist.max() < 1e-2 and len(set(ix)) == len(ix), name; assert np.abs(wrap(t[ix, 4] - T.heading.values)).max() < 1e-2, name
    p = p[ix]; kw = {}
    if 'pred_distance_head_m' in d: kw['dist_head'] = d['pred_distance_head_m'][ix]
    add(name, dict(x=p[:, 0], y=p[:, 1], z=p[:, 2], tilt=p[:, 3], heading=p[:, 4]), **kw)

# ---- regression on pooled frozen-MASt3R statistics (row 1 of Table 3): original heads and the sin/cos MLP (prior branch of DreamNav)
from models_src.distance_head import DistanceHead
F = pd.read_csv(f'{ROOT}/outputs/pairuav5d_60k/features/test_xyz.csv').set_index('manifest_index'); MD = f'{ROOT}/outputs/pairuav5d_60k/models'; X = F.loc[T.manifest_index.values]
def head(path):
    ck = torch.load(path, map_location='cpu', weights_only=False); m = DistanceHead(ck['feature_dim'], ck['hidden_dim'], ck['dropout']); m.load_state_dict(ck['model_state_dict']); m.eval()
    x = X[ck['feature_names']].values.astype(np.float32); x = (x - np.asarray(ck['feature_mean'], np.float32)) / np.maximum(np.asarray(ck['feature_std'], np.float32), 1e-6)
    with torch.inference_mode(): return m(torch.as_tensor(x)).numpy().reshape(-1) * float(ck['target_scale'])
r = {a: head(f'{MD}/coordinate_head_{a}.pt') for a in 'xyz'}; r['tilt'] = head(f'{MD}/tilt_head.pt')
h = json.load(open(f'{MD}/heading_baseline.json')); hx = (X[h['feature_names']].values.astype(np.float64) - np.asarray(h['mean'])) / np.where(np.asarray(h['std']) < 1e-8, 1.0, np.asarray(h['std'])); hx = np.c_[hx, np.ones(len(hx))]
r['heading'] = np.rad2deg(np.arctan2(hx @ np.asarray(h['sin_weights']), hx @ np.asarray(h['cos_weights']))); add('MASt3R-statistics regression (RASR heads)', r)
add('MASt3R-statistics MLP, world-frame x/y', dict(x=T.n_x.values, y=T.n_y.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values))
add('MASt3R-statistics MLP, camera-frame x/y', dict(x=T.n_lx.values, y=T.n_ly.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values), frame='cam')
z0 = np.zeros(n); add('Constant (zero motion)', dict(x=z0, y=z0, z=z0, tilt=z0, heading=z0))

# ---- DreamNav
for nm, pre in (('DreamNav Stage I', 's1_'), ('DreamNav Stage II', 'p_')):
    add(f'{nm} (ENU, learned heading)', dreamnav(T, pre, 'net')); add(f'{nm} (camera frame)', dreamnav(T, pre, 'cam'), frame='cam'); add(f'{nm} (ENU, true heading)', dreamnav(T, pre, 'true'))

R = {}; E = {}
for name, m in M.items():
    e = errors(m['pred'], gc if m['frame'] == 'cam' else gw); E[name] = e; s = summarize(e); s['frame'] = m['frame']
    s['ci'] = {k: ci(fn, boots) for k, fn in (('d_mae', lambda b: e['d'][b].mean()), ('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('epe_med', lambda b: np.median(e['epe'][b])), ('epe_mae', lambda b: e['epe'][b].mean()),
                                              ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('sr_5dof_1m_1deg', lambda b: ((e['epe'][b] < 1) & (e['theta'][b] < 1) & (e['psi'][b] < 1)).mean()),
                                              ('theta_mae', lambda b: e['theta'][b].mean()), ('psi_mae', lambda b: e['psi'][b].mean()))}
    if 'dist_head' in m:   # ViNT / NoMaD: Table 2 reports d and SR from the model's own distance head
        dh = np.abs(m['dist_head'] - np.sqrt(T.x.values**2 + T.y.values**2 + T.z.values**2)); s['dist_head'] = dict(d_mae=float(dh.mean()), sr_d_1m=float((dh <= 1).mean()))
    R[name] = s
# paired differences Stage II - Stage I
e1, e2 = E['DreamNav Stage I (ENU, learned heading)'], E['DreamNav Stage II (ENU, learned heading)']; c1, c2 = E['DreamNav Stage I (camera frame)'], E['DreamNav Stage II (camera frame)']
R['_paired_stage2_minus_stage1'] = {k: dict(delta=float(fn(np.arange(n))), ci=ci(fn, boots)) for k, fn in (
    ('sr_d_1m', lambda b: (e2['d'][b] <= 1).mean() - (e1['d'][b] <= 1).mean()), ('d_mae', lambda b: e2['d'][b].mean() - e1['d'][b].mean()),
    ('sr_epe_1m_enu', lambda b: (e2['epe'][b] < 1).mean() - (e1['epe'][b] < 1).mean()), ('sr_epe_1m_cam', lambda b: (c2['epe'][b] < 1).mean() - (c1['epe'][b] < 1).mean()),
    ('epe_mae_cam', lambda b: c2['epe'][b].mean() - c1['epe'][b].mean()), ('theta_mae', lambda b: e2['theta'][b].mean() - e1['theta'][b].mean()), ('psi_mae', lambda b: e2['psi'][b].mean() - e1['psi'][b].mean()))}
# consistency check quoted in the text: a mean Euclidean error can never be below a coordinate MAE, the range error can
for nm in ('DreamNav Stage II (ENU, learned heading)',): s = R[nm]; print(nm, '| coordinate MAEs', round(s['x_mae'], 3), round(s['y_mae'], 3), round(s['z_mae'], 3), '| range error d', round(s['d_mae'], 3), '| EPE mean', round(s['epe_mae'], 3), 'median', round(s['epe_med'], 3))
json.dump(dict(n=n, tours=int(T.tour.nunique()), methods=R), open(f'{O}/main.json', 'w'), indent=1)
np.savez_compressed(f'{O}/main_errors.npz', **{f'{k}|{kk}': v for k, e in E.items() for kk, v in e.items()})
L = [f'# All methods, explicit definitions ({n} test pairs, {T.tour.nunique()} buildings); 95% CI = bootstrap over buildings\n',
     '| Method | frame | d MAE (m) | SR_d@1m | EPE mean | EPE med. | SR_EPE@1m | SR_EPE@2m | EPE after ideal yaw (med.) | SR after ideal yaw | z MAE | dir. med. | e_theta MAE | e_psi MAE | 5-DoF (1m,1deg) | 5-DoF (1m,5deg) | AUC@5/10/20 |', '|---|---|' + '---:|' * 15]
f1 = lambda v: f'{100 * v:.1f}'
for k, s in R.items():
    if k.startswith('_'): continue
    c = s['ci']; L.append(f"| {k} | {s['frame']} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | "
                     f"{s['epe_yaw_oracle_med']:.2f} | {f1(s['sr_epe_yaw_oracle_1m'])} | {s['z_mae']:.2f} | {s['dir_med']:.1f} | {s['theta_mae']:.2f} | {s['psi_mae']:.2f} | {f1(s['sr_5dof_1m_1deg'])} | {f1(s['sr_5dof_1m_5deg'])} | {f1(s['auc_5'])}/{f1(s['auc_10'])}/{f1(s['auc_20'])} |"
                     + (f"  (distance head: d {s['dist_head']['d_mae']:.2f}, SR_d {f1(s['dist_head']['sr_d_1m'])})" if 'dist_head' in s else ''))
L.append('\nPaired Stage II - Stage I: ' + '; '.join(f"{k} {v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" for k, v in R['_paired_stage2_minus_stage1'].items()))
open(f'{O}/main.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
