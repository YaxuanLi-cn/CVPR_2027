#!/usr/bin/env python3
"""Track E evaluation: Table 2 baselines re-trained with camera-frame targets vs their east-north originals.

All rows are scored on the 6,364 common test pairs with supp/common.py. Camera-frame predictions are compared with
common.gt_cam (lx, ly, z); east-north predictions with common.gt_world. "after ideal yaw" is the endpoint error left when
the predicted horizontal displacement is rotated about the vertical axis onto the true one (no heading source can do better).
Outputs: outputs/supp/trackE/camframe.json, camframe.md
"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/supp/trackE')
from common import *
from scipy.spatial import cKDTree
O = Path(f'{OUT}/trackE'); R = O / 'runs'; T = load_test(); n = len(T); gw, gc = gt_world(T), gt_cam(T); boots = boot_indices(T.tour.values, 1000, 0)
PAIRS = pd.read_pickle(f'{SG}/pairs.pkl').set_index('pair_id').manifest_index
qtree = T[['x', 'y', 'z', 'tilt']].values


def P(p): return dict(x=p[:, 0], y=p[:, 1], z=p[:, 2], tilt=p[:, 3], heading=p[:, 4])
def by_manifest(d, key='pred_xyz_tilt_heading'):
    mi = d['manifest_index'] if 'manifest_index' in d.files else PAIRS.loc[d['pair_id']].values
    ix = pd.Series(np.arange(len(mi)), index=mi).reindex(T.manifest_index.values).values; assert not np.isnan(ix).any(); return ix.astype(int)
def by_labels(d):
    t = d['true_xyz_tilt_heading']; dist, ix = cKDTree(t[:, :4]).query(qtree); assert dist.max() < 1e-2 and len(set(ix)) == n; return ix
def check_cam(d, ix):  # the file's own labels must be the camera-frame labels of the common frame
    t = d['true_xyz_tilt_heading'][ix]; assert np.abs(t[:, 0] - T.lx.values).max() < 1e-3 and np.abs(t[:, 1] - T.ly.values).max() < 1e-3 and np.abs(t[:, 2] - T.z.values).max() < 1e-3


def score(pred, frame, extra=None):
    e = errors(pred, gc if frame == 'cam' else gw); s = summarize(e); s['frame'] = frame; g = gc if frame == 'cam' else gw
    s['ci'] = {'sr_d_1m': ci(lambda b: (e['d'][b] <= 1).mean(), boots), 'sr_epe_1m': ci(lambda b: (e['epe'][b] < 1).mean(), boots), 'd_mae': ci(lambda b: e['d'][b].mean(), boots), 'epe_med': ci(lambda b: np.median(e['epe'][b]), boots)}
    s['corr'] = {k: float(np.corrcoef(pred[k], g[k])[0, 1]) if np.std(pred[k]) > 0 else float('nan') for k in ('x', 'y', 'z', 'tilt')}
    s['pred_std'] = {k: float(np.std(pred[k])) for k in ('x', 'y', 'z')}; s['pred_norm_med'] = float(np.median(np.sqrt(pred['x']**2 + pred['y']**2 + pred['z']**2)))
    if extra: s.update(extra)
    return s


def with_spread(runs):
    s = dict(runs[0]); s['seeds'] = len(runs); s['seed_range'] = {k: [float(min(r[k] for r in runs)), float(max(r[k] for r in runs))] for k in ('d_mae', 'sr_d_1m', 'epe_mae', 'epe_med', 'sr_epe_1m', 'sr_epe_2m', 'x_mae', 'y_mae', 'z_mae', 'theta_mae', 'psi_mae', 'dir_med')}
    s['seed_mean'] = {k: float(np.mean([r[k] for r in runs])) for k in s['seed_range']}; s['seed_mean']['epe_yaw_oracle_med'] = float(np.mean([r['epe_yaw_oracle_med'] for r in runs]))
    s['seed_mean']['corr_h'] = float(np.mean([(r['corr']['x'] + r['corr']['y']) / 2 for r in runs])); return s


rows = {}
gd = np.sqrt(T.x.values**2 + T.y.values**2 + T.z.values**2)
def head_stats(dh): e = np.abs(dh - gd); return dict(dist_head=dict(d_mae=float(e.mean()), sr_d_1m=float((e <= 1).mean()), ci_sr=ci(lambda b: (e[b] <= 1).mean(), boots)))
# ---------------- ViNT / NoMaD
for name, tag in (('ViNT', 'vint'), ('NoMaD', 'nomad')):
    d = np.load(f'{BL}/visualnav-transformer/pairuav5d_v2/runs/{tag}_native/test_predictions.npz'); ix = by_labels(d)
    rows[f'{name} | ENU-trained (Table 2)'] = score(P(d['pred_xyz_tilt_heading'][ix]), 'world', head_stats(d['pred_distance_head_m'][ix]))
    f = R / f'{tag}_camframe/test_predictions.npz'
    if f.exists():
        d = np.load(f); ix = by_manifest(d); check_cam(d, ix); m = json.load(open(R / f'{tag}_camframe/metrics.json'))
        ex = dict(head_stats(d['pred_distance_head_m'][ix]), train_minutes=m['history'][-1]['minutes'], epochs=m['epochs'], xy_scale=m.get('xy_scale'))
        rows[f'{name} | camera-frame-trained'] = score(P(d['pred_xyz_tilt_heading'][ix]), 'cam', ex)
        if tag == 'nomad': rows[f'{name} | camera-frame-trained, mean of 8 samples'] = score(P(d['pred_mean_xyz_tilt_heading'][ix]), 'cam', head_stats(d['pred_distance_head_m'][ix]))
# ---------------- DINO-WM
d = np.load(f'{BL}/gaoyuezhou-dino_wm/outputs_pairuav5d/test_predictions.npz'); rows['DINO-WM | ENU-trained (Table 2)'] = score(P(d['pred_xyz_tilt_heading'][by_labels(d)]), 'world')
for fr, lab in (('enu', 'DINO-WM | ENU-trained, re-run (control)'), ('cam', 'DINO-WM | camera-frame-trained')):
    runs = []
    for sub in (f'dino_wm_{fr}', f'dino_wm_{fr}_seed18'):       # seed 17 (as the Table 2 run) first, then seed 18
        f = R / sub / 'test_predictions.npz'
        if f.exists():
            d = np.load(f); ix = by_manifest(d)
            if fr == 'cam': check_cam(d, ix)
            runs.append(score(P(d['pred_xyz_tilt_heading'][ix]), 'cam' if fr == 'cam' else 'world'))
    if runs: rows[lab] = with_spread(runs) if len(runs) > 1 else runs[0]
# ---------------- Visual Foresight
d = np.load(f'{BL}/visualnav-transformer/outputs_pairuav5d_visual_foresight_v2/test_predictions.npz'); rows['Visual Foresight | ENU-trained (Table 2)'] = score(P(d['pred_xyz_tilt_heading'][by_labels(d)]), 'world')
for fr, lab in (('enu', 'Visual Foresight | ENU-trained, re-run (control)'), ('cam', 'Visual Foresight | camera-frame-trained')):
    f = R / f'visual_foresight_{fr}/test_predictions.npz'
    if f.exists():
        d = np.load(f); ix = by_manifest(d)
        if fr == 'cam': check_cam(d, ix)
        rows[lab] = score(P(d['pred_xyz_tilt_heading'][ix]), 'cam' if fr == 'cam' else 'world')
# ---------------- DINOv3 ViT-7B (5 seeds per frame; seed 0 is the row, the others give the spread)
d = np.load(f'{BL}/dinov3/outputs_xyz7b_mast3r_split/test_predictions.npz'); rows['DINOv3 ViT-7B | ENU-trained (Table 2)'] = score(P(d['pred_xyz_tilt_heading'][by_labels(d)]), 'world')
def seeds(pattern, frame, nmax=8):
    out = []
    for s in range(nmax):
        f = Path(str(pattern).format(s))
        if not f.exists(): break
        d = np.load(f); ix = by_manifest(d)
        if frame == 'cam': check_cam(d, ix)
        out.append(score(P(d['pred_xyz_tilt_heading'][ix]), frame))
    return out
for fr, lab in (('enu', 'DINOv3 ViT-7B | ENU-trained, re-run (control)'), ('cam', 'DINOv3 ViT-7B | camera-frame-trained')):
    runs = seeds(R / ('dinov3_vit7b/test_predictions_' + fr + '_seed{}.npz'), 'cam' if fr == 'cam' else 'world')
    if runs: rows[lab] = with_spread(runs)
# ---------------- Sample4Geo (head replayed on one recorded feature stream; batch 8 = run.sh, batch 32 = argparse default)
d = np.load(f'{BL}/sample4geo/outputs_xyz_mast3r_split_rerun/train_test_predictions.npz'); rows['Sample4Geo | ENU-trained (Table 2)'] = score(P(d['pred_xyz_tilt_heading'][by_labels(d)]), 'world')
for bs in (8, 32):
    for fr, lab in (('enu', f'Sample4Geo | ENU-trained, re-run (control), batch {bs}'), ('cam', f'Sample4Geo | camera-frame-trained, batch {bs}')):
        runs = seeds(R / (f'sample4geo/test_predictions_bs{bs}_' + fr + '_seed{}.npz'), 'cam' if fr == 'cam' else 'world')
        if runs: rows[lab] = with_spread(runs)
# ---------------- pi0.5 / pi0-FAST: not re-trained (originals for reference)
for name, f in (('pi0.5', 'openpi/pairuav_v2/results/pi05/test_predictions.npz'), ('pi0-FAST', 'openpi/pairuav_v2/results/pi0_fast/test_predictions.npz')):
    d = np.load(f'{BL}/{f}'); rows[f'{name} | ENU-trained (Table 2); not re-trained'] = score(P(d['pred_xyz_tilt_heading'][by_labels(d)]), 'world')
# ---------------- references
z0 = np.zeros(n); rows['Constant (zero motion) | scored in the camera frame'] = score(dict(x=z0, y=z0, z=z0, tilt=z0, heading=z0), 'cam')
rows['Constant (zero motion) | scored in east-north'] = score(dict(x=z0, y=z0, z=z0, tilt=z0, heading=z0), 'world')
Ptr = pd.read_pickle(f'{SG}/pairs.pkl'); Ptr = Ptr[Ptr.split == 'train']; dmed = float(np.median(np.sqrt(Ptr.x**2 + Ptr.y**2 + Ptr.z**2)))
rows[f'Constant distance ({dmed:.2f} m, train median), direction unknown | only d and SR_d are meaningful'] = score(dict(x=z0, y=z0, z=z0 + dmed, tilt=z0, heading=z0), 'cam')
rows['MASt3R-statistics MLP | ENU-trained x/y'] = score(dict(x=T.n_x.values, y=T.n_y.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values), 'world')
rows['MASt3R-statistics MLP | camera-frame-trained x/y'] = score(dict(x=T.n_lx.values, y=T.n_ly.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values), 'cam')
rows['DreamNav Stage I (camera frame)'] = score(dreamnav(T, 's1_', 'cam'), 'cam'); rows['DreamNav Stage II (camera frame)'] = score(dreamnav(T, 'p_', 'cam'), 'cam')
json.dump(dict(n=n, tours=int(T.tour.nunique()), rows=rows), open(O / 'camframe.json', 'w'), indent=1)

f1 = lambda v: f'{100 * v:.1f}'
L = [f'# Table 2 baselines re-trained with camera-frame targets ({n} test pairs, {T.tour.nunique()} buildings; 95% CI = bootstrap over buildings)\n',
     'x/y columns are east/north MAE for ENU-trained rows and lx/ly (right / forward of the source camera) MAE for camera-frame rows. "ideal yaw" = endpoint error after rotating the prediction about the vertical axis onto the truth.\n',
     '| Method | training target | scored in | d MAE (m) | SR_d@1m | EPE mean | EPE med. | SR_EPE@1m | SR_EPE@2m | EPE med. ideal yaw | SR ideal yaw | x / lx | y / ly | z | dir. med. | e_theta | e_psi | AUC@5/10/20 | corr x/y/z | dist. head d / SR_d |', '|---|---|---|' + '---:|' * 17]
for k, s in rows.items():
    c = s['ci']; dh = f"{s['dist_head']['d_mae']:.2f} / {f1(s['dist_head']['sr_d_1m'])}" if 'dist_head' in s else '--'
    mth, var = (k.split(' | ', 1) + ['--'])[:2]
    L.append(f"| {mth} | {var} | {s['frame']} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | "
             f"{s['epe_yaw_oracle_med']:.2f} | {f1(s['sr_epe_yaw_oracle_1m'])} | {s['x_mae']:.2f} | {s['y_mae']:.2f} | {s['z_mae']:.2f} | {s['dir_med']:.1f} | {s['theta_mae']:.2f} | {s['psi_mae']:.2f} | {f1(s['auc_5'])}/{f1(s['auc_10'])}/{f1(s['auc_20'])} | "
             f"{s['corr']['x']:.2f}/{s['corr']['y']:.2f}/{s['corr']['z']:.2f} | {dh} |")
L.append('\nSeed spread (min-max over seeds; the table row is seed 0):')
for k, s in rows.items():
    if 'seed_range' in s: r = s['seed_range']; L.append(f"- {k} ({s['seeds']} seeds): d {r['d_mae'][0]:.2f}-{r['d_mae'][1]:.2f}, SR_d {f1(r['sr_d_1m'][0])}-{f1(r['sr_d_1m'][1])}, EPE med. {r['epe_med'][0]:.2f}-{r['epe_med'][1]:.2f}, x {r['x_mae'][0]:.2f}-{r['x_mae'][1]:.2f}, y {r['y_mae'][0]:.2f}-{r['y_mae'][1]:.2f}, z {r['z_mae'][0]:.2f}-{r['z_mae'][1]:.2f}")
open(O / 'camframe.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
