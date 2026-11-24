#!/usr/bin/env python3
"""Table metrics for the zero-shot baselines on the real Mezieres pairs, within the training range (300 pairs).

Predictions come from
  /root/baseline/visualnav-transformer/pairuav5d_v2/eval_real.py  (ViNT, NoMaD)
  /root/baseline/openpi/pairuav_v2/eval_real.py                   (pi0-FAST)
Columns are those of docs/paper/real_table.md: median position error, SR at 1 m / 2 m (|norm(pred) - norm(gt)|),
MAE of the heading change and of the tilt change. The DreamNav / RASR / constant rows are recomputed from pred.pkl
with the same code so that all rows share one definition.

The baselines output the PairUAV-5D label (x east, y north, z up, in metres), so their position error is taken
against the world-frame displacement; DreamNav and RASR output the displacement in the frame of the first camera.
Both are the same vector up to a rotation, so the SR (a difference of norms) does not depend on the frame.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REAL = Path('/root/realdata/mezieres'); BASE = REAL / 'baselines'
K = 60.0 / 220.18   # flight height above ground / camera height the training data was rendered at
wrap = lambda a: (np.asarray(a) + 180.0) % 360.0 - 180.0


def auc(e, thr):
    e = np.sort(e); r = (np.arange(len(e)) + 1) / len(e); i = np.searchsorted(e, thr)
    x = np.r_[0.0, e[:i], thr]; y = np.r_[0.0, r[:i], r[i - 1] if i else 0.0]
    return float(np.trapezoid(y, x) / thr)


def row(v, g, tilt, heading, d, scale=1.0, distance=None):
    """v: predicted displacement [n, 3]; g: true displacement in the same frame; distance: optional distance-head output."""
    v = np.asarray(v, float) * scale; gd = d.dist.values; pe = np.linalg.norm(v - g, axis=1); pn = np.linalg.norm(v, axis=1)
    he = np.abs(wrap(heading - d.heading.values)); te = np.abs(np.asarray(tilt) - d.tilt.values); de = np.abs(pn - gd); pose = np.maximum(he, te)
    cos = (v * g).sum(1) / (pn.clip(1e-9) * gd)
    r = {'pos_med': np.median(pe), 'pos_mean': pe.mean(), 'sr1': 100 * (de <= 1).mean(), 'sr2': 100 * (de <= 2).mean(), 'dist_mae': de.mean(),
         'heading_mae': he.mean(), 'heading_med': np.median(he), 'tilt_mae': te.mean(), 'tilt_med': np.median(te),
         'pos_within_1m': 100 * (pe < 1).mean(), 'pos_within_2m': 100 * (pe < 2).mean(), 'tdir_med': np.median(np.rad2deg(np.arccos(np.clip(cos, -1, 1)))),
         'auc5': 100 * auc(pose, 5), 'auc10': 100 * auc(pose, 10), 'auc20': 100 * auc(pose, 20)}
    if distance is not None:
        dh = np.abs(np.asarray(distance) * scale - gd); r.update(head_sr1=100 * (dh <= 1).mean(), head_sr2=100 * (dh <= 2).mean(), head_dist_mae=dh.mean())
    return {k: float(x) for k, x in r.items()}


def main():
    D = pd.read_pickle(REAL / 'pred.pkl'); d = D[D.dist_equiv <= 38]
    gl = d[['lx', 'ly', 'z']].values; gw = d[['x', 'y', 'z']].values; z0 = np.zeros(len(d)); R = {}
    R['Constant (zero motion)'] = row(np.zeros((len(d), 3)), gl, z0, z0, d)
    R['RASR + sin/cos'] = row(d[['n_lx', 'n_ly', 'n_z']].values, gl, d.n_tilt, d.n_heading, d)
    R['RASR + sin/cos, rescaled'] = row(d[['n_lx', 'n_ly', 'n_z']].values, gl, d.n_tilt, d.n_heading, d, K)
    files = {'ViNT': 'vint', 'NoMaD': 'nomad', 'pi0-FAST': 'pi0_fast'}; extra = {}
    for name, f in files.items():
        if not (BASE / f'{f}_predictions.npz').exists(): print('missing', f); continue
        p = np.load(BASE / f'{f}_predictions.npz'); assert np.array_equal(p['pair_index'], d.index.values) and p['a'].tolist() == d.a.tolist() and p['b'].tolist() == d.b.tolist()
        assert np.abs(p['true_xyz_tilt_heading'] - d[['x', 'y', 'z', 'tilt', 'heading']].values).max() < 1e-9
        q = p['pred_xyz_tilt_heading']; dist = p['pred_distance_head_m'] if 'pred_distance_head_m' in p else None
        R[name] = row(q[:, :3], gw, q[:, 3], q[:, 4], d, 1.0, dist); R[name + ', rescaled'] = row(q[:, :3], gw, q[:, 3], q[:, 4], d, K, dist)
        # diagnostics: the same output read as a first-camera-frame displacement; correlation of each output with its label
        extra[name] = {'pos_med_if_read_in_camera_frame': row(q[:, :3], gl, q[:, 3], q[:, 4], d)['pos_med'], 'pos_med_if_read_in_camera_frame_rescaled': row(q[:, :3], gl, q[:, 3], q[:, 4], d, K)['pos_med'],
                       'pred_mean': q.mean(0).tolist(), 'pred_std': q.std(0).tolist(), 'pred_norm_median_m': float(np.median(np.linalg.norm(q[:, :3], axis=1))),
                       'corr_with_label_x_y_z_tilt': [float(np.corrcoef(q[:, i], gw[:, i] if i < 3 else d.tilt.values)[0, 1]) for i in range(4)],
                       'corr_sin_heading': float(np.corrcoef(np.sin(np.deg2rad(q[:, 4])), np.sin(np.deg2rad(d.heading.values)))[0, 1]), 'corr_cos_heading': float(np.corrcoef(np.cos(np.deg2rad(q[:, 4])), np.cos(np.deg2rad(d.heading.values)))[0, 1])}
        if dist is not None: extra[name]['distance_head_median_m'] = float(np.median(dist)); extra[name]['corr_distance_head'] = float(np.corrcoef(dist, d.dist.values)[0, 1])
    R['Ours (Stage II), no flight height'] = row(d[['s2_lx', 's2_ly', 's2_z']].values, gl, d.s2_tilt, d.s2_heading, d)
    R['Ours (Stage I)'] = row(d[['s1_lx', 's1_ly', 's1_z']].values, gl, d.s1_tilt, d.s1_heading, d, K)
    R['Ours (Stage II)'] = row(d[['s2_lx', 's2_ly', 's2_z']].values, gl, d.s2_tilt, d.s2_heading, d, K)
    # reference for the SR: a constant distance guess (PairUAV train-median distance, 20.11 m), with and without rescaling
    for s, tag in ((1.0, 'constant_distance_20.11m'), (K, 'constant_distance_20.11m_rescaled')):
        e = np.abs(20.11 * s - d.dist.values); extra[tag] = {'sr1': float(100 * (e <= 1).mean()), 'sr2': float(100 * (e <= 2).mean())}
    out = {'pairs': int(len(d)), 'true_distance_m': {'mean': float(d.dist.mean()), 'median': float(d.dist.median())}, 'rescale_factor': K, 'methods': R, 'baseline_diagnostics': extra}
    json.dump(out, open(BASE / 'real_baseline_metrics.json', 'w'), indent=1)
    L = [f'### Within the training range ({len(d)} pairs, mean true distance {d.dist.mean():.1f} m)\n', '| Method | Pos. median (m) | SR@1 m | SR@2 m | heading MAE (deg) | tilt MAE (deg) | SR@1 m, dist. head | SR@2 m, dist. head | pose AUC@5 / 10 / 20 deg |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, r in R.items():
        h = (f"{r['head_sr1']:.1f} | {r['head_sr2']:.1f}" if 'head_sr1' in r else '-- | --')
        L.append(f"| {m} | {r['pos_med']:.2f} | {r['sr1']:.1f} | {r['sr2']:.1f} | {r['heading_mae']:.2f} | {r['tilt_mae']:.2f} | {h} | {r['auc5']:.1f} / {r['auc10']:.1f} / {r['auc20']:.1f} |")
    open(BASE / 'real_baseline_metrics.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print(json.dumps(extra, indent=1))


if __name__ == '__main__':
    main()
