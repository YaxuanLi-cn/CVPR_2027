"""Shared definitions for the supplementary experiments: metric formulas, the common test frame, bootstrap.

Every number of the supplementary material is computed with the functions of this file, so that one definition
is used for all methods.

Notation (pair = source image a, target image b; labels are "b minus a"):
  dp      = (dx, dy, dz)  true displacement in the local east-north-up frame, metres
  dp_hat  = predicted displacement
  range error   d    = | ||dp_hat|| - ||dp|| |              (column "d" of Table 2 of the paper)
  endpoint err  EPE  = || dp_hat - dp ||                     (3-D position error of the executed motion)
  e_theta = | wrap(dtheta_hat - dtheta) |,  e_psi = | dpsi_hat - dpsi |      (degrees; wrap to (-180, 180])
  direction error    = angle between dp_hat and dp           (degrees)
  rotation error     e_rot = max(e_theta, e_psi)
  Pose AUC@t         = (1/t) * integral_0^t  recall(e_rot <= s) ds           (recall-vs-threshold curve, trapezoid)
  SR_d@1m   = mean(d <= 1)            (what the SR column of Table 2 reports)
  SR_EPE@1m = mean(EPE < 1)           (3-D endpoint success)
  SR_5DoF(a m, b deg) = mean(EPE < a  and  e_theta < b  and  e_psi < b)
"""
import os, numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; SG = f'{ROOT}/outputs/pairuav5d_60k/superglue'; S2 = f'{SG}/stage2'; OUT = f'{ROOT}/outputs/supp'; DATA = '/root/pairUAV-5D-temp'; BL = '/root/baseline'
os.makedirs(f'{OUT}/common', exist_ok=True)


def wrap(a): return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def world(h_deg, lx, ly):
    """Camera-a levelled frame (right, forward) -> east, north. h = heading of camera a, clockwise from north."""
    h = np.deg2rad(np.asarray(h_deg, float)); return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly


def level(h_deg, x, y):
    """East, north -> camera-a levelled frame (right, forward)."""
    h = np.deg2rad(np.asarray(h_deg, float)); return np.cos(h) * x - np.sin(h) * y, np.sin(h) * x + np.cos(h) * y


def auc(e, thr):
    e = np.sort(np.asarray(e, float)); r = (np.arange(len(e)) + 1) / len(e); i = int(np.searchsorted(e, thr)); x = np.r_[0.0, e[:i], thr]; y = np.r_[0.0, r[:i], r[i - 1] if i else 0.0]
    return float(np.trapezoid(y, x) / thr)


def errors(pred, gt):
    """pred / gt: dicts of arrays with x, y, z (metres, same frame) and optionally tilt, heading (degrees).
    Returns per-pair error arrays."""
    px, py, pz = (np.asarray(pred[k], float) for k in 'xyz'); gx, gy, gz = (np.asarray(gt[k], float) for k in 'xyz')
    e = dict(x=np.abs(px - gx), y=np.abs(py - gy), z=np.abs(pz - gz)); pn = np.sqrt(px**2 + py**2 + pz**2); gn = np.sqrt(gx**2 + gy**2 + gz**2)
    e['d'] = np.abs(pn - gn); e['epe'] = np.sqrt(e['x']**2 + e['y']**2 + e['z']**2); e['epe_h'] = np.hypot(e['x'], e['y'])
    cos = (px * gx + py * gy + pz * gz) / (pn.clip(1e-9) * gn.clip(1e-9)); e['dir'] = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
    # error that remains after the best rotation of the prediction about the vertical axis (a perfect compass cannot do better)
    e['epe_yaw_oracle'] = np.sqrt((np.hypot(px, py) - np.hypot(gx, gy))**2 + (pz - gz)**2)
    if 'heading' in pred and pred['heading'] is not None:
        e['theta'] = np.abs(wrap(np.asarray(pred['heading'], float) - np.asarray(gt['heading'], float))); e['psi'] = np.abs(np.asarray(pred['tilt'], float) - np.asarray(gt['tilt'], float)); e['rot'] = np.maximum(e['theta'], e['psi'])
    return e


def summarize(e):
    """Scalar summary of one method from its per-pair errors."""
    s = {}
    for k in ('x', 'y', 'z', 'd', 'epe', 'epe_h', 'dir', 'epe_yaw_oracle') + (('theta', 'psi') if 'theta' in e else ()):
        s[f'{k}_mae'] = float(np.mean(e[k])); s[f'{k}_med'] = float(np.median(e[k]))
    s['sr_d_1m'] = float(np.mean(e['d'] <= 1.0)); s['sr_d_2m'] = float(np.mean(e['d'] <= 2.0))
    for t in (0.5, 1, 2, 5): s[f'sr_epe_{t}m'] = float(np.mean(e['epe'] < t))
    s['sr_epe_yaw_oracle_1m'] = float(np.mean(e['epe_yaw_oracle'] < 1.0))
    if 'theta' in e:
        for a, b in ((1, 1), (1, 5), (2, 2), (5, 5)): s[f'sr_5dof_{a}m_{b}deg'] = float(np.mean((e['epe'] < a) & (e['theta'] < b) & (e['psi'] < b)))
        s['sr_d1m_and_5deg'] = float(np.mean((e['d'] <= 1.0) & (e['theta'] < 5) & (e['psi'] < 5)))
        for t in (5, 10, 20): s[f'auc_{t}'] = auc(e['rot'], t)
        s['rot_lt_1deg'] = float(np.mean(e['rot'] < 1)); s['rot_lt_5deg'] = float(np.mean(e['rot'] < 5))
    s['n'] = int(len(e['d'])); return s


def boot_indices(groups, n_boot=1000, seed=0):
    """Cluster bootstrap: resample groups (buildings / tours, or episodes) with replacement."""
    groups = np.asarray(groups); ug = np.unique(groups); by = {g: np.where(groups == g)[0] for g in ug}; rng = np.random.default_rng(seed)
    return [np.concatenate([by[g] for g in rng.choice(ug, len(ug))]) for _ in range(n_boot)]


def ci(fn, boots):
    v = np.array([fn(b) for b in boots], float); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load_test(refresh=False):
    """The common test frame: 6,364 test pairs of the result set, labels, per-camera poses and the archived DreamNav predictions.
    Columns: labels x y z tilt heading dist lx ly, heading_a/b, tilt_a/b; s1_* (coarse stage), p_* (refined stage) with
    heading tilt dist z lx ly; n_* (prior MLP); absolute-heading estimates v2h_/v2k_ (raw-view network) and bvh_/bvk_ (nadir-rectified network) per side."""
    f = f'{OUT}/common/T.pkl'
    if os.path.exists(f) and not refresh: return pd.read_pickle(f)
    import sys; sys.path.insert(0, f'{ROOT}/superglue_fusion'); from fusion_lib import add_targets
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); F = pd.read_pickle(f'{S2}/s2_pred.pkl'); Z = pd.read_pickle(f'{SG}/half_zoom_pred.pkl')
    D = P.merge(F.drop(columns=['split', 'tour']), on='manifest_index').merge(Z[['manifest_index', 'n_x', 'n_y', 'n_z', 'n_tilt', 'n_heading', 'n_dist']], on='manifest_index')
    D = D.merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl')[['manifest_index', 'zm_inl']], on='manifest_index', how='left')
    N = pd.read_pickle(f'{SG}/stage0_mlp.pkl')[['manifest_index', 'n_lx', 'n_ly', 'n_tilt_a', 'n_tilt_b']]; D = D.merge(N, on='manifest_index', how='left')
    A = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image'); B = pd.read_pickle(f'{S2}/pred_bev_s.pkl')
    for s in 'ab':
        D[f'v2h_{s}'] = A.ah_pred.reindex(D[f'image_{s}']).values; D[f'v2k_{s}'] = A.ah_kappa.reindex(D[f'image_{s}']).values
        b = B[B.side == s].set_index('manifest_index'); D[f'bvh_{s}'] = b.ah_est.reindex(D.manifest_index).values; D[f'bvk_{s}'] = b.ak_est.reindex(D.manifest_index).values
    T = D[D.split == 'test'].reset_index(drop=True); T.to_pickle(f); return T


V2 = [('v2h_a', 'v2k_a', 'v2h_b', 'v2k_b')]; BV = [('bvh_a', 'bvk_a', 'bvh_b', 'bvk_b')]


def fuse_heading(d, src, dh):
    """Absolute heading of camera a (Eq. 4 of the paper): sum of von Mises natural parameters of a's estimate and of b's estimate rotated by -dh."""
    s = 0; c = 0
    for ha, ka, hb, kb in src:
        ta = np.deg2rad(d[ha].values); tb = np.deg2rad(d[hb].values - dh); s = s + d[ka].values * np.sin(ta) + d[kb].values * np.sin(tb); c = c + d[ka].values * np.cos(ta) + d[kb].values * np.cos(tb)
    return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)


def dreamnav(T, pre, heading='net', al=None):
    """Assemble a DreamNav prediction. pre: 's1_' (coarse) or 'p_' (refined).
    heading: 'net' = learned absolute heading (as in the paper: coarse uses the raw-view network, refined the two-network ensemble),
             'true' = true heading of camera a (e.g. an onboard compass), 'cam' = no rotation (output stays in the camera-a levelled frame),
             or an array of headings in degrees.
    al: exponent of the norm rescaling towards the fused distance (paper: 0.75 coarse, 0.5 refined, chosen on validation tours)."""
    if al is None: al = 0.75 if pre == 's1_' else 0.5
    lx, ly, z = T[pre + 'lx'].values, T[pre + 'ly'].values, T[pre + 'z'].values
    if isinstance(heading, str):
        if heading == 'net': h = fuse_heading(T, V2 if pre == 's1_' else BV + V2, T[pre + 'heading'].values)[0]
        elif heading == 'true': h = T.heading_a.values
        elif heading == 'cam': h = np.zeros(len(T))
        else: raise ValueError(heading)
    else: h = np.asarray(heading, float)
    x, y = world(h, lx, ly); n = np.sqrt(x**2 + y**2 + z**2).clip(1e-6); sc = (T[pre + 'dist'].values / n) ** al
    return dict(x=x * sc, y=y * sc, z=z * sc, tilt=T[pre + 'tilt'].values, heading=T[pre + 'heading'].values, abs_heading=h)


def gt_world(T): return dict(x=T.x.values, y=T.y.values, z=T.z.values, tilt=T.tilt.values, heading=T.heading.values)
def gt_cam(T): return dict(x=T.lx.values, y=T.ly.values, z=T.z.values, tilt=T.tilt.values, heading=T.heading.values)
