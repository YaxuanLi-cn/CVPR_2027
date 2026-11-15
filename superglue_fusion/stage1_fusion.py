#!/usr/bin/env python3
"""Stage 1: fuse SuperGlue relative-pose geometry with MASt3R-feature predictions (select-and-refine per target)."""
import sys, json, argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
ap = argparse.ArgumentParser(); ap.add_argument('--geom', default=f'{SG}/geom_rot.pkl'); ap.add_argument('--partial', action='store_true'); ap.add_argument('--tag', default='fusion')
ap.add_argument('--all-mast3r', action='store_true'); ap.add_argument('--map', default=None); ap.add_argument('--zoom', default=None, help='zoom-pass pose features (zm_*, zmap_*) used as extra candidates'); ap.add_argument('--neural', default=None, help='stage0c neural-fusion predictions (q_*) used as extra candidates'); ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--sel', default='sqrt', choices=['log', 'abs', 'sqrt']); ap.add_argument('--blend-p', type=float, default=2.0, help='0 = hard selection; >0 = blend candidates with weights predicted_error^-p'); ap.add_argument('--abs-heading', default=None); a = ap.parse_args()

P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); S0 = pd.read_pickle(f'{SG}/stage0.pkl').merge(pd.read_pickle(f'{SG}/stage0_mlp.pkl'), on='manifest_index'); G = pd.read_pickle(a.geom)
D = P.merge(S0, on='manifest_index').merge(M.drop(columns='split'), on='manifest_index').merge(G, on='manifest_index', how='inner' if a.partial else 'left')
if a.map: D = D.merge(pd.read_pickle(a.map), on='manifest_index', how='left')
else:
    for c in ('map_ok', 'map_dh', 'map_dtilt', 'map_vx', 'map_vy', 'map_vz', 'map_hgta', 'map_hgtb', 'map_ta', 'map_tb', 'map_par'): D[c] = np.nan
D['map_ok'] = D.map_ok.fillna(0.0)
if a.zoom:
    D = D.merge(pd.read_pickle(a.zoom), on='manifest_index', how='left'); D['zm_ok'] = D.zm_ok.fillna(0.0); D['zmap_ok'] = D.zmap_ok.fillna(0.0)
    D['zm_dh'] = np.where(D.zm_ok == 1, D.zm_dh, np.nan); D['zm_dhdiff_sg'] = wrap(D.zm_dh - D.sg_dh); D['zm_dhdiff_map'] = wrap(D.zmap_dh - D.map_dh); D['zm_dtiltdiff_map'] = D.zmap_dtilt - D.map_dtilt
    D['zm_dircos_sg'] = D.zm_vx * D.sg_vx + D.zm_vy * D.sg_vy + D.zm_vz * D.sg_vz; D['zm_dircos_zmap'] = D.zm_vx * D.zmap_vx + D.zm_vy * D.zmap_vy + D.zm_vz * D.zmap_vz
    D['s_zmhgtab'] = -np.log(((D.zm_hgta_p75 + D.zm_hgtb_p75) / 2).clip(lower=1e-3)); D['s_zmaphgtab'] = -np.log(((D.zmap_hgta + D.zmap_hgtb) / 2).clip(lower=1e-3)); D['zm_logpar'] = np.log(D.zm_par.clip(lower=1e-6))
if a.neural: D = D.merge(pd.read_pickle(a.neural), on='manifest_index', how='left')
D['sg_ok'] = D.sg_ok.fillna(0.0); print('rows', len(D), 'sg_ok', D.sg_ok.mean(), flush=True)
mfeats = [c for c in M.columns if c not in ('manifest_index', 'split')]
ok = D.sg_ok.values == 1

# ---------- derived features ----------
D['f_dh_diff'] = wrap(D.sg_dh - D.n_heading); D['f_dtilt_diff'] = D.sg_dtilt - D.n_tilt; D['f_ta_diff'] = D.sg_ta - D.n_tilt_a; D['f_tb_diff'] = D.sg_tb - D.n_tilt_b
D['f_dh_diff_m'] = wrap(D.sg_dh - D.m_heading); D['f_mn_dh'] = wrap(D.m_heading - D.n_heading)
D['f_abs_sin_dh'] = np.abs(D.sg_dh_sin); D['f_tsum_diff'] = D.sg_tsum - D.n_tilt_a - D.n_tilt_b
D['c_tilt_s1'] = D.sg_tsum - 2 * D.n_tilt_a; D['c_tilt_s2'] = 2 * D.n_tilt_b - D.sg_tsum; D['c_tilt_ab'] = D.n_tilt_b - D.n_tilt_a
D['alt_dh'] = np.rad2deg(np.arctan2(D.alt_dh_sin, D.alt_dh_cos)); D['alt_dtilt'] = D.alt_tb - D.alt_ta
for k in 'xyz': D[f'f_u{k}_diff'] = D[f'sg_v{k}'] - D[f'n_u{k}']
D['f_dir_cos_m'] = (D.sg_vx * D.n_ux + D.sg_vy * D.n_uy + D.sg_vz * D.n_uz) / np.sqrt(D.n_ux**2 + D.n_uy**2 + D.n_uz**2).clip(lower=1e-3)
D['f_logpar'] = np.log(D.sg_par.clip(lower=1e-6)); D['f_alt_logpar'] = np.log(D.alt_par.clip(lower=1e-6))
depth_stats = [c for c in mfeats if c.startswith(('pts3d_a_c2_', 'pts3d_a_norm_', 'pts3d_b_in_a_c2_', 'pts3d_b_c2_')) and c.endswith(('p25', 'p50', 'p75', 'mean'))]
for c in depth_stats: D['s_' + c] = D.f_logpar + np.log(D[c].clip(lower=1e-3))   # geometric scale recovery: baseline = (baseline/depth) x metric depth
# camera-height prior: cameras in this dataset fly ~210 m above ground, so  baseline ~= H0 / (ground depth below camera in baseline units)
for c in ('a', 'b'):
    for q in (50, 75, 90): D[f's_hgt{c}_p{q}'] = -np.log(D[f'sg_hgt{c}_p{q}'].clip(lower=1e-3))
D['s_hgtab_p75'] = -np.log(((D.sg_hgta_p75 + D.sg_hgtb_p75) / 2).clip(lower=1e-3)); D['s_hgtab_p50'] = -np.log(((D.sg_hgta_p50 + D.sg_hgtb_p50) / 2).clip(lower=1e-3))
D['f_hgt_ratio'] = D.sg_hgtb_p75 / D.sg_hgta_p75.clip(lower=1e-3); D['f_hgt_diff'] = D.sg_hgtb_p75 - D.sg_hgta_p75
D['s_maphgtab'] = -np.log(((D.map_hgta + D.map_hgtb) / 2).clip(lower=1e-3)); D['s_maphgta'] = -np.log(D.map_hgta.clip(lower=1e-3))
D['f_map_dh_diff'] = wrap(D.map_dh - D.sg_dh); D['f_map_dtilt_diff'] = D.map_dtilt - D.sg_dtilt; D['f_map_dir_cos'] = D.map_vx * D.sg_vx + D.map_vy * D.sg_vy + D.map_vz * D.sg_vz
D = D.copy()
sg_feats = [c for c in D.columns if c.startswith(('sg_', 'alt_', 'map_', 'zm_', 'zmap_', 'f_', 's_', 'c_tilt')) and c not in ('sg_dh',)] + ['sg_dh']
m_feats0 = [c for c in S0.columns if c.startswith(('m_', 'n_'))] + [c for c in D.columns if c.startswith('q_')]
base_feats = sg_feats + m_feats0 + (mfeats if a.all_mast3r else depth_stats + [c for c in mfeats if c.startswith(('conf_', 'delta_pts3d_ba_norm', 'match_', 'pose_'))])
base_feats = list(dict.fromkeys(base_feats)); print('n feats', len(base_feats), flush=True)

tr = (D.split == 'train').values; te = ~tr; g = D.tour.values[tr]
def fit(X, y, obj='l1', mask=None, **kw):
    """OOF on train rows + test prediction, returned as one array aligned with D."""
    o, p = oof_fit(X[tr], y[tr], g, X[te], dict(objective=obj, n_jobs=a.jobs, **kw), mask=None if mask is None else mask[tr])
    out = np.zeros(len(D)); out[tr] = o; out[te] = p; return out

def select_refine(name, y, cands, circular=False, extra=()):
    """cands: {name: (values, valid_mask)}. 1) learn each candidate's log-error, 2) pick the best per row, 3) L1 residual GBDT."""
    diff = (lambda u, v: wrap(u - v)) if circular else (lambda u, v: u - v)
    cands = {k: v for k, v in cands.items() if (v[1] & tr).sum() >= 50}   # drop candidates that do not exist for this match source
    X0 = D[base_feats + list(extra)].values.astype(np.float32); names = list(cands); E = np.full((len(D), len(names)), np.inf); eps = 1e-2
    for j, k in enumerate(names):
        v, valid = cands[k]; v = np.where(valid, v, 0.0); err = np.abs(diff(y, v))
        tf = {'log': lambda e: np.log(e + eps), 'abs': lambda e: e, 'sqrt': np.sqrt}[a.sel]   # 'abs' = expected |error| (tail-risk aware, matches the MAE metric)
        E[:, j] = np.where(valid, fit(X0, tf(err), 'l2', mask=valid, n_estimators=500), np.inf)
    pick = E.argmin(1); V = np.stack([np.where(cands[k][1], cands[k][0], np.nan) for k in names], 1); base = V[np.arange(len(D)), pick]
    if a.blend_p > 0:   # soft combination: weights from the predicted absolute error of each candidate
        inv = {'log': lambda e: np.exp(e) - eps, 'abs': lambda e: e, 'sqrt': lambda e: np.square(np.maximum(e, 0))}[a.sel]
        Wt = np.where(np.isfinite(E), np.maximum(inv(np.where(np.isfinite(E), E, 0.0)), 1e-3) ** (-a.blend_p), 0.0); Wt = Wt / Wt.sum(1, keepdims=True); Vz = np.nan_to_num(V)
        base = np.rad2deg(np.arctan2((Wt * np.sin(np.deg2rad(Vz))).sum(1), (Wt * np.cos(np.deg2rad(Vz))).sum(1))) if circular else (Wt * Vz).sum(1)
    Xr = np.column_stack([X0, np.where(np.isfinite(E), E, 20.0), pick] + [np.nan_to_num(diff(V[:, j], base), nan=0.0) for j in range(len(names))])
    res = fit(Xr, diff(y, base), 'l1'); final = base + res; final = wrap(final) if circular else final
    msg = {k: float(np.mean(np.abs(diff(y, np.where(cands[k][1], cands[k][0], 0.0)))[te & cands[k][1]])) for k in names}
    print(f'[{name}] test MAE per candidate (valid rows): ' + ' '.join(f'{k}={v:.3f}' for k, v in msg.items()) +
          f' | picked {dict(zip(names, np.bincount(pick[te], minlength=len(names)).tolist()))} | selected-base {np.mean(np.abs(diff(y, base))[te]):.4f} | +residual {np.mean(np.abs(diff(y, final))[te]):.4f}'
          f' | OOF {np.mean(np.abs(diff(y, final))[tr]):.4f}', flush=True)
    return final, base

allv = np.ones(len(D), bool); altok = (D.alt_ok.fillna(0).values == 1); mapok = (D.map_ok.values == 1) & bool(a.map)
zmok = (D.zm_ok.values == 1) if a.zoom else None; zmapok = (D.zmap_ok.values == 1) if a.zoom else None
def addmap(c, v, q=None, zm=None, zmap=None):
    if a.map: c['map'] = (np.asarray(v), mapok)
    if a.zoom and zm is not None: c['zoom'] = (np.asarray(zm), zmok); c['zoom_map'] = (np.asarray(zmap), zmapok)
    if a.neural and q is not None: c['neural'] = (np.asarray(q), allv)
    return c
heading, _ = select_refine('heading', D.heading.values, addmap({'sg': (D.sg_dh.values, ok), 'alt': (D.alt_dh.values, altok), 'mast3r': (D.m_heading.values, allv), 'mlp': (D.n_heading.values, allv)}, D.map_dh.values, D.q_heading.values if a.neural else None, zm=D.zm_dh.values if a.zoom else None, zmap=D.zmap_dh.values if a.zoom else None), circular=True)
tilt, _ = select_refine('tilt', D.tilt.values, addmap({'sg': (D.sg_dtilt.values, ok), 'alt': (D.alt_dtilt.values, altok), 'mast3r': (D.m_tilt.values, allv), 'mlp': (D.n_tilt.values, allv), 'mlp_ab': (D.c_tilt_ab.values, allv), 'sum1': (D.c_tilt_s1.values, ok), 'sum2': (D.c_tilt_s2.values, ok)}, D.map_dtilt.values, D.q_tilt.values if a.neural else None, zm=D.zm_dtilt.values if a.zoom else None, zmap=D.zmap_dtilt.values if a.zoom else None))
# distance (log space): MASt3R direct regression vs geometric scale recovery (offset-calibrated on train)
dc = {'mast3r': (D.m_logd.values, allv), 'mlp': (D.n_logd.values, allv)}
if a.neural: dc['neural'] = (D.q_logd.values, allv)
for c in ('s_pts3d_a_c2_p50', 's_pts3d_a_norm_p50', 's_pts3d_b_in_a_c2_p50', 's_hgta_p75', 's_hgtb_p75', 's_hgtab_p75', 's_hgtab_p50') + (('s_maphgtab', 's_maphgta') if a.map else ()) + (('s_zmhgtab', 's_zmaphgtab') if a.zoom else ()):
    okc = (zmapok if 'zmap' in c else zmok if 'zm' in c else mapok if 'map' in c else ok) & np.isfinite(D[c].values); off = np.nanmedian((D.logd - D[c])[tr & okc]); dc[c[2:]] = (D[c].values + off, okc)
logd, _ = select_refine('logd', D.logd.values, dc); dist = np.exp(logd); D['p_dist'] = dist; D['p_logd'] = logd
print(f'[dist] test MAE: mast3r-stage0 {np.mean(np.abs(D.m_dist - D.dist)[te]):.4f} -> fused {np.mean(np.abs(dist - D.dist)[te]):.4f}', flush=True)
for k in 'xyz': D[f'g_{k}'] = dist * D[f'sg_v{k}']
D['mz_fromu'] = dist * D.n_uz
z, _ = select_refine('z', D.z.values, addmap({'sg': (D.g_z.values, ok), 'alt': ((dist * D.alt_vz).values, altok), 'mast3r': (D.m_z.values, allv), 'mlp': (D.n_z.values, allv), 'mlp_u': (D.mz_fromu.values, allv)}, dist * D.map_vz, D.q_z.values if a.neural else None, zm=dist * D.zm_vz if a.zoom else None, zmap=dist * D.zmap_vz if a.zoom else None), extra=['p_dist', 'p_logd', 'g_x', 'g_y', 'g_z'])
lx, _ = select_refine('lx', D.lx.values, addmap({'sg': (D.g_x.values, ok), 'alt': ((dist * D.alt_vx).values, altok), 'mast3r': (D.m_lx.values, allv), 'mlp': (D.n_lx.values, allv)}, dist * D.map_vx, D.q_lx.values if a.neural else None, zm=dist * D.zm_vx if a.zoom else None, zmap=dist * D.zmap_vx if a.zoom else None), extra=['p_dist', 'p_logd', 'g_x', 'g_y', 'g_z'])
ly, _ = select_refine('ly', D.ly.values, addmap({'sg': (D.g_y.values, ok), 'alt': ((dist * D.alt_vy).values, altok), 'mast3r': (D.m_ly.values, allv), 'mlp': (D.n_ly.values, allv)}, dist * D.map_vy, D.q_ly.values if a.neural else None, zm=dist * D.zm_vy if a.zoom else None, zmap=dist * D.zmap_vy if a.zoom else None), extra=['p_dist', 'p_logd', 'g_x', 'g_y', 'g_z'])
out = D[['manifest_index', 'split', 'tour']].copy()
out['p_heading'] = heading; out['p_tilt'] = tilt; out['p_dist'] = dist; out['p_z'] = z; out['p_lx'] = lx; out['p_ly'] = ly
for c in ('m_x', 'm_y', 'm_z', 'm_tilt', 'm_heading', 'm_dist', 'n_x', 'n_y', 'n_z', 'n_tilt', 'n_heading', 'n_dist'): out[c] = D[c].values
out.to_pickle(f'{SG}/{a.tag}_pred.pkl')
T = D[te]; o = out[te]
res = {'rows_test': int(te.sum()), 'rows_train': int(tr.sum()),
       'stage0_mast3r_only': report(T, o.m_x.values, o.m_y.values, o.m_z.values, o.m_tilt.values, o.m_heading.values, 'MASt3R-only GBDT      '),
       'stage0_mlp_only': report(T, o.n_x.values, o.n_y.values, o.n_z.values, o.n_tilt.values, o.n_heading.values, 'MASt3R-only MLP       '),
       'fusion_xy_from_mlp': report(T, o.n_x.values, o.n_y.values, o.p_z.values, o.p_tilt.values, o.p_heading.values, 'fusion (xy=mlp)       '),
       'aux': {'MAE_dist_direct': float(np.mean(np.abs(o.p_dist - T.dist))), 'MAE_lx': float(np.mean(np.abs(o.p_lx - T.lx))), 'MAE_ly': float(np.mean(np.abs(o.p_ly - T.ly)))}}
# oracle check: if cam-a absolute heading were known, what would x/y be?
ha = np.deg2rad(T.heading_a.values); xo = np.cos(ha) * o.p_lx + np.sin(ha) * o.p_ly; yo = -np.sin(ha) * o.p_lx + np.cos(ha) * o.p_ly
res['oracle_abs_heading'] = report(T, xo.values, yo.values, o.p_z.values, o.p_tilt.values, o.p_heading.values, 'fusion (oracle abs-h) ')
print(res['aux']); json.dump(res, open(f'{SG}/{a.tag}_metrics.json', 'w'), indent=2)
