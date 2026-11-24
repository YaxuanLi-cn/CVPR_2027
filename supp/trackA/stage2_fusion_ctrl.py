#!/usr/bin/env python3
"""Refinement-stage fusion with switches for the control experiments.

Copy of stage2_imagination/stage2_fusion.py (same features, same LightGBM settings, same tour-grouped OOF protocol);
with no switch it is the archived Stage II fusion. Added switches:
  --imag PKL          geometry of the (control) re-matching pass; its rows define the pairs that are used
  --no-imag-cands     no new candidates and no new features: a second learned fusion over the coarse-stage outputs only
  --no-ver            drop the hypothesis-verification scores
  --no-scale          keep the coarse-stage distance and drop every feature that carries a re-triangulated scale
  --uniform           equal weights for all valid candidates instead of predicted-error weights
  --no-residual       no residual regressor: output = weighted blend of the candidates
Writes <out>/<tag>_pred.pkl with p_* (final) and b_* (blend before the residual).
"""
import sys, json, argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
ap = argparse.ArgumentParser(); ap.add_argument('--imag', default=f'{SG}/stage2/geom_imag_9k.pkl'); ap.add_argument('--tag', required=True); ap.add_argument('--jobs', type=int, default=1)
ap.add_argument('--out', default='/root/rasr-pairuav/outputs/supp/trackA'); ap.add_argument('--blend-p', type=float, default=2.0); ap.add_argument('--ids', default=None, help='npy file: restrict the rows (train and test) to these pairs'); ap.add_argument('--sift', default=None, help='pkl with SIFT + essential-matrix geometry (sf_*): its pose and scale cues become additional candidates')
for f in ('--no-imag-cands', '--no-ver', '--no-scale', '--uniform', '--no-residual'): ap.add_argument(f, action='store_true')
a = ap.parse_args()
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); S0 = pd.read_pickle(f'{SG}/stage0.pkl').merge(pd.read_pickle(f'{SG}/stage0_mlp.pkl'), on='manifest_index')
I = pd.read_pickle(a.imag); F1 = pd.read_pickle(f'{SG}/half_zoom_pred.pkl')[['manifest_index', 'p_heading', 'p_tilt', 'p_dist', 'p_z', 'p_lx', 'p_ly']].rename(columns=lambda c: c.replace('p_', 's1_'))
D = P.merge(I, on='manifest_index').merge(F1, on='manifest_index').merge(S0, on='manifest_index').merge(M.drop(columns='split'), on='manifest_index') \
     .merge(pd.read_pickle(f'{SG}/geom_half.pkl'), on='manifest_index', how='left').merge(pd.read_pickle(f'{SG}/map_half.pkl'), on='manifest_index', how='left').merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl'), on='manifest_index', how='left')
for c in ('sg_ok', 'map_ok', 'zm_ok', 'zmap_ok', 'im_ok', 'imap_ok'): D[c] = D[c].fillna(0.0)
if a.ids: D = D[D.manifest_index.isin(set(np.load(a.ids).tolist()))].reset_index(drop=True)
if a.sift:
    D = D.merge(pd.read_pickle(a.sift), on='manifest_index', how='left'); D['sf_ok'] = D.sf_ok.fillna(0.0)
    D['ss_hgtab_p75'] = -np.log(((D.sf_hgta_p75 + D.sf_hgtb_p75) / 2).clip(lower=1e-3)); D['ss_hgtab_p50'] = -np.log(((D.sf_hgta_p50 + D.sf_hgtb_p50) / 2).clip(lower=1e-3)); D['sf_logpar'] = np.log(D.sf_par.clip(lower=1e-6))
mfeats = [c for c in M.columns if c not in ('manifest_index', 'split')]; print('rows', len(D), D.split.value_counts().to_dict(), 'im_ok', D.im_ok.mean(), flush=True)

# ---------- features (identical to stage2_fusion.py)
D['s1_logd'] = np.log(D.s1_dist); D['f_logpar'] = np.log(D.sg_par.clip(lower=1e-6)); D['zm_logpar'] = np.log(D.zm_par.clip(lower=1e-6)); D['im_logpar'] = np.log(D.im_par.clip(lower=1e-6)); D['imap_logpar'] = np.log(D.imap_par.clip(lower=1e-6))
depth_stats = [c for c in mfeats if c.startswith(('pts3d_a_c2_', 'pts3d_a_norm_', 'pts3d_b_in_a_c2_', 'pts3d_b_c2_')) and c.endswith(('p25', 'p50', 'p75', 'mean'))]
for c in depth_stats: D['si_' + c] = D.imap_logpar + np.log(D[c].clip(lower=1e-3))
D['si_hgtab_p75'] = -np.log(((D.im_hgta_p75 + D.im_hgtb_p75) / 2).clip(lower=1e-3)); D['si_hgtab_p50'] = -np.log(((D.im_hgta_p50 + D.im_hgtb_p50) / 2).clip(lower=1e-3))
D['si_maphgtab'] = -np.log(((D.imap_hgta + D.imap_hgtb) / 2).clip(lower=1e-3)); D['si_maphgta'] = -np.log(D.imap_hgta.clip(lower=1e-3)); D['si_maphgtb'] = -np.log(D.imap_hgtb.clip(lower=1e-3))
D['s_zmaphgtab'] = -np.log(((D.zmap_hgta + D.zmap_hgtb) / 2).clip(lower=1e-3))
D['fi_dh_s1'] = wrap(D.im_dh - D.s1_heading); D['fi_dhmap_s1'] = wrap(D.imap_dh - D.s1_heading); D['fi_dh_map'] = wrap(D.imap_dh - D.im_dh); D['fi_dtilt_s1'] = D.im_dtilt - D.s1_tilt; D['fi_dtiltmap_s1'] = D.imap_dtilt - D.s1_tilt
D['fi_dtilt_map'] = D.imap_dtilt - D.im_dtilt; D['fi_dircos_map'] = D.im_vx * D.imap_vx + D.im_vy * D.imap_vy + D.im_vz * D.imap_vz
n1 = np.sqrt(D.s1_lx**2 + D.s1_ly**2 + D.s1_z**2).clip(lower=1e-6); D['fi_dircos_s1'] = (D.imap_vx * D.s1_lx + D.imap_vy * D.s1_ly + D.imap_vz * D.s1_z) / n1; D['fi_dircos_zmap'] = D.imap_vx * D.zmap_vx + D.imap_vy * D.zmap_vy + D.imap_vz * D.zmap_vz
D['fi_inl_gain'] = np.log1p(D.im_inl) - np.log1p(D.zm_inl.fillna(0)); D['fi_hgt_ratio'] = D.imap_hgtb / D.imap_hgta.clip(lower=1e-3); D['fi_ta_n'] = D.imap_ta - D.n_tilt_a; D['fi_tb_n'] = D.imap_tb - D.n_tilt_b
if a.sift:
    D['fs_dh_s1'] = wrap(D.sf_dh - D.s1_heading); D['fs_dtilt_s1'] = D.sf_dtilt - D.s1_tilt; n1s = np.sqrt(D.s1_lx**2 + D.s1_ly**2 + D.s1_z**2).clip(lower=1e-6); D['fs_dircos_s1'] = (D.sf_vx * D.s1_lx + D.sf_vy * D.s1_ly + D.sf_vz * D.s1_z) / n1s
    D['fs_dircos_imap'] = D.sf_vx * D.imap_vx + D.sf_vy * D.imap_vy + D.sf_vz * D.imap_vz; D['fs_dh_imap'] = wrap(D.sf_dh - D.imap_dh); D['ss_pts3d_a_c2_p50'] = D.sf_logpar + np.log(D['pts3d_a_c2_p50'].clip(lower=1e-3))
D = D.copy()
s1_feats = ['s1_heading', 's1_tilt', 's1_dist', 's1_logd', 's1_z', 's1_lx', 's1_ly', 'sg_inl', 'sg_inl_ratio', 'sg_resid', 'sg_tinfo', 'sg_n', 'sg_cov0', 'sg_cov1', 'sg_zspread', 'f_logpar', 'map_inl', 'map_resid', 'map_from_prior',
            'zm_inl', 'zm_inl_ratio', 'zm_resid', 'zm_tinfo', 'zm_zspread', 'zm_logpar', 'zm_zoom', 'zm_cov0', 'zm_cov1', 'zmap_inl', 'zmap_resid', 'zmap_from_prior', 'zmap_ta', 'zmap_tb', 'zmap_dh', 'zmap_vx', 'zmap_vy', 'zmap_vz', 's_zmaphgtab',
            'n_heading', 'n_tilt', 'n_tilt_a', 'n_tilt_b', 'n_logd', 'n_z', 'n_lx', 'n_ly', 'n_ux', 'n_uy', 'n_uz', 'm_heading', 'm_tilt', 'm_logd', 'm_z'] + depth_stats + [c for c in mfeats if c.startswith(('conf_', 'match_'))]
im_feats = [c for c in D.columns if c.startswith(('im_', 'imap_', 'fi_', 'si_'))] + ([] if a.no_ver else [c for c in D.columns if c.startswith('ver_')])
if a.no_scale: im_feats = [c for c in im_feats if not (c.startswith('si_') or 'par' in c or 'hgt' in c or 'zspread' in c)]   # nothing that carries a re-triangulated scale
sift_feats = ([c for c in ('sf_ok', 'sf_n', 'sf_inl', 'sf_inl_ratio', 'sf_resid', 'sf_tinfo', 'sf_zspread', 'sf_cov0', 'sf_cov1', 'sf_cheir', 'sf_logpar', 'sf_dh', 'sf_dtilt', 'sf_ta', 'sf_tb', 'sf_vx', 'sf_vy', 'sf_vz', 'ss_hgtab_p75', 'ss_hgtab_p50', 'ss_pts3d_a_c2_p50', 'fs_dh_s1', 'fs_dtilt_s1', 'fs_dircos_s1') if c in D.columns] + ([] if a.no_imag_cands else ['fs_dircos_imap', 'fs_dh_imap'])) if a.sift else []
base_feats = list(dict.fromkeys([c for c in s1_feats if c in D.columns] + ([] if a.no_imag_cands else im_feats) + sift_feats)); print('n feats', len(base_feats), flush=True)
tr = (D.split == 'train').values; te = ~tr; g = D.tour.values[tr]
def fit(X, y, obj='l1', mask=None, **kw):
    o, p = oof_fit(X[tr], y[tr], g, X[te], dict(objective=obj, n_jobs=a.jobs, **kw), mask=None if mask is None else mask[tr]); out = np.zeros(len(D)); out[tr] = o; out[te] = p; return out

def select_refine(name, y, cands, circular=False, extra=()):
    diff = (lambda u, v: wrap(u - v)) if circular else (lambda u, v: u - v)
    cands = {k: (np.asarray(v[0], float), np.asarray(v[1], bool) & np.isfinite(np.asarray(v[0], float))) for k, v in cands.items()}; cands = {k: v for k, v in cands.items() if (v[1] & tr).sum() >= 50}
    X0 = D[base_feats + list(extra)].values.astype(np.float32); names = list(cands); E = np.full((len(D), len(names)), np.inf)
    V = np.stack([np.where(cands[k][1], cands[k][0], np.nan) for k in names], 1); valid = np.isfinite(V)
    if a.uniform: Wt = valid.astype(float); E = np.where(valid, 1.0, np.inf)
    else:
        for j, k in enumerate(names):
            v, ok = cands[k]; err = np.abs(diff(y, np.where(ok, v, 0.0))); E[:, j] = np.where(ok, fit(X0, np.sqrt(err), 'l2', mask=ok, n_estimators=500), np.inf)
        Wt = np.where(np.isfinite(E), np.maximum(np.square(np.maximum(np.where(np.isfinite(E), E, 0.0), 0)), 1e-3) ** (-a.blend_p), 0.0)
    pick = E.argmin(1); Wt = Wt / Wt.sum(1, keepdims=True); Vz = np.nan_to_num(V)
    base = np.rad2deg(np.arctan2((Wt * np.sin(np.deg2rad(Vz))).sum(1), (Wt * np.cos(np.deg2rad(Vz))).sum(1))) if circular else (Wt * Vz).sum(1)
    if a.no_residual: final = base
    else:
        Xr = np.column_stack([X0, np.where(np.isfinite(E), E, 20.0), pick] + [np.nan_to_num(diff(V[:, j], base), nan=0.0) for j in range(len(names))])
        res = fit(Xr, diff(y, base), 'l1'); final = base + res; final = wrap(final) if circular else final
    msg = {k: float(np.mean(np.abs(diff(y, np.where(cands[k][1], cands[k][0], 0.0)))[te & cands[k][1]])) for k in names}
    print(f'[{name}] test MAE per candidate (valid rows): ' + ' '.join(f'{k}={v:.4f}' for k, v in msg.items()) + f' | blended {np.mean(np.abs(diff(y, base))[te]):.4f} | final {np.mean(np.abs(diff(y, final))[te]):.4f} | OOF {np.mean(np.abs(diff(y, final))[tr]):.4f}', flush=True)
    return final, base

allv = np.ones(len(D), bool); imok = (D.im_ok.values == 1) & (not a.no_imag_cands); imapok = (D.imap_ok.values == 1) & (not a.no_imag_cands)
sfok = (D.sf_ok.values == 1) if a.sift else None
def withsift(c, v):
    if a.sift: c['sift'] = (v, sfok)
    return c
heading, b_heading = select_refine('heading', D.heading.values, withsift({'stage1': (D.s1_heading, allv), 'imag': (D.im_dh, imok), 'imag_map': (D.imap_dh, imapok)}, D.sf_dh if a.sift else None), circular=True)
tilt, b_tilt = select_refine('tilt', D.tilt.values, withsift({'stage1': (D.s1_tilt, allv), 'imag': (D.im_dtilt, imok), 'imag_map': (D.imap_dtilt, imapok)}, D.sf_dtilt if a.sift else None))
if a.no_scale: logd = b_logd = D.s1_logd.values
else:
    dc = {'stage1': (D.s1_logd, allv)}
    for c in ('si_pts3d_a_c2_p50', 'si_pts3d_b_in_a_c2_p50', 'si_hgtab_p75', 'si_hgtab_p50', 'si_maphgtab', 'si_maphgta', 'si_maphgtb'):
        okc = (imapok if ('map' in c or 'pts3d' in c) else imok) & np.isfinite(D[c].values); off = np.nanmedian((D.logd - D[c])[tr & okc]) if (tr & okc).sum() else 0.0; dc[c[3:]] = (D[c].values + off, okc)
    if a.sift:
        for c in ('ss_hgtab_p75', 'ss_hgtab_p50', 'ss_pts3d_a_c2_p50'):
            okc = sfok & np.isfinite(D[c].values); off = np.nanmedian((D.logd - D[c])[tr & okc]) if (tr & okc).sum() else 0.0; dc['sift_' + c[3:]] = (D[c].values + off, okc)
    logd, b_logd = select_refine('logd', D.logd.values, dc)
dist = np.exp(logd); D['p_dist'] = dist; D['p_logd'] = logd
print(f'[dist] test MAE: stage1 {np.mean(np.abs(D.s1_dist - D.dist)[te]):.4f} -> stage2 {np.mean(np.abs(dist - D.dist)[te]):.4f}', flush=True)
for k in 'xyz': D[f'gi_{k}'] = dist * D[f'imap_v{k}']
ex = ['p_dist', 'p_logd'] + ([] if a.no_imag_cands else ['gi_x', 'gi_y', 'gi_z'])
z, b_z = select_refine('z', D.z.values, withsift({'stage1': (D.s1_z, allv), 'imag': (dist * D.im_vz, imok), 'imag_map': (dist * D.imap_vz, imapok)}, (dist * D.sf_vz) if a.sift else None), extra=ex)
lx, b_lx = select_refine('lx', D.lx.values, withsift({'stage1': (D.s1_lx, allv), 'imag': (dist * D.im_vx, imok), 'imag_map': (dist * D.imap_vx, imapok)}, (dist * D.sf_vx) if a.sift else None), extra=ex)
ly, b_ly = select_refine('ly', D.ly.values, withsift({'stage1': (D.s1_ly, allv), 'imag': (dist * D.im_vy, imok), 'imag_map': (dist * D.imap_vy, imapok)}, (dist * D.sf_vy) if a.sift else None), extra=ex)
out = D[['manifest_index', 'split', 'tour']].copy(); out['p_heading'] = heading; out['p_tilt'] = tilt; out['p_dist'] = dist; out['p_z'] = z; out['p_lx'] = lx; out['p_ly'] = ly
out['b_heading'] = b_heading; out['b_tilt'] = b_tilt; out['b_dist'] = np.exp(b_logd); out['b_z'] = b_z; out['b_lx'] = b_lx; out['b_ly'] = b_ly
for c in ('s1_heading', 's1_tilt', 's1_dist', 's1_z', 's1_lx', 's1_ly', 'imap_ok', 'im_ok', 'im_inl', 'im_dh', 'im_dtilt', 'im_vx', 'im_vy', 'im_vz', 'imap_dh', 'imap_dtilt', 'imap_vx', 'imap_vy', 'imap_vz', 'im_disp_med', 'im_ntiles'): out[c] = D[c].values
out.to_pickle(f'{a.out}/{a.tag}_pred.pkl'); T = D[te]; o = out[te]; res = {'rows_test': int(te.sum()), 'rows_train': int(tr.sum()), 'args': vars(a)}
for nm, pre in (('stage1', 's1_'), ('blend', 'b_'), ('final', 'p_')):
    res[nm] = dict(heading=float(np.mean(np.abs(wrap(o[pre + 'heading'] - T.heading)))), tilt=float(np.mean(np.abs(o[pre + 'tilt'] - T.tilt))), dist=float(np.mean(np.abs(o[pre + 'dist'] - T.dist))),
                   z=float(np.mean(np.abs(o[pre + 'z'] - T.z))), lx=float(np.mean(np.abs(o[pre + 'lx'] - T.lx))), ly=float(np.mean(np.abs(o[pre + 'ly'] - T.ly))))
    print(nm, {k: round(v, 4) for k, v in res[nm].items()})
json.dump(res, open(f'{a.out}/{a.tag}_metrics.json', 'w'), indent=2)
