"""Stage-1 / Stage-2 reliability-weighted fusion in a form that can be fitted once and then applied to new pairs.

The logic is a line-for-line port of superglue_fusion/stage1_fusion.py (config used for the reported result:
--map --zoom --sel sqrt --blend-p 2) and stage2_imagination/stage2_fusion.py. The only change is that every
LightGBM model fitted on the full training rows, every train-calibrated constant and every feature list goes
through a `Store`, so the same code path can run in "predict" mode on rows that have no labels.
"""
import sys, pickle, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import LGB, wrap


def _fit(X, y, params):
    m = lgb.LGBMRegressor(**params); m.fit(X, y); return m


class Store:
    """fit mode: tour-grouped OOF on train rows + full-train model (saved) applied to the remaining rows.
    predict mode: saved full-train model applied to every row."""
    def __init__(self, path=None, fitting=False, jobs=4):
        self.fitting = fitting; self.jobs = jobs; self.d = {}
        if not fitting: self.d = pickle.load(open(path, 'rb')); self._boosters = {}
    def save(self, path): pickle.dump(self.d, open(path, 'wb'))
    def set_rows(self, tr, groups): self.tr = np.asarray(tr, bool); self.te = ~self.tr; self.g = np.asarray(groups)
    def const(self, key, fn):
        if self.fitting: self.d[key] = fn()
        return self.d[key]
    def booster(self, key):
        if key not in self._boosters: self._boosters[key] = lgb.Booster(model_str=self.d[key])
        return self._boosters[key]
    def fit(self, key, X, y, obj, mask=None, **kw):
        X = np.asarray(X, np.float32)
        if not self.fitting: return self.booster(key).predict(X)
        tr, te = self.tr, self.te; params = {**LGB, 'objective': obj, 'n_jobs': self.jobs, **kw}
        Xtr, ytr = X[tr], np.asarray(y, np.float64)[tr]; mk = np.ones(tr.sum(), bool) if mask is None else np.asarray(mask, bool)[tr]
        folds = list(GroupKFold(5).split(Xtr, ytr, self.g[tr]))
        def job(a, b): a = a[mk[a]]; return _fit(Xtr[a], ytr[a], params).predict(Xtr[b])
        def full(): m = _fit(Xtr[mk], ytr[mk], params); return m.booster_.model_to_string(), (m.predict(X[te]) if te.any() else np.zeros(0))
        res = Parallel(n_jobs=len(folds) + 1)([delayed(job)(a, b) for a, b in folds] + [delayed(full)()])
        oof = np.zeros(tr.sum())
        for (a, b), p in zip(folds, res[:-1]): oof[b] = p
        self.d[key] = res[-1][0]; out = np.zeros(len(X)); out[tr] = oof; out[te] = res[-1][1]; return out


def _select_refine(D, st, pre, name, y, cands, base_feats, circular=False, extra=(), blend_p=2.0, finite_valid=False, log=print):
    """Predict each candidate's sqrt-|error|, blend candidates with weights (predicted error)^-2p, add an L1 residual GBDT."""
    diff = (lambda u, v: wrap(u - v)) if circular else (lambda u, v: u - v)
    if finite_valid: cands = {k: (np.asarray(v[0], float), np.asarray(v[1], bool) & np.isfinite(np.asarray(v[0], float))) for k, v in cands.items()}
    else: cands = {k: (np.asarray(v[0], float), np.asarray(v[1], bool)) for k, v in cands.items()}
    names = st.const(f'{pre}/{name}/cands', lambda: [k for k, v in cands.items() if (v[1] & st.tr).sum() >= 50])
    X0 = D[base_feats + list(extra)].values.astype(np.float32); E = np.full((len(D), len(names)), np.inf)
    for j, k in enumerate(names):
        v, valid = cands[k]; v0 = np.where(valid, v, 0.0)
        tgt = np.sqrt(np.abs(diff(y, v0))) if st.fitting else None
        E[:, j] = np.where(valid, st.fit(f'{pre}/{name}/err/{k}', X0, tgt, 'l2', mask=valid, n_estimators=500), np.inf)
    pick = E.argmin(1); V = np.stack([np.where(cands[k][1], cands[k][0], np.nan) for k in names], 1)
    Wt = np.where(np.isfinite(E), np.maximum(np.square(np.maximum(np.where(np.isfinite(E), E, 0.0), 0)), 1e-3) ** (-blend_p), 0.0)
    Wt = Wt / Wt.sum(1, keepdims=True); Vz = np.nan_to_num(V)
    base = np.rad2deg(np.arctan2((Wt * np.sin(np.deg2rad(Vz))).sum(1), (Wt * np.cos(np.deg2rad(Vz))).sum(1))) if circular else (Wt * Vz).sum(1)
    Xr = np.column_stack([X0, np.where(np.isfinite(E), E, 20.0), pick] + [np.nan_to_num(diff(V[:, j], base), nan=0.0) for j in range(len(names))])
    res = st.fit(f'{pre}/{name}/res', Xr, diff(y, base) if st.fitting else None, 'l1')
    final = base + res; final = wrap(final) if circular else final
    # predicted |error| of the blended estimate: weights are ~1/var, so var(blend) ~ 1/sum(1/var_k); kept as a per-pair reliability score
    with np.errstate(all='ignore'):
        e2 = np.maximum(np.square(np.maximum(np.where(np.isfinite(E), E, np.inf), 0)), 1e-3)   # predicted |err| per candidate
        rel = 1.0 / np.sqrt(np.sum(np.where(np.isfinite(e2), e2 ** -2.0, 0.0), 1))
    if st.fitting:
        te, tr = st.te, st.tr
        msg = {k: float(np.mean(np.abs(diff(y, np.where(cands[k][1], cands[k][0], 0.0)))[te & cands[k][1]])) for k in names} if te.any() else {}
        log(f'[{pre}/{name}] test MAE per candidate: ' + ' '.join(f'{k}={v:.4f}' for k, v in msg.items()) +
            (f' | blended {np.mean(np.abs(diff(y, base))[te]):.4f} | +residual {np.mean(np.abs(diff(y, final))[te]):.4f}' if te.any() else '') +
            f' | OOF {np.mean(np.abs(diff(y, final))[tr]):.4f}', flush=True)
    return final, rel


# --------------------------------------------------------------------------------------------- stage 1
def stage1(D, st, log=print):
    """D: one row per pair with stage-0 predictions (m_*, n_*), MASt3R stats, pass-1 geometry (sg_*, alt_*), MAP (map_*), zoom (zm_*, zmap_*).
    Returns a frame with p_heading, p_tilt, p_dist, p_z, p_lx, p_ly (+ predicted-error scores r_*)."""
    D = D.copy(); F = st.fitting
    mfeats = st.const('s1/mfeats', lambda: [c for c in D.columns if c in set(D.attrs['mfeats'])])
    D['map_ok'] = D.map_ok.fillna(0.0); D['zm_ok'] = D.zm_ok.fillna(0.0); D['zmap_ok'] = D.zmap_ok.fillna(0.0)
    D['zm_dh'] = np.where(D.zm_ok == 1, D.zm_dh, np.nan); D['zm_dhdiff_sg'] = wrap(D.zm_dh - D.sg_dh); D['zm_dhdiff_map'] = wrap(D.zmap_dh - D.map_dh); D['zm_dtiltdiff_map'] = D.zmap_dtilt - D.map_dtilt
    D['zm_dircos_sg'] = D.zm_vx * D.sg_vx + D.zm_vy * D.sg_vy + D.zm_vz * D.sg_vz; D['zm_dircos_zmap'] = D.zm_vx * D.zmap_vx + D.zm_vy * D.zmap_vy + D.zm_vz * D.zmap_vz
    D['s_zmhgtab'] = -np.log(((D.zm_hgta_p75 + D.zm_hgtb_p75) / 2).clip(lower=1e-3)); D['s_zmaphgtab'] = -np.log(((D.zmap_hgta + D.zmap_hgtb) / 2).clip(lower=1e-3)); D['zm_logpar'] = np.log(D.zm_par.clip(lower=1e-6))
    D['sg_ok'] = D.sg_ok.fillna(0.0); ok = D.sg_ok.values == 1
    D['f_dh_diff'] = wrap(D.sg_dh - D.n_heading); D['f_dtilt_diff'] = D.sg_dtilt - D.n_tilt; D['f_ta_diff'] = D.sg_ta - D.n_tilt_a; D['f_tb_diff'] = D.sg_tb - D.n_tilt_b
    D['f_dh_diff_m'] = wrap(D.sg_dh - D.m_heading); D['f_mn_dh'] = wrap(D.m_heading - D.n_heading)
    D['f_abs_sin_dh'] = np.abs(D.sg_dh_sin); D['f_tsum_diff'] = D.sg_tsum - D.n_tilt_a - D.n_tilt_b
    D['c_tilt_s1'] = D.sg_tsum - 2 * D.n_tilt_a; D['c_tilt_s2'] = 2 * D.n_tilt_b - D.sg_tsum; D['c_tilt_ab'] = D.n_tilt_b - D.n_tilt_a
    D['alt_dh'] = np.rad2deg(np.arctan2(D.alt_dh_sin, D.alt_dh_cos)); D['alt_dtilt'] = D.alt_tb - D.alt_ta
    for k in 'xyz': D[f'f_u{k}_diff'] = D[f'sg_v{k}'] - D[f'n_u{k}']
    D['f_dir_cos_m'] = (D.sg_vx * D.n_ux + D.sg_vy * D.n_uy + D.sg_vz * D.n_uz) / np.sqrt(D.n_ux**2 + D.n_uy**2 + D.n_uz**2).clip(lower=1e-3)
    D['f_logpar'] = np.log(D.sg_par.clip(lower=1e-6)); D['f_alt_logpar'] = np.log(D.alt_par.clip(lower=1e-6))
    depth_stats = [c for c in mfeats if c.startswith(('pts3d_a_c2_', 'pts3d_a_norm_', 'pts3d_b_in_a_c2_', 'pts3d_b_c2_')) and c.endswith(('p25', 'p50', 'p75', 'mean'))]
    for c in depth_stats: D['s_' + c] = D.f_logpar + np.log(D[c].clip(lower=1e-3))
    for c in ('a', 'b'):
        for q in (50, 75, 90): D[f's_hgt{c}_p{q}'] = -np.log(D[f'sg_hgt{c}_p{q}'].clip(lower=1e-3))
    D['s_hgtab_p75'] = -np.log(((D.sg_hgta_p75 + D.sg_hgtb_p75) / 2).clip(lower=1e-3)); D['s_hgtab_p50'] = -np.log(((D.sg_hgta_p50 + D.sg_hgtb_p50) / 2).clip(lower=1e-3))
    D['f_hgt_ratio'] = D.sg_hgtb_p75 / D.sg_hgta_p75.clip(lower=1e-3); D['f_hgt_diff'] = D.sg_hgtb_p75 - D.sg_hgta_p75
    D['s_maphgtab'] = -np.log(((D.map_hgta + D.map_hgtb) / 2).clip(lower=1e-3)); D['s_maphgta'] = -np.log(D.map_hgta.clip(lower=1e-3))
    D['f_map_dh_diff'] = wrap(D.map_dh - D.sg_dh); D['f_map_dtilt_diff'] = D.map_dtilt - D.sg_dtilt; D['f_map_dir_cos'] = D.map_vx * D.sg_vx + D.map_vy * D.sg_vy + D.map_vz * D.sg_vz
    D = D.copy()
    def _base():
        sg_feats = [c for c in D.columns if c.startswith(('sg_', 'alt_', 'map_', 'zm_', 'zmap_', 'f_', 's_', 'c_tilt')) and c not in ('sg_dh',)] + ['sg_dh']
        m_feats0 = [c for c in D.attrs['s0cols'] if c.startswith(('m_', 'n_'))]
        return list(dict.fromkeys(sg_feats + m_feats0 + depth_stats + [c for c in mfeats if c.startswith(('conf_', 'delta_pts3d_ba_norm', 'match_', 'pose_'))]))
    base_feats = st.const('s1/base_feats', _base)
    sr = lambda name, y, cands, **kw: _select_refine(D, st, 's1', name, y, cands, base_feats, log=log, **kw)
    Y = (lambda c: D[c].values) if F else (lambda c: None)
    allv = np.ones(len(D), bool); altok = (D.alt_ok.fillna(0).values == 1); mapok = (D.map_ok.values == 1); zmok = (D.zm_ok.values == 1); zmapok = (D.zmap_ok.values == 1)
    def addmap(c, v, zm, zmap): c['map'] = (np.asarray(v), mapok); c['zoom'] = (np.asarray(zm), zmok); c['zoom_map'] = (np.asarray(zmap), zmapok); return c
    heading, r_h = sr('heading', Y('heading'), addmap({'sg': (D.sg_dh.values, ok), 'alt': (D.alt_dh.values, altok), 'mast3r': (D.m_heading.values, allv), 'mlp': (D.n_heading.values, allv)}, D.map_dh.values, D.zm_dh.values, D.zmap_dh.values), circular=True)
    tilt, r_t = sr('tilt', Y('tilt'), addmap({'sg': (D.sg_dtilt.values, ok), 'alt': (D.alt_dtilt.values, altok), 'mast3r': (D.m_tilt.values, allv), 'mlp': (D.n_tilt.values, allv), 'mlp_ab': (D.c_tilt_ab.values, allv), 'sum1': (D.c_tilt_s1.values, ok), 'sum2': (D.c_tilt_s2.values, ok)}, D.map_dtilt.values, D.zm_dtilt.values, D.zmap_dtilt.values))
    dc = {'mast3r': (D.m_logd.values, allv), 'mlp': (D.n_logd.values, allv)}
    for c in ('s_pts3d_a_c2_p50', 's_pts3d_a_norm_p50', 's_pts3d_b_in_a_c2_p50', 's_hgta_p75', 's_hgtb_p75', 's_hgtab_p75', 's_hgtab_p50', 's_maphgtab', 's_maphgta', 's_zmhgtab', 's_zmaphgtab'):
        okc = (zmapok if 'zmap' in c else zmok if 'zm' in c else mapok if 'map' in c else ok) & np.isfinite(D[c].values)
        off = st.const(f's1/off/{c}', lambda: float(np.nanmedian((D.logd - D[c])[st.tr & okc]))); dc[c[2:]] = (D[c].values + off, okc)
    logd, r_d = sr('logd', Y('logd'), dc); dist = np.exp(logd); D['p_dist'] = dist; D['p_logd'] = logd
    for k in 'xyz': D[f'g_{k}'] = dist * D[f'sg_v{k}']
    D['mz_fromu'] = dist * D.n_uz; ex = ['p_dist', 'p_logd', 'g_x', 'g_y', 'g_z']
    z, r_z = sr('z', Y('z'), addmap({'sg': (D.g_z.values, ok), 'alt': ((dist * D.alt_vz).values, altok), 'mast3r': (D.m_z.values, allv), 'mlp': (D.n_z.values, allv), 'mlp_u': (D.mz_fromu.values, allv)}, dist * D.map_vz, dist * D.zm_vz, dist * D.zmap_vz), extra=ex)
    lx, r_x = sr('lx', Y('lx'), addmap({'sg': (D.g_x.values, ok), 'alt': ((dist * D.alt_vx).values, altok), 'mast3r': (D.m_lx.values, allv), 'mlp': (D.n_lx.values, allv)}, dist * D.map_vx, dist * D.zm_vx, dist * D.zmap_vx), extra=ex)
    ly, r_y = sr('ly', Y('ly'), addmap({'sg': (D.g_y.values, ok), 'alt': ((dist * D.alt_vy).values, altok), 'mast3r': (D.m_ly.values, allv), 'mlp': (D.n_ly.values, allv)}, dist * D.map_vy, dist * D.zm_vy, dist * D.zmap_vy), extra=ex)
    out = D[['manifest_index']].copy()
    out['p_heading'] = heading; out['p_tilt'] = tilt; out['p_dist'] = dist; out['p_z'] = z; out['p_lx'] = lx; out['p_ly'] = ly
    for k, r in (('heading', r_h), ('tilt', r_t), ('logd', r_d), ('z', r_z), ('lx', r_x), ('ly', r_y)): out['r_' + k] = r
    return out


# --------------------------------------------------------------------------------------------- stage 2
S1_FEATS = ['s1_heading', 's1_tilt', 's1_dist', 's1_logd', 's1_z', 's1_lx', 's1_ly', 'sg_inl', 'sg_inl_ratio', 'sg_resid', 'sg_tinfo', 'sg_n', 'sg_cov0', 'sg_cov1', 'sg_zspread', 'f_logpar', 'map_inl', 'map_resid', 'map_from_prior',
            'zm_inl', 'zm_inl_ratio', 'zm_resid', 'zm_tinfo', 'zm_zspread', 'zm_logpar', 'zm_zoom', 'zm_cov0', 'zm_cov1', 'zmap_inl', 'zmap_resid', 'zmap_from_prior', 'zmap_ta', 'zmap_tb', 'zmap_dh', 'zmap_vx', 'zmap_vy', 'zmap_vz', 's_zmaphgtab',
            'n_heading', 'n_tilt', 'n_tilt_a', 'n_tilt_b', 'n_logd', 'n_z', 'n_lx', 'n_ly', 'n_ux', 'n_uy', 'n_uz', 'm_heading', 'm_tilt', 'm_logd', 'm_z']

def stage2(D, st, log=print):
    """D: stage-1 inputs + stage-1 prediction (s1_*) + imagination re-matching geometry (im_*, imap_*, ver_*)."""
    D = D.copy(); F = st.fitting
    mfeats = st.const('s2/mfeats', lambda: [c for c in D.columns if c in set(D.attrs['mfeats'])])
    for c in ('sg_ok', 'map_ok', 'zm_ok', 'zmap_ok', 'im_ok', 'imap_ok'): D[c] = D[c].fillna(0.0)
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
    D = D.copy()
    def _base():
        s1_feats = S1_FEATS + depth_stats + [c for c in mfeats if c.startswith(('conf_', 'match_'))]
        im_feats = [c for c in D.columns if c.startswith(('im_', 'imap_', 'fi_', 'si_'))] + [c for c in D.columns if c.startswith('ver_')]
        return list(dict.fromkeys([c for c in s1_feats if c in D.columns] + im_feats))
    base_feats = st.const('s2/base_feats', _base)
    sr = lambda name, y, cands, **kw: _select_refine(D, st, 's2', name, y, cands, base_feats, finite_valid=True, log=log, **kw)
    Y = (lambda c: D[c].values) if F else (lambda c: None)
    allv = np.ones(len(D), bool); imok = (D.im_ok.values == 1); imapok = (D.imap_ok.values == 1)
    heading, r_h = sr('heading', Y('heading'), {'stage1': (D.s1_heading, allv), 'imag': (D.im_dh, imok), 'imag_map': (D.imap_dh, imapok)}, circular=True)
    tilt, r_t = sr('tilt', Y('tilt'), {'stage1': (D.s1_tilt, allv), 'imag': (D.im_dtilt, imok), 'imag_map': (D.imap_dtilt, imapok)})
    dc = {'stage1': (D.s1_logd, allv)}
    for c in ('si_pts3d_a_c2_p50', 'si_pts3d_b_in_a_c2_p50', 'si_hgtab_p75', 'si_hgtab_p50', 'si_maphgtab', 'si_maphgta', 'si_maphgtb'):
        okc = (imapok if ('map' in c or 'pts3d' in c) else imok) & np.isfinite(D[c].values)
        off = st.const(f's2/off/{c}', lambda: float(np.nanmedian((D.logd - D[c])[st.tr & okc])) if (st.tr & okc).sum() else 0.0); dc[c[3:]] = (D[c].values + off, okc)
    logd, r_d = sr('logd', Y('logd'), dc); dist = np.exp(logd); D['p_dist'] = dist; D['p_logd'] = logd
    for k in 'xyz': D[f'gi_{k}'] = dist * D[f'imap_v{k}']
    ex = ['p_dist', 'p_logd', 'gi_x', 'gi_y', 'gi_z']
    z, r_z = sr('z', Y('z'), {'stage1': (D.s1_z, allv), 'imag': (dist * D.im_vz, imok), 'imag_map': (dist * D.imap_vz, imapok)}, extra=ex)
    lx, r_x = sr('lx', Y('lx'), {'stage1': (D.s1_lx, allv), 'imag': (dist * D.im_vx, imok), 'imag_map': (dist * D.imap_vx, imapok)}, extra=ex)
    ly, r_y = sr('ly', Y('ly'), {'stage1': (D.s1_ly, allv), 'imag': (dist * D.im_vy, imok), 'imag_map': (dist * D.imap_vy, imapok)}, extra=ex)
    out = D[['manifest_index']].copy(); out['p_heading'] = heading; out['p_tilt'] = tilt; out['p_dist'] = dist; out['p_z'] = z; out['p_lx'] = lx; out['p_ly'] = ly
    for k, r in (('heading', r_h), ('tilt', r_t), ('logd', r_d), ('z', r_z), ('lx', r_x), ('ly', r_y)): out['r_' + k] = r
    return out
