#!/usr/bin/env python3
"""Track B, part 3: every geometric two-view baseline on the same random test pairs of PairUAV, scored with supp/common.py.

Rows = matcher x {plain, rot. TTA} x metric-scale variant. Nothing is trained on PairUAV; fitted on the 300 calibration
(training-tour) pairs only: one scalar per (method, scale variant) and, per matcher, the essential-matrix inlier threshold.
  plain     target image as is (k = 0)
  rot. TTA  target rotated by k*90 deg, keep the rotation with the most essential-matrix inliers
  scale     native  = MASt3R's own metric depth / pointmap, untouched
            calib   = native x one global constant (median of true / predicted distance on the calibration pairs)
            height_p75ab = flight-height prior: baseline = H / h, with h the camera height above the triangulated scene in baseline units,
                      taken as the mean over both cameras of the 75th percentile of the inlier heights (the ground level; the statistic of
                      DreamNav's own flight-height cue) and H fitted on the calibration pairs (median of true distance x h)
            height_p50a  = the same with h = median inlier height below the source camera only (statistic of the archived direct
                      baselines; biased by roofs; kept in the JSON for reference)
            height_pointmap (PnP rows) = H / (median height of the source pointmap), translation and height in the model's units
            blend   = log-linear blend of the two metric cues above: log d = w log(H / h_p75ab) + (1 - w) log(d_MLP) + b, with the two
                      scalars (w, b) fitted on the calibration pairs (least absolute log-error); a minimal "geometry + learned scale" fusion
            learned = unit translation direction of the method x distance regressed by the frozen-MASt3R-statistics MLP
                      (trained on the 47,687 training pairs; the learned-prior branch of DreamNav)
Evaluation frame: levelled frame of the source camera (right, forward, up); no absolute heading is involved.
Failed pairs (no pose) predict zero motion and are counted in "fail %".
Output: outputs/supp/trackB/pairuav.json, pairuav.md
"""
import sys, os, json, glob
import numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/supp', f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
import common as C
from fusion_lib import add_targets
from rb_recon import subset
OUT = f'{ROOT}/outputs/supp/trackB'; RB = f'{ROOT}/outputs/reviewer_baselines'; GRID = (0.2, 0.4, 0.75, 1.5, 3.0, 6.0, 12.0)      # candidate inlier thresholds, pixels of the 3686-px image
NAMES = {'sift': 'SIFT + ratio test', 'sgplain': 'SuperPoint + SuperGlue', 'sgrot': 'SuperPoint + SuperGlue, rotation-aware matching of the paper', 'splg': 'SuperPoint + LightGlue', 'aliked': 'ALIKED + LightGlue',
         'loftr': 'LoFTR', 'dedode': 'DeDoDe-B + SO(2) steerer (C8)', 'roma': 'RoMa', 'mast3r': 'MASt3R matches'}
HAS_TTA = {'sift': False, 'sgplain': False, 'sgrot': False, 'splg': True, 'aliked': True, 'loftr': True, 'dedode': True, 'roma': True, 'mast3r': True}


def pick(G, ids, thr, tta):
    """One row per id (reindexed; ids the matcher was not run on -> NaN). plain = rotation 0; rot. TTA = the rotation with the most
    essential-matrix inliers among the solved ones. Sources with a single rotation per pair (the paper's rotation search) use that row."""
    g = G[G.thr == thr]
    if tta or g.groupby('id').size().max() == 1:
        g = g.assign(score=np.where(g.ok.fillna(0) == 1, g.inl.fillna(0), -1.0)); g = g.loc[g.groupby('id').score.idxmax()]
    else: g = g[g.k == 0]
    return g.drop_duplicates('id').set_index('id').reindex(ids)


def unit(g):
    return pd.DataFrame(dict(ok=(g.ok.fillna(0).values == 1).astype(float), have=g.n.notna().values, dh=g.dh.values, ta=g.ta.values, tb=g.tb.values, ux=g.vx.values, uy=g.vy.values, uz=g.vz.values, hgt=g.hgt.values,
                             hgt75ab=g.hgt75ab.values if 'hgt75ab' in g else np.full(len(g), np.nan), sc=g.scale.values, inl=g.inl.values, k=g.k.values))


def to_metric(m, mode, c, ndist=None):
    u = m[['ux', 'uy', 'uz']].values.astype(float)
    if mode == 'native': t = u * m.sc.values[:, None]
    elif mode == 'calib': t = u * (c * m.sc.values)[:, None]
    elif mode in ('height', 'height_p50a', 'height_pointmap'): t = u * (c / m.hgt.values)[:, None]
    elif mode == 'height_p75ab': t = u * (c / m.hgt75ab.values)[:, None]
    elif mode == 'learned': t = u * np.asarray(ndist, float)[:, None]
    elif mode == 'blend': t = u * np.exp(c[0] * np.log((c[2] / m.hgt75ab.values).clip(1e-3)) + (1 - c[0]) * np.log(np.asarray(ndist, float).clip(1e-3)) + c[1])[:, None]
    ok = (m.ok.values == 1) & np.isfinite(t).all(1) & np.isfinite(m.dh.values) & (np.linalg.norm(t, axis=1) < 500); t = np.where(ok[:, None], t, 0.0)
    return dict(x=t[:, 0], y=t[:, 1], z=t[:, 2], heading=np.where(ok, m.dh.values, 0.0), tilt=np.where(ok, m.tb.values - m.ta.values, 0.0)), ok


def fit_const(m, gd, mode):
    r = gd / m.sc.values if mode == 'calib' else gd * (m.hgt75ab.values if mode == 'height_p75ab' else m.hgt.values); ok = (m.ok.values == 1) & np.isfinite(r) & (r > 0); return (float(np.median(r[ok])) if ok.sum() >= 10 else np.nan), int(ok.sum())


def fit_blend(m, gd, ndist, H):
    """(w, b) of  log d = w log(H / h_p75ab) + (1 - w) log(d_MLP) + b  by least absolute log-error on the calibration pairs that have a pose."""
    with np.errstate(all='ignore'): x1 = np.log(H / m.hgt75ab.values); x2 = np.log(np.asarray(ndist, float)); y = np.log(gd)
    ok = (m.ok.values == 1) & np.isfinite(x1) & np.isfinite(x2); best = None
    for w in np.linspace(0, 1, 21):
        r = y[ok] - (w * x1[ok] + (1 - w) * x2[ok]); b = float(np.median(r)); e = float(np.mean(np.abs(r - b)))
        if best is None or e < best[0]: best = (e, float(w), b)
    return best[1], best[2], int(ok.sum())


def dir_err(m, T):
    u = m[['ux', 'uy', 'uz']].values.astype(float); g = np.c_[T.lx.values, T.ly.values, T.z.values]; g = g / np.linalg.norm(g, axis=1, keepdims=True); ok = (m.ok.values == 1) & np.isfinite(u).all(1)
    e = np.full(len(m), 180.0); e[ok] = np.rad2deg(np.arccos(np.clip((u[ok] * g[ok]).sum(1) / np.linalg.norm(u[ok], axis=1).clip(1e-9), -1, 1))); return e


def main():
    T = add_targets(subset(1000, split='test')); Tc = add_targets(subset(300, split='train')); ids, idc = T.manifest_index.values, Tc.manifest_index.values; n = len(T)
    T6 = C.load_test().set_index('manifest_index').loc[ids].reset_index(); assert np.abs(T6.x.values - T.x.values).max() < 1e-9
    N0 = pd.read_pickle(f'{C.SG}/stage0_mlp.pkl').set_index('manifest_index'); nd = N0.n_dist.reindex(ids).values; ndc = N0.n_dist.reindex(idc).values; gdc = Tc.dist.values      # training rows of the MLP are out-of-fold predictions
    rows = []; consts = {}; thr_sel = {}; timing = {}; sens = {}

    def score(name, variant, scale, pred, ok, sub=None, **extra):
        """sub: boolean mask of the pairs this method was run on (None = all)."""
        sub = np.ones(n, bool) if sub is None else sub; Ts = T6[sub].reset_index(drop=True); gt = C.gt_cam(Ts); p = {k: np.asarray(v)[sub] for k, v in pred.items()}
        e = C.errors(p, gt); s = C.summarize(e); boots = C.boot_indices(Ts.tour.values, 1000, 0)
        s['ci'] = {k: C.ci(fn, boots) for k, fn in (('d_mae', lambda b: e['d'][b].mean()), ('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('epe_med', lambda b: np.median(e['epe'][b])))}
        s.update(method=name, variant=variant, scale=scale, fail=float(100 * (1 - np.mean(np.asarray(ok)[sub]))), **extra); rows.append(s); return s

    # ------------------------------------------------------------ matcher + essential matrix
    for m, nm in NAMES.items():
        ft, fc = f'{OUT}/geom/{m}_test.pkl', f'{OUT}/geom/{m}_train.pkl'
        if not (os.path.exists(ft) and os.path.exists(fc)): print('skip (no geometry yet):', m); continue
        Gt, Gc = pd.read_pickle(ft), pd.read_pickle(fc); thrs = sorted(set(Gc.thr.unique()) & set(GRID))      # the threshold is chosen on the calibration pairs among GRID; the test set only needs the chosen one
        for tta in ((False, True) if HAS_TTA[m] else (False,)):
            # inlier threshold: the one with the largest share of calibration pairs whose translation direction is within 5 deg
            # (failed pairs count as misses; ties -> lower median direction error)
            cal = {t: unit(pick(Gc, idc, t, tta)) for t in thrs}; have_c = cal[thrs[0]].have.values
            de = {t: dir_err(cal[t], Tc)[have_c] for t in thrs}; frac = {t: float((de[t] < 5).mean()) for t in thrs}; med = {t: float(np.median(de[t])) for t in thrs}; thr = max(thrs, key=lambda t: (frac[t], -med[t]))
            thr_sel[f'{m}|{"tta" if tta else "plain"}'] = dict(chosen=thr, share_dir_err_below_5deg_calibration=frac, median_dir_err_calibration=med, calibration_pairs=int(have_c.sum()))
            assert thr in set(Gt.thr.unique()), f'{m}: threshold {thr} not computed on the test pairs'
            mt, mc = unit(pick(Gt, ids, thr, tta)), cal[thr]; sub = mt.have.values; variant = 'rot. TTA' if tta else 'plain'
            if m in ('sift',): variant = 'plain (rotation-invariant descriptor)'
            if m == 'sgrot': variant = 'rotation search of the paper'
            if m == 'dedode' and not tta: variant = 'plain (rotation-equivariant matching)'
            extra = dict(matcher=m, thr_px=thr, n_eval=int(sub.sum()), inl_med=float(np.nanmedian(mt.inl.values[sub])), rot_hist=np.bincount(mt.k.values[sub & np.isfinite(mt.k.values)].astype(int), minlength=4).tolist())
            Hc = np.nan
            for mode in ('native', 'calib', 'height_p75ab', 'height_p50a', 'learned', 'blend'):
                c = np.nan
                if mode in ('calib', 'height_p75ab', 'height_p50a'):
                    c, nfit = fit_const(mc[mc.have.values], gdc[mc.have.values], mode); consts[f'{nm} | {variant} | {mode}'] = dict(const=c, fitted_on=nfit)
                    if mode == 'height_p75ab': Hc = c
                if mode == 'blend':
                    if not np.isfinite(Hc): continue
                    w, b, nfit = fit_blend(mc[mc.have.values], gdc[mc.have.values], ndc[mc.have.values], Hc); c = (w, b, Hc); consts[f'{nm} | {variant} | blend'] = dict(const=w, offset=b, H=Hc, fitted_on=nfit)
                p, ok = to_metric(mt, mode, c, nd); score(f'{nm} + E', variant, mode, p, ok, None if sub.all() else sub, const=(c if np.isscalar(c) else list(c)), **extra)
        # sensitivity to the inlier threshold on the TEST pairs (flight-height scale; reported, never used for selection)
        for tta in ((False, True) if HAS_TTA[m] else (False,)):
            for t in sorted(set(Gt.thr.unique()) & set(Gc.thr.unique())):
                mt2, mc2 = unit(pick(Gt, ids, t, tta)), unit(pick(Gc, idc, t, tta)); sub2 = mt2.have.values; c2, _ = fit_const(mc2[mc2.have.values], gdc[mc2.have.values], 'height_p75ab')
                if not np.isfinite(c2): continue
                p2, ok2 = to_metric(mt2, 'height_p75ab', c2); e2 = C.errors({k: np.asarray(v)[sub2] for k, v in p2.items()}, C.gt_cam(T6[sub2].reset_index(drop=True)))
                sens.setdefault(f'{m}|{"tta" if tta else "plain"}', {})[str(t)] = dict(n=int(sub2.sum()), H=c2, fail=float(100 * (1 - ok2[sub2].mean())), sr_d_1m=float((e2['d'] <= 1).mean()), sr_epe_1m=float((e2['epe'] < 1).mean()), d_mae=float(e2['d'].mean()), epe_med=float(np.median(e2['epe'])), dir_med=float(np.median(e2['dir'])), theta_med=float(np.median(e2['theta'])), psi_med=float(np.median(e2['psi'])))
        tj = f'{OUT}/matches/{m}/timing_test.json'
        if os.path.exists(tj): timing[m] = json.load(open(tj))
        timing.setdefault(m, {})['solver_s_per_pose'] = float(Gt[Gt.thr == thr].sec.mean()) if 'sec' in Gt else None

    # ------------------------------------------------------------ pointmap + PnP (archived forward passes)
    for which, key in (('mast3r', 'mm_inl'), ('dust3r', 'conf2')):
        Rt, Rc = pd.read_pickle(f'{RB}/recon_{which}_test.pkl'), pd.read_pickle(f'{RB}/recon_{which}_train.pkl')
        for tta in (False, True):
            def sel(R, idx):
                r = R.loc[R.fillna({key: -1}).groupby('manifest_index')[key].idxmax()] if tta else R[R.k == 0]; r = r.set_index('manifest_index').reindex(idx)
                return pd.DataFrame(dict(ok=r.pnp_ok.fillna(0).values, have=np.ones(len(r), bool), dh=r.pnp_dh.values, ta=r.pnp_ta.values, tb=r.pnp_tb.values, ux=r.pnp_lx.values, uy=r.pnp_ly.values, uz=r.pnp_z.values, hgt=r.pnp_hgt.values, hgt75ab=np.full(len(r), np.nan), sc=np.ones(len(r)), inl=r.pnp_inl.values, k=r.k.values))
            mt, mc = sel(Rt, ids), sel(Rc, idc); nm = f'{"MASt3R" if which == "mast3r" else "DUSt3R"} pointmap + PnP'; variant = 'rot. TTA' if tta else 'plain'
            for mode in ((('native', 'calib') if which == 'mast3r' else ()) + ('height_pointmap', 'learned')):      # DUSt3R has no metric scale: no native / calib rows
                c = np.nan
                if mode == 'calib':   # one global constant on the model's metric translation: median of true distance / predicted distance
                    r = gdc / np.linalg.norm(mc[['ux', 'uy', 'uz']].values.astype(float), axis=1); okc = (mc.ok.values == 1) & np.isfinite(r) & (r > 0); c = float(np.median(r[okc])); consts[f'{nm} | {variant} | calib'] = dict(const=c, fitted_on=int(okc.sum()))
                if mode == 'height_pointmap':   # translation and height are in the same (model) units: baseline = |t| * H / hgt
                    r = gdc * mc.hgt.values / np.linalg.norm(mc[['ux', 'uy', 'uz']].values.astype(float), axis=1); okc = (mc.ok.values == 1) & np.isfinite(r) & (r > 0); c = float(np.median(r[okc])); consts[f'{nm} | {variant} | height_pointmap'] = dict(const=c, fitted_on=int(okc.sum()))
                    m2 = mt.copy(); nrm = np.linalg.norm(mt[['ux', 'uy', 'uz']].values.astype(float), axis=1); m2['hgt'] = mt.hgt.values / nrm; m2[['ux', 'uy', 'uz']] = mt[['ux', 'uy', 'uz']].values / nrm[:, None]; p, ok = to_metric(m2, 'height_pointmap', c)
                elif mode == 'learned':
                    m2 = mt.copy(); nrm = np.linalg.norm(mt[['ux', 'uy', 'uz']].values.astype(float), axis=1); m2[['ux', 'uy', 'uz']] = mt[['ux', 'uy', 'uz']].values / nrm[:, None]; p, ok = to_metric(m2, 'learned', c, nd)
                else: p, ok = to_metric(mt, mode, c)
                score(nm, variant, mode, p, ok, None, const=c, matcher=f'{which}_pnp', thr_px=None, n_eval=n)

    # ------------------------------------------------------------ learned references on the same pairs
    one = np.ones(n, bool)
    score('Regression: MLP on frozen MASt3R statistics (camera-frame targets)', '-', 'learned', dict(x=T6.n_lx.values, y=T6.n_ly.values, z=T6.n_z.values, tilt=T6.n_tilt.values, heading=T6.n_heading.values), one, None, matcher='mlp', n_eval=n)
    for nm, pre in (('DreamNav Stage I', 's1_'), ('DreamNav Stage II', 'p_')): score(nm, '-', 'learned', C.dreamnav(T6, pre, 'cam'), one, None, matcher='dreamnav', n_eval=n)
    z0 = np.zeros(n); score('Constant (zero motion)', '-', '-', dict(x=z0, y=z0, z=z0, tilt=z0, heading=z0), one, None, matcher='zero', n_eval=n)
    # DreamNav on the subset of any matcher that was run on fewer pairs
    for m in NAMES:
        ft = f'{OUT}/geom/{m}_test.pkl'
        if os.path.exists(ft):
            have = np.isin(ids, pd.read_pickle(ft).id.unique())
            if not have.all():
                for nm, pre in (('DreamNav Stage I', 's1_'), ('DreamNav Stage II', 'p_')): score(f'{nm} [same {int(have.sum())} pairs as {NAMES[m]}]', '-', 'learned', C.dreamnav(T6, pre, 'cam'), one, have, matcher='dreamnav_subset', n_eval=int(have.sum()))

    json.dump(dict(n_test=n, n_calibration=len(Tc), rows=rows, constants=consts, threshold_selection=thr_sel, threshold_sensitivity_test=sens, timing=timing), open(f'{OUT}/pairuav.json', 'w'), indent=1)
    f1 = lambda v: f'{100 * v:.1f}'
    L = ['| Method | variant | scale | n | fail % | d MAE [95% CI] | d med. | SR_d@1m [CI] | EPE mean | EPE med. | SR_EPE@1m [CI] | SR_EPE@2m | e_theta MAE / med. | e_psi MAE / med. | dir. med. | AUC@5/10/20 | 5-DoF (1m,1deg) | thr (px) |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in rows:
        c = s['ci']; L.append(f"| {s['method']} | {s['variant']} | {s['scale']} | {s['n']} | {s['fail']:.1f} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {s['d_med']:.2f} | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} | "
                              f"{f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | {s['theta_mae']:.2f} / {s['theta_med']:.3f} | {s['psi_mae']:.2f} / {s['psi_med']:.3f} | {s['dir_med']:.1f} | {f1(s['auc_5'])}/{f1(s['auc_10'])}/{f1(s['auc_20'])} | {f1(s['sr_5dof_1m_1deg'])} | {s.get('thr_px') or '-'} |")
    open(f'{OUT}/pairuav.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print(json.dumps(thr_sel, indent=1)); print(json.dumps(consts, indent=1))


if __name__ == '__main__':
    main()
