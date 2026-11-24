#!/usr/bin/env python3
"""Track B, part 4: geometric two-view baselines on the real drone photographs (Mezieres, DJI Phantom 4 RTK), zero-shot.

Pairs: the 300 pairs of Table 4 ("within the training range": dist_equiv <= 38) and, if matches exist, the 300 pairs beyond it.
Nothing is fitted on these photographs. The essential-matrix inlier threshold of each matcher is the one selected on the
PairUAV calibration pairs (outputs/supp/trackB/pairuav.json). Metric scale:
  height60_p75ab  baseline = 60 m / h, h = camera height above the triangulated scene in baseline units (mean over both cameras of the
            75th percentile of the inlier heights, the same statistic as on PairUAV); 60 m is the flight height reported by the
            aircraft (XMP RelativeAltitude), i.e. the same side information the DreamNav rows use (they rescale by 60 / 220.18)
  height60_p50a   the same with h = median inlier height below the source camera (kept in the JSON for reference)
  height60_pointmap (PnP rows) 60 m / (median height of the source pointmap)
  native    MASt3R's own metric depth / pointmap, untouched
Columns follow Table 4 (median position error in the frame of the first camera; SR@1m / SR@2m on the range error d) plus
the endpoint success rates, angle errors and confidence intervals (bootstrap over source photographs).
Output: outputs/supp/trackB/real.json, real.md
"""
import sys, os, json
import numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/supp', f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
import common as C
from eval_pairuav import NAMES, HAS_TTA, pick, unit
OUT = f'{ROOT}/outputs/supp/trackB'; K = 60.0 / 220.18; FLIGHT_H = 60.0


def main():
    D = pd.read_pickle('/root/realdata/mezieres/pred.pkl'); PU = json.load(open(f'{OUT}/pairuav.json')) if os.path.exists(f'{OUT}/pairuav.json') else {'threshold_selection': {}}
    res = {}; L = []
    for which, mask, title in (('real_in', D.dist_equiv <= 38, 'Within the training range'), ('real_out', D.dist_equiv > 38, 'Beyond the training range')):
        d = D[mask]; ids = d.index.values; n = len(d); has_depth = np.array([os.path.exists(f'{OUT}/mast3r/depth_{which}/{i}.npy') for i in ids]); gt = dict(x=d.lx.values, y=d.ly.values, z=d.z.values, tilt=d.tilt.values, heading=d.heading.values); rows = []
        e_ref = C.errors(dict(x=d.s2_lx.values * K, y=d.s2_ly.values * K, z=d.s2_z.values * K, tilt=d.s2_tilt.values, heading=d.s2_heading.values), gt)      # DreamNav Stage II, for the paired differences
        def score(name, variant, scale, pred, ok, sub=None, **extra):
            sub = np.ones(n, bool) if sub is None else sub; g = {k: v[sub] for k, v in gt.items()}; p = {k: np.asarray(v, float)[sub] for k, v in pred.items()}
            e = C.errors(p, g); s = C.summarize(e); boots = C.boot_indices(d.a.values[sub], 1000, 0); r = {k: v[sub] for k, v in e_ref.items()}
            s['paired_minus_stage2'] = {k: dict(delta=float(f(np.arange(int(sub.sum())))), ci=C.ci(f, boots)) for k, f in (('sr_d_1m_points', lambda b: 100 * ((e['d'][b] <= 1).mean() - (r['d'][b] <= 1).mean())), ('sr_d_2m_points', lambda b: 100 * ((e['d'][b] <= 2).mean() - (r['d'][b] <= 2).mean())),
                                                                                                                                      ('sr_epe_1m_points', lambda b: 100 * ((e['epe'][b] < 1).mean() - (r['epe'][b] < 1).mean())), ('epe_med', lambda b: np.median(e['epe'][b]) - np.median(r['epe'][b])))}
            s['ci'] = {k: C.ci(fn, boots) for k, fn in (('epe_med', lambda b: np.median(e['epe'][b])), ('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_d_2m', lambda b: (e['d'][b] <= 2).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('sr_epe_2m', lambda b: (e['epe'][b] < 2).mean()))}
            s.update(method=name, variant=variant, scale=scale, fail=float(100 * (1 - np.mean(np.asarray(ok)[sub]))), **extra); rows.append(s)
        def metric(m, mode):
            u = m[['ux', 'uy', 'uz']].values.astype(float); t = u * (FLIGHT_H / m.hgt75ab.values)[:, None] if mode == 'height60_p75ab' else u * (FLIGHT_H / m.hgt.values)[:, None] if mode == 'height60_p50a' else u * m.sc.values[:, None]
            ok = (m.ok.values == 1) & np.isfinite(t).all(1) & np.isfinite(m.dh.values) & (np.linalg.norm(t, axis=1) < 500); t = np.where(ok[:, None], t, 0.0)
            return dict(x=t[:, 0], y=t[:, 1], z=t[:, 2], heading=np.where(ok, m.dh.values, 0.0), tilt=np.where(ok, m.tb.values - m.ta.values, 0.0)), ok
        any_row = False
        for m, nm in NAMES.items():
            f = f'{OUT}/geom/{m}_{which}.pkl'
            if not os.path.exists(f): continue
            G = pd.read_pickle(f); any_row = True
            for tta in ((False, True) if HAS_TTA[m] else (False,)):
                sel = PU['threshold_selection'].get(f'{m}|{"tta" if tta else "plain"}', {}).get('chosen', 3.0); thr = sel if sel in set(G.thr.unique()) else sorted(G.thr.unique())[0]
                mt = unit(pick(G, ids, thr, tta)); sub = mt.have.values; variant = 'rot. TTA' if tta else 'plain'
                if m == 'sift': variant = 'plain (rotation-invariant descriptor)'
                if m == 'sgrot': variant = 'rotation search of the paper'
                if m == 'dedode' and not tta: variant = 'plain (rotation-equivariant matching)'
                for mode in ('height60_p75ab', 'height60_p50a', 'native'):
                    sub_m = sub & has_depth if mode == 'native' else sub      # the native scale needs the MASt3R depth map, which exists for the pairs MASt3R was run on
                    if not sub_m.any(): continue
                    p, ok = metric(mt, mode); score(f'{nm} + E', variant, mode, p, ok, None if sub_m.all() else sub_m, matcher=m, thr_px=thr, n_eval=int(sub_m.sum()), inl_med=float(np.nanmedian(mt.inl.values[sub_m])))
        fr = f'{OUT}/mast3r/recon_{which}.pkl'
        if os.path.exists(fr):
            R = pd.read_pickle(fr); G = pd.read_pickle(f'{OUT}/geom/mast3r_{which}.pkl') if os.path.exists(f'{OUT}/geom/mast3r_{which}.pkl') else None
            for tta in (False, True):
                if tta:   # rotation chosen like in the archived PairUAV runs: most reciprocal-match inliers of the essential matrix (threshold 3 px)
                    if G is None: continue
                    g3 = G[G.thr == (3.0 if 3.0 in set(G.thr.unique()) else G.thr.min())]; g3 = g3.assign(score=np.where(g3.ok.fillna(0) == 1, g3.inl.fillna(0), -1.0)); kb = g3.loc[g3.groupby('id').score.idxmax()].set_index('id').k; r = R.set_index(['id', 'k']).reindex(list(zip(ids, kb.reindex(ids).fillna(0).astype(int)))).reset_index()
                else: r = R[R.k == 0].set_index('id').reindex(ids).reset_index()
                t = r[['pnp_lx', 'pnp_ly', 'pnp_z']].values.astype(float); ok0 = (r.pnp_ok.fillna(0).values == 1) & np.isfinite(t).all(1); sub_p = np.isin(ids, R.id.unique())      # pairs MASt3R was run on
                for mode in ('native', 'height60_pointmap'):
                    tt = t if mode == 'native' else t * (FLIGHT_H / r.pnp_hgt.values)[:, None]; ok = ok0 & np.isfinite(tt).all(1) & (np.linalg.norm(tt, axis=1) < 500); tt = np.where(ok[:, None], tt, 0.0)
                    score('MASt3R pointmap + PnP', 'rot. TTA' if tta else 'plain', mode, dict(x=tt[:, 0], y=tt[:, 1], z=tt[:, 2], heading=np.where(ok, r.pnp_dh.values, 0.0), tilt=np.where(ok, r.pnp_tb.values - r.pnp_ta.values, 0.0)), ok, None if sub_p.all() else sub_p, matcher='mast3r_pnp', n_eval=int(sub_p.sum()))
        if not any_row: continue
        one = np.ones(n, bool); z0 = np.zeros(n)
        score('Constant (zero motion)', '-', '-', dict(x=z0, y=z0, z=z0, tilt=z0, heading=z0), one, None, matcher='zero', n_eval=n)
        score('Regression: MLP on frozen MASt3R statistics', '-', 'rescaled by 60/220.18', dict(x=d.n_lx.values * K, y=d.n_ly.values * K, z=d.n_z.values * K, tilt=d.n_tilt.values, heading=d.n_heading.values), one, None, matcher='mlp', n_eval=n)
        for nm, pre in (('DreamNav Stage I', 's1'), ('DreamNav Stage II', 's2')):
            score(nm, '-', 'rescaled by 60/220.18', dict(x=d[f'{pre}_lx'].values * K, y=d[f'{pre}_ly'].values * K, z=d[f'{pre}_z'].values * K, tilt=d[f'{pre}_tilt'].values, heading=d[f'{pre}_heading'].values), one, None, matcher='dreamnav', n_eval=n)
        score('DreamNav Stage II', '-', 'no flight height', dict(x=d.s2_lx.values, y=d.s2_ly.values, z=d.s2_z.values, tilt=d.s2_tilt.values, heading=d.s2_heading.values), one, None, matcher='dreamnav_noheight', n_eval=n)
        for m in NAMES:   # DreamNav on the subset of a matcher that was run on fewer pairs
            f = f'{OUT}/geom/{m}_{which}.pkl'
            if os.path.exists(f):
                have = np.isin(ids, pd.read_pickle(f).id.unique())
                if not have.all():
                    for nm, pre in (('DreamNav Stage I', 's1'), ('DreamNav Stage II', 's2')):
                        score(f'{nm} [same {int(have.sum())} pairs as {NAMES[m]}]', '-', 'rescaled by 60/220.18', dict(x=d[f'{pre}_lx'].values * K, y=d[f'{pre}_ly'].values * K, z=d[f'{pre}_z'].values * K, tilt=d[f'{pre}_tilt'].values, heading=d[f'{pre}_heading'].values), one, have, matcher='dreamnav_subset', n_eval=int(have.sum()))
        res[which] = dict(n=n, mean_true_distance=float(d.dist.mean()), rows=rows); f1 = lambda v: f'{100 * v:.1f}'
        L += [f'\n### {title} ({n} pairs, mean true distance {d.dist.mean():.1f} m)\n', '| Method | variant | scale | n | fail % | Pos. med. (m) [95% CI] | EPE mean | SR@1m (range) [CI] | SR@2m (range) | SR_EPE@1m [CI] | SR_EPE@2m | d MAE | e_theta MAE / med. | e_psi MAE / med. | dir. med. | AUC@5/10/20 | thr |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for s in rows:
            c = s['ci']; L.append(f"| {s['method']} | {s['variant']} | {s['scale']} | {s['n']} | {s['fail']:.1f} | {s['epe_med']:.2f} [{c['epe_med'][0]:.2f}, {c['epe_med'][1]:.2f}] | {s['epe_mae']:.2f} | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {f1(s['sr_d_2m'])} | "
                                  f"{f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | {s['d_mae']:.2f} | {s['theta_mae']:.2f} / {s['theta_med']:.3f} | {s['psi_mae']:.2f} / {s['psi_med']:.3f} | {s['dir_med']:.1f} | {f1(s['auc_5'])}/{f1(s['auc_10'])}/{f1(s['auc_20'])} | {s.get('thr_px') or '-'} |")
    json.dump(dict(flight_height_m=FLIGHT_H, dreamnav_rescale=K, sets=res), open(f'{OUT}/real.json', 'w'), indent=1); open(f'{OUT}/real.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
