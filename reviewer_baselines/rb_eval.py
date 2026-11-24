#!/usr/bin/env python3
"""Reviewer baselines, part 2b: assemble every direct baseline on the same sampled test pairs and score them like Table 2.

Baselines (none uses a learned fusion; nothing is trained on PairUAV):
  DUSt3R / MASt3R pointmap + PnP                     (rb_recon.py)         plain and with 4-rotation test-time augmentation
  MASt3R matches + essential matrix + MASt3R depth   (rb_recon.py)
  SuperGlue + essential matrix + MASt3R depth        (rb_sg.py, plain)     and with the paper's rotation-aware matching (cached pass-1 matches)
Metric scale variants:
  native   the model's own metric output, untouched (MASt3R metric pointmap / depth)
  calib    native x ONE global constant, the median of GT/predicted distance on 300 training pairs
  height   baseline = H / (camera height above the triangulated scene in baseline units), H fitted on the same 300 training pairs
           (flight-height prior of the dataset; the only option for scale-free DUSt3R)
World-frame x/y need the absolute heading of the source camera, which no two-view method observes: we report them with the
TRUE heading (favourable to the baselines; SR is computed this way) and with the paper's heading network.
"""
import sys, os, json
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
from fusion_lib import add_targets, wrap
from rb_recon import subset, geom_from_matches, SG, OUT
S2 = f'{SG}/stage2'; NT, NC = 1000, 300


def sg_work(a):
    idx, kind = a
    try:
        if kind == 'plain': d = np.load(f'{OUT}/matches_sg_plain/{idx}.npz'); src = None; k = 0
        else: d = np.load(f'{SG}/matches_rot/{idx}.npz'); src = d['src']; k = int(d['rot'])
        dep = np.load(f'{OUT}/depth_mast3r/{idx}.npy')[k].astype(np.float32)
        g = geom_from_matches(d['p0'].astype(np.float64), d['p1'].astype(np.float64), d['conf'].astype(np.float64), dep, multi_src=src)
    except Exception as e:
        g = dict(ok=0.0, inl=0.0, n=0.0, err=str(e))
    g.update(manifest_index=idx, kind=kind); return g


def unit_methods(split):
    """-> {name: DataFrame indexed like the subset with ok, dh, ta, tb, (ux,uy,uz) translation in the model's native units, hgt, inl}"""
    T = subset(NT if split == 'test' else NC, split=split); ids = T.manifest_index.values; M = {}
    for which in ('dust3r', 'mast3r'):
        f = f'{OUT}/recon_{which}_{split}.pkl'
        if not os.path.exists(f): continue
        R = pd.read_pickle(f)
        for tta in (False, True):
            if tta:
                key = 'mm_inl' if which == 'mast3r' else 'conf2'; pick = R.loc[R.fillna({key: -1}).groupby('manifest_index')[key].idxmax()]
            else: pick = R[R.k == 0]
            pick = pick.set_index('manifest_index').reindex(ids); tag = ' + rot. TTA' if tta else ''
            M[f'{"DUSt3R" if which == "dust3r" else "MASt3R"} pointmap + PnP{tag}'] = pd.DataFrame(dict(ok=pick.pnp_ok.fillna(0).values, dh=pick.pnp_dh.values, ta=pick.pnp_ta.values, tb=pick.pnp_tb.values, ux=pick.pnp_lx.values, uy=pick.pnp_ly.values, uz=pick.pnp_z.values, hgt=pick.pnp_hgt.values, inl=pick.pnp_inl.values))
            if which == 'mast3r':
                s = pick.mm_scale.values
                M[f'MASt3R matches + E + MASt3R depth{tag}'] = pd.DataFrame(dict(ok=(pick.mm_ok.fillna(0).values == 1).astype(float), dh=pick.mm_dh.values, ta=pick.mm_ta.values, tb=pick.mm_tb.values, ux=pick.mm_vx.values * s, uy=pick.mm_vy.values * s, uz=pick.mm_vz.values * s, hgt=pick.mm_hgt.values * s, inl=pick.mm_inl.values))
    with Pool(14) as pool: G = pd.DataFrame(pool.map(sg_work, [(int(i), k) for k in ('plain', 'rot') for i in ids], chunksize=8))
    for kind, name in (('plain', 'SuperGlue + E + MASt3R depth'), ('rot', 'Rot.-aware SuperGlue + E + MASt3R depth')):
        g = G[G.kind == kind].set_index('manifest_index').reindex(ids); s = g.scale.values
        M[name] = pd.DataFrame(dict(ok=(g.ok.fillna(0).values == 1).astype(float), dh=g.dh.values, ta=g.ta.values, tb=g.tb.values, ux=g.vx.values * s, uy=g.vy.values * s, uz=g.vz.values * s, hgt=g.hgt.values * s, inl=g.inl.values, unit_x=g.vx.values, unit_y=g.vy.values, unit_z=g.vz.values, unit_hgt=g.hgt.values))
    return T, M


def to_metric(m, mode, c):
    """native / calib: translation x c; height: translation x c / hgt (c = flight height). Failed pairs predict zero motion."""
    u = m[['ux', 'uy', 'uz']].values.astype(float); h = m.hgt.values.astype(float)
    if mode == 'height':
        if 'unit_x' in m: u = m[['unit_x', 'unit_y', 'unit_z']].values.astype(float); h = m.unit_hgt.values.astype(float)   # scale-free: unit direction and height in baseline units
        t = u * (c / h)[:, None]
    else: t = u * c
    ok = (m.ok.values == 1) & np.isfinite(t).all(1) & np.isfinite(m.dh.values) & (np.linalg.norm(t, axis=1) < 500)
    t = np.where(ok[:, None], t, 0.0); return dict(ok=ok, lx=t[:, 0], ly=t[:, 1], z=t[:, 2], heading=np.where(ok, m.dh.values, 0.0), tilt=np.where(ok, m.tb.values - m.ta.values, 0.0))


def fit_const(m, T, mode):
    gd = np.sqrt(T.x**2 + T.y**2 + T.z**2).values; u = m[['ux', 'uy', 'uz']].values.astype(float); h = m.hgt.values.astype(float)
    if mode == 'height' and 'unit_x' in m: u = m[['unit_x', 'unit_y', 'unit_z']].values.astype(float); h = m.unit_hgt.values.astype(float)
    n = np.linalg.norm(u, axis=1); r = gd / n if mode == 'calib' else gd * h / n; ok = (m.ok.values == 1) & np.isfinite(r) & (r > 0)
    return float(np.median(r[ok])), int(ok.sum())


def main():
    Tc, Mc = unit_methods('train'); T, M = unit_methods('test'); T = add_targets(T); n = len(T); print('test pairs', n, '| calibration pairs', len(Tc))
    # ---- absolute heading of the source camera from the paper's network (same recipe as analysis/extended_metrics.py)
    A = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image'); B = pd.read_pickle(f'{S2}/pred_bev_s.pkl')
    for s in 'ab':
        T[f'v2h_{s}'] = A.ah_pred.reindex(T[f'image_{s}']).values; T[f'v2k_{s}'] = A.ah_kappa.reindex(T[f'image_{s}']).values
        b = B[B.side == s].set_index('manifest_index'); T[f'bvh_{s}'] = b.ah_est.reindex(T.manifest_index).values; T[f'bvk_{s}'] = b.ak_est.reindex(T.manifest_index).values
    V2 = [('v2h_a', 'v2k_a', 'v2h_b', 'v2k_b')]; BV = [('bvh_a', 'bvk_a', 'bvh_b', 'bvk_b')]
    def combine(hk, dh):
        s = 0; c = 0
        for ha, ka, hb, kb in hk:
            ta = np.deg2rad(T[ha].values); tb = np.deg2rad(T[hb].values - dh); s = s + T[ka].values * np.sin(ta) + T[kb].values * np.sin(tb); c = c + T[ka].values * np.cos(ta) + T[kb].values * np.cos(tb)
        return np.rad2deg(np.arctan2(s, c))
    def world(h, lx, ly): h = np.deg2rad(h); return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly
    gt = {k: T[k].values for k in ('x', 'y', 'z', 'tilt', 'heading', 'lx', 'ly')}; gd = T.dist.values
    def auc(e, thr):
        e = np.sort(e); r = (np.arange(len(e)) + 1) / len(e); i = np.searchsorted(e, thr); x = np.r_[0.0, e[:i], thr]; y = np.r_[0.0, r[:i], r[i - 1] if i else 0.0]; return float(np.trapezoid(y, x) / thr)
    def score(p, hk, al=0.0, dist=None):
        """p: lx, ly, z, heading, tilt (+ok). World x/y with the true source heading and with the heading network hk."""
        o = {}; lx, ly, z = p['lx'], p['ly'], p['z']
        if dist is not None and al: nn = np.sqrt(lx**2 + ly**2 + z**2).clip(1e-6); sc = (dist / nn) ** al; lx, ly, z = lx * sc, ly * sc, z * sc
        for tag, h in (('', T.heading_a.values), ('_net', combine(hk, p['heading']))):
            x, y = world(h, lx, ly); o['x' + tag] = np.abs(x - gt['x']).mean(); o['y' + tag] = np.abs(y - gt['y']).mean(); e3 = np.sqrt((x - gt['x'])**2 + (y - gt['y'])**2 + (z - gt['z'])**2); o['sr1' + tag] = 100 * (e3 < 1).mean(); o['sr2' + tag] = 100 * (e3 < 2).mean(); o['pos_med' + tag] = np.median(e3)
        eh = np.abs(wrap(p['heading'] - gt['heading'])); et = np.abs(p['tilt'] - gt['tilt']); pdn = np.sqrt(lx**2 + ly**2 + z**2)
        cos = (lx * gt['lx'] + ly * gt['ly'] + z * gt['z']) / (pdn.clip(1e-9) * gd.clip(1e-9))
        o.update(sr_d=100 * (np.abs(pdn - gd) <= 1).mean(), sr_d2=100 * (np.abs(pdn - gd) <= 2).mean(), z=np.abs(z - gt['z']).mean(), d=np.abs(pdn - gd).mean(), d_med=np.median(np.abs(pdn - gd)), heading=eh.mean(), heading_med=np.median(eh), tilt=et.mean(), tilt_med=np.median(et), tdir_med=np.median(np.rad2deg(np.arccos(np.clip(cos, -1, 1)))),
                 auc5=100 * auc(np.maximum(eh, et), 5), auc10=100 * auc(np.maximum(eh, et), 10), auc20=100 * auc(np.maximum(eh, et), 20), fail=100 * (1 - np.mean(p['ok'])) if 'ok' in p else 0.0)
        return {k: float(v) for k, v in o.items()}
    rows = []; consts = {}
    for name, m in M.items():
        modes = ['height'] if name.startswith('DUSt3R') else ['native', 'calib', 'height']
        for mode in modes:
            c = 1.0 if mode == 'native' else fit_const(Mc[name], Tc, mode)[0]; consts[f'{name} | {mode}'] = c
            rows.append(dict(method=name, scale=mode, const=c, **score(to_metric(m, mode, c), V2)))
    # ---- learned regression baselines: MLP on frozen MASt3R features (archived, trained on the 47,687 training pairs) and the end-to-end Siamese correlation regressor (rb_rpr.py)
    N0 = pd.read_pickle(f'{SG}/stage0_mlp.pkl').set_index('manifest_index').reindex(T.manifest_index)
    rows.append(dict(method='Regression: MLP on frozen MASt3R features', scale='learned', const=np.nan, **score(dict(lx=N0.n_lx.values, ly=N0.n_ly.values, z=N0.n_z.values, heading=N0.n_heading.values, tilt=N0.n_tilt.values, ok=np.ones(n, bool)), V2)))
    if os.path.exists(f'{OUT}/rpr_test.pkl'):
        Q = pd.read_pickle(f'{OUT}/rpr_test.pkl').set_index('manifest_index').reindex(T.manifest_index)
        rows.append(dict(method=f'Regression: Siamese + correlation, end-to-end ({Q.epochs.iloc[0]:.0f} epochs, {Q.minutes.iloc[0]:.0f} min)', scale='learned', const=np.nan, **score(dict(lx=Q.lx.values, ly=Q.ly.values, z=Q.z.values, heading=Q.dh.values, tilt=(Q.tb - Q.ta).values, ok=np.ones(n, bool)), V2)))
    # ---- the paper's two stages on the same pairs (archived predictions)
    F = pd.read_pickle(f'{S2}/s2_pred.pkl').set_index('manifest_index').reindex(T.manifest_index)
    for name, pre, hk, al in (('DreamNav Stage I', 's1_', V2, 0.75), ('DreamNav Stage II', 'p_', BV + V2, 0.5)):
        p = dict(lx=F[pre + 'lx'].values, ly=F[pre + 'ly'].values, z=F[pre + 'z'].values, heading=F[pre + 'heading'].values, tilt=F[pre + 'tilt'].values, ok=np.ones(n, bool))
        rows.append(dict(method=name, scale='learned', const=np.nan, **score(p, hk, al, F[pre + 'dist'].values)))
    rows.append(dict(method='Constant (zero motion)', scale='-', const=np.nan, **score(dict(lx=np.zeros(n), ly=np.zeros(n), z=np.zeros(n), heading=np.zeros(n), tilt=np.zeros(n), ok=np.ones(n, bool)), V2)))
    R = pd.DataFrame(rows); R.to_pickle(f'{OUT}/offline_metrics.pkl'); json.dump(dict(rows=rows, consts=consts, n_test=n, n_calib=len(Tc)), open(f'{OUT}/offline_metrics.json', 'w'), indent=1)
    L = [f'# Direct two-view baselines on PairUAV ({n} random test pairs of the Table-2 test set; scale constants from {len(Tc)} training pairs)\n',
         'SR@1m / SR@2m follow Table 2: |predicted distance - true distance| <= 1 m (2 m). "3-D<1m" is the stricter 3-D position error < 1 m. x/y and 3-D<1m use the TRUE absolute heading of the source camera for every row (favourable to the baselines); "net" columns use the paper\'s heading network instead. Failed pairs predict zero motion.\n',
         '| Method | scale | x | y | z | d | d med. | Δθ | Δθ med. | Δψ | Δψ med. | dir. med. | SR@1m | SR@2m | AUC@5 | AUC@10 | AUC@20 | fail % | 3-D<1m | 3-D pos. med. | x net | y net | 3-D<1m net |', '|---|---|' + '---:|' * 21]
    for r in rows: L.append(f"| {r['method']} | {r['scale']} | {r['x']:.2f} | {r['y']:.2f} | {r['z']:.2f} | {r['d']:.2f} | {r['d_med']:.2f} | {r['heading']:.2f} | {r['heading_med']:.2f} | {r['tilt']:.2f} | {r['tilt_med']:.2f} | {r['tdir_med']:.1f} | {r['sr_d']:.1f} | {r['sr_d2']:.1f} | {r['auc5']:.1f} | {r['auc10']:.1f} | {r['auc20']:.1f} | {r['fail']:.1f} | {r['sr1']:.1f} | {r['pos_med']:.2f} | {r['x_net']:.2f} | {r['y_net']:.2f} | {r['sr1_net']:.1f} |")
    L.append('\nScale constants fitted on the training pairs: ' + '; '.join(f'{k}: {v:.1f}' for k, v in consts.items() if not k.endswith('native')))
    open(f'{OUT}/offline_metrics.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
