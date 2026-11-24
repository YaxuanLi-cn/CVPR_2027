#!/usr/bin/env python3
"""How planar are the scenes, and does the refinement still help when they are not?

For every test pair, the matches of the imagination pass (archived) are triangulated with the GROUND-TRUTH relative pose
(used here only to describe the scene, never by the method). In the source camera frame the depth below the camera along
gravity is  b_i = X_i . (0, sin psi_S, cos psi_S).  The ground is the lowest dominant surface: g = 85th percentile of b.
  height above ground of a matched point   r_i = g - b_i
  structure height                         S   = g - P5(b)          (top 5 % of the matched points)
  relief ratio                             S / g                    (building height over flight height)
  off-plane fraction                       mean(r_i > 3 m), mean(r_i > 10 m)
  parallax of the structure in the target  S / g * baseline * f / g  (pixels; first-order)
Errors of the coarse and refined estimates (camera frame, supp/common.py) are then binned by these quantities.
Output: outputs/supp/trackA/planarity.json, planarity_pairs.pkl
"""
import sys, os, json, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from common import *
from geom import gt_relative
from pose_from_matches import K0, _sampson
O = f'{OUT}/trackA'; MI = f'{S2}/matches_imag'; Ki = np.linalg.inv(K0); F0 = K0[0, 0]


def work(r):
    try:
        d = np.load(f'{MI}/{r["manifest_index"]}.npz'); keep = d['conf'].astype(np.float64) >= 0.2; p0, p1 = d['p0'][keep].astype(np.float64), d['p1'][keep].astype(np.float64)
        R, t = gt_relative(r['heading_a'], r['tilt_a'], r['heading_b'], r['tilt_b'], [r['x'], r['y'], r['z']])
        x0 = np.c_[p0, np.ones(len(p0))] @ Ki.T; x1 = np.c_[p1, np.ones(len(p1))] @ Ki.T; res = np.abs(_sampson(R, t / np.linalg.norm(t), x0, x1)) * F0; inl = res < 2.0
        if inl.sum() < 50: return dict(manifest_index=r['manifest_index'], n_tri=int(inl.sum()))
        X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], x0[inl, :2].T.copy(), x1[inl, :2].T.copy()); X = (X[:3] / X[3]).T; Xb = X @ R.T + t; ok = (X[:, 2] > 0) & (Xb[:, 2] > 0) & np.isfinite(X).all(1); X = X[ok]
        ta = np.deg2rad(r['tilt_a']); b = X @ np.array([0.0, np.sin(ta), np.cos(ta)]); g = np.percentile(b, 85); rr = g - b; S = g - np.percentile(b, 5)
        # least-squares plane through the points near the ground (within 2 m of g): slope of the terrain
        near = np.abs(rr) < 2.0; slope = np.nan
        if near.sum() >= 30:
            c = X[near].mean(0); _, _, vt = np.linalg.svd(X[near] - c, full_matrices=False); nrm = vt[2]; slope = float(np.rad2deg(np.arccos(abs(nrm @ np.array([0.0, np.sin(ta), np.cos(ta)])))))
        return dict(manifest_index=r['manifest_index'], n_tri=int(len(X)), ground=float(g), struct=float(S), relief=float(S / g), off3=float((rr > 3).mean()), off10=float((rr > 10).mean()), off20=float((rr > 20).mean()),
                    below2=float((rr < -2).mean()), near2=float((np.abs(rr) < 2).mean()), slope=slope, gt_resid_med=float(np.median(res[inl])), gt_inl_frac=float(inl.mean()))
    except Exception as e:
        return dict(manifest_index=r['manifest_index'], n_tri=0, err=str(e))


if __name__ == '__main__':
    cv2.setNumThreads(1); T = load_test(); rows = T[['manifest_index', 'heading_a', 'tilt_a', 'heading_b', 'tilt_b', 'x', 'y', 'z']].to_dict('records')
    with Pool(int(os.environ.get('PROCS', '4'))) as p: Rr = pd.DataFrame(p.map(work, rows, chunksize=32))
    D = T.merge(Rr, on='manifest_index', how='left'); I = pd.read_pickle(f'{S2}/geom_imag.pkl')[['manifest_index', 'im_disp_med', 'im_inl', 'im_ntiles']]; D = D.drop(columns=[c for c in ('im_disp_med', 'im_inl', 'im_ntiles') if c in D.columns]).merge(I, on='manifest_index', how='left')
    # terrain-based height of the source camera (independent of any matching): KML altitude - terrain altitude of the building anchor
    terr = {}
    for t in D.tour.unique():
        with open(f'{DATA}/pairs/{t}.jsonl') as f: terr[t] = json.loads(f.readline())['anchor']['terrain_altitude_msl']
    D['h_terrain'] = D.altitude_a - D.tour.map(terr)
    e1 = errors(dreamnav(D, 's1_', 'cam'), gt_cam(D)); e2 = errors(dreamnav(D, 'p_', 'cam'), gt_cam(D))
    for k in ('d', 'epe', 'theta', 'psi', 'dir'): D[f'e1_{k}'] = e1[k]; D[f'e2_{k}'] = e2[k]
    D['parallax_px'] = D.relief * D.dist * F0 / D.ground; D['tilt_max'] = np.maximum(D.tilt_a, D.tilt_b); D['b_over_h'] = D.dist / D.ground; D['dz_over_h'] = np.abs(D.z) / D.ground
    D.to_pickle(f'{O}/planarity_pairs.pkl'); V = D[D.n_tri >= 200].copy(); print('pairs', len(D), '| with >= 200 triangulated matches', len(V))
    res = dict(n=int(len(D)), n_valid=int(len(V)))
    q = lambda s: {k: float(v) for k, v in zip(('p5', 'p25', 'p50', 'p75', 'p95', 'max'), np.r_[np.percentile(s.dropna(), [5, 25, 50, 75, 95]), s.max()])}
    res['dist'] = {k: q(V[k]) for k in ('ground', 'h_terrain', 'struct', 'relief', 'off3', 'off10', 'off20', 'near2', 'slope', 'parallax_px', 'gt_resid_med', 'tilt_max', 'b_over_h')}
    res['ground_vs_terrain'] = dict(corr=float(np.corrcoef(V.ground, V.h_terrain)[0, 1]), median_abs_diff=float(np.median(np.abs(V.ground - V.h_terrain))), median_diff=float(np.median(V.ground - V.h_terrain)))
    def table(col, edges, labels):
        out = []
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            m = (V[col] >= lo) & (V[col] < hi); s = V[m]
            if len(s) < 20: continue
            out.append(dict(bin=lab, n=int(len(s)), relief_med=float(s.relief.median()), struct_med=float(s.struct.median()), disp_med=float(s.im_disp_med.median()), inl_med=float(s.im_inl.median()),
                            **{f's{i}_{k}': float(fn(s[f'e{i}_{c}'])) for i in (1, 2) for k, c, fn in (('d_mae', 'd', np.mean), ('epe_mae', 'epe', np.mean), ('epe_med', 'epe', np.median), ('sr_d', 'd', lambda v: (v <= 1).mean()), ('sr_epe', 'epe', lambda v: (v < 1).mean()), ('theta_mae', 'theta', np.mean), ('psi_mae', 'psi', np.mean), ('dir_med', 'dir', np.median))}))
        return out
    qs = lambda col, n: np.r_[-np.inf, np.quantile(V[col].dropna(), np.linspace(0, 1, n + 1))[1:-1], np.inf]
    res['by_relief'] = table('relief', [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 10], ['<2%', '2-5%', '5-10%', '10-15%', '15-20%', '20-30%', '>=30%'])
    res['by_struct_m'] = table('struct', [0, 5, 10, 20, 30, 45, 1e9], ['<5', '5-10', '10-20', '20-30', '30-45', '>=45'])
    res['by_off10'] = table('off10', [0, 0.01, 0.05, 0.15, 0.30, 1.01], ['<1%', '1-5%', '5-15%', '15-30%', '>=30%'])
    res['by_parallax_px'] = table('parallax_px', [0, 10, 25, 50, 100, 1e9], ['<10', '10-25', '25-50', '50-100', '>=100'])
    res['by_tilt_max'] = table('tilt_max', [0, 15, 25, 35, 46], ['0-15', '15-25', '25-35', '35-45'])
    res['by_ground_m'] = table('ground', [0, 190, 210, 230, 250, 1e9], ['<190', '190-210', '210-230', '230-250', '>=250'])
    res['by_baseline_over_height'] = table('b_over_h', [0, 0.03, 0.06, 0.09, 0.12, 1], ['<3%', '3-6%', '6-9%', '9-12%', '>=12%'])
    # does the gain of the refinement shrink with relief?  paired SR gain per relief bin with tour bootstrap
    V['gain_epe'] = (V.e2_epe < 1).astype(float) - (V.e1_epe < 1).astype(float); V['gain_d'] = (V.e2_d <= 1).astype(float) - (V.e1_d <= 1).astype(float); res['gain_by_relief'] = []
    for lo, hi, lab in zip([0, 0.05, 0.10, 0.20, 10][:-1], [0, 0.05, 0.10, 0.20, 10][1:], ['<5%', '5-10%', '10-20%', '>=20%']):
        s = V[(V.relief >= lo) & (V.relief < hi)]; bt = boot_indices(s.tour.values, 1000, 0); ge = s.gain_epe.values; gd = s.gain_d.values
        res['gain_by_relief'].append(dict(bin=lab, n=int(len(s)), gain_sr_epe=float(ge.mean()), ci_epe=ci(lambda b: ge[b].mean(), bt), gain_sr_d=float(gd.mean()), ci_d=ci(lambda b: gd[b].mean(), bt)))
    from scipy.stats import spearmanr
    res['spearman'] = {k: dict(s1=float(spearmanr(V.relief, V[f'e1_{k}']).statistic), s2=float(spearmanr(V.relief, V[f'e2_{k}']).statistic)) for k in ('epe', 'd', 'psi', 'theta')}
    res['spearman']['relief_vs_residual_px'] = float(spearmanr(V.relief, V.im_disp_med, nan_policy='omit').statistic)
    json.dump(res, open(f'{O}/planarity.json', 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('dist', 'ground_vs_terrain', 'spearman', 'gain_by_relief')}, indent=1))
    for nm in ('by_relief', 'by_struct_m', 'by_off10', 'by_tilt_max', 'by_ground_m', 'by_baseline_over_height'):
        print('\n', nm); print(pd.DataFrame(res[nm])[['bin', 'n', 'relief_med', 'disp_med', 's1_d_mae', 's2_d_mae', 's1_epe_med', 's2_epe_med', 's1_sr_d', 's2_sr_d', 's1_sr_epe', 's2_sr_epe', 's1_psi_mae', 's2_psi_mae']].round(3).to_string(index=False))
