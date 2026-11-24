#!/usr/bin/env python3
"""Classical planar baseline: plane-induced homography from the matches, decomposed with the known calibration.
H ~ R + t n^T / h  ->  (R, t/h, n). The solution whose plane normal is closest to the gravity direction of its own
zero-roll decomposition (and whose tilts are plausible) is kept; the metric translation is (t/h) x H0, with the flight
height H0 a constant fitted on training pairs (median ratio of the true distance to |t/h|). No learned component.
Run on the cached matches of each pass (first pass, zoom pass, imagination pass) for the test pairs.
Output: outputs/supp/trackA/homography.json, homography_pred.pkl"""
import sys, os, json, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from common import *
from geom import decompose_R, R_wc
from pose_from_matches import K0
O = f'{OUT}/trackA'; Ki = np.linalg.inv(K0); F0 = K0[0, 0]
SRC = {'P1': (f'{SG}/matches_rot', True), 'Z': (f'{SG}/matches_zoom', False), 'IM': (f'{S2}/matches_imag', False)}

def solve(a):
    idx, src, thr = a; path, fine = SRC[src]; out = dict(manifest_index=idx, src=src, thr=thr, ok=0)
    try:
        d = np.load(f'{path}/{idx}.npz'); p0, p1, cf = d['p0'].astype(np.float64), d['p1'].astype(np.float64), d['conf'].astype(np.float64); keep = cf >= 0.2
        if fine and 'src' in d.files and (d['src'] >= 10).sum() >= 100: keep &= d['src'] >= 10          # fine matches first, as in the coarse stage
        p0, p1 = p0[keep], p1[keep]
        if len(p0) < 12: return out
        x0 = (np.c_[p0, np.ones(len(p0))] @ Ki.T)[:, :2]; x1 = (np.c_[p1, np.ones(len(p1))] @ Ki.T)[:, :2]
        H, mask = cv2.findHomography(x0, x1, cv2.USAC_MAGSAC, thr / F0, maxIters=5000, confidence=0.9999)
        if H is None or mask is None or mask.sum() < 12: return out
        n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H, np.eye(3)); best = None; inl = mask.ravel().astype(bool)
        for R, t, n in zip(Rs, ts, ns):
            t = t.ravel(); n = n.ravel(); dh, ta, tb = decompose_R(R); g = np.array([0.0, np.sin(np.deg2rad(ta)), np.cos(np.deg2rad(ta))])
            if n @ g < 0: n = -n; t = -t                                              # plane in front of the camera: normal along gravity
            front = np.mean((np.c_[x0[inl], np.ones(inl.sum())] @ n) > 0)            # points must lie in front of the plane normal
            pen = max(0.0, -8 - ta, ta - 53) + max(0.0, -8 - tb, tb - 53); score = (n @ g) + front - 0.05 * pen
            if best is None or score > best[0]: best = (score, R, t, n, dh, ta, tb)
        _, R, t, n, dh, ta, tb = best; l = R_wc(0.0, ta) @ (-R.T @ t)                 # displacement in the levelled frame, in units of the plane distance
        out.update(ok=1, inl=int(inl.sum()), dh=dh, ta=ta, tb=tb, wx=l[0], wy=l[1], wz=l[2], ncos=float(n @ np.array([0.0, np.sin(np.deg2rad(ta)), np.cos(np.deg2rad(ta))])))
    except Exception as e: out['err'] = str(e)[:80]
    return out

if __name__ == '__main__':
    cv2.setNumThreads(1); T = load_test(); P = pd.read_pickle(f'{SG}/pairs.pkl'); Z = pd.read_pickle(f'{SG}/half_zoom_pred.pkl'); I = pd.read_pickle(f'{S2}/geom_imag.pkl')
    tr_all = Z[Z.split == 'train'].manifest_index.values; rng = np.random.default_rng(0); tr = rng.choice(np.intersect1d(tr_all, I.manifest_index.values), 2000, replace=False)   # training pairs that have all three passes
    THR = (3.0, 8.0, 20.0); jobs = [(int(i), s, th) for s in SRC for th in THR for i in np.r_[T.manifest_index.values, tr]]
    with Pool(int(os.environ.get('PROCS', '6'))) as pool: R = pd.DataFrame(pool.map(solve, jobs, chunksize=64))
    R.to_pickle(f'{O}/homography_pred.pkl'); Pd = P.set_index('manifest_index'); dist = np.sqrt(Pd.x**2 + Pd.y**2 + Pd.z**2); res = {}; gc = gt_cam(T); boots = boot_indices(T.tour.values, 1000, 0)
    Ptr = add_t = None
    import fusion_lib; Ptr = fusion_lib.add_targets(P).set_index('manifest_index').reindex(tr); gtr = dict(x=Ptr.lx.values, y=Ptr.ly.values, z=Ptr.z.values, tilt=Ptr.tilt.values, heading=Ptr.heading.values)
    def predict(r, ids, H0):
        g = r.reindex(ids); ok = ((g.ok == 1) & np.isfinite(g.wx)).values; t = np.where(ok[:, None], g[['wx', 'wy', 'wz']].values.astype(float) * H0, 0.0); t = np.where((np.linalg.norm(t, axis=1) < 500)[:, None], t, 0.0)
        return dict(x=t[:, 0], y=t[:, 1], z=t[:, 2], tilt=np.where(ok, g.tb - g.ta, 0.0), heading=np.where(ok, g.dh, 0.0)), ok, g
    for s in SRC:
        cand = {}
        for th in THR:
            r = R[(R.src == s) & (R.thr == th)].set_index('manifest_index'); rt = r.reindex(tr); w = np.sqrt(rt.wx**2 + rt.wy**2 + rt.wz**2); okt = (rt.ok == 1) & np.isfinite(w) & (w > 1e-6); H0 = float(np.median((dist.reindex(tr)[okt] / w[okt])))
            ptr, _, _ = predict(r, tr, H0); etr = errors(ptr, gtr); cand[th] = (float((etr['d'] <= 1).mean()), H0, r)
        th = max(cand, key=lambda k: cand[k][0]); sr_tr, H0, r = cand[th]            # inlier threshold chosen on the TRAINING pairs
        pt, ok, g = predict(r, T.manifest_index.values, H0); e = errors(pt, gc); sm = summarize(e); sm.update(fail=float(1 - ok.mean()), H0=H0, inl_med=float(g.inl.median()), thr_px=th, train_sr_d_by_thr={str(k): v[0] for k, v in cand.items()})
        sm['ci'] = {k: ci(fn, boots) for k, fn in (('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()))}; res[s] = sm
        print(f"{s:3s} thr {th:g}px (train SR_d by thr {sm['train_sr_d_by_thr']}) fail {100*sm['fail']:.1f}% H0 {H0:.1f} | d {sm['d_mae']:.2f} SR_d {100*sm['sr_d_1m']:.1f} | EPE mean {sm['epe_mae']:.2f} med {sm['epe_med']:.2f} SR_EPE {100*sm['sr_epe_1m']:.1f} | theta {sm['theta_mae']:.2f}/{sm['theta_med']:.3f} psi {sm['psi_mae']:.2f}/{sm['psi_med']:.3f} dir {sm['dir_med']:.2f}")
    json.dump(res, open(f'{O}/homography.json', 'w'), indent=1)
