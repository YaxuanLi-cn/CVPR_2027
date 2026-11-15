#!/usr/bin/env python3
"""MAP relative pose: SuperGlue epipolar residuals + soft priors from the MASt3R-feature MLP (tilt_a, tilt_b, heading).

In near-planar, small-baseline pairs the epipolar constraint alone leaves a pitch-rotation <-> forward-translation
ambiguity. The MLP's per-camera tilt (~1.5 deg MAE) pins that direction; where the matches are informative the data
term dominates and the prior is irrelevant. Usage: map_pose.py <matches_dir> <geom.pkl> <out.pkl> [limit]
"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from geom import R_wc, decompose_R, level_dir_from_t
from pose_from_matches import K0, _sampson, _cheir
F0 = K0[0, 0]; SIG_PX, SIG_T, SIG_H = 1.5, 2.0, 5.0

def R_of(dh, ta, tb): return R_wc(dh, tb).T @ R_wc(0.0, ta)
def t_from_level(R, u, ta):
    d_cam_a = R_wc(0.0, ta).T @ u; t = -R @ d_cam_a; return t / (np.linalg.norm(t) + 1e-12)

def refine(x0, x1, dh, ta, tb, t, prior, w_prior=1.0):
    t0 = t / np.linalg.norm(t); b1 = np.cross(t0, [1, 0, 0] if abs(t0[0]) < 0.9 else [0, 1, 0]); b1 /= np.linalg.norm(b1); b2 = np.cross(t0, b1); p0 = np.array([dh, ta, tb])
    def unpack(p):
        tn = t0 + p[3] * b1 + p[4] * b2; return R_of(*(p0 + p[:3])), tn / np.linalg.norm(tn)
    def fun(p):
        R, tn = unpack(p); r = _sampson(R, tn, x0, x1) * F0 / SIG_PX
        r = np.sign(r) * np.sqrt(2 * np.sqrt(1 + r * r) - 2)          # pseudo-Huber (soft-L1), applied to the data term only
        q = p0 + p[:3]; pr = np.array([(q[1] - prior['ta']) / SIG_T, (q[2] - prior['tb']) / SIG_T, ((q[0] - prior['h'] + 180) % 360 - 180) / SIG_H]) * w_prior
        return np.r_[r, pr]
    res = least_squares(fun, np.zeros(5), x_scale=np.array([0.5, 0.5, 0.5, 0.05, 0.05]), max_nfev=40)
    R, tn = unpack(res.x); q = p0 + res.x[:3]
    return q[0], q[1], q[2], R, tn, float(res.cost)

def map_features(p0, p1, conf, prim, prior):
    out = {k: np.nan for k in ('map_dh', 'map_ta', 'map_tb', 'map_dtilt', 'map_vx', 'map_vy', 'map_vz', 'map_par', 'map_hgta', 'map_hgtb', 'map_shift_t', 'map_shift_dir', 'map_resid')}
    out.update(map_ok=0.0, map_inl=0.0, map_from_prior=0.0)
    keep = conf >= 0.2; p0, p1 = p0[keep], p1[keep]
    if len(p0) < 6: return out
    Ki = np.linalg.inv(K0); x0 = np.c_[p0, np.ones(len(p0))] @ Ki.T; x1 = np.c_[p1, np.ones(len(p1))] @ Ki.T
    u = prior['u'] / (np.linalg.norm(prior['u']) + 1e-9); inits = []
    if prim is not None: inits.append((prim['dh'], prim['ta'], prim['tb'], prim['t'], 3.0, 0))
    Rp = R_of(prior['h'], prior['ta'], prior['tb']); inits.append((prior['h'], prior['ta'], prior['tb'], t_from_level(Rp, u, prior['ta']), 8.0, 1))
    best = None
    for dh, ta, tb, t, thr, fp in inits:
        try:
            for th in ((thr, 3.0) if thr > 3 else (3.0, 3.0)):
                r = np.abs(_sampson(R_of(dh, ta, tb), t, x0, x1)) * F0; inl = r < th
                if inl.sum() < 6: raise ValueError
                dh, ta, tb, R, t, cost = refine(x0[inl], x1[inl], dh, ta, tb, t, prior)
            r = np.abs(_sampson(R, t, x0, x1)) * F0; inl = r < 3.0; n = int(inl.sum())
            pen = ((ta - prior['ta']) / SIG_T)**2 + ((tb - prior['tb']) / SIG_T)**2
            score = n - 2.0 * pen
            if best is None or score > best[0]: best = (score, dh, ta, tb, R, t, inl, r, fp)
        except Exception:
            continue
    if best is None or best[6].sum() < 6: return out
    _, dh, ta, tb, R, t, inl, r, fp = best
    ch, z0 = _cheir(R, t, x0[inl], x1[inl])
    if ch < 0.5:
        ch2, z02 = _cheir(R, -t, x0[inl], x1[inl])
        if ch2 > ch: t, z0 = -t, z02
    zz = z0[(z0 > 0) & np.isfinite(z0)]; v = level_dir_from_t(R, t, ta, tb)
    out.update(map_ok=1.0, map_inl=float(inl.sum()), map_from_prior=float(fp), map_dh=(dh + 180) % 360 - 180, map_ta=ta, map_tb=tb, map_dtilt=tb - ta, map_vx=v[0], map_vy=v[1], map_vz=v[2],
               map_par=float(1 / np.median(zz)) if len(zz) else np.nan, map_resid=float(np.median(r[inl])))
    with np.errstate(all='ignore'):
        X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], x0[inl, :2].T.copy(), x1[inl, :2].T.copy()); X = (X[:3] / X[3]).T
    Xb = X @ R.T + t; good = (X[:, 2] > 0) & (Xb[:, 2] > 0) & np.isfinite(X).all(1)
    if good.sum() >= 6:
        for c, pts, tl in (('a', X[good], ta), ('b', Xb[good], tb)):
            up = np.array([0.0, -np.sin(np.deg2rad(tl)), -np.cos(np.deg2rad(tl))]); out[f'map_hgt{c}'] = float(np.percentile(-(pts @ up), 75))
    if prim is not None:
        out['map_shift_t'] = float(abs(ta - prim['ta']) + abs(tb - prim['tb'])); out['map_shift_dir'] = float(np.rad2deg(np.arccos(np.clip(v @ prim['v'], -1, 1))))
    return out

def work(a):
    path, g, pr = a; d = np.load(path); src = d['src']; p0 = d['p0'].astype(np.float64); p1 = d['p1'].astype(np.float64); conf = d['conf'].astype(np.float64)
    prim = None
    if g is not None and g['sg_ok'] == 1:
        prim = dict(dh=g['sg_dh'], ta=g['sg_ta'], tb=g['sg_tb'], t=np.array([g['sg_t0'], g['sg_t1'], g['sg_t2']]), v=np.array([g['sg_vx'], g['sg_vy'], g['sg_vz']]))
        if g['sg_prim_fine'] == 1: m = src >= 10; p0, p1, conf = p0[m], p1[m], conf[m]
    o = map_features(p0, p1, conf, prim, pr); o['manifest_index'] = int(os.path.basename(path)[:-4]); return o

if __name__ == '__main__':
    cv2.setNumThreads(1); src_dir, geom, dst = sys.argv[1:4]; SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'
    G = pd.read_pickle(geom).set_index('manifest_index'); N = pd.read_pickle(f'{SGD}/stage0_mlp.pkl').set_index('manifest_index')
    files = sorted(glob.glob(f'{src_dir}/*.npz')); files = [f for f in files if int(os.path.basename(f)[:-4]) in G.index]
    if len(sys.argv) > 4: files = files[:int(sys.argv[4])]
    cols = ['sg_ok', 'sg_dh', 'sg_ta', 'sg_tb', 'sg_t0', 'sg_t1', 'sg_t2', 'sg_vx', 'sg_vy', 'sg_vz', 'sg_prim_fine']; Gd = G[cols].to_dict('index'); jobs = []
    for f in files:
        i = int(os.path.basename(f)[:-4]); n = N.loc[i]
        jobs.append((f, Gd[i], dict(h=float(n.n_heading), ta=float(np.clip(n.n_tilt_a, 0, 47)), tb=float(np.clip(n.n_tilt_b, 0, 47)), u=np.array([n.n_ux, n.n_uy, n.n_uz], float))))
    with Pool(int(os.environ.get('GEOM_PROCS', '12'))) as p: rows = p.map(work, jobs, chunksize=32)
    M = pd.DataFrame(rows); M.to_pickle(dst); print(len(M), 'map_ok', M.map_ok.mean(), 'from_prior', M.map_from_prior.mean())
