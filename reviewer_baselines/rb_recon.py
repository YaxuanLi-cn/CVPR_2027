#!/usr/bin/env python3
"""Reviewer baselines, part 1: feed-forward reconstruction models used directly as two-view pose estimators.

For every sampled test pair and every in-plane rotation k*90deg of the target image (k=0 is the plain model):
  pnp_*  pose from the predicted pointmap of the target view (expressed in the source frame) by PnP-RANSAC with the
         calibrated intrinsics; MASt3R's pointmap is metric, so the translation is used as is (no scale fitting).
  mm_*   (MASt3R only) reciprocal dense-descriptor matches -> essential matrix -> metric scale from MASt3R depth.
Nothing here is trained or fitted on PairUAV. The source-view metric depth map is saved for rb_eval.py, which uses it
to give SuperGlue + essential-matrix poses a metric scale.

Usage: rb_recon.py {mast3r|dust3r} [n_pairs] [test|train]
"""
import sys, os, time
from multiprocessing import get_context
import numpy as np, pandas as pd, cv2
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
from geom import decompose_R, R_wc
from pose_from_matches import K0, pose_features, _sampson
SG = f'{ROOT}/outputs/pairuav5d_60k/superglue'; OUT = f'{ROOT}/outputs/reviewer_baselines'; DATA = '/root/pairUAV-5D-temp'
W0, H0, W5, H5 = 3686, 2073, 512, 288; SX, SY = W5 / W0, H5 / H0
K5 = np.array([[K0[0, 0] * SX, 0, (K0[0, 2] + 0.5) * SX - 0.5], [0, K0[1, 1] * SY, (K0[1, 2] + 0.5) * SY - 0.5], [0, 0, 1.0]])
to_full = lambda xy: np.c_[(xy[:, 0] + 0.5) / SX - 0.5, (xy[:, 1] + 0.5) / SY - 0.5]


def subset(n=1000, seed=0, split='test'):
    """Random test pairs of the archived result set (the 6,364 test pairs behind Table 2); split='train' gives a calibration sample from the training tours."""
    f = f'{OUT}/subset_{split}_{n}.pkl'
    if os.path.exists(f): return pd.read_pickle(f)
    P = pd.read_pickle(f'{SG}/pairs.pkl'); S = pd.read_pickle(f'{SG}/stage2/s2_pred.pkl'); ids = S[S.split == split].manifest_index
    T = P[P.manifest_index.isin(ids)].sample(n, random_state=seed).sort_values('manifest_index').reset_index(drop=True); T.to_pickle(f); return T


def level(R, t):
    """X_b = R X_a + t  ->  d_heading, tilt_a, tilt_b and the displacement of b from a in a's levelled frame (right, forward, up)."""
    dh, ta, tb = decompose_R(R); d = R_wc(0.0, ta) @ (-R.T @ t); return dh, ta, tb, d


def scale_from_depth(R, t, p0, p1, depth, thr_px=3.0):
    """Metric length of the baseline: median over epipolar inliers of (network depth at the source keypoint) / (triangulated depth for a unit baseline)."""
    Ki = np.linalg.inv(K0); x0 = np.c_[p0, np.ones(len(p0))] @ Ki.T; x1 = np.c_[p1, np.ones(len(p1))] @ Ki.T
    inl = np.abs(_sampson(R, t, x0, x1)) * K0[0, 0] < thr_px
    if inl.sum() < 8: return np.nan, 0
    with np.errstate(all='ignore'):
        X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], x0[inl, :2].T.copy(), x1[inl, :2].T.copy()); X = (X[:3] / X[3]).T
    z = X[:, 2]; u = np.clip(np.rint((p0[inl, 0] + 0.5) * depth.shape[1] / W0 - 0.5).astype(int), 0, depth.shape[1] - 1); v = np.clip(np.rint((p0[inl, 1] + 0.5) * depth.shape[0] / H0 - 0.5).astype(int), 0, depth.shape[0] - 1)
    D = depth[v, u].astype(np.float64); ok = (z > 0) & np.isfinite(z) & (D > 0) & ((X @ R.T + t)[:, 2] > 0)
    return (float(np.median(D[ok] / z[ok])), int(ok.sum())) if ok.sum() >= 8 else (np.nan, 0)


def geom_from_matches(p0, p1, conf, depth, multi_src=None):
    """Essential-matrix pose (the solver of the paper's coarse stage, without priors/MAP) + metric scale from a depth map."""
    from pose_from_matches import pose_features_multi
    F = pose_features(p0, p1, conf) if multi_src is None else pose_features_multi(p0, p1, conf, multi_src)
    o = dict(ok=F['sg_ok'], inl=F['sg_inl'], n=float(len(p0)), dh=F['sg_dh'], ta=F['sg_ta'], tb=F['sg_tb'], vx=F['sg_vx'], vy=F['sg_vy'], vz=F['sg_vz'], hgt=F['sg_hgta_p50'], scale=np.nan, nsc=0)
    if F['sg_ok'] == 1 and depth is not None:
        R = np.array([[F[f'sg_R{i}{j}'] for j in range(3)] for i in range(3)]); t = np.array([F[f'sg_t{i}'] for i in range(3)])
        keep = conf >= 0.2; o['scale'], o['nsc'] = scale_from_depth(R, t, p0[keep], p1[keep], depth)
    return o


def cpu_work(a):
    """PnP on the target pointmap and, if matches are given, the essential-matrix pose. One (pair, rotation) item."""
    idx, k, X2, uv2, c2, X1, xy1, xy2 = a; r = dict(manifest_index=idx, k=k, pnp_ok=0.0, conf2=float(c2.mean()))
    try:
        m = c2 >= np.quantile(c2, 0.3); ok, rv, tv, inl = cv2.solvePnPRansac(X2[m].astype(np.float64), uv2[m].astype(np.float64), K5, None, iterationsCount=500, reprojectionError=2.0, confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
        if ok and inl is not None and len(inl) >= 20:
            ii = inl.ravel(); rv, tv = cv2.solvePnPRefineLM(X2[m][ii].astype(np.float64), uv2[m][ii].astype(np.float64), K5, None, rv, tv); R = cv2.Rodrigues(rv)[0]; dh, ta, tb, d = level(R, tv.ravel())
            r.update(pnp_ok=1.0, pnp_inl=len(ii) / m.sum(), pnp_dh=dh, pnp_ta=ta, pnp_tb=tb, pnp_lx=d[0], pnp_ly=d[1], pnp_z=d[2])
            up = np.array([0.0, -np.sin(np.deg2rad(ta)), -np.cos(np.deg2rad(ta))]); r['pnp_hgt'] = float(np.median(-(X1.reshape(-1, 3)[::7] @ up)))   # camera height above the reconstructed scene, in the model's own units
    except cv2.error:
        pass
    if xy1 is not None:
        g = geom_from_matches(to_full(xy1.astype(np.float64)), to_full(xy2.astype(np.float64)), np.ones(len(xy1)), X1[..., 2]) if len(xy1) >= 8 else dict(ok=0.0, inl=0.0, n=float(len(xy1)))
        r.update({f'mm_{k_}': v for k_, v in g.items()})
    return r


class Pairs:
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        from dust3r.utils.image import load_images
        r = self.df.iloc[i]; a, b = (load_images([f'{DATA}/{p}'], size=512, verbose=False)[0]['img'] for p in (r.image_a, r.image_b)); return int(r.manifest_index), a, b


def main(which, n=1000, split='test'):
    T = subset(n, split=split); os.makedirs(f'{OUT}/depth_{which}', exist_ok=True); pool = get_context('fork').Pool(14)
    sys.path.insert(0, f'{ROOT}/superglue_fusion'); import matched_mast3r_ablation as mm; mm.load_runtime()
    import torch
    from mast3r.model import AsymmetricMASt3R
    from dust3r.model import AsymmetricCroCo3DStereo
    from mast3r.fast_nn import fast_reciprocal_NNs
    dev = 'cuda'
    model = (AsymmetricMASt3R.from_pretrained(str(mm.MODEL_PATH)) if which == 'mast3r' else AsymmetricCroCo3DStereo.from_pretrained('naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt')).to(dev).eval().requires_grad_(False)
    dl = torch.utils.data.DataLoader(Pairs(T), batch_size=None, num_workers=8, prefetch_factor=4); ys, xs = torch.meshgrid(torch.arange(H5), torch.arange(W5), indexing='ij'); jobs = []; t0 = time.time()
    for n_, (idx, ia, ib) in enumerate(dl):
        assert ia.shape[-2:] == (H5, W5), ia.shape
        ia, ib = ia.to(dev), ib.to(dev); deps = []
        for k in range(4):
            ibr = torch.rot90(ib, k, (2, 3)); v1 = dict(img=ia, true_shape=torch.tensor([[H5, W5]]), idx=[0], instance=['0']); v2 = dict(img=ibr, true_shape=torch.tensor([list(ibr.shape[-2:])]), idx=[1], instance=['1'])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                p1, p2 = model(v1, v2)
            X1 = p1['pts3d'][0].float(); X2 = p2['pts3d_in_other_view'][0].float(); c2 = p2['conf'][0].float()
            mx, my = torch.rot90(xs, k, (0, 1)), torch.rot90(ys, k, (0, 1))                       # unrotated target pixel of every rotated pixel
            sl = (slice(2, None, 4), slice(2, None, 4)); uv2 = torch.stack([mx[sl], my[sl]], -1).reshape(-1, 2).numpy().astype(np.float32)
            xy1 = xy2 = None
            if which == 'mast3r':
                a1, a2 = fast_reciprocal_NNs(p1['desc'][0].float(), p2['desc'][0].float(), subsample_or_initxy1=4, device=dev, dist='dot', block_size=2**13)
                Hr, Wr = ibr.shape[-2:]; keep = (a1[:, 0] >= 3) & (a1[:, 0] < W5 - 3) & (a1[:, 1] >= 3) & (a1[:, 1] < H5 - 3) & (a2[:, 0] >= 3) & (a2[:, 0] < Wr - 3) & (a2[:, 1] >= 3) & (a2[:, 1] < Hr - 3); a1, a2 = a1[keep], a2[keep]
                xy1 = a1.astype(np.float32); xy2 = np.stack([mx.numpy()[a2[:, 1], a2[:, 0]], my.numpy()[a2[:, 1], a2[:, 0]]], 1).astype(np.float32)
            X1n = X1.cpu().numpy(); deps.append(X1n[..., 2].astype(np.float16))
            jobs.append(pool.apply_async(cpu_work, ((idx, k, X2[sl].reshape(-1, 3).cpu().numpy(), uv2, c2[sl].reshape(-1).cpu().numpy(), X1n, xy1, xy2),)))
        np.save(f'{OUT}/depth_{which}/{idx}.npy', np.stack(deps))
        if (n_ + 1) % 100 == 0: print(f'{n_+1}/{len(T)} {(time.time()-t0)/(n_+1):.2f}s/pair', flush=True)
    R = pd.DataFrame([j.get() for j in jobs]); R.to_pickle(f'{OUT}/recon_{which}_{split}.pkl'); print('saved', R.shape, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000, sys.argv[3] if len(sys.argv) > 3 else 'test')
