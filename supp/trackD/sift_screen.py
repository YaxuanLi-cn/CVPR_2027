#!/usr/bin/env python3
"""Re-implementation of the visual screen named in the pair metadata ("SIFT mutual ratio + USAC_MAGSAC", model "fundamental").
The original screening code is not part of the release, so image size, feature budget and ratio are unknown; `calib` searches the
configuration that best reproduces the (matches, inliers, grid_cells, hull_fraction) stored with selected pairs.
Usage: sift_screen.py calib [n_pairs]   |   sift_screen.py apply <pairs.pkl> <out.pkl> [procs]
"""
import sys, os, json, itertools, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
OUT = '/root/rasr-pairuav/outputs/supp/trackD'; DATA = '/root/pairUAV-5D-temp'
BEST = dict(width=1024, nfeat=0, ratio=0.75, thr=1.0)          # overwritten from cache/sift_calib.json when present
if os.path.exists(f'{OUT}/cache/sift_calib.json'): BEST = json.load(open(f'{OUT}/cache/sift_calib.json'))['best']


def feats(path, width, nfeat):
    if width < 0: im = cv2.imread(path, {-2: cv2.IMREAD_REDUCED_GRAYSCALE_2, -4: cv2.IMREAD_REDUCED_GRAYSCALE_4, -8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[width]); h = im.shape[0]; width = im.shape[1]      # libjpeg reduced-size decoding
    else: im = cv2.imread(path, cv2.IMREAD_GRAYSCALE); h = int(round(im.shape[0] * width / im.shape[1])); im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
    k, d = cv2.SIFT_create(nfeatures=nfeat).detectAndCompute(im, None); return np.float32([p.pt for p in k]).reshape(-1, 2), d, (width, h)


def screen(pa, pb, width, nfeat, ratio, thr):
    ka, da, sz = feats(pa, width, nfeat); kb, db, _ = feats(pb, width, nfeat)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8: return dict(matches=0, inliers=0, grid_cells=0, hull_fraction=0.0)
    bf = cv2.BFMatcher(cv2.NORM_L2); m12 = bf.knnMatch(da, db, k=2); m21 = bf.knnMatch(db, da, k=2)
    g12 = {m.queryIdx: m.trainIdx for m, n in (x for x in m12 if len(x) == 2) if m.distance < ratio * n.distance}; g21 = {m.queryIdx: m.trainIdx for m, n in (x for x in m21 if len(x) == 2) if m.distance < ratio * n.distance}
    pairs = [(i, j) for i, j in g12.items() if g21.get(j) == i]; nm = len(pairs)
    if nm < 8: return dict(matches=nm, inliers=0, grid_cells=0, hull_fraction=0.0)
    p0 = ka[[i for i, _ in pairs]]; p1 = kb[[j for _, j in pairs]]
    try: F, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, thr, 0.999, 10000)
    except cv2.error: F, mask = None, None
    if mask is None: return dict(matches=nm, inliers=0, grid_cells=0, hull_fraction=0.0)
    inl = mask.ravel().astype(bool); q = p0[inl]; cells = len(set(zip((q[:, 0] * 4 // sz[0]).astype(int).clip(0, 3), (q[:, 1] * 4 // sz[1]).astype(int).clip(0, 3)))) if inl.sum() else 0
    hull = float(cv2.contourArea(cv2.convexHull(q)) / (sz[0] * sz[1])) if inl.sum() >= 3 else 0.0
    return dict(matches=nm, inliers=int(inl.sum()), grid_cells=cells, hull_fraction=hull)


def _job(a):
    cv2.setNumThreads(1); pa, pb, cfg = a
    try: return screen(pa, pb, **cfg)
    except Exception as e: return dict(matches=-1, inliers=-1, grid_cells=-1, hull_fraction=-1.0)


def calib(n=48, procs=3):
    P = pd.read_pickle(f'{OUT}/cache/pairs_selected.pkl').sample(n, random_state=0).reset_index(drop=True); grid = [dict(width=w, nfeat=nf, ratio=r, thr=t) for w in (-4, 800, 864, 960) for nf in (0,) for r in (0.7, 0.75, 0.8) for t in (1.0,)]; rows = []
    with Pool(procs) as pool:
        for cfg in grid:
            R = pd.DataFrame(pool.map(_job, [(f'{DATA}/images/{t}/{a:04d}.jpg', f'{DATA}/images/{t}/{b:04d}.jpg', cfg) for t, a, b in zip(P.tour, P.a, P.b)]))
            lm = np.log((R.matches + 1) / (P.matches + 1)); li = np.log((R.inliers + 1) / (P.inliers + 1))
            rows.append(dict(**cfg, med_ratio_matches=float(np.exp(lm.median())), med_ratio_inliers=float(np.exp(li.median())), mad_log_matches=float((lm - lm.median()).abs().median()), mad_log_inliers=float((li - li.median()).abs().median()),
                             corr_log_inliers=float(np.corrcoef(np.log1p(R.inliers), np.log1p(P.inliers))[0, 1]), grid_equal=float((R.grid_cells == P.grid_cells).mean()), hull_mae=float((R.hull_fraction - P.hull_fraction).abs().mean()), score=float(np.abs(li).median() + np.abs(lm).median())))
            print(rows[-1], flush=True)
    C = pd.DataFrame(rows).sort_values('score'); best = {k: (int(v) if k in ('width', 'nfeat') else float(v)) for k, v in C.iloc[0][['width', 'nfeat', 'ratio', 'thr']].items()}
    json.dump(dict(best=best, table=C.to_dict('records'), n=n), open(f'{OUT}/cache/sift_calib.json', 'w'), indent=1); print(C.head(6).to_string()); print('best', best)


def apply(src, dst, procs=3):
    P = pd.read_pickle(src)
    with Pool(procs) as pool: R = pd.DataFrame(pool.map(_job, [(a, b, BEST) for a, b in zip(P.image_a, P.image_b)], chunksize=4))
    R.columns = ['sift_' + c for c in R.columns]; pd.concat([P.reset_index(drop=True), R], axis=1).to_pickle(dst); print(R.describe().round(2).to_string())


if __name__ == '__main__':
    if sys.argv[1] == 'calib': calib(int(sys.argv[2]) if len(sys.argv) > 2 else 48)
    else: apply(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
