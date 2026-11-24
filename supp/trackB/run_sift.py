#!/usr/bin/env python3
"""Track B: classical baseline. SIFT (OpenCV, 8000 features, half-resolution grayscale 1843 x 1037) + Lowe ratio 0.8 + mutual
nearest neighbours; matches in original pixel coordinates. SIFT is rotation invariant by construction, so only k = 0 is run.
Usage: run_sift.py <set> [procs]"""
import sys, os, time, numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/supp/trackB')
from run_matchers import pair_set, to_res, OUT, W0, H0

def feats(path):
    g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2); kp, de = cv2.SIFT_create(nfeatures=8000).detectAndCompute(g, None)
    return np.float32([k.pt for k in kp]).reshape(-1, 2), de, g.shape

def work(a):
    cv2.setNumThreads(1); which, idx, pa, pb = a; f = f'{OUT}/matches/sift/{which}/{idx}.npz'
    if os.path.exists(f): return 0.0
    if which == 'test_full' and os.path.exists(f'{OUT}/matches/sift/test/{idx}.npz'):      # the 1000-pair subset is part of the full test set
        import shutil; shutil.copyfile(f'{OUT}/matches/sift/test/{idx}.npz', f); return 0.0
    t0 = time.time(); (ka, da, sa), (kb, db, sb) = feats(pa), feats(pb); xa = np.zeros((0, 2)); xb = np.zeros((0, 2)); c = np.zeros(0)
    if da is not None and db is not None and len(ka) >= 2 and len(kb) >= 2:
        bf = cv2.BFMatcher(cv2.NORM_L2); m12 = bf.knnMatch(da, db, k=2); m21 = bf.knnMatch(db, da, k=1); back = {m[0].queryIdx: m[0].trainIdx for m in m21 if m}
        good = [m for m, n in (p for p in m12 if len(p) == 2) if m.distance < 0.8 * n.distance and back.get(m.trainIdx, -1) == m.queryIdx]
        if good:
            ia = [m.queryIdx for m in good]; ib = [m.trainIdx for m in good]; xa, xb = ka[ia].astype(np.float64), kb[ib].astype(np.float64); c = np.ones(len(good))
    q = lambda v: np.clip(np.rint(4 * v), 0, 65535).astype(np.uint16)
    np.savez_compressed(f, p0_0=q(to_res(xa, sa[1], sa[0], W0, H0)), p1_0=q(to_res(xb, sb[1], sb[0], W0, H0)), c_0=np.full(len(c), 255, np.uint8)); return time.time() - t0

if __name__ == '__main__':
    which = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 2; df = pair_set(which).sample(frac=1.0, random_state=0); os.makedirs(f'{OUT}/matches/sift/{which}', exist_ok=True)      # fixed random order
    with Pool(procs) as p: t = [x for x in p.imap_unordered(work, [(which, int(r.id), r.a, r.b) for r in df.itertuples()], chunksize=4)]
    t = [x for x in t if x > 0]; print(which, 'sift done', len(df), 'pairs;', f'{np.mean(t):.2f} s/pair (1 CPU thread)' if t else 'cached')
    if t:
        import json; json.dump(dict(set=which, pairs=len(t), plain_s=float(np.mean(t)), tta_s=float(np.mean(t)), device='CPU, 1 thread', settings=dict(detector='OpenCV SIFT, nfeatures=8000, half-resolution grayscale (1843 x 1037)', matching='brute-force L2, ratio 0.8, mutual')), open(f'{OUT}/matches/sift/timing_{which}.json', 'w'), indent=1)
