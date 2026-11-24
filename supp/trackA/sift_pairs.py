#!/usr/bin/env python3
"""SIFT matches (same settings as supp/trackB/run_sift.py: OpenCV SIFT, 8000 features, half-resolution grayscale, ratio 0.8, mutual)
for the pairs of the control subset (ids_ctrl.npy), then the essential-matrix pose with the solver used everywhere
(pose_from_matches.pose_features, inlier threshold 0.2 px as selected by Track B on its calibration pairs).
Matches already computed by Track B are reused. Output: outputs/supp/trackA/geom_sift.pkl   Usage: sift_pairs.py match|geom [procs]"""
import sys, os, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/supp']
from common import SG, OUT, DATA
O = f'{OUT}/trackA'; MD = f'{O}/matches_sift'; TB = f'{OUT}/trackB/matches/sift'; W0, H0 = 3686, 2073; THR = 0.2
def find(i):
    for f in (f'{TB}/test_full/{i}.npz', f'{TB}/test/{i}.npz', f'{TB}/train/{i}.npz', f'{MD}/{i}.npz'):
        if os.path.exists(f): return f
    return None
def feats(path):
    g = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2); kp, de = cv2.SIFT_create(nfeatures=8000).detectAndCompute(g, None)
    return np.float32([k.pt for k in kp]).reshape(-1, 2), de, g.shape
def to_res(xy, w, h): return np.c_[(xy[:, 0] + 0.5) * W0 / w - 0.5, (xy[:, 1] + 0.5) * H0 / h - 0.5] if len(xy) else np.zeros((0, 2))
def match(a):
    cv2.setNumThreads(1); idx, pa, pb = a
    if find(idx): return 0
    (ka, da, sa), (kb, db, sb) = feats(pa), feats(pb); xa = np.zeros((0, 2)); xb = np.zeros((0, 2))
    if da is not None and db is not None and len(ka) >= 2 and len(kb) >= 2:
        bf = cv2.BFMatcher(cv2.NORM_L2); m12 = bf.knnMatch(da, db, k=2); m21 = bf.knnMatch(db, da, k=1); back = {m[0].queryIdx: m[0].trainIdx for m in m21 if m}
        good = [m for m, n in (p for p in m12 if len(p) == 2) if m.distance < 0.8 * n.distance and back.get(m.trainIdx, -1) == m.queryIdx]
        if good: xa, xb = ka[[m.queryIdx for m in good]].astype(np.float64), kb[[m.trainIdx for m in good]].astype(np.float64)
    q = lambda v: np.clip(np.rint(4 * v), 0, 65535).astype(np.uint16)
    np.savez_compressed(f'{MD}/{idx}.npz', p0_0=q(to_res(xa, sa[1], sa[0])), p1_0=q(to_res(xb, sb[1], sb[0])), c_0=np.full(len(xa), 255, np.uint8)); return 1
def geom(idx):
    from pose_from_matches import pose_features
    cv2.setNumThreads(1); f = find(idx)
    if f is None: return dict(manifest_index=idx, sf_ok=0.0)
    d = np.load(f); p0 = d['p0_0'].astype(np.float64) / 4; p1 = d['p1_0'].astype(np.float64) / 4; g = pose_features(p0, p1, np.ones(len(p0)), thr_px=THR)
    out = {'sf_' + k[3:]: v for k, v in g.items() if k.startswith('sg_') and not k.startswith(('sg_R', 'sg_t0', 'sg_t1', 'sg_t2'))}; out['manifest_index'] = idx; return out
if __name__ == '__main__':
    mode = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 4; os.makedirs(MD, exist_ok=True)
    ids = np.load(f'{O}/ids_ctrl.npy'); P = pd.read_pickle(f'{SG}/pairs.pkl').set_index('manifest_index').loc[ids]
    if mode == 'match':
        Q = P.sample(frac=1.0, random_state=0); Q = pd.concat([Q[Q.split == 'train'], Q[Q.split == 'test']])      # training pairs first: the test pairs are also being matched by Track B
        jobs = [(int(i), f'{DATA}/{r.image_a}', f'{DATA}/{r.image_b}') for i, r in Q.iterrows()]
        with Pool(procs) as p: n = sum(p.imap_unordered(match, jobs, chunksize=4))
        print('matched', n, 'new pairs; available', sum(find(int(i)) is not None for i in ids), 'of', len(ids))
    else:
        with Pool(procs) as p: G = pd.DataFrame(p.map(geom, [int(i) for i in ids], chunksize=16))
        G.to_pickle(f'{O}/geom_sift.pkl'); print(len(G), 'sf_ok', G.sf_ok.mean(), 'median inliers', G.sf_inl.median())
