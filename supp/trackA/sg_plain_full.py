#!/usr/bin/env python3
"""Off-the-shelf SuperGlue (outdoor weights, 1280x720, 2048 keypoints, ONE pass, no rotation handling) on all 6,364 test pairs,
then the essential-matrix pose with the same solver as everywhere else (pose_from_matches.pose_features).
This is the literal "frozen SuperGlue" configuration. Output: outputs/supp/trackA/geom_sg_plain_test.pkl"""
import sys, os, numpy as np, pandas as pd, torch, cv2
from multiprocessing import Pool
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/supp']
import run_superglue_rot as rs
from pose_from_matches import pose_features
from common import load_test, OUT
O = f'{OUT}/trackA'; MD = f'{O}/matches_sg_plain'; OLD = f'{ROOT}/outputs/reviewer_baselines/matches_sg_plain'; os.makedirs(MD, exist_ok=True)
def path(i): return f'{OLD}/{i}.npz' if os.path.exists(f'{OLD}/{i}.npz') else f'{MD}/{i}.npz'
def work(i):
    cv2.setNumThreads(1); d = np.load(path(i)); f = pose_features(d['p0'].astype(np.float64), d['p1'].astype(np.float64), d['conf'].astype(np.float64)); f['manifest_index'] = i; return f
if __name__ == '__main__':
    T = load_test(); todo = T[[not os.path.exists(path(i)) for i in T.manifest_index]]; print('pairs to match', len(todo), flush=True)
    if len(todo):
        size = (1280, 720); dev = 'cuda'; sp = rs.SuperPoint({'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 2048}).eval().to(dev); sg = rs.SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev)
        dl = torch.utils.data.DataLoader(rs.Pairs(todo, [size]), batch_size=None, num_workers=5, prefetch_factor=4); sc = np.array([rs.W0 / size[0], rs.H0 / size[1]], dtype=np.float32)
        with torch.inference_mode():
            for n, (idx, tour, pid, a, b) in enumerate(dl):
                f = lambda t: (t.to(dev).float() / 255.0)[None, None]
                with torch.autocast('cuda', dtype=torch.float16): (k0, k1, conf), = rs.match_rot(sp, sg, f(a[0]), f(b[0]), [0])[0]
                np.savez(f'{MD}/{idx}.npz', p0=k0.float().cpu().numpy() * sc, p1=k1.float().cpu().numpy() * sc, conf=conf.float().cpu().numpy())
                if (n + 1) % 1000 == 0: print(n + 1, flush=True)
        del sp, sg; torch.cuda.empty_cache()
    with Pool(6) as p: G = pd.DataFrame(p.map(work, [int(i) for i in T.manifest_index], chunksize=32))
    G.to_pickle(f'{O}/geom_sg_plain_test.pkl'); print('solved', G.sg_ok.mean(), 'median inliers', G.sg_inl.median())
