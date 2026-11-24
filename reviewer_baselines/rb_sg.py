#!/usr/bin/env python3
"""Reviewer baselines, part 2a: off-the-shelf SuperGlue (outdoor weights, 1280x720, 2048 keypoints, single pass, no rotation
handling) on the sampled pairs. Output: one npz per pair with matches in original pixel coordinates.
Usage: rb_sg.py [n_pairs] [test|train]"""
import sys, os, numpy as np, torch
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
import run_superglue_rot as rs
from rb_recon import subset, OUT
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; split = sys.argv[2] if len(sys.argv) > 2 else 'test'
T = subset(n, split=split); out = f'{OUT}/matches_sg_plain'; os.makedirs(out, exist_ok=True); size = (1280, 720); dev = 'cuda'
sp = rs.SuperPoint({'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': 2048}).eval().to(dev); sg = rs.SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev)
dl = torch.utils.data.DataLoader(rs.Pairs(T, [size]), batch_size=None, num_workers=8, prefetch_factor=4); sc = np.array([rs.W0 / size[0], rs.H0 / size[1]], dtype=np.float32)
for i, (idx, tour, pid, a, b) in enumerate(dl):
    f = lambda t: (t.to(dev).float() / 255.0)[None, None]
    with torch.autocast('cuda', dtype=torch.float16):
        (k0, k1, conf), = rs.match_rot(sp, sg, f(a[0]), f(b[0]), [0])[0]
    np.savez(f'{out}/{idx}.npz', p0=k0.float().cpu().numpy() * sc, p1=k1.float().cpu().numpy() * sc, conf=conf.float().cpu().numpy())
print('done', len(T))
