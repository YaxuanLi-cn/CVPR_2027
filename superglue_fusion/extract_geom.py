#!/usr/bin/env python3
"""Pose features from rotation-aware SuperGlue match files (<dir>/<manifest_index>.npz)."""
import sys, glob, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from pose_from_matches import pose_features_multi
import cv2; cv2.setNumThreads(1)
def work(path):
    d = np.load(path)
    f = pose_features_multi(d['p0'].astype(np.float64), d['p1'].astype(np.float64), d['conf'].astype(np.float64), d['src'])
    f['manifest_index'] = int(os.path.basename(path)[:-4]); f['sg_rot'] = int(d['rot'])
    rc = np.sort(d['rot_counts'])[::-1]; f['sg_rot_margin'] = float((rc[0] - rc[1]) / max(rc[0], 1)); f['sg_rot_best'] = float(rc[0])
    return f
if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(f'{src}/*.npz'))
    if len(sys.argv) > 3: files = files[:int(sys.argv[3])]
    with Pool(int(os.environ.get("GEOM_PROCS", "14"))) as p: rows = p.map(work, files, chunksize=64)
    g = pd.DataFrame(rows); g.to_pickle(dst); print(len(g), 'ok', g.sg_ok.mean())
