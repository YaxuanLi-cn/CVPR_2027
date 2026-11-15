#!/usr/bin/env python3
"""Pose features from the ORIGINAL matches_superglue npz files (640x480, no rotation handling) for the 60k-split subset they cover.
Same feature schema as extract_geom.py (the 'alt_' estimate does not exist here and is left empty)."""
import sys, os, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from pose_from_matches import pose_features, load_existing_npz, ALT_KEYS
import cv2; cv2.setNumThreads(1)
OUT = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'
def work(a):
    idx, tour, pid = a
    p0, p1, c = load_existing_npz(f'/root/pairUAV-5D-temp/matches_superglue/{tour}/{pid}.npz')
    f = pose_features(p0, p1, c); f['manifest_index'] = idx; f['sg_prim_fine'] = 0.0
    for k in ALT_KEYS: f['alt_' + k[3:]] = 0.0 if k == 'sg_ok' else np.nan
    return f
if __name__ == '__main__':
    df = pd.read_pickle(f'{OUT}/pairs.pkl'); s = df[df.has_sg]
    with Pool(int(os.environ.get('GEOM_PROCS', '9'))) as p: rows = p.map(work, list(zip(s.manifest_index, s.tour, s.pair_id)), chunksize=32)
    g = pd.DataFrame(rows); g.to_pickle(f'{OUT}/geom_existing.pkl'); print(len(g), 'ok', g.sg_ok.mean())
