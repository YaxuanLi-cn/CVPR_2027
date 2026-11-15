#!/usr/bin/env python3
"""Pose features from the zoom re-matching pass: plain E-matrix estimate (zm_*) and MAP estimate with MLP priors (zmap_*).
Usage: extract_geom_zoom.py <zoom_matches_dir> <out.pkl>"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from pose_from_matches import pose_features
from map_pose import map_features
def work(a):
    path, pr = a; d = np.load(path); p0 = d['p0'].astype(np.float64); p1 = d['p1'].astype(np.float64); conf = d['conf'].astype(np.float64)
    f = pose_features(p0, p1, conf); prim = None
    if f['sg_ok'] == 1: prim = dict(dh=f['sg_dh'], ta=f['sg_ta'], tb=f['sg_tb'], t=np.array([f['sg_t0'], f['sg_t1'], f['sg_t2']]), v=np.array([f['sg_vx'], f['sg_vy'], f['sg_vz']]))
    m = map_features(p0, p1, conf, prim, pr)
    out = {'zm_' + k[3:]: v for k, v in f.items()}; out.update({'zmap_' + k[4:]: v for k, v in m.items()})
    out['manifest_index'] = int(os.path.basename(path)[:-4]); out['zm_zoom'] = float(d['zoom']); out['zm_guide'] = float(d['guide_src']); return out
if __name__ == '__main__':
    cv2.setNumThreads(1); src, dst = sys.argv[1:3]; N = pd.read_pickle('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/stage0_mlp.pkl').set_index('manifest_index'); jobs = []
    for f in sorted(glob.glob(f'{src}/*.npz')):
        n = N.loc[int(os.path.basename(f)[:-4])]
        jobs.append((f, dict(h=float(n.n_heading), ta=float(np.clip(n.n_tilt_a, 0, 47)), tb=float(np.clip(n.n_tilt_b, 0, 47)), u=np.array([n.n_ux, n.n_uy, n.n_uz], float))))
    with Pool(int(os.environ.get('GEOM_PROCS', '12'))) as p: rows = p.map(work, jobs, chunksize=32)
    Z = pd.DataFrame(rows); Z.to_pickle(dst); print(len(Z), 'zm_ok', Z.zm_ok.mean(), 'zmap_ok', Z.zmap_ok.mean())
