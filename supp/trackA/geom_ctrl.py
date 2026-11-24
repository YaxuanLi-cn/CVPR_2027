#!/usr/bin/env python3
"""Pose + verification features from a directory of (control) imagination matches: same worker as stage2_imagination/imagine_geom.py.
Usage: geom_ctrl.py <matches_dir> <out.pkl> [guide.pkl]"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/stage2_imagination'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
import imagine_geom as ig
SGD = ig.SGD
if __name__ == '__main__':
    cv2.setNumThreads(1); src, dst = sys.argv[1:3]; guide = sys.argv[3] if len(sys.argv) > 3 else f'{SGD}/stage2/guide.pkl'
    N = pd.read_pickle(f'{SGD}/stage0_mlp.pkl').set_index('manifest_index'); G1 = pd.read_pickle(f'{SGD}/geom_half.pkl').set_index('manifest_index'); M1 = pd.read_pickle(f'{SGD}/map_half.pkl').set_index('manifest_index')
    Z = pd.read_pickle(f'{SGD}/geom_zoom_half.pkl').set_index('manifest_index'); U = pd.read_pickle(guide).set_index('manifest_index'); jobs = []
    for f in sorted(glob.glob(f'{src}/*.npz')):
        i = int(os.path.basename(f)[:-4]); n = N.loc[i]; g = G1.loc[i]; m = M1.loc[i]; z = Z.loc[i]; u = U.loc[i]; fu = np.array([u.p_lx, u.p_ly, u.p_z]); fu = fu / (np.linalg.norm(fu) + 1e-9)
        cands = dict(sg=(g.sg_dh, g.sg_ta, g.sg_tb, (g.sg_vx, g.sg_vy, g.sg_vz)), map=(m.map_dh, m.map_ta, m.map_tb, (m.map_vx, m.map_vy, m.map_vz)),
                     zm=(z.zm_dh, z.zm_ta, z.zm_tb, (z.zm_vx, z.zm_vy, z.zm_vz)), zmap=(z.zmap_dh, z.zmap_ta, z.zmap_tb, (z.zmap_vx, z.zmap_vy, z.zmap_vz)), s1=(u.g_dh, u.g_ta, u.g_tb, tuple(fu)))
        jobs.append((f, dict(h=float(n.n_heading), ta=float(np.clip(n.n_tilt_a, 0, 47)), tb=float(np.clip(n.n_tilt_b, 0, 47)), u=np.array([n.n_ux, n.n_uy, n.n_uz], float)), cands))
    with Pool(int(os.environ.get('GEOM_PROCS', '6'))) as p: rows = p.map(ig.work, jobs, chunksize=16)
    R = pd.DataFrame(rows); R.to_pickle(dst); print(len(R), 'im_ok', R.im_ok.mean(), 'imap_ok', R.imap_ok.mean(), 'median inliers', R.im_inl.median(), 'median residual px', R.im_disp_med.median())
