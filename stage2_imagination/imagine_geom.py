#!/usr/bin/env python3
"""Pose from the imagination-pass matches (E-matrix `im_*`, MAP `imap_*`) + candidate verification scores.

Candidate verification: every pose candidate that stage 1 produced (pass-1 E / MAP, zoom E / MAP, the fused stage-1 pose) is scored
against the imagination-vs-reality correspondences (share of matches within 1.5 px / 3 px of its epipolar geometry, median residual).
Also exported: statistics of the residual displacement between imagined and real view (how wrong the imagined view was).
Usage: imagine_geom.py <matches_dir> <out.pkl>"""
import sys, os, glob, numpy as np, pandas as pd, cv2
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from pose_from_matches import pose_features, K0, _sampson
from map_pose import map_features, R_of, t_from_level
SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; F0 = K0[0, 0]

def cand_scores(x0, x1, dh, ta, tb, v):
    if not np.isfinite([dh, ta, tb, *v]).all() or len(x0) < 8: return np.nan, np.nan, np.nan
    R = R_of(dh, ta, tb); t = t_from_level(R, np.asarray(v, float), ta); r = np.abs(_sampson(R, t, x0, x1)) * F0
    return float((r < 1.5).mean()), float((r < 3.0).mean()), float(np.median(r))

def work(a):
    path, pr, cands = a; d = np.load(path); p0 = d['p0'].astype(np.float64); p1 = d['p1'].astype(np.float64); conf = d['conf'].astype(np.float64); disp = d['disp'].astype(np.float64)
    f = pose_features(p0, p1, conf); prim = None
    if f['sg_ok'] == 1: prim = dict(dh=f['sg_dh'], ta=f['sg_ta'], tb=f['sg_tb'], t=np.array([f['sg_t0'], f['sg_t1'], f['sg_t2']]), v=np.array([f['sg_vx'], f['sg_vy'], f['sg_vz']]))
    m = map_features(p0, p1, conf, prim, pr)
    out = {'im_' + k[3:]: v for k, v in f.items() if not k.startswith(('sg_R', 'sg_t0', 'sg_t1', 'sg_t2'))}; out.update({'imap_' + k[4:]: v for k, v in m.items()})
    out['manifest_index'] = int(os.path.basename(path)[:-4]); out['im_zoom'] = float(d['zoom']); out['im_grid'] = float(d['grid']); out['im_ntiles'] = float(d['ntiles'])
    keep = conf >= 0.2; n = int(keep.sum())
    if n:   # imagination-vs-reality discrepancy, in original-a pixels
        dd = disp[keep]; mag = np.hypot(dd[:, 0], dd[:, 1]); out.update(im_disp_med=float(np.median(mag)), im_disp_p90=float(np.percentile(mag, 90)), im_disp_mx=float(np.median(dd[:, 0])), im_disp_my=float(np.median(dd[:, 1])),
                                                                 im_disp_lt8=float((mag < 8).mean()), im_disp_lt30=float((mag < 30).mean()))
    else: out.update(im_disp_med=np.nan, im_disp_p90=np.nan, im_disp_mx=np.nan, im_disp_my=np.nan, im_disp_lt8=np.nan, im_disp_lt30=np.nan)
    Ki = np.linalg.inv(K0); x0 = np.c_[p0[keep], np.ones(n)] @ Ki.T; x1 = np.c_[p1[keep], np.ones(n)] @ Ki.T
    for nm, c in cands.items():
        s15, s3, med = cand_scores(x0, x1, *c); out[f'ver_{nm}_in15'] = s15; out[f'ver_{nm}_in3'] = s3; out[f'ver_{nm}_med'] = med
    return out

if __name__ == '__main__':
    cv2.setNumThreads(1); src, dst = sys.argv[1:3]; N = pd.read_pickle(f'{SGD}/stage0_mlp.pkl').set_index('manifest_index')
    G1 = pd.read_pickle(f'{SGD}/geom_half.pkl').set_index('manifest_index'); M1 = pd.read_pickle(f'{SGD}/map_half.pkl').set_index('manifest_index'); Z = pd.read_pickle(f'{SGD}/geom_zoom_half.pkl').set_index('manifest_index')
    U = pd.read_pickle(f'{SGD}/stage2/guide.pkl').set_index('manifest_index'); jobs = []
    for f in sorted(glob.glob(f'{src}/*.npz')):
        i = int(os.path.basename(f)[:-4]); n = N.loc[i]; g = G1.loc[i]; m = M1.loc[i]; z = Z.loc[i]; u = U.loc[i]; fu = np.array([u.p_lx, u.p_ly, u.p_z]); fu = fu / (np.linalg.norm(fu) + 1e-9)
        cands = dict(sg=(g.sg_dh, g.sg_ta, g.sg_tb, (g.sg_vx, g.sg_vy, g.sg_vz)), map=(m.map_dh, m.map_ta, m.map_tb, (m.map_vx, m.map_vy, m.map_vz)),
                     zm=(z.zm_dh, z.zm_ta, z.zm_tb, (z.zm_vx, z.zm_vy, z.zm_vz)), zmap=(z.zmap_dh, z.zmap_ta, z.zmap_tb, (z.zmap_vx, z.zmap_vy, z.zmap_vz)), s1=(u.g_dh, u.g_ta, u.g_tb, tuple(fu)))
        jobs.append((f, dict(h=float(n.n_heading), ta=float(np.clip(n.n_tilt_a, 0, 47)), tb=float(np.clip(n.n_tilt_b, 0, 47)), u=np.array([n.n_ux, n.n_uy, n.n_uz], float)), cands))
    with Pool(int(os.environ.get('GEOM_PROCS', '12'))) as p: rows = p.map(work, jobs, chunksize=16)
    R = pd.DataFrame(rows); R.to_pickle(dst); print(len(R), 'im_ok', R.im_ok.mean(), 'imap_ok', R.imap_ok.mean(), 'median inliers', R.im_inl.median())
