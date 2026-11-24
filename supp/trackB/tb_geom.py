#!/usr/bin/env python3
"""Track B, part 2: matches -> calibrated essential matrix (MAGSAC++) -> 5-DoF relative pose, for every pair, rotation and inlier threshold.

The solver is the one behind the archived direct baselines (reviewer_baselines/rb_recon.geom_from_matches -> superglue_fusion/
pose_from_matches.pose_features: cv2.findEssentialMat(USAC_MAGSAC), cheirality + zero-roll/tilt-range selection among the four
decompositions, Sampson refinement, zero-roll decomposition into heading change and the two tilts, translation direction in the
levelled frame of the source camera). The only addition is that the inlier threshold (in ORIGINAL pixels) is a parameter:
matchers that work on 512-1150 px images cannot be judged with one fixed 3-px threshold at 3686 px.
All matches returned by a matcher under its own recommended filtering are given to the solver (confidence = 1).

Usage: tb_geom.py <matcher> <set> <thr[,thr...]>      -> outputs/supp/trackB/geom/<matcher>_<set>.pkl  (one row per pair x rotation x threshold)
"""
import sys, os, glob
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
from pose_from_matches import pose_features, pose_features_multi
from rb_recon import scale_from_depth
OUT = f'{ROOT}/outputs/supp/trackB'; RB = f'{ROOT}/outputs/reviewer_baselines'; SGD = f'{ROOT}/outputs/pairuav5d_60k/superglue'


def depth_for(which, idx, k):
    """Source-view metric depth of MASt3R (288 x 512), from the forward pass with the target rotated by k."""
    f = f'{RB}/depth_mast3r/{idx}.npy' if which in ('test', 'train') else f'{OUT}/mast3r/depth_{which}/{idx}.npy'
    return np.load(f)[k].astype(np.float32) if os.path.exists(f) else None


def solve(p0, p1, conf, depth, thr, multi_src=None):
    F = pose_features(p0, p1, conf, thr_px=thr) if multi_src is None else pose_features_multi(p0, p1, conf, multi_src)
    # camera height above the triangulated scene, in baseline units: `hgt` = median over the inliers below the source camera (the statistic of the
    # archived direct baselines); `hgt75ab` = mean over both cameras of the 75th percentile, i.e. the ground level rather than the roofs
    # (the statistic DreamNav's own flight-height scale cue uses: s_hgtab_p75 in superglue_fusion/stage1_fusion.py)
    o = dict(ok=F['sg_ok'], inl=F['sg_inl'], n=float(len(p0)), dh=F['sg_dh'], ta=F['sg_ta'], tb=F['sg_tb'], vx=F['sg_vx'], vy=F['sg_vy'], vz=F['sg_vz'], hgt=F['sg_hgta_p50'],
             hgta75=F['sg_hgta_p75'], hgtb75=F['sg_hgtb_p75'], hgtb50=F['sg_hgtb_p50'], hgt75ab=0.5 * (F['sg_hgta_p75'] + F['sg_hgtb_p75']), resid=F['sg_resid'], scale=np.nan, nsc=0)
    if F['sg_ok'] == 1 and depth is not None:
        R = np.array([[F[f'sg_R{i}{j}'] for j in range(3)] for i in range(3)]); t = np.array([F[f'sg_t{i}'] for i in range(3)]); keep = conf >= 0.2
        o['scale'], o['nsc'] = scale_from_depth(R, t, p0[keep], p1[keep], depth, thr_px=thr)
    return o


def load(matcher, which, idx):
    """-> list of (k, p0, p1, conf, multi_src) in original pixel coordinates."""
    if matcher == 'sgplain':   # archived plain SuperGlue (1280 x 720, 2048 keypoints, single pass); SuperGlue confidences kept as in the archived run
        d = np.load(f'{RB}/matches_sg_plain/{idx}.npz') if which in ('test', 'train') else np.load(f'{OUT}/matches/sgplain/{which}/{idx}.npz')
        return [(0, d['p0'].astype(np.float64), d['p1'].astype(np.float64), d['conf'].astype(np.float64), None)]
    if matcher == 'sgrot':     # the paper's rotation-aware pass-1 matches (coarse rotation search + fine pass), no learned part
        d = np.load(f'{SGD}/matches_rot/{idx}.npz') if which in ('test', 'train') else np.load(f'{OUT}/matches/sgrot/{which}/{idx}.npz')
        return [(int(d['rot']), d['p0'].astype(np.float64), d['p1'].astype(np.float64), d['conf'].astype(np.float64), d['src'])]
    d = np.load(f'{OUT}/matches/{matcher}/{which}/{idx}.npz'); ks = sorted(int(k[3:]) for k in d.files if k.startswith('p0_'))
    return [(k, d[f'p0_{k}'].astype(np.float64) / 4.0, d[f'p1_{k}'].astype(np.float64) / 4.0, np.ones(len(d[f'p0_{k}'])), None) for k in ks]


def work(a):
    import time
    matcher, which, idx, thrs = a; cv2.setNumThreads(1); rows = []
    try: items = load(matcher, which, idx)
    except Exception as e: return [dict(id=idx, k=0, thr=t, ok=0.0, inl=0.0, n=0.0, err=str(e)[:80]) for t in thrs]
    for k, p0, p1, conf, src in items:
        dep = depth_for(which, idx, k)
        for t in thrs:
            t0 = time.time()
            try: g = solve(p0, p1, conf, dep, t, src)
            except Exception as e: g = dict(ok=0.0, inl=0.0, n=float(len(p0)), err=str(e)[:80])
            g.update(id=idx, k=k, thr=t, sec=time.time() - t0); rows.append(g)
    return rows


def ids_of(matcher, which):
    """Pairs of the set for which this matcher has been run (a long-running matcher may cover a random subset only)."""
    sys.path.append(f'{ROOT}/supp/trackB'); from run_matchers import pair_set
    ids = pair_set(which).id.tolist()
    if matcher in ('sgplain', 'sgrot') and which in ('test', 'train'): return ids
    return [i for i in ids if os.path.exists(f'{OUT}/matches/{matcher}/{which}/{i}.npz')]


if __name__ == '__main__':
    matcher, which = sys.argv[1:3]; thrs = [float(x) for x in sys.argv[3].split(',')]; os.makedirs(f'{OUT}/geom', exist_ok=True); dst = f'{OUT}/geom/{matcher}_{which}.pkl'
    old = pd.read_pickle(dst) if os.path.exists(dst) else None; ids = ids_of(matcher, which)
    done = set() if old is None else set(zip(old.id, old.thr)); jobs = [(matcher, which, int(i), [t for t in thrs if (int(i), t) not in done]) for i in ids]; jobs = [j for j in jobs if j[3]]
    if jobs:
        with Pool(int(os.environ.get('GEOM_PROCS', '3'))) as p: rows = [r for rr in p.imap_unordered(work, jobs, chunksize=4) for r in rr]
        R = pd.DataFrame(rows); R = R if old is None else pd.concat([old, R], ignore_index=True); R.to_pickle(dst)
    else: R = old
    for t, g in R.groupby('thr'): print(f'{matcher} {which} thr {t}: rows {len(g)} ok {g.ok.mean():.3f} median inliers (k=0) {g[g.k == g.k.min()].inl.median():.0f}')
