#!/usr/bin/env python3
"""Track B: SuperGlue on the real pairs, in the two configurations of the archived PairUAV baselines.
  sgplain  off-the-shelf SuperGlue (outdoor, 1280 x 720, 2048 keypoints, single pass)                     = reviewer_baselines/rb_sg.py
  sgrot    the paper's rotation-aware pass-1 matching (4-rotation search at 640 x 360, fine pass at 1280 x 720) = engine._pass1, no learned part
Usage: run_sg_real.py <real_in|real_out>"""
import sys, os, numpy as np, pandas as pd, torch
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion']; sys.path.append(f'{ROOT}/supp/trackB')
import run_superglue_rot as rs
OUT = f'{ROOT}/outputs/supp/trackB'; which = sys.argv[1]; D = pd.read_pickle('/root/realdata/mezieres/pred.pkl'); d = D[(D.dist_equiv <= 38) if which == 'real_in' else (D.dist_equiv > 38)]
T = pd.DataFrame(dict(manifest_index=d.index.values, tour='', pair_id='', image_a=d.image_a.values, image_b=d.image_b.values))
for n in ('sgplain', 'sgrot'): os.makedirs(f'{OUT}/matches/{n}/{which}', exist_ok=True)
T = T[[not (os.path.exists(f'{OUT}/matches/sgplain/{which}/{i}.npz') and os.path.exists(f'{OUT}/matches/sgrot/{which}/{i}.npz')) for i in T.manifest_index]]; print(which, len(T), 'pairs to do', flush=True)
if not len(T): sys.exit(0)
dev = 'cuda'; mk = lambda n: rs.SuperPoint({'nms_radius': 4 if n <= 1024 else 3, 'keypoint_threshold': 0.005, 'max_keypoints': n}).eval().to(dev); sp_c, sp_f = mk(768), mk(2048)
sg = rs.SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev); coarse, fine = (640, 360), (1280, 720)
sc_f = np.array([rs.W0 / fine[0], rs.H0 / fine[1]], np.float32); sc_c = np.array([rs.W0 / coarse[0], rs.H0 / coarse[1]], np.float32)
dl = torch.utils.data.DataLoader(rs.Pairs(T, [coarse, fine]), batch_size=None, num_workers=2, prefetch_factor=2)
for idx, tour, pid, a, b in dl:
    f = lambda t: (t.to(dev).float() / 255.0)[None, None]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        (k0, k1, conf), = rs.match_rot(sp_f, sg, f(a[1]), f(b[1]), [0])[0]
        np.savez(f'{OUT}/matches/sgplain/{which}/{idx}.npz', p0=k0.float().cpu().numpy() * sc_f, p1=k1.float().cpu().numpy() * sc_f, conf=conf.float().cpu().numpy())
        cs, _ = rs.match_rot(sp_c, sg, f(a[0]), f(b[0]), [0, 1, 2, 3]); cnt = [int(len(c[0])) for c in cs]; csum = [float(c[2].sum()) for c in cs]; order = np.argsort(csum)[::-1]; kbest = int(order[0])
        ks = [kbest] + ([int(order[1])] if csum[order[1]] >= 0.5 * csum[kbest] else []); fn, _ = rs.match_rot(sp_f, sg, f(a[1]), f(b[1]), ks)
    P0, P1, C, S = [], [], [], []
    for k, (k0, k1, conf) in zip(ks, fn): P0.append(k0.float().cpu().numpy() * sc_f); P1.append(k1.float().cpu().numpy() * sc_f); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), 10 + k))
    for k, (k0, k1, conf) in enumerate(cs): P0.append(k0.float().cpu().numpy() * sc_c); P1.append(k1.float().cpu().numpy() * sc_c); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), k))
    np.savez(f'{OUT}/matches/sgrot/{which}/{idx}.npz', p0=np.concatenate(P0).astype(np.float32), p1=np.concatenate(P1).astype(np.float32), conf=np.concatenate(C).astype(np.float16), src=np.concatenate(S).astype(np.int8), rot=kbest, rot_counts=np.array(cnt, dtype=np.int32))
print('done', which, len(T))
