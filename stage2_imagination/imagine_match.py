#!/usr/bin/env python3
"""Stage 2 / module A: imagine the other view from the stage-1 pose, then compare imagination and reality with SuperGlue.

Imagination: with the calibrated intrinsics, the stage-1 pose (relative heading, both tilts, metric displacement) and the height of the
camera above the ground, the ground-plane-induced homography  H_ab = K (R + t n^T / H) K^-1  renders one image into the other camera.
If the pose is right the imagined and the real image coincide on the ground (roofs keep a small parallax), so
  * corresponding TILES can be cut from both and matched at (up to) native resolution -- impossible without the alignment,
  * the residual displacement of the matches is the imagination-vs-reality discrepancy (the 'MSE' of the reference figure, but sub-pixel
    and robust), used downstream to verify pose candidates and to re-estimate the pose.
Canvas = image a's pixel frame; image b is the one that is re-rendered (so the keypoints of a stay un-interpolated when scale = 1)."""
import argparse, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, '/root/PairUAV/baseline/SuperGlue'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from geom import R_wc
from pose_from_matches import K0
DATA = Path('/root/pairUAV-5D-temp'); W0, H0 = 3686, 2073; CW, CH = 1280, 720

def homography_b2a(r, plane=True):
    """original b px -> original a px, from the stage-1 pose. plane=False: rotation only (what the stage-1 zoom pass used)."""
    Ra = R_wc(0.0, r.g_ta); R = R_wc(r.g_dh, r.g_tb).T @ Ra                       # X_b = R X_a + t
    Hab = R
    if plane:
        t = -R @ (Ra.T @ np.array([r.p_lx, r.p_ly, r.p_z], float)); n_down = np.array([0.0, math.sin(math.radians(r.g_ta)), math.cos(math.radians(r.g_ta))])
        Hab = R + np.outer(t, n_down) / r.g_H
    return np.linalg.inv(K0 @ Hab @ np.linalg.inv(K0))

class Pairs(torch.utils.data.Dataset):
    def __init__(self, df, a): self.df = df.reset_index(drop=True); self.a = a
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        cv2.setNumThreads(1); r = self.df.iloc[i]; a = self.a; Hb2a = homography_b2a(r, not a.rot_only)
        cor = cv2.perspectiveTransform(np.float32([[[0, 0], [W0, 0], [W0, H0], [0, H0]]]), Hb2a)[0]
        poly = np.zeros((H0 // 8, W0 // 8), np.uint8)
        if np.isfinite(cor).all() and np.abs(cor).max() < 1e6: cv2.fillConvexPoly(poly, np.round(cor / 8).astype(np.int32), 255)
        ys, xs = np.nonzero(poly)
        if len(xs) < 20: x0, x1, y0, y1 = 0, W0, 0, H0
        else: x0, x1, y0, y1 = xs.min() * 8 - a.pad, xs.max() * 8 + a.pad, ys.min() * 8 - a.pad, ys.max() * 8 + a.pad
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W0), min(y1, H0); w, h = max(x1 - x0, 640), max(y1 - y0, 360)
        if w / h > CW / CH: h = w * CH / CW                                    # grow to 16:9
        else: w = h * CW / CH
        w, h = min(w, W0), min(h, H0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0 = float(np.clip(cx - w / 2, 0, W0 - w)); y0 = float(np.clip(cy - h / 2, 0, H0 - h))
        s = max(CW / w, a.scale); n = max(1, math.ceil(w * s / CW - 0.08)); n = min(n, a.max_grid); s = min(s, n * CW / w)   # canvas px per original px; n x n tiles of CWxCH
        ia = cv2.imread(str(DATA / r.image_a), cv2.IMREAD_GRAYSCALE); ib = cv2.imread(str(DATA / r.image_b), cv2.IMREAD_GRAYSCALE); ones = np.full(ib.shape, 255, np.uint8)
        TA, TB, MK, ZI, ZHI = [], [], [], [], []
        for ty in range(n):
            for tx in range(n):
                ox = x0 * s + (0 if n == 1 else tx * (w * s - CW) / (n - 1)); oy = y0 * s + (0 if n == 1 else ty * (h * s - CH) / (n - 1))
                Z = np.array([[s, 0, -ox], [0, s, -oy], [0, 0, 1.0]])
                mask = cv2.warpPerspective(ones, Z @ Hb2a, (CW, CH), flags=cv2.INTER_NEAREST)
                if (mask > 0).mean() < 0.03: continue
                TA.append(cv2.warpPerspective(ia, Z, (CW, CH), flags=cv2.INTER_AREA if s < 0.7 else cv2.INTER_LINEAR)); TB.append(cv2.warpPerspective(ib, Z @ Hb2a, (CW, CH), flags=cv2.INTER_LINEAR))
                MK.append(cv2.erode(mask, np.ones((17, 17), np.uint8))); ZI.append(np.linalg.inv(Z)); ZHI.append(np.linalg.inv(Z @ Hb2a))
        if not TA: return int(r.manifest_index), None, None, None, None, None, float(s), n
        return int(r.manifest_index), torch.from_numpy(np.stack(TA)), torch.from_numpy(np.stack(TB)), torch.from_numpy(np.stack(MK)), np.stack(ZI), np.stack(ZHI), float(s), n

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--guide', default='/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/stage2/guide.pkl'); ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default=None, help='npy file with the manifest indices to process'); ap.add_argument('--pad', type=int, default=150); ap.add_argument('--kp', type=int, default=2048)
    ap.add_argument('--scale', type=float, default=1.0, help='minimum canvas px per original px (1 = native resolution)'); ap.add_argument('--max-grid', type=int, default=3)
    ap.add_argument('--rot-only', action='store_true', help='ablation: rotation-only alignment (no ground-plane imagination)')
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1]); ap.add_argument('--workers', type=int, default=5); a = ap.parse_args()
    df = pd.read_pickle(a.guide)
    if a.ids: df = df[df.manifest_index.isin(set(np.load(a.ids).tolist()))]
    df = df.sample(frac=1.0, random_state=1).iloc[a.shard[0]::a.shard[1]]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = df[[not (out / f'{i}.npz').exists() for i in df.manifest_index]]; print(f'{len(df)} pairs to do', flush=True)
    dev = 'cuda'; sp = SuperPoint({'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': a.kp}).eval().to(dev)
    sg = SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev)
    dl = torch.utils.data.DataLoader(Pairs(df, a), batch_size=None, num_workers=a.workers, prefetch_factor=3); t0 = time.time()
    def back(k, Hm):
        if not len(k): return k
        p = np.c_[k, np.ones(len(k))] @ np.asarray(Hm).T; return p[:, :2] / p[:, 2:3]
    for nn_, (idx, TA, TB, MK, ZI, ZHI, s, n) in enumerate(dl):
        P0, P1, CF, TI, DS = [np.zeros((0, 2))], [np.zeros((0, 2))], [np.zeros(0)], [np.zeros(0)], [np.zeros((0, 2))]
        for ti in range(0 if TA is None else len(TA)):
            f = lambda t: (t.to(dev).float() / 255.0)[None, None]; ia, ib = f(TA[ti]), f(TB[ti]); mask = MK[ti].to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                fa = sp({'image': ia}); fb = sp({'image': ib}); kb = fb['keypoints'][0]; keep = mask[kb[:, 1].long(), kb[:, 0].long()] > 0
                if keep.sum() < 2 or len(fa['keypoints'][0]) < 2: continue
                d = {'image0': ia, 'image1': ib, 'keypoints0': fa['keypoints'][0][None], 'scores0': fa['scores'][0][None], 'descriptors0': fa['descriptors'][0][None],
                     'keypoints1': kb[keep][None], 'scores1': fb['scores'][0][keep][None], 'descriptors1': fb['descriptors'][0][:, keep][None]}
                o = sg(d); m = o['matches0'][0]; ok = m > -1
            k0 = d['keypoints0'][0][ok].float().cpu().numpy().astype(np.float64); k1 = d['keypoints1'][0][m[ok]].float().cpu().numpy().astype(np.float64)
            P0.append(back(k0, ZI[ti])); P1.append(back(k1, ZHI[ti])); CF.append(o['matching_scores0'][0][ok].float().cpu().numpy()); TI.append(np.full(len(k0), ti)); DS.append((k1 - k0) / s)
        np.savez(out / f'{idx}.npz', p0=np.concatenate(P0).astype(np.float32), p1=np.concatenate(P1).astype(np.float32), conf=np.concatenate(CF).astype(np.float16),
                 tile=np.concatenate(TI).astype(np.int8), disp=np.concatenate(DS).astype(np.float32), zoom=float(s), grid=int(n), ntiles=0 if TA is None else len(TA))
        if (nn_ + 1) % 100 == 0: print(f'{nn_+1}/{len(df)} {(time.time()-t0)/(nn_+1):.3f}s/pair', flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
