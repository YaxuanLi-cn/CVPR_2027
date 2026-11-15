#!/usr/bin/env python3
"""Pass 2 for weak pairs: pose-guided ZOOM re-matching.
Weak pairs are the ones whose view frusta overlap only slightly, so most of the 2048-keypoint budget is wasted outside the overlap.
Using the pass-1/MAP pose (or the MLP prior) we predict the overlap region, crop BOTH images to it at up to native resolution
(image b additionally warped into image a's orientation by the rotation-only homography), match there, and map the keypoints back."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, '/root/PairUAV/baseline/SuperGlue'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from geom import R_wc
from pose_from_matches import K0
DATA = Path('/root/pairUAV-5D-temp'); W0, H0 = 3686, 2073; CW, CH = 1280, 720

def guide_rotation(r):
    if r.map_ok == 1: return R_wc(r.map_dh, r.map_tb).T @ R_wc(0.0, r.map_ta), 2
    return R_wc(r.n_heading, float(np.clip(r.n_tilt_b, 0, 50))).T @ R_wc(0.0, float(np.clip(r.n_tilt_a, 0, 50))), 0

class Pairs(torch.utils.data.Dataset):
    def __init__(self, df, pad): self.df = df.reset_index(drop=True); self.pad = pad
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        cv2.setNumThreads(1); r = self.df.iloc[i]; R, src = guide_rotation(r)
        Hb2a = K0 @ R.T @ np.linalg.inv(K0)                                   # original b px -> original a px (rotation-only)
        cor = cv2.perspectiveTransform(np.float32([[[0, 0], [W0, 0], [W0, H0], [0, H0]]]), Hb2a)[0]
        poly = np.zeros((H0 // 8, W0 // 8), np.uint8); cv2.fillConvexPoly(poly, np.round(cor / 8).astype(np.int32), 255)
        ys, xs = np.nonzero(poly)
        if len(xs) < 20: x0, x1, y0, y1 = 0, W0, 0, H0
        else: x0, x1, y0, y1 = xs.min() * 8 - self.pad, xs.max() * 8 + self.pad, ys.min() * 8 - self.pad, ys.max() * 8 + self.pad
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W0), min(y1, H0); w, h = max(x1 - x0, 640), max(y1 - y0, 360)
        if w / h > CW / CH: h = w * CH / CW                                    # grow to 16:9
        else: w = h * CW / CH
        w, h = min(w, W0), min(h, H0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0 = float(np.clip(cx - w / 2, 0, W0 - w)); y0 = float(np.clip(cy - h / 2, 0, H0 - h)); s = CW / w    # zoom: original a px -> canvas px
        Z = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1.0]])
        ia = cv2.imread(str(DATA / r.image_a), cv2.IMREAD_GRAYSCALE); ib = cv2.imread(str(DATA / r.image_b), cv2.IMREAD_GRAYSCALE)
        interp = cv2.INTER_AREA if s < 0.7 else cv2.INTER_LINEAR
        za = cv2.warpPerspective(ia, Z, (CW, CH), flags=interp); zb = cv2.warpPerspective(ib, Z @ Hb2a, (CW, CH), flags=cv2.INTER_LINEAR)
        mask = cv2.erode(cv2.warpPerspective(np.full(ib.shape, 255, np.uint8), Z @ Hb2a, (CW, CH), flags=cv2.INTER_NEAREST), np.ones((17, 17), np.uint8))
        return int(r.manifest_index), torch.from_numpy(za), torch.from_numpy(zb), torch.from_numpy(mask), np.linalg.inv(Z), np.linalg.inv(Z @ Hb2a), src, float(s)

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--geom', required=True); ap.add_argument('--map', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-inl', type=float, default=150); ap.add_argument('--min-inl', type=float, default=-1); ap.add_argument('--pad', type=int, default=300); ap.add_argument('--kp', type=int, default=2048)
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1]); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args(); SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'
    df = pd.read_pickle(f'{SGD}/pairs.pkl').merge(pd.read_pickle(a.geom), on='manifest_index').merge(pd.read_pickle(a.map), on='manifest_index').merge(pd.read_pickle(f'{SGD}/stage0_mlp.pkl'), on='manifest_index')
    df = df[(df.sg_inl.fillna(0) < a.max_inl) & (df.sg_inl.fillna(0) >= a.min_inl)].sample(frac=1.0, random_state=1)
    if a.limit: df = df.iloc[:a.limit]
    df = df.iloc[a.shard[0]::a.shard[1]]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = df[[not (out / f'{i}.npz').exists() for i in df.manifest_index]]; print(f'{len(df)} weak pairs to do', flush=True)
    dev = 'cuda'; sp = SuperPoint({'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': a.kp}).eval().to(dev)
    sg = SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev)
    dl = torch.utils.data.DataLoader(Pairs(df, a.pad), batch_size=None, num_workers=a.workers, prefetch_factor=4); t0 = time.time()
    for n, (idx, ia, ib, mask, Zi, ZHi, src, s) in enumerate(dl):
        f = lambda t: (t.to(dev).float() / 255.0)[None, None]; ia, ib = f(ia), f(ib); mask = mask.to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            fa = sp({'image': ia}); fb = sp({'image': ib}); kb = fb['keypoints'][0]; keep = mask[kb[:, 1].long(), kb[:, 0].long()] > 0
            if keep.sum() >= 2:
                d = {'image0': ia, 'image1': ib, 'keypoints0': fa['keypoints'][0][None], 'scores0': fa['scores'][0][None], 'descriptors0': fa['descriptors'][0][None],
                     'keypoints1': kb[keep][None], 'scores1': fb['scores'][0][keep][None], 'descriptors1': fb['descriptors'][0][:, keep][None]}
                o = sg(d); m = o['matches0'][0]; ok = m > -1
                k0 = d['keypoints0'][0][ok].float().cpu().numpy().astype(np.float64); k1 = d['keypoints1'][0][m[ok]].float().cpu().numpy().astype(np.float64); conf = o['matching_scores0'][0][ok].float().cpu().numpy()
            else: k0 = np.zeros((0, 2)); k1 = np.zeros((0, 2)); conf = np.zeros(0, np.float32)
        def back(k, Hm):
            if not len(k): return k
            p = np.c_[k, np.ones(len(k))] @ np.asarray(Hm).T; return p[:, :2] / p[:, 2:3]
        np.savez(out / f'{idx}.npz', p0=back(k0, Zi).astype(np.float32), p1=back(k1, ZHi).astype(np.float32), conf=conf.astype(np.float16), guide_src=int(src), zoom=float(s))
        if (n + 1) % 200 == 0: print(f'{n+1}/{len(df)} {(time.time()-t0)/(n+1):.3f}s/pair', flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
