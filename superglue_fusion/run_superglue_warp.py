#!/usr/bin/env python3
"""Pass 2: pose-guided SuperGlue. Warp image b into image a's orientation with the rotation-only homography
H = K R^T K^-1 (R from pass-1 geometry, or from the MASt3R-feature MLP when pass 1 is weak), so the two images
are aligned up to small parallax; match; map the keypoints back to original image-b pixels."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, '/root/PairUAV/baseline/SuperGlue'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from geom import R_wc
from pose_from_matches import K0
DATA = Path('/root/pairUAV-5D-temp'); W0, H0 = 3686, 2073

def guide_rotation(r):
    """Rotation a->b used for the warp, and which source it came from (1 = pass-1 SuperGlue geometry, 0 = MLP prior)."""
    dh_gap = abs((r.sg_dh - r.n_heading + 180) % 360 - 180) if r.sg_ok == 1 else 999
    if r.sg_ok == 1 and (r.sg_inl >= 150 or (r.sg_inl >= 30 and dh_gap < 10)):
        return np.array([[r[f'sg_R{i}{j}'] for j in range(3)] for i in range(3)]), 1
    return R_wc(r.n_heading, float(np.clip(r.n_tilt_b, 0, 50))).T @ R_wc(0.0, float(np.clip(r.n_tilt_a, 0, 50))), 0

class Pairs(torch.utils.data.Dataset):
    def __init__(self, df, size): self.df = df.reset_index(drop=True); self.size = size
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        cv2.setNumThreads(1); r = self.df.iloc[i]; w, h = self.size
        ia = cv2.resize(cv2.imread(str(DATA / r.image_a), cv2.IMREAD_REDUCED_GRAYSCALE_2), (w, h), interpolation=cv2.INTER_AREA)
        ib = cv2.imread(str(DATA / r.image_b), cv2.IMREAD_REDUCED_GRAYSCALE_2); hb, wb = ib.shape
        R, src = guide_rotation(r)
        Sa = np.diag([w / W0, h / H0, 1.0]); Sb = np.diag([wb / W0, hb / H0, 1.0])
        Hm = Sa @ K0 @ R.T @ np.linalg.inv(K0) @ np.linalg.inv(Sb)          # half-res b pixels -> fine-res a-frame pixels
        wb_img = cv2.warpPerspective(ib, Hm, (w, h), flags=cv2.INTER_AREA if False else cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((hb, wb), 255, np.uint8), Hm, (w, h), flags=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((17, 17), np.uint8))
        Hback = K0 @ R @ np.linalg.inv(K0) @ np.linalg.inv(Sa)               # fine-res a-frame pixels -> ORIGINAL b pixels
        return int(r.manifest_index), torch.from_numpy(ia), torch.from_numpy(wb_img), torch.from_numpy(mask), Hback.astype(np.float64), src, float((mask > 0).mean())

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--geom', required=True); ap.add_argument('--out', required=True); ap.add_argument('--fine', type=int, nargs=2, default=[1280, 720])
    ap.add_argument('--kp', type=int, default=2048); ap.add_argument('--shard', type=int, nargs=2, default=[0, 1]); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args(); SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'
    df = pd.read_pickle(f'{SGD}/pairs.pkl').merge(pd.read_pickle(a.geom), on='manifest_index').merge(pd.read_pickle(f'{SGD}/stage0_mlp.pkl'), on='manifest_index')
    df = df.sample(frac=1.0, random_state=1)
    if a.limit: df = df.iloc[:a.limit]
    df = df.iloc[a.shard[0]::a.shard[1]]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = df[[not (out / f'{i}.npz').exists() for i in df.manifest_index]]; print(f'{len(df)} pairs to do', flush=True)
    dev = 'cuda'; sp = SuperPoint({'nms_radius': 3, 'keypoint_threshold': 0.005, 'max_keypoints': a.kp}).eval().to(dev)
    sg = SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev)
    dl = torch.utils.data.DataLoader(Pairs(df, tuple(a.fine)), batch_size=None, num_workers=a.workers, prefetch_factor=8)
    sc = np.array([W0 / a.fine[0], H0 / a.fine[1]]); t0 = time.time()
    for n, (idx, ia, ib, mask, Hback, src, cover) in enumerate(dl):
        f = lambda t: (t.to(dev).float() / 255.0)[None, None]; ia, ib = f(ia), f(ib); mask = mask.to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            fa = sp({'image': ia}); fb = sp({'image': ib})
            kb = fb['keypoints'][0]; keep = mask[kb[:, 1].long(), kb[:, 0].long()] > 0
            if keep.sum() >= 2:
                d = {'image0': ia, 'image1': ib, 'keypoints0': fa['keypoints'][0][None], 'scores0': fa['scores'][0][None], 'descriptors0': fa['descriptors'][0][None],
                     'keypoints1': kb[keep][None], 'scores1': fb['scores'][0][keep][None], 'descriptors1': fb['descriptors'][0][:, keep][None]}
                o = sg(d); m = o['matches0'][0]; ok = m > -1
                k0 = d['keypoints0'][0][ok].float().cpu().numpy().astype(np.float64); k1 = d['keypoints1'][0][m[ok]].float().cpu().numpy().astype(np.float64); conf = o['matching_scores0'][0][ok].float().cpu().numpy()
            else:
                k0 = np.zeros((0, 2)); k1 = np.zeros((0, 2)); conf = np.zeros(0, np.float32)
        Hb = np.asarray(Hback); pb = np.c_[k1, np.ones(len(k1))] @ Hb.T; pb = pb[:, :2] / pb[:, 2:3] if len(k1) else k1
        np.savez(out / f'{idx}.npz', p0=(k0 * sc).astype(np.float32), p1=pb.astype(np.float32), conf=conf.astype(np.float16), guide_src=int(src), cover=float(cover), k0w=k0.astype(np.float32), k1w=k1.astype(np.float32))
        if (n + 1) % 500 == 0: print(f'{n+1}/{len(df)} {(time.time()-t0)/(n+1):.3f}s/pair', flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
