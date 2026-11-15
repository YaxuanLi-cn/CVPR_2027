#!/usr/bin/env python3
"""Rotation-aware SuperGlue: match image a against image b rotated by k*90deg, keep the best rotation.

Writes one npz per pair with matched keypoints in ORIGINAL pixel coordinates (3686x2073).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2
sys.path.insert(0, '/root/PairUAV/baseline/SuperGlue')
from models.superpoint import SuperPoint
from models.superglue import SuperGlue

DATA = Path('/root/pairUAV-5D-temp'); W0, H0 = 3686, 2073

class Pairs(torch.utils.data.Dataset):
    def __init__(self, df, sizes): self.df = df.reset_index(drop=True); self.sizes = sizes
    def __len__(self): return len(self.df)
    def load(self, rel):
        im = cv2.imread(str(DATA / rel), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        return [torch.from_numpy(cv2.resize(im, s, interpolation=cv2.INTER_AREA)) for s in self.sizes]
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return int(r.manifest_index), r.tour, r.pair_id, self.load(r.image_a), self.load(r.image_b)

def rot_coord_maps(h, w, k, device):
    """For an (h,w) image rotated by torch.rot90(k): maps giving original (x,y) for every rotated pixel."""
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return torch.rot90(xs, k, (0, 1)), torch.rot90(ys, k, (0, 1))

@torch.inference_mode()
def match_rot(sp, sg, ia, ib, rots, feat_a=None):
    """ia, ib: float tensors (1,1,H,W) in [0,1]. Returns list over rots of (kp_a, kp_b_in_unrotated_coords, conf)."""
    if feat_a is None: feat_a = sp({'image': ia})
    res = []
    for k in rots:
        ibr = torch.rot90(ib, k, (2, 3))
        fb = sp({'image': ibr})
        d = {'image0': ia, 'image1': ibr, 'keypoints0': torch.stack(feat_a['keypoints']), 'scores0': torch.stack(feat_a['scores']), 'descriptors0': torch.stack(feat_a['descriptors']),
             'keypoints1': torch.stack(fb['keypoints']), 'scores1': torch.stack(fb['scores']), 'descriptors1': torch.stack(fb['descriptors'])}
        out = sg(d); m = out['matches0'][0]; ok = m > -1
        k0 = d['keypoints0'][0][ok]; k1 = d['keypoints1'][0][m[ok]]; conf = out['matching_scores0'][0][ok]
        if k % 4:
            mx, my = rot_coord_maps(ib.shape[2], ib.shape[3], k, ib.device)
            xi, yi = k1[:, 0].long(), k1[:, 1].long()
            k1 = torch.stack([mx[yi, xi], my[yi, xi]], 1).float()
        res.append((k0, k1, conf))
    return res, feat_a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/pairs.pkl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--coarse', type=int, nargs=2, default=[640, 360]); ap.add_argument('--coarse-kp', type=int, default=768)
    ap.add_argument('--fine', type=int, nargs=2, default=[1024, 576]); ap.add_argument('--fine-kp', type=int, default=2048)
    ap.add_argument('--second-rot-ratio', type=float, default=0.5, help='also run the fine pass on the runner-up rotation if its coarse score is at least this fraction of the best')
    ap.add_argument('--shard', type=int, nargs=2, default=[0, 1]); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--sample-seed', type=int, default=None)
    ap.add_argument('--workers', type=int, default=12); ap.add_argument('--shuffle', action='store_true')
    a = ap.parse_args()
    df = pd.read_pickle(a.pairs)
    if a.sample_seed is not None: df = df.sample(a.limit, random_state=a.sample_seed)
    elif a.limit: df = df.iloc[:a.limit]
    if a.shuffle: df = df.sample(frac=1.0, random_state=0)
    df = df.iloc[a.shard[0]::a.shard[1]]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = df[[not (out / f'{i}.npz').exists() for i in df.manifest_index]]
    print(f'{len(df)} pairs to do', flush=True)
    dev = 'cuda'
    mk_sp = lambda n: SuperPoint({'nms_radius': 4 if n <= 1024 else 3, 'keypoint_threshold': 0.005, 'max_keypoints': n}).eval().to(dev)
    sp_c, sp_f = mk_sp(a.coarse_kp), mk_sp(a.fine_kp)
    sg = SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev)
    dl = torch.utils.data.DataLoader(Pairs(df, [tuple(a.coarse), tuple(a.fine)]), batch_size=None, num_workers=a.workers, prefetch_factor=8)
    sc_f = np.array([W0 / a.fine[0], H0 / a.fine[1]], dtype=np.float32); sc_c = np.array([W0 / a.coarse[0], H0 / a.coarse[1]], dtype=np.float32)
    t0 = time.time()
    for n, (idx, tour, pid, ims_a, ims_b) in enumerate(dl):
        f = lambda t: (t.to(dev).float() / 255.0)[None, None]
        with torch.autocast('cuda', dtype=torch.float16):
            coarse, _ = match_rot(sp_c, sg, f(ims_a[0]), f(ims_b[0]), [0, 1, 2, 3])
            cnt = [int(len(c[0])) for c in coarse]; csum = [float(c[2].sum()) for c in coarse]
            order = np.argsort(csum)[::-1]; kbest = int(order[0])
            ks = [kbest] + ([int(order[1])] if csum[order[1]] >= a.second_rot_ratio * csum[kbest] else [])
            fine, _ = match_rot(sp_f, sg, f(ims_a[1]), f(ims_b[1]), ks)
        # union: fine matches (selected rotations) + coarse matches (all rotations); src: 10+k = fine, k = coarse
        P0, P1, C, S = [], [], [], []
        for k, (k0, k1, conf) in zip(ks, fine):
            P0.append(k0.float().cpu().numpy() * sc_f); P1.append(k1.float().cpu().numpy() * sc_f); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), 10 + k))
        for k, (k0, k1, conf) in enumerate(coarse):
            P0.append(k0.float().cpu().numpy() * sc_c); P1.append(k1.float().cpu().numpy() * sc_c); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), k))
        np.savez(out / f'{idx}.npz', p0=np.concatenate(P0).astype(np.float32), p1=np.concatenate(P1).astype(np.float32), conf=np.concatenate(C).astype(np.float16),
                 src=np.concatenate(S).astype(np.int8), rot=kbest, rot_counts=np.array(cnt, dtype=np.int32), fine_counts=np.array([len(x[0]) for x in fine], dtype=np.int32))
        if (n + 1) % 500 == 0: print(f'{n+1}/{len(df)} {(time.time()-t0)/(n+1):.3f}s/pair', flush=True)
    print('done', flush=True)

if __name__ == '__main__':
    main()
