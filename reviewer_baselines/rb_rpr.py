#!/usr/bin/env python3
"""Reviewer baselines, part 4: an end-to-end relative-pose REGRESSION network of the kind used in cross-view localisation
(Siamese encoder -> dense correlation between the two views -> soft correspondence field -> pose regression), trained on the
PairUAV training tours. The cited ground-to-satellite methods cannot be run as they are (they need a satellite map, a
coarse pose prior and a known ground camera), so this is a re-implementation of their shared recipe for UAV-to-UAV pairs.

Outputs: d_heading (sin/cos), tilt of each camera, displacement in the source camera's levelled frame (right, forward, up).
Usage: rb_rpr.py [minutes]      (training time budget; preliminary numbers only)
"""
import sys, os, time, math
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
from fusion_lib import add_targets
from rb_recon import subset, SG, OUT, DATA
W, H = 384, 216; CACHE = '/dev/shm/rb_rpr_images.npy'

def load(p):
    im = cv2.imread(f'{DATA}/{p}', cv2.IMREAD_REDUCED_COLOR_4); return cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def main(minutes=25.0):
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); T = add_targets(subset(1000, split='test')); tr = P[P.split == 'train'].reset_index(drop=True)
    names = sorted(set(tr.image_a) | set(tr.image_b) | set(T.image_a) | set(T.image_b)); pos = {n: i for i, n in enumerate(names)}
    if os.path.exists(CACHE) and np.load(CACHE, mmap_mode='r').shape[0] == len(names): X = np.load(CACHE, mmap_mode='r')
    else:
        X = np.lib.format.open_memmap(CACHE, mode='w+', dtype=np.uint8, shape=(len(names), H, W, 3)); t0 = time.time()
        with Pool(12) as pool:
            for i, im in enumerate(pool.imap(load, names, chunksize=64)):
                X[i] = im
                if i % 10000 == 0: print(f'images {i}/{len(names)} {time.time()-t0:.0f}s', flush=True)
        X.flush()
    import torch, torch.nn as nn, torch.nn.functional as F, timm
    dev = 'cuda'; torch.manual_seed(0); np.random.seed(0)
    def targets(d): return np.c_[np.sin(np.deg2rad(d.heading)), np.cos(np.deg2rad(d.heading)), d.tilt_a / 45, d.tilt_b / 45, d.lx / 10, d.ly / 10, d.z / 15].astype(np.float32)
    tours = np.array(sorted(tr.tour.unique())); val_t = set(np.random.default_rng(0).choice(tours, 30, replace=False)); va = tr[tr.tour.isin(val_t)].reset_index(drop=True); tr = tr[~tr.tour.isin(val_t)].reset_index(drop=True)
    ix = lambda d: (np.array([pos[n] for n in d.image_a]), np.array([pos[n] for n in d.image_b])); print(f'train pairs {len(tr)}, val {len(va)}, test {len(T)}, images {len(names)}', flush=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
    def batch(ii): ii = np.sort(ii); return ((torch.from_numpy(X[ii]).to(dev).permute(0, 3, 1, 2).float() - mean) / std)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.enc = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, features_only=True, out_indices=(2,)); s.proj = nn.Conv2d(384, 256, 1); s.tau = nn.Parameter(torch.tensor(20.0))
            s.cnv = nn.Sequential(nn.Conv2d(256 + 5, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, 256, 3, padding=1), nn.GELU())
            s.mlp = nn.Sequential(nn.Linear(4 * 256, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 7))
        def field(s, fa, fb, grid):
            """Soft correspondence of every location of a in b: expected coordinates, peak probability, own coordinates."""
            B, C, h, w = fa.shape; c = torch.einsum('bci,bcj->bij', F.normalize(fa, dim=1).flatten(2), F.normalize(fb, dim=1).flatten(2)) * s.tau; p = c.float().softmax(-1)
            e = (p @ grid).transpose(1, 2).reshape(B, 2, h, w); mx = p.max(-1).values.reshape(B, 1, h, w); own = grid.t().reshape(1, 2, h, w).expand(B, -1, -1, -1)
            return s.cnv(torch.cat([fa, e, mx, own], 1)).mean((2, 3))
        def forward(s, a, b):
            f = s.proj(s.enc(torch.cat([a, b]))[0]); fa, fb = f.chunk(2); h, w = fa.shape[-2:]
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=a.device), torch.linspace(-1, 1, w, device=a.device), indexing='ij'); grid = torch.stack([xs.flatten(), ys.flatten()], 1)
            return s.mlp(torch.cat([s.field(fa, fb, grid), s.field(fb, fa, grid), fa.mean((2, 3)), fb.mean((2, 3))], 1).float())

    net = Net().to(dev).to(memory_format=torch.channels_last); bs = 48; ia, ib = ix(tr); Y = torch.tensor(targets(tr), device=dev); n = len(tr)
    opt = torch.optim.AdamW([{'params': net.enc.parameters(), 'lr': 1e-4}, {'params': [p for k, p in net.named_parameters() if not k.startswith('enc.')], 'lr': 4e-4}], weight_decay=0.05)
    def fwd(ja, jb):
        A = torch.from_numpy(X[np.r_[ja, jb]]).to(dev).permute(0, 3, 1, 2).float(); A = ((A - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.bfloat16): return net(A[:len(ja)], A[len(ja):])
    def predict(d):
        net.eval(); ja, jb = ix(d); o = []
        with torch.inference_mode():
            for i in range(0, len(d), 128): o.append(fwd(ja[i:i + 128], jb[i:i + 128]).float().cpu().numpy())
        net.train(); o = np.concatenate(o); return pd.DataFrame(dict(manifest_index=d.manifest_index.values, dh=np.rad2deg(np.arctan2(o[:, 0], o[:, 1])), ta=o[:, 2] * 45, tb=o[:, 3] * 45, lx=o[:, 4] * 10, ly=o[:, 5] * 10, z=o[:, 6] * 15))
    def mae(d, p): w = lambda a: (a + 180) % 360 - 180; return dict(heading=float(np.abs(w(p.dh.values - d.heading.values)).mean()), tilt=float(np.abs(p.tb.values - p.ta.values - d.tilt.values).mean()), lx=float(np.abs(p.lx.values - d.lx.values).mean()), ly=float(np.abs(p.ly.values - d.ly.values).mean()), z=float(np.abs(p.z.values - d.z.values).mean()))
    # time-boxed schedule: measure the speed, then fix the number of steps for a cosine schedule
    t0 = time.time(); total = None; step = 0; base = [g['lr'] for g in opt.param_groups]; ep = 0
    while total is None or step < total:
        perm = np.random.permutation(n)
        for i in range(0, n - bs + 1, bs):
            j = perm[i:i + bs]; lr = min(1.0, (step + 1) / 300) * (0.5 * (1 + math.cos(math.pi * step / total)) if total else 1.0)
            for g, b0 in zip(opt.param_groups, base): g['lr'] = b0 * lr
            loss = F.smooth_l1_loss(fwd(ia[j], ib[j]).float(), Y[j], beta=0.1); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); step += 1
            if step == 100: rate = 100 / (time.time() - t0); total = int(min(minutes * 60 * rate, 30 * n / bs)); print(f'{rate:.1f} it/s -> {total} steps = {total * bs / n:.1f} epochs', flush=True)
            if total and step >= total: break
            if step % 200 == 0: print(f'step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s', flush=True)
        ep += 1; print(f'epoch {ep} val {mae(va, predict(va))} | test {mae(T, predict(T))}', flush=True)
    Pd = predict(T); Pd['epochs'] = step * bs / n; Pd['minutes'] = (time.time() - t0) / 60; Pd.to_pickle(f'{OUT}/rpr_test.pkl'); torch.save(net.state_dict(), f'{OUT}/rpr.pt'); print('final test', mae(T, Pd), f'{(time.time()-t0)/60:.1f} min', flush=True)

if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 25.0)
