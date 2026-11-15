#!/usr/bin/env python3
"""Stage 2 / module B: absolute heading from an IMAGINED canonical bird's-eye view.

Imagination step: the tilted image is re-rendered as the view of a virtual nadir camera at the same position
(rotation-only homography -> exact for any scene depth). The tilt comes from the label for train tours and from the
SuperGlue geometry (X-Z-X decomposition of the relative rotation, fused in stage 1) for val/test pairs.
In the nadir view a heading change of delta is exactly an in-plane rotation, so
  * training uses exact continuous-rotation augmentation,
  * inference verifies K heading candidates: for each candidate rotation the imagined view is scored by the network and the
    von Mises outputs are combined.
Everything (rectification + rotation + scale + shift) is one grid_sample on the GPU from the cached 448x256 images."""
import argparse, math, time, threading, queue, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--model', default='convnext_small.fb_in22k_ft_in1k'); ap.add_argument('--tag', default='bev_s'); ap.add_argument('--S', type=int, default=256)
ap.add_argument('--gsd', type=float, default=0.6, help='metres per BEV pixel assuming the camera is 212 m above ground'); ap.add_argument('--tilt-noise', type=float, default=1.0)
ap.add_argument('--no-rectify', action='store_true', help='ablation: keep the tilt (rotation augmentation only)'); ap.add_argument('--krot', type=int, default=12)
ap.add_argument('--eval-only', action='store_true'); ap.add_argument('--train-frac', type=float, default=1.0); a = ap.parse_args()
SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; D = f'{SGD}/absheading'; OUT = f'{SGD}/stage2'
m1 = pd.read_pickle(f'{D}/images.pkl'); m2 = pd.read_pickle(f'{D}/images2.pkl'); I1 = np.load(f'{D}/images_448x256.npy', mmap_mode='r'); I2 = np.load(f'{D}/images2_448x256.npy', mmap_mode='r')
rng = np.random.default_rng(0); tr_tours = np.array(sorted(m1[m1.split == 'train'].tour.unique())); val_tours = set(rng.choice(tr_tours, len(tr_tours) // 10, replace=False))  # same split as v1/v2
m1['part'] = np.where(m1.split == 'test', 'test', np.where(m1.tour.isin(val_tours), 'val', 'train')); m2['part'] = np.where(m2.tour.isin(val_tours), 'val2', 'train')
meta = pd.concat([m1.assign(src=0, row=np.arange(len(m1))), m2.assign(src=1, row=np.arange(len(m2)))], ignore_index=True)
idx_tr = np.where(meta.part == 'train')[0]
if a.train_frac < 1: idx_tr = np.random.default_rng(1).choice(idx_tr, int(len(idx_tr) * a.train_frac), replace=False)
print(meta.part.value_counts().to_dict(), 'train images used', len(idx_tr), flush=True)
dev = 'cuda'; mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
HEAD = np.deg2rad(meta.heading.values).astype(np.float32); TILT = np.deg2rad(meta.tilt.values).astype(np.float32); SRC = meta.src.values; ROW = meta.row.values; S = a.S
WC, HC = 448, 256; FX, FY, CX, CY = 3324.9 * WC / 3686, 3324.9 * HC / 2073, 1841.57 * WC / 3686, 1078.04 * HC / 2073   # intrinsics of the cached (anisotropically resized) image
FV = 212.0 / a.gsd                                                                                                    # virtual nadir camera focal length in BEV px

def raw_np(ix): return np.stack([(I1 if SRC[i] == 0 else I2)[ROW[i]] for i in ix])

def R_wc(h, t):
    """camera (x right, y down, z fwd) -> world ENU; heading h clockwise from north, tilt t (0 = nadir). Batched torch version of geom.R_wc."""
    z = torch.zeros_like(h); right = torch.stack([torch.cos(h), -torch.sin(h), z], 1)
    fwd = torch.stack([torch.sin(t) * torch.sin(h), torch.sin(t) * torch.cos(h), -torch.cos(t)], 1)
    up = torch.stack([torch.cos(t) * torch.sin(h), torch.cos(t) * torch.cos(h), torch.sin(t)], 1)
    return torch.stack([right, -up, fwd], 2)

def imagine_bev(x, tilt, delta, flip=None, scale=None, shift=None):
    """x: Bx3xHCxWC cached images. Returns the imagined view of a virtual nadir camera whose heading is (true heading + delta), centred where
    the real optical axis meets the ground. flip: bool tensor, mirror the SOURCE image first (heading -> -heading)."""
    B = len(x); z = torch.zeros(B, device=dev)
    if flip is not None: x = torch.where(flip.view(B, 1, 1, 1), torch.flip(x, dims=[3]), x)
    cx = torch.full((B,), CX, device=dev) if flip is None else torch.where(flip, WC - CX, CX)
    Rt = R_wc(z, z if a.no_rectify else tilt); Rn = R_wc(delta, z)
    M = Rt.transpose(1, 2) @ Rn                                   # ray in virtual nadir camera -> ray in the real camera
    g = Rn.transpose(1, 2) @ Rt[:, :, 2:3]; c = g[:, :2, 0] / g[:, 2:3, 0]   # where the real optical axis lands in the virtual image (normalised coords)
    if shift is not None: c = c + shift
    fv = FV * (torch.ones(B, device=dev) if a.no_rectify else torch.cos(tilt)) * (1.0 if scale is None else scale)   # coarser GSD for oblique views (their ground resolution is lower, and the window then covers more)
    u = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5 - S / 2); yy, xx = torch.meshgrid(u, u, indexing='ij')
    rn = torch.stack([xx[None] / fv.view(B, 1, 1) + c[:, 0].view(B, 1, 1), yy[None] / fv.view(B, 1, 1) + c[:, 1].view(B, 1, 1), torch.ones(B, S, S, device=dev)], -1)
    rc = rn @ M.transpose(1, 2).unsqueeze(1); zc = rc[..., 2].clamp(min=1e-6)
    px = FX * rc[..., 0] / zc + cx.view(B, 1, 1); py = FY * rc[..., 1] / zc + CY
    grid = torch.stack([2 * px / WC - 1, 2 * py / HC - 1], -1); grid = torch.where((rc[..., 2:3] > 1e-6), grid, torch.full_like(grid, -5.0))
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model(a.model, pretrained=True, num_classes=0, drop_path_rate=0.2); s.h = nn.Linear(s.b.num_features, 3)
    def forward(s, x):
        o = s.h(s.b(x)); return F.normalize(o[:, :2], dim=1), F.softplus(o[:, 2]) + 1e-3
def vm_nll(mu, kappa, th): return (-kappa * (mu[:, 0] * torch.sin(th) + mu[:, 1] * torch.cos(th)) + torch.log(torch.special.i0e(kappa)) + kappa).mean()
def norm(x): return ((x - mean) / std).contiguous(memory_format=torch.channels_last)

def train_batch(ix, xnp):
    x = torch.from_numpy(xnp).to(dev).permute(0, 3, 1, 2).float(); n = len(ix); th = torch.from_numpy(HEAD[ix]).to(dev); tl = torch.from_numpy(TILT[ix]).to(dev)
    tl = (tl + torch.randn(n, device=dev) * math.radians(a.tilt_noise)).clamp(0, math.radians(50))
    flip = torch.rand(n, device=dev) < 0.5; th = torch.where(flip, -th, th); delta = torch.rand(n, device=dev) * 2 * math.pi
    scale = torch.empty(n, device=dev).uniform_(0.8, 1.25); shift = torch.empty(n, 2, device=dev).uniform_(-0.15, 0.15)
    x = imagine_bev(x, tl, delta, flip, scale, shift)
    x = x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2) + torch.empty(n, 1, 1, 1, device=dev).uniform_(-20, 20)
    return norm(x), th + delta

@torch.inference_mode()
def predict(ix, tilt_rad, krot=None, flips=(False, True)):
    """Candidate verification: K imagined rotations (x mirror) of the nadir view; returns combined heading (deg) and concentration."""
    krot = krot or a.krot; net.eval(); Sx, Cx = [], []
    for i in range(0, len(ix), 128):
        j = ix[i:i + 128]; x = torch.from_numpy(raw_np(j)).to(dev).permute(0, 3, 1, 2).float(); n = len(j); tl = torch.from_numpy(np.asarray(tilt_rad[i:i + 128], np.float32)).to(dev)
        s = torch.zeros(n, device=dev); c = torch.zeros(n, device=dev)
        for fl in flips:
            for k in range(krot):
                d = torch.full((n,), 2 * math.pi * k / krot, device=dev)
                xt = norm(imagine_bev(x, tl, d, torch.full((n,), fl, device=dev, dtype=torch.bool)))
                with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = net(xt)
                ang = torch.atan2(mu[:, 0].float(), mu[:, 1].float()) - d
                if fl: ang = -ang
                s += kap.float() * torch.sin(ang); c += kap.float() * torch.cos(ang)
        nt = krot * len(flips); Sx.append((s / nt).cpu()); Cx.append((c / nt).cpu())
    net.train(); s, c = torch.cat(Sx).numpy(), torch.cat(Cx).numpy(); return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)

def evaluate(part, n=3000, krot=1, flips=(False,)):
    ix = np.where(meta.part == part)[0]; ix = ix[::max(1, len(ix) // n)]
    ang, _ = predict(ix, TILT[ix], krot, flips); e = np.abs((ang - meta.heading.values[ix] + 180) % 360 - 180)
    return f'{part}: MAE {e.mean():.1f} med {np.median(e):.1f} <30deg {(e < 30).mean():.2f}'

def loader(perm):
    q = queue.Queue(maxsize=6)
    def work():
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            ix = np.sort(perm[i:i + a.bs]); q.put((ix, raw_np(ix)))
        q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while (b := q.get()) is not None: yield b

net = Net().to(dev).to(memory_format=torch.channels_last)
if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    if not a.eval_only:
        steps = a.epochs * (len(idx_tr) // a.bs); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05); t0 = time.time(); step = 0; net.train()
        for ep in range(a.epochs):
            tot = 0.0; nb = 0
            for ix, xnp in loader(np.random.default_rng(ep).permutation(idx_tr)):
                x, th = train_batch(ix, xnp)
                with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = net(x)
                loss = vm_nll(mu.float(), kap.float(), th); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
                tot += loss.item(); nb += 1; step += 1
                if step % 400 == 0: print(f'ep {ep} step {step}/{steps} loss {tot/nb:.4f} {time.time()-t0:.0f}s', flush=True)
            print(f'== epoch {ep} loss {tot/nb:.4f} | {evaluate("val")} | {evaluate("test")} (single view, GT tilt, subsample)', flush=True)
            torch.save(net.state_dict(), f'{OUT}/model_{a.tag}.pt')
    else: net.load_state_dict(torch.load(f'{OUT}/model_{a.tag}.pt'))
    # ---- per-pair inference on the val/test pairs of the stage-1 result set, tilt taken from the stage-1 SuperGlue geometry ----
    it = pd.read_pickle(f'{OUT}/bev_items.pkl'); row = pd.Series(np.arange(len(meta)), index=meta.image.values); ix = row.reindex(it.image).values
    for nm, tl in (('est', it.tilt_est.values), ('gt', it.tilt_gt.values)):
        ang, kap = predict(ix, np.deg2rad(tl)); it[f'ah_{nm}'] = ang; it[f'ak_{nm}'] = kap
        for part in ('val', 'test'):
            m = (it.part == part).values; e = np.abs((ang[m] - it.heading_gt.values[m] + 180) % 360 - 180)
            print(f'[pair items, tilt={nm}, K={a.krot}x2 candidates] {part}: MAE {e.mean():.2f} med {np.median(e):.2f} <30deg {(e < 30).mean():.3f}', flush=True)
    ang1, kap1 = predict(ix, np.deg2rad(it.tilt_est.values), 1, (False,)); it['ah_single'] = ang1; it['ak_single'] = kap1
    for part in ('val', 'test'):
        m = (it.part == part).values; e = np.abs((ang1[m] - it.heading_gt.values[m] + 180) % 360 - 180); print(f'[pair items, tilt=est, no candidates (1 view)] {part}: MAE {e.mean():.2f} med {np.median(e):.2f}', flush=True)
    it.to_pickle(f'{OUT}/pred_{a.tag}.pkl'); print('saved', flush=True)
