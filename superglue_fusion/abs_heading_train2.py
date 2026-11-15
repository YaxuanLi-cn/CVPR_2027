#!/usr/bin/env python3
"""Absolute-heading CNN v2: square crops + full dihedral augmentation (rot90 k => heading + 90k; mirror => -heading),
all images of the train tours, dihedral x 3-crop test-time averaging. von Mises output (direction + concentration)."""
import argparse, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--model', default='convnext_small.fb_in22k_ft_in1k'); ap.add_argument('--tag', default='cnx_s_d8'); ap.add_argument('--rot-sign', type=int, default=1); a = ap.parse_args()
D = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading'
m1 = pd.read_pickle(f'{D}/images.pkl'); m2 = pd.read_pickle(f'{D}/images2.pkl'); I1 = np.load(f'{D}/images_448x256.npy', mmap_mode='r'); I2 = np.load(f'{D}/images2_448x256.npy', mmap_mode='r')
rng = np.random.default_rng(0); tr_tours = np.array(sorted(m1[m1.split == 'train'].tour.unique())); val_tours = set(rng.choice(tr_tours, len(tr_tours) // 10, replace=False))  # same split as v1
m1['part'] = np.where(m1.split == 'test', 'test', np.where(m1.tour.isin(val_tours), 'val', 'train')); m2['part'] = np.where(m2.tour.isin(val_tours), 'val2', 'train')
meta = pd.concat([m1.assign(src=0, row=np.arange(len(m1))), m2.assign(src=1, row=np.arange(len(m2)))], ignore_index=True)
idx_tr = np.where(meta.part == 'train')[0]; print(meta.part.value_counts().to_dict(), flush=True)
dev = 'cuda'; mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
head = torch.tensor(np.deg2rad(meta.heading.values), dtype=torch.float32); SRC = meta.src.values; ROW = meta.row.values; S = 256

def raw(ix): return torch.from_numpy(np.stack([(I1 if SRC[i] == 0 else I2)[ROW[i]] for i in ix])).to(dev).permute(0, 3, 1, 2).float()
def dihedral(x, th, k, flip):
    """flip (mirror left-right): heading -> -heading; then rot90 CCW k times: heading -> heading + rot_sign*90k."""
    if flip: x = torch.flip(x, dims=[3]); th = -th
    return torch.rot90(x, k, (2, 3)), th + a.rot_sign * k * math.pi / 2

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model(a.model, pretrained=True, num_classes=0, drop_path_rate=0.2); s.h = nn.Linear(s.b.num_features, 3)
    def forward(s, x):
        o = s.h(s.b(x)); return F.normalize(o[:, :2], dim=1), F.softplus(o[:, 2]) + 1e-3
def vm_nll(mu, kappa, th): return (-kappa * (mu[:, 0] * torch.sin(th) + mu[:, 1] * torch.cos(th)) + torch.log(torch.special.i0e(kappa)) + kappa).mean()

def train_batch(ix):
    ix = np.sort(ix); x = raw(ix); th = head[ix].to(dev); n = len(ix)
    hc = np.random.randint(170, 257); x0 = np.random.randint(0, 448 - hc + 1); y0 = np.random.randint(0, 256 - hc + 1)
    x = x[:, :, y0:y0 + hc, x0:x0 + hc]
    if hc != S: x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False, antialias=True)
    x, th = dihedral(x, th, np.random.randint(4), np.random.rand() < 0.5)
    x = x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2) + torch.empty(n, 1, 1, 1, device=dev).uniform_(-20, 20)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last), th

net = Net().to(dev).to(memory_format=torch.channels_last); steps = a.epochs * (len(idx_tr) // a.bs)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)

@torch.inference_mode()
def predict(ix, full_tta=True):
    net.eval(); Sx, Cx = [], []
    for i in range(0, len(ix), 128):
        x = raw(np.sort(ix[i:i + 128])); s = torch.zeros(len(x), device=dev); c = torch.zeros(len(x), device=dev)
        for x0 in ((0, 96, 192) if full_tta else (96,)):
            xc = (x[:, :, :, x0:x0 + S] - mean) / std
            for flip in ((False, True) if full_tta else (False,)):
                for k in (range(4) if full_tta else (0,)):
                    xt, off = dihedral(xc, torch.zeros(len(x), device=dev), k, flip)
                    with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = net(xt.contiguous(memory_format=torch.channels_last))
                    ang = torch.atan2(mu[:, 0].float(), mu[:, 1].float()) - a.rot_sign * k * math.pi / 2   # undo rotation
                    if flip: ang = -ang                                                                     # undo mirror
                    s += kap.float() * torch.sin(ang); c += kap.float() * torch.cos(ang)
        nt = 24 if full_tta else 1; Sx.append((s / nt).cpu()); Cx.append((c / nt).cpu())
    net.train(); s, c = torch.cat(Sx).numpy(), torch.cat(Cx).numpy(); return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)

def evaluate(part, full_tta=False, n=None):
    ix = np.where(meta.part == part)[0]; ix = ix if n is None else ix[::max(1, len(ix) // n)]
    ang, _ = predict(ix, full_tta); e = np.abs((ang - meta.heading.values[np.sort(ix)] + 180) % 360 - 180)
    return f'{part}: MAE {e.mean():.1f} med {np.median(e):.1f} <30deg {(e < 30).mean():.2f}'

if __name__ == '__main__':
    t0 = time.time(); step = 0; net.train()
    for ep in range(a.epochs):
        perm = np.random.default_rng(ep).permutation(idx_tr); tot = 0.0; nb = 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            x, th = train_batch(perm[i:i + a.bs])
            with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = net(x)
            loss = vm_nll(mu.float(), kap.float(), th); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
            tot += loss.item(); nb += 1; step += 1
            if step % 400 == 0: print(f'ep {ep} step {step}/{steps} loss {tot/nb:.4f} {time.time()-t0:.0f}s', flush=True)
        print(f'== epoch {ep} loss {tot/nb:.4f} | {evaluate("val")} | {evaluate("test")}', flush=True)
        torch.save(net.state_dict(), f'{D}/model_{a.tag}.pt')
    print('full TTA |', evaluate('val', True), '|', evaluate('test', True), flush=True)
    ix = np.where(meta.part.isin(['val', 'test']))[0]; ang, kap = predict(ix, True); o = meta.iloc[np.sort(ix)][['image', 'part']].copy(); o['ah_pred'] = ang; o['ah_kappa'] = kap
    o.to_pickle(f'{D}/pred_{a.tag}.pkl'); print('saved', flush=True)
