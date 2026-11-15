#!/usr/bin/env python3
"""Single-image absolute-heading CNN (cue: shadows / sun azimuth in Google Earth aerial textures).

Predicts a von Mises distribution over the camera's absolute heading. Trained on train-split tours only
(10% of train tours held out for validation/calibration); predictions are written for every cached image.
"""
import argparse, math, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=96); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--model', default='convnext_tiny.fb_in22k_ft_in1k'); ap.add_argument('--tag', default='cnx_t'); a = ap.parse_args()
D = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading'
meta = pd.read_pickle(f'{D}/images.pkl'); imgs = np.load(f'{D}/images_448x256.npy', mmap_mode='r')
rng = np.random.default_rng(0); tr_tours = np.array(sorted(meta[meta.split == 'train'].tour.unique())); val_tours = set(rng.choice(tr_tours, len(tr_tours) // 10, replace=False))
meta['part'] = np.where(meta.split == 'test', 'test', np.where(meta.tour.isin(val_tours), 'val', 'train'))
idx_tr = np.where(meta.part == 'train')[0]; print(meta.part.value_counts().to_dict(), flush=True)
dev = 'cuda'; mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
head_rad = torch.tensor(np.deg2rad(meta.heading.values), dtype=torch.float32)

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model(a.model, pretrained=True, num_classes=0, drop_path_rate=0.1); s.h = nn.Linear(s.b.num_features, 3)
    def forward(s, x):
        o = s.h(s.b(x)); return F.normalize(o[:, :2], dim=1), F.softplus(o[:, 2]) + 1e-3   # unit mean direction (sin, cos), concentration kappa

def vm_nll(mu, kappa, th):
    cosd = mu[:, 0] * torch.sin(th) + mu[:, 1] * torch.cos(th)
    return (-kappa * cosd + torch.log(torch.special.i0e(kappa)) + kappa).mean()

def batch(ix, train):
    x = torch.from_numpy(np.stack([imgs[i] for i in np.sort(ix)])).to(dev).permute(0, 3, 1, 2).float(); th = head_rad[np.sort(ix)].to(dev)
    if train:
        n = len(ix); s = torch.empty(n, device=dev).uniform_(0.55, 1.0); flip = torch.rand(n, device=dev) < 0.5
        tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - s); ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - s)
        theta = torch.zeros(n, 2, 3, device=dev); theta[:, 0, 0] = s * torch.where(flip, -1.0, 1.0); theta[:, 1, 1] = s; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode='bilinear', padding_mode='reflection', align_corners=False)
        th = torch.where(flip, -th, th)   # mirror image <=> heading -> -heading
        x = x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2) + torch.empty(n, 1, 1, 1, device=dev).uniform_(-20, 20)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last), th

net = Net().to(dev).to(memory_format=torch.channels_last)
steps = a.epochs * (len(idx_tr) // a.bs); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.06)

@torch.inference_mode()
def predict(ix):
    net.eval(); S, C, K = [], [], []
    for i in range(0, len(ix), 256):
        x, _ = batch(ix[i:i + 256], False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            m1, k1 = net(x); m2, k2 = net(torch.flip(x, dims=[3]))
        m1, k1, m2, k2 = m1.float(), k1.float(), m2.float(), k2.float()
        s = k1 * m1[:, 0] - k2 * m2[:, 0]; c = k1 * m1[:, 1] + k2 * m2[:, 1]   # flip-TTA: un-mirror (sin -> -sin), combine as natural params
        S.append(s.cpu()); C.append(c.cpu()); K.append(((k1 + k2) / 2).cpu())
    net.train(); s, c = torch.cat(S).numpy(), torch.cat(C).numpy()
    return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c) / 2, torch.cat(K).numpy()

def evaluate(part):
    ix = np.where(meta.part == part)[0]; ang, kap, _ = predict(ix); e = np.abs((ang - meta.heading.values[ix] + 180) % 360 - 180)
    return f'{part}: MAE {e.mean():.1f} med {np.median(e):.1f} <30deg {(e < 30).mean():.2f} <90deg {(e < 90).mean():.2f}'

t0 = time.time(); step = 0; net.train()
for ep in range(a.epochs):
    perm = np.random.default_rng(ep).permutation(idx_tr); tot = 0.0; nb = 0
    for i in range(0, len(perm) - a.bs + 1, a.bs):
        x, th = batch(perm[i:i + a.bs], True)
        with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = net(x)
        loss = vm_nll(mu.float(), kap.float(), th); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
        tot += loss.item(); nb += 1; step += 1
        if step % 200 == 0: print(f'ep {ep} step {step}/{steps} loss {tot/nb:.4f} {time.time()-t0:.0f}s', flush=True)
    print(f'== epoch {ep} loss {tot/nb:.4f} | {evaluate("val")} | {evaluate("test")}', flush=True)
ang, kap, kraw = predict(np.arange(len(meta)))
meta['ah_pred'] = ang; meta['ah_kappa'] = kap; meta[['image', 'part', 'ah_pred', 'ah_kappa']].to_pickle(f'{D}/pred_{a.tag}.pkl'); torch.save(net.state_dict(), f'{D}/model_{a.tag}.pt'); print('saved', flush=True)
