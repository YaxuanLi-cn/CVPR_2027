#!/usr/bin/env python3
"""Track C: re-train the raw-view absolute-heading network (recipe of superglue_fusion/abs_heading_train2.py, unchanged)
with a list of training buildings removed (site-disjoint split, or the size-matched control).

Differences from the archived script, none of which changes the recipe:
  * --exclude: training buildings (tours) dropped from the training set; --out: output directory (archived files are not touched)
  * the 128-image batch is pushed through the network in chunks whose gradients are accumulated before the single optimiser
    step (GPU-memory budget of this shared machine). ConvNeXt has no batch statistics and the augmentation parameters are
    drawn once per 128-image batch exactly as before, so the update equals the one of a 128-image batch.
  * seeds are fixed; per-epoch monitoring also prints the error on the test buildings of the held-out sites.
"""
import argparse, math, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--model', default='convnext_small.fb_in22k_ft_in1k'); ap.add_argument('--tag', required=True); ap.add_argument('--rot-sign', type=int, default=1)
ap.add_argument('--exclude', default=None); ap.add_argument('--out', default='/root/rasr-pairuav/outputs/supp/trackC'); ap.add_argument('--chunk', type=int, default=22); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
D = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading'
m1 = pd.read_pickle(f'{D}/images.pkl'); m2 = pd.read_pickle(f'{D}/images2.pkl'); I1 = np.load(f'{D}/images_448x256.npy', mmap_mode='r'); I2 = np.load(f'{D}/images2_448x256.npy', mmap_mode='r')
rng = np.random.default_rng(0); tr_tours = np.array(sorted(m1[m1.split == 'train'].tour.unique())); val_tours = set(rng.choice(tr_tours, len(tr_tours) // 10, replace=False))  # same split as v1 / v2
m1['part'] = np.where(m1.split == 'test', 'test', np.where(m1.tour.isin(val_tours), 'val', 'train')); m2['part'] = np.where(m2.tour.isin(val_tours), 'val2', 'train')
meta = pd.concat([m1.assign(src=0, row=np.arange(len(m1))), m2.assign(src=1, row=np.arange(len(m2)))], ignore_index=True)
excl = set(open(a.exclude).read().split()) if a.exclude else set()
idx_tr = np.where((meta.part == 'train') & ~meta.tour.isin(excl))[0]
ST = pd.read_pickle(f'{a.out}/sites.pkl').set_index('tour'); meta['heldout'] = ST.heldout.reindex(meta.tour).values
print(meta.part.value_counts().to_dict(), '| excluded training buildings', len(excl), '| training images used', len(idx_tr), 'from', meta.tour.values[idx_tr].astype(str).tolist().__len__() and len(set(meta.tour.values[idx_tr])), 'buildings', flush=True)
assert not (set(meta.tour.values[idx_tr]) & excl)
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
    for i in range(0, len(ix), 64):
        x = raw(np.sort(ix[i:i + 64])); s = torch.zeros(len(x), device=dev); c = torch.zeros(len(x), device=dev)
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

def evaluate(part, full_tta=False, n=None, held=None):
    m = (meta.part == part) if held is None else ((meta.part == part) & (meta.heldout == held)); ix = np.where(m)[0]; ix = ix if n is None else ix[::max(1, len(ix) // n)]
    ang, _ = predict(ix, full_tta); e = np.abs((ang - meta.heading.values[np.sort(ix)] + 180) % 360 - 180)
    return f'{part}{"" if held is None else ("/heldout" if held else "/other")}: MAE {e.mean():.1f} med {np.median(e):.1f} <30deg {(e < 30).mean():.2f}'

if __name__ == '__main__':
    t0 = time.time(); step = 0; net.train(); torch.cuda.reset_peak_memory_stats()
    for ep in range(a.epochs):
        perm = np.random.default_rng(ep).permutation(idx_tr); tot = 0.0; nb = 0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            x, th = train_batch(perm[i:i + a.bs]); n = len(x); opt.zero_grad(set_to_none=True); lsum = 0.0
            for j in range(0, n, a.chunk):
                with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = net(x[j:j + a.chunk])
                l = vm_nll(mu.float(), kap.float(), th[j:j + a.chunk]) * (len(mu) / n); l.backward(); lsum += l.item()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
            tot += lsum; nb += 1; step += 1
            if step % 400 == 0: print(f'ep {ep} step {step}/{steps} loss {tot/nb:.4f} {time.time()-t0:.0f}s peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GB', flush=True)
        print(f'== epoch {ep} loss {tot/nb:.4f} | {evaluate("val")} | {evaluate("test")} | {evaluate("test", held=True)} | {evaluate("test", held=False)}', flush=True)
        torch.save(net.state_dict(), f'{a.out}/model_{a.tag}.pt')
    train_s = time.time() - t0
    print('full TTA |', evaluate('val', True), '|', evaluate('test', True), '|', evaluate('test', True, held=True), '|', evaluate('test', True, held=False), flush=True)
    ix = np.where(meta.part.isin(['val', 'test']))[0]; ang, kap = predict(ix, True); o = meta.iloc[np.sort(ix)][['image', 'part', 'tour', 'heading', 'tilt', 'heldout']].copy(); o['ah_pred'] = ang; o['ah_kappa'] = kap
    o.to_pickle(f'{a.out}/pred_{a.tag}.pkl')
    json.dump(dict(tag=a.tag, exclude=a.exclude, excluded_buildings=len(excl), train_images=int(len(idx_tr)), train_buildings=int(len(set(meta.tour.values[idx_tr]))), epochs=a.epochs, bs=a.bs, chunk=a.chunk, lr=a.lr, steps=steps, train_seconds=train_s,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, seed=a.seed), open(f'{a.out}/train_{a.tag}.json', 'w'), indent=1); print('saved', flush=True)
