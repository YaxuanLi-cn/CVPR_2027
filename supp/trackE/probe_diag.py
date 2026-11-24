#!/usr/bin/env python3
"""Diagnostic beyond the Table 2 recipes: is the frame the bottleneck of the frozen-feature baselines, or the features?

The DINOv3 ViT-7B and Sample4Geo recipes fit a linear head for ONE epoch. Here the head is trained to convergence on the same
frozen pair features [f(A), f(B)], in both frames: (i) ridge regression in closed form, (ii) a two-hidden-layer MLP.
Model selection (ridge strength, MLP epoch) uses 10 % of the training tours; the test pairs are scored once.
This is not a row of Table 2; it bounds what any head on these features could gain from a camera-frame target.
Output: outputs/supp/trackE/probe_diag.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, str(Path(__file__).parent))
from common import *
from splitdata import ordered
from camlabels import table
O = Path(f'{OUT}/trackE'); dev = torch.device('cuda'); T = load_test(); tab = table(); gw, gc = gt_world(T), gt_cam(T)
def lab(mi, frame): t = tab.loc[mi]; return np.c_[t.lx.values if frame == 'cam' else t.x.values, t.ly.values if frame == 'cam' else t.y.values, t.z.values].astype(np.float32)

def features(which):
    if which == 'DINOv3 ViT-7B (CLS of each image)':
        tr, te = ordered('train'), ordered('test'); paths = sorted({p for r in tr + te for p in (r['image_a'], r['image_b'])}); ix = {p: i for i, p in enumerate(paths)}
        f = torch.load('/root/baseline/dinov3/outputs_xyz7b_mast3r_split/unique_image_features.pt', map_location='cpu', weights_only=True).float()
        g = lambda rows: torch.cat((f[[ix[r['image_a']] for r in rows]], f[[ix[r['image_b']] for r in rows]]), 1)
        return g(tr), np.array([r['manifest_index'] for r in tr]), g(te), np.array([r['manifest_index'] for r in te])
    d = np.load(O / 'runs/sample4geo/features.npz')   # Sample4Geo: features of the augmented training stream / plain test images
    return torch.tensor(np.concatenate([d['train_a'], d['train_b']], 1)), d['train_manifest'], torch.tensor(np.concatenate([d['test_a'], d['test_b']], 1)), d['test_manifest']

res = {}
for which in ('DINOv3 ViT-7B (CLS of each image)', 'Sample4Geo (ConvNeXt-B, University-1652)'):
    Xtr, mtr, Xte, mte = features(which); tours = tab.loc[mtr].tour.values; ut = np.unique(tours); rng = np.random.default_rng(0); val_t = set(rng.choice(ut, len(ut) // 10, replace=False).tolist())
    va = np.array([t in val_t for t in tours]); mu, sd = Xtr[~va].mean(0), Xtr[~va].std(0).clamp_min(1e-6); Z = lambda X: (X - mu) / sd      # standardised features stay on the CPU (shared GPU); batches are moved on demand
    Ztr, Zva, Zte = Z(Xtr[~va]), Z(Xtr[va]), Z(Xte); order = pd.Series(np.arange(len(mte)), index=mte).reindex(T.manifest_index.values).values.astype(int)
    for frame in ('enu', 'cam'):
        Ytr = torch.tensor(lab(mtr[~va], frame), device=dev); Yva = lab(mtr[va], frame); g = gc if frame == 'cam' else gw
        def final(pred_te, tag, info):
            p = pred_te[order]; e = errors(dict(x=p[:, 0], y=p[:, 1], z=p[:, 2]), g)
            r = dict(x_mae=float(e['x'].mean()), y_mae=float(e['y'].mean()), z_mae=float(e['z'].mean()), d_mae=float(e['d'].mean()), sr_d_1m=float((e['d'] <= 1).mean()), epe_med=float(np.median(e['epe'])), sr_epe_1m=float((e['epe'] < 1).mean()), dir_med=float(np.median(e['dir'])),
                     corr=[float(np.corrcoef(p[:, i], np.c_[g['x'], g['y'], g['z']][:, i])[0, 1]) for i in range(3)], **info)
            res[f'{which} | {tag} | {frame}'] = r; print(which, '|', tag, '|', frame, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        # ---- (i) ridge, closed form in float64 (bias via centred targets)
        ym = Ytr.double().mean(0); G = torch.zeros((Ztr.shape[1],) * 2, device=dev, dtype=torch.float64); b = torch.zeros((Ztr.shape[1], 3), device=dev, dtype=torch.float64); best = None
        for s0 in range(0, len(Ztr), 4096): a_ = Ztr[s0:s0 + 4096].to(dev).double(); G += a_.T @ a_; b += a_.T @ (Ytr[s0:s0 + 4096].double() - ym)
        apply = lambda X, W: torch.cat([(X[s0:s0 + 4096].to(dev).double() @ W + ym).cpu() for s0 in range(0, len(X), 4096)]).numpy()
        for lam in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=dev, dtype=torch.float64), b); pv = apply(Zva, W); s = float(np.abs(pv[:, :2] - Yva[:, :2]).mean())
            if best is None or s < best[0]: best = (s, lam, W)
        final(apply(Zte, best[2]), 'linear head to convergence (ridge)', dict(ridge_lambda=best[1], val_horizontal_mae=best[0])); del G, b; torch.cuda.empty_cache()
        # ---- (ii) MLP probe
        torch.manual_seed(0); net = nn.Sequential(nn.Linear(Ztr.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 3)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sc = torch.tensor([15.0, 15.0, 33.0], device=dev); ep, best = 30, None; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=ep * ((len(Ztr) + 511) // 512))
        for e_ in range(ep):
            net.train(); perm = torch.randperm(len(Ztr))
            for s0 in range(0, len(Ztr), 512):
                r_ = perm[s0:s0 + 512]; loss = F.smooth_l1_loss(net(Ztr[r_].to(dev)), Ytr[r_.to(dev)] / sc, beta=0.05); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            net.eval()
            fwd = lambda X: torch.cat([(net(X[s0:s0 + 4096].to(dev)) * sc).cpu() for s0 in range(0, len(X), 4096)]).numpy()
            with torch.inference_mode(): pv = fwd(Zva); s = float(np.abs(pv[:, :2] - Yva[:, :2]).mean())
            if best is None or s < best[0]:
                with torch.inference_mode(): best = (s, e_ + 1, fwd(Zte))
        final(best[2], 'MLP head (2 x 1024), best validation epoch', dict(epoch=best[1], val_horizontal_mae=best[0]))
    del Ztr, Zva, Zte; torch.cuda.empty_cache()
json.dump(res, open(O / 'probe_diag.json', 'w'), indent=1)
