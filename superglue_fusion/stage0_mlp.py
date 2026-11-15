#!/usr/bin/env python3
"""Stage 0b: the baseline's MLP head recipe (LayerNorm-256-128-1, SmoothL1, AdamW 3e-4, 30 epochs, bs 256) on MASt3R features,
re-trained with tour-grouped 5-fold OOF so its predictions can be stacked. One independent head per target, trained in parallel."""
import sys, json, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
from sklearn.model_selection import GroupKFold
dev = 'cuda'

class Heads(nn.Module):
    """T independent copies of the baseline DistanceHead, evaluated with batched matmuls."""
    def __init__(s, T, d, h=256, p=0.1):
        super().__init__(); s.T = T; s.p = p
        s.g = nn.Parameter(torch.ones(T, 1, d)); s.b = nn.Parameter(torch.zeros(T, 1, d))
        def lin(i, o):
            w = torch.empty(T, i, o); bound = 1 / math.sqrt(i); nn.init.uniform_(w, -bound, bound); return nn.Parameter(w), nn.Parameter(torch.empty(T, 1, o).uniform_(-bound, bound))
        s.w1, s.b1 = lin(d, h); s.w2, s.b2 = lin(h, h // 2); s.w3, s.b3 = lin(h // 2, 1)
    def forward(s, x):                      # x: (B, d)
        x = F.layer_norm(x, x.shape[-1:])[None] * s.g + s.b
        x = F.dropout(F.gelu(x @ s.w1 + s.b1), s.p, s.training); x = F.dropout(F.gelu(x @ s.w2 + s.b2), s.p, s.training)
        return (x @ s.w3 + s.b3).squeeze(-1).T   # (B, T)

def train_predict(Xtr, Ytr, Xs, seed, epochs=30, bs=256):
    torch.manual_seed(seed); mu = Xtr.mean(0); sd = Xtr.std(0); sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    ys = Ytr.abs().max(0).values.clamp(min=1.0); X = (Xtr - mu) / sd; Y = Ytr / ys
    net = Heads(Y.shape[1], X.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            ix = perm[i:i + bs]; loss = F.smooth_l1_loss(net(X[ix]), Y[ix], reduction='none').mean(0).sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.inference_mode(): return [(net((x - mu) / sd) * ys).cpu().numpy() for x in Xs]

if __name__ == '__main__':
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); D = P.merge(M.drop(columns='split'), on='manifest_index')
    feats = [c for c in M.columns if c not in ('manifest_index', 'split')]; targets = ['x', 'y', 'z', 'tilt', 'dist', 'hsin', 'hcos', 'tilt_a', 'tilt_b', 'lx', 'ly', 'ux', 'uy', 'uz']
    tr, te = D[D.split == 'train'].reset_index(drop=True), D[D.split == 'test'].reset_index(drop=True)
    Xtr = torch.tensor(tr[feats].values, dtype=torch.float32, device=dev); Xte = torch.tensor(te[feats].values, dtype=torch.float32, device=dev)
    Ytr = torch.tensor(tr[targets].values, dtype=torch.float32, device=dev); oof = np.zeros((len(tr), len(targets)))
    for f, (a, b) in enumerate(GroupKFold(5).split(tr, groups=tr.tour.values)):
        a = torch.tensor(a, device=dev); oof[b] = train_predict(Xtr[a], Ytr[a], [Xtr[torch.tensor(b, device=dev)]], seed=100 + f)[0]; print('fold', f, flush=True)
    pte = train_predict(Xtr, Ytr, [Xte], seed=61)[0]
    o_tr = pd.DataFrame(oof, columns=['n_' + t for t in targets]); o_tr['manifest_index'] = tr.manifest_index.values
    o_te = pd.DataFrame(pte, columns=['n_' + t for t in targets]); o_te['manifest_index'] = te.manifest_index.values
    for o in (o_tr, o_te): o['n_heading'] = np.rad2deg(np.arctan2(o.n_hsin, o.n_hcos)); o['n_logd'] = np.log(o.n_dist.clip(lower=0.3))
    pd.concat([o_tr, o_te]).to_pickle(f'{SG}/stage0_mlp.pkl')
    res = {'train_oof': report(tr, o_tr.n_x, o_tr.n_y, o_tr.n_z, o_tr.n_tilt, o_tr.n_heading, 'mlp OOF '), 'test': report(te, o_te.n_x, o_te.n_y, o_te.n_z, o_te.n_tilt, o_te.n_heading, 'mlp TEST')}
    for nm, o, d in (('oof', o_tr, tr), ('test', o_te, te)):
        res[f'{nm}_aux'] = {k: float(np.mean(np.abs(o['n_' + k] - d[k]))) for k in ('dist', 'tilt_a', 'tilt_b', 'lx', 'ly', 'uz')}; print(nm, res[f'{nm}_aux'])
    json.dump(res, open(f'{SG}/stage0_mlp_metrics.json', 'w'), indent=2)
