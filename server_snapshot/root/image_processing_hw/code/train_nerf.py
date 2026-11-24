"""Compact NeRF trainer in the style of nerfacto (Tancik et al. 2023):
multi-resolution hash grid field (Instant-NGP, via tiny-cuda-nn), Mip-NeRF 360 scene contraction,
two proposal networks with histogram (interlevel) loss, and distortion regularisation.

Degradation axes: --views N (fewer training views); --render_steps (fewer iterations, from one run).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from common import ROOT, load_scene, subsample_views, save_png

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--views", type=int, default=0)
p.add_argument("--max_steps", type=int, default=30000)
p.add_argument("--render_steps", type=int, nargs="+", default=[30000])
p.add_argument("--rays", type=int, default=8192)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--log2T", type=int, default=21)
p.add_argument("--max_res", type=int, default=4096)
p.add_argument("--test_limit", type=int, default=0, help="render only the first N test views (debug)")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
scene = load_scene(args.scene)
train_idx = subsample_views(scene, args.views)
out_dir = os.path.join(ROOT, "runs", args.scene, args.tag)
os.makedirs(out_dir, exist_ok=True)
W, H = scene.W, scene.H

# ---------------- scene normalisation: all training cameras inside the unit ball
c2w_np = np.linalg.inv(scene.w2c)
ctr = c2w_np[scene.train_idx, :3, 3].mean(0)
scl = 1.0 / np.linalg.norm(c2w_np[scene.train_idx, :3, 3] - ctr, axis=1).max()
c2w_np[:, :3, 3] = (c2w_np[:, :3, 3] - ctr) * scl
c2w = torch.from_numpy(c2w_np).float().to(dev)
fx, fy, cx, cy = scene.K[0, 0], scene.K[1, 1], scene.K[0, 2], scene.K[1, 2]
imgs = torch.stack([torch.from_numpy(scene.load_image(i)) for i in train_idx]).to(dev)  # N,H,W,3 uint8
train_idx_t = torch.from_numpy(np.asarray(train_idx)).to(dev)
NEAR, FAR = 0.05, 1000.0


def get_rays(cam_ids, px, py):
    d = torch.stack([(px + 0.5 - cx) / fx, (py + 0.5 - cy) / fy, torch.ones_like(px)], -1)
    R = c2w[cam_ids, :3, :3]
    d = (R @ d[..., None])[..., 0]
    d = d / d.norm(dim=-1, keepdim=True)
    return c2w[cam_ids, :3, 3], d


def contract(x):
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.where(n <= 1, x, (2 - 1 / n) * (x / n))


def hash_cfg(levels, log2T, base, max_res, feat=2):
    return {"otype": "HashGrid", "n_levels": levels, "n_features_per_level": feat, "log2_hashmap_size": log2T,
            "base_resolution": base, "per_level_scale": math.exp(math.log(max_res / base) / (levels - 1))}


def mlp_cfg(width, layers):
    return {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None",
            "n_neurons": width, "n_hidden_layers": layers}


def trunc_exp(x):
    return torch.exp(x.float().clamp(max=15.0) - 1.0)


class Field(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo = tcnn.NetworkWithInputEncoding(3, 16, hash_cfg(16, args.log2T, 16, args.max_res), mlp_cfg(64, 1))
        self.dir = tcnn.Encoding(3, {"otype": "SphericalHarmonics", "degree": 4})
        self.rgb = tcnn.Network(self.dir.n_output_dims + 15, 3, mlp_cfg(64, 2))

    def forward(self, x, d):
        """x: (...,3) world positions, d: (...,3) unit view directions"""
        sh = x.shape[:-1]
        h = self.geo(((contract(x) + 2) / 4).reshape(-1, 3))
        sigma = trunc_exp(h[:, 0])
        de = self.dir(((d + 1) / 2).reshape(-1, 3))
        rgb = torch.sigmoid(self.rgb(torch.cat([de, h[:, 1:]], -1)).float())
        return sigma.reshape(sh), rgb.reshape(*sh, 3)


class Proposal(nn.Module):
    def __init__(self, max_res):
        super().__init__()
        self.net = tcnn.NetworkWithInputEncoding(3, 1, hash_cfg(5, 17, 16, max_res), mlp_cfg(16, 1))

    def forward(self, x):
        sh = x.shape[:-1]
        return trunc_exp(self.net(((contract(x) + 2) / 4).reshape(-1, 3))[:, 0]).reshape(sh)


field = Field().to(dev)
props = nn.ModuleList([Proposal(128), Proposal(256)]).to(dev)
opt = torch.optim.Adam(list(field.parameters()) + list(props.parameters()), lr=1e-2, eps=1e-15)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 500) * 0.03 ** (s / args.max_steps))
scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)


# ---------------- sampling (nerfacto: piecewise linear / linear-in-disparity spacing)
def g(x):
    return torch.where(x < 1, x / 2, 1 - 1 / (2 * x))


def g_inv(y):
    return torch.where(y < 0.5, 2 * y, 1 / (2 - 2 * y))


S_NEAR, S_FAR = float(g(torch.tensor(NEAR))), float(g(torch.tensor(FAR)))


def s_to_t(s):
    return g_inv(s * S_FAR + (1 - s) * S_NEAR)


def weights_from_sigma(sigma, t):
    delta = t[:, 1:] - t[:, :-1]
    alpha = 1 - torch.exp(-sigma * delta)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1)[:, :-1]
    return alpha * T


def pdf_sample(s, w, n, train):
    """s: (R,m+1) bin edges in [0,1], w: (R,m) weights -> (R,n+1) new sorted edges"""
    w = w + 0.01
    pdf = w / w.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1).clamp(max=1.0)
    nb = n + 1
    u = torch.linspace(0, 1 - 1 / nb, nb, device=s.device)[None].expand(s.shape[0], nb)
    u = u + (torch.rand(s.shape[0], 1, device=s.device) / nb if train else 0.5 / nb)
    inds = torch.searchsorted(cdf, u.contiguous(), right=True)
    below = (inds - 1).clamp(0, s.shape[1] - 1)
    above = inds.clamp(0, s.shape[1] - 1)
    c0, c1 = torch.gather(cdf, 1, below), torch.gather(cdf, 1, above)
    s0, s1 = torch.gather(s, 1, below), torch.gather(s, 1, above)
    tt = torch.nan_to_num((u - c0) / (c1 - c0), nan=0.0).clamp(0, 1)
    return s0 + tt * (s1 - s0)


def outer(t0s, t0e, t1s, t1e, y1):
    cy1 = torch.cat([torch.zeros_like(y1[:, :1]), torch.cumsum(y1, -1)], -1)
    lo = (torch.searchsorted(t1s.contiguous(), t0s.contiguous(), right=True) - 1).clamp(0, y1.shape[1] - 1)
    hi = torch.searchsorted(t1e.contiguous(), t0e.contiguous(), right=True).clamp(0, y1.shape[1] - 1)
    return torch.gather(cy1[:, 1:], 1, hi) - torch.gather(cy1[:, :-1], 1, lo)


def interlevel(s, w, s_env, w_env):
    w_outer = outer(s[:, :-1], s[:, 1:], s_env[:, :-1], s_env[:, 1:], w_env)
    return ((w - w_outer).clamp_min(0) ** 2 / (w + 1e-7)).sum(-1).mean()


def distortion(s, w):
    mid = (s[:, 1:] + s[:, :-1]) / 2
    inter = (w * (w[:, None, :] * (mid[:, :, None] - mid[:, None, :]).abs()).sum(-1)).sum(-1)
    intra = (w ** 2 * (s[:, 1:] - s[:, :-1])).sum(-1) / 3
    return (inter + intra).mean()


N_PROP = [256, 96]
N_FINAL = 48


def render_rays(o, d, train, step=0):
    R = o.shape[0]
    s = torch.linspace(0, 1, N_PROP[0] + 1, device=dev)[None].expand(R, -1)
    if train:
        s = s + (torch.rand(R, 1, device=dev) - 0.5) / N_PROP[0]
        s = s.clamp(0, 1)
    anneal = 1.0
    if train:
        f = min(step / 1000, 1.0)
        anneal = 10 * f / (9 * f + 1)
    prop_s, prop_w = [], []
    for k, net in enumerate(props):
        t = s_to_t(s)
        tm = (t[:, 1:] + t[:, :-1]) / 2
        x = o[:, None] + d[:, None] * tm[..., None]
        with torch.set_grad_enabled(train):
            sig = net(x)
        w = weights_from_sigma(sig, t)
        prop_s.append(s)
        prop_w.append(w)
        n_next = N_PROP[k + 1] if k + 1 < len(props) else N_FINAL
        s = pdf_sample(s.detach(), (w.detach() ** anneal), n_next, train)
    t = s_to_t(s)
    tm = (t[:, 1:] + t[:, :-1]) / 2
    x = o[:, None] + d[:, None] * tm[..., None]
    sig, rgb = field(x, d[:, None].expand_as(x))
    w = weights_from_sigma(sig, t)
    acc = w.sum(-1, keepdim=True)
    col = (w[..., None] * rgb).sum(1) + (1 - acc) * rgb[:, -1]   # "last sample" background
    return col, s, w, prop_s, prop_w


@torch.no_grad()
def render_image(cam):
    py, px = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    px, py = px.reshape(-1).float(), py.reshape(-1).float()
    out = []
    for a in range(0, H * W, 32768):
        o, d = get_rays(torch.full((len(px[a:a + 32768]),), cam, device=dev), px[a:a + 32768], py[a:a + 32768])
        out.append(render_rays(o, d, False)[0])
    return (torch.cat(out).reshape(H, W, 3).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()


@torch.no_grad()
def render_test(step):
    field.eval(); props.eval()
    d = os.path.join(out_dir, f"step_{step:06d}")
    os.makedirs(d, exist_ok=True)
    ids = scene.test_idx[: args.test_limit] if args.test_limit else scene.test_idx
    for j, i in enumerate(ids):
        save_png(os.path.join(d, f"{j:04d}.png"), render_image(int(i)))
    field.train(); props.train()


log = {"scene": args.scene, "tag": args.tag, "method": "nerf", "views": int(len(train_idx)), "seed": args.seed,
       "time": {}}
t0 = time.time()
NI = len(train_idx)
for step in range(args.max_steps):
    ii = torch.randint(0, NI, (args.rays,), device=dev)
    px = torch.randint(0, W, (args.rays,), device=dev)
    py = torch.randint(0, H, (args.rays,), device=dev)
    gt = imgs[ii, py, px].float() / 255.0
    o, d = get_rays(train_idx_t[ii], px.float(), py.float())
    col, s, w, prop_s, prop_w = render_rays(o, d, True, step)
    loss_rgb = ((col - gt) ** 2).mean()
    loss_prop = sum(interlevel(s.detach(), w.detach(), ps, pw) for ps, pw in zip(prop_s, prop_w))
    loss_dist = distortion(s, w)
    loss = loss_rgb + loss_prop + 0.002 * loss_dist
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()
    if (step + 1) % 1000 == 0:
        print(f"[{args.scene}/{args.tag}] step {step + 1} mse {loss_rgb.item():.5f} "
              f"psnr {-10 * math.log10(loss_rgb.item()):.2f} t {time.time() - t0:.0f}s", flush=True)
    if (step + 1) in args.render_steps:
        torch.cuda.synchronize()
        log["time"][step + 1] = time.time() - t0
        render_test(step + 1)
        json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=1)
print("DONE", flush=True)
