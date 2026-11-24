"""3D Gaussian Splatting trainer (gsplat) with controlled degradations.

Degradation axes:
  --views N      train on N evenly spaced training views (default: all)
  --cap N        cap the number of Gaussians at N (uses the MCMC densification strategy)
  --render_steps renders the held-out views / reference camera path at intermediate iterations,
                 which gives the "fewer iterations" axis from a single run.
Training follows the gsplat `simple_trainer` defaults (Kerbl et al. 2023 hyper-parameters).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

from common import ROOT, load_scene, subsample_views, save_png

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--views", type=int, default=0)
p.add_argument("--cap", type=int, default=0)
p.add_argument("--max_steps", type=int, default=30000)
p.add_argument("--render_steps", type=int, nargs="+", default=[30000])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sh_degree", type=int, default=3)
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda"
scene = load_scene(args.scene)
train_idx = subsample_views(scene, args.views)
out_dir = os.path.join(ROOT, "runs", args.scene, args.tag)
os.makedirs(out_dir, exist_ok=True)

# ---------------- data on GPU
imgs = torch.stack([torch.from_numpy(scene.load_image(i)) for i in train_idx]).to(dev)  # N,H,W,3 uint8
w2c_all = torch.from_numpy(scene.w2c).float().to(dev)
K = torch.from_numpy(scene.K).float().to(dev)
W, H = scene.W, scene.H
centers = scene.cam_centers()[scene.train_idx]
scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1


# ---------------- model init from SfM points
def knn_dist(x, k=4):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(x).query(x, k=k)
    return torch.from_numpy((d[:, 1:] ** 2).mean(1)).float()


def rgb_to_sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814


mcmc = args.cap > 0
pts = torch.from_numpy(scene.points).float()
rgbs = torch.from_numpy(scene.points_rgb).float() / 255.0
if mcmc and len(pts) > args.cap:
    sel = torch.randperm(len(pts))[: args.cap]
    pts, rgbs = pts[sel], rgbs[sel]
dist_avg = torch.sqrt(knn_dist(pts.numpy()).clamp_min(1e-12))
init_scale = 0.1 if mcmc else 1.0
init_opa = 0.5 if mcmc else 0.1
N = len(pts)
K_sh = (args.sh_degree + 1) ** 2
colors = torch.zeros(N, K_sh, 3)
colors[:, 0] = rgb_to_sh(rgbs)
params = {
    "means": pts,
    "scales": torch.log(dist_avg * init_scale)[:, None].repeat(1, 3),
    "quats": torch.rand(N, 4),
    "opacities": torch.logit(torch.full((N,), init_opa)),
    "sh0": colors[:, :1],
    "shN": colors[:, 1:],
}
lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
       "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(dev)) for k, v in params.items()})
optimizers = {k: torch.optim.Adam([{"params": splats[k], "lr": lrs[k], "name": k}], eps=1e-15)
              for k in splats}
sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps))

if mcmc:
    strategy = MCMCStrategy(cap_max=args.cap)
    state = strategy.initialize_state()
else:
    strategy = DefaultStrategy()
    state = strategy.initialize_state(scene_scale=scene_scale)
strategy.check_sanity(splats, optimizers)


def render(viewmat, sh_deg):
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    out, alpha, info = rasterization(
        splats["means"], splats["quats"], torch.exp(splats["scales"]), torch.sigmoid(splats["opacities"]),
        colors, viewmat[None], K[None], W, H, sh_degree=sh_deg, packed=True,
        absgrad=False, rasterize_mode="classic")
    return out[0], info


# SSIM (same Gaussian-window formulation as the 3DGS reference code)
def _win(size=11, sigma=1.5):
    g = torch.tensor([math.exp(-((x - size // 2) ** 2) / (2 * sigma ** 2)) for x in range(size)])
    g = (g / g.sum())[:, None]
    return (g @ g.t())[None, None].repeat(3, 1, 1, 1).to(dev)


WIN = _win()


def ssim(a, b):
    mu1 = F.conv2d(a, WIN, padding=5, groups=3)
    mu2 = F.conv2d(b, WIN, padding=5, groups=3)
    s1 = F.conv2d(a * a, WIN, padding=5, groups=3) - mu1 ** 2
    s2 = F.conv2d(b * b, WIN, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, WIN, padding=5, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()


@torch.no_grad()
def render_test(step):
    d = os.path.join(out_dir, f"step_{step:06d}")
    os.makedirs(d, exist_ok=True)
    for j, i in enumerate(scene.test_idx):
        img, _ = render(w2c_all[i], args.sh_degree)
        img = (img.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
        save_png(os.path.join(d, f"{j:04d}.png"), img)


log = {"scene": args.scene, "tag": args.tag, "method": "3dgs", "views": int(len(train_idx)), "cap": args.cap,
       "seed": args.seed, "strategy": "mcmc" if mcmc else "adc", "num_gs": {}, "time": {}}
t0 = time.time()
perm = torch.randperm(len(train_idx))
pi = 0
for step in range(args.max_steps):
    if pi >= len(perm):
        perm = torch.randperm(len(train_idx))
        pi = 0
    k = int(perm[pi])
    pi += 1
    gt = imgs[k].float() / 255.0
    sh_deg = min(step // 1000, args.sh_degree)
    out, info = render(w2c_all[train_idx[k]], sh_deg)
    strategy.step_pre_backward(splats, optimizers, state, step, info)
    l1 = F.l1_loss(out, gt)
    loss = 0.8 * l1 + 0.2 * (1 - ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
    if mcmc:
        loss = loss + 0.01 * torch.sigmoid(splats["opacities"]).mean() + 0.01 * torch.exp(splats["scales"]).mean()
    loss.backward()
    for o in optimizers.values():
        o.step()
        o.zero_grad(set_to_none=True)
    sched.step()
    if mcmc:
        strategy.step_post_backward(splats, optimizers, state, step, info, lr=sched.get_last_lr()[0])
    else:
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=True)
    if (step + 1) in args.render_steps:
        torch.cuda.synchronize()
        log["time"][step + 1] = time.time() - t0
        log["num_gs"][step + 1] = int(len(splats["means"]))
        render_test(step + 1)
        print(f"[{args.scene}/{args.tag}] step {step + 1} loss {loss.item():.4f} n_gs {len(splats['means'])} "
              f"t {time.time() - t0:.0f}s", flush=True)
        json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=1)
print("DONE", flush=True)
