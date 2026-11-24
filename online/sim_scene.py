#!/usr/bin/env python3
"""Closed-loop simulator scenes: one 3-D Gaussian-splatting reconstruction per dataset tour.

Each PairUAV-5D tour has 150 posed images of one site (camera positions inside a ~30 m box, ~210 m above ground, all
headings, tilt 0-45 deg). A splat model fitted to them renders the view from ANY nearby pose with correct parallax
(buildings included), which is what a closed-loop test needs. Every 10th image is held out: never used for fitting,
used as navigation goals and to measure rendering fidelity.

Usage: sim_scene.py build <tour> [iters] [cap]     -> outputs/sim/scenes/<tour>.pt (+ fidelity numbers)
"""
import sys, os, re, math, json, time
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from geom import R_wc, gt_relative
from pose_from_matches import K0
DATA = Path('/root/pairUAV-5D-temp'); SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; OUT = Path('/root/rasr-pairuav/outputs/sim/scenes'); W0, H0 = 3686, 2073
_pat = {k: re.compile(rf'<{k}>([-\d.eE+]+)</{k}>') for k in ('longitude', 'latitude', 'altitude', 'heading', 'tilt')}

def load_cameras(tour):
    rows = []
    for f in sorted((DATA / 'kml' / tour).glob('*.kml')):
        s = f.read_text(); rows.append(dict(i=int(f.stem), image=str(DATA / 'images' / tour / f'{f.stem}.jpg'), **{k: float(p.search(s).group(1)) for k, p in _pat.items()}))
    C = pd.DataFrame(rows); lat0, lon0 = C.latitude.mean(), C.longitude.mean(); la = np.deg2rad(lat0); N = 6378137.0 / np.sqrt(1 - 6.69437999014e-3 * np.sin(la)**2)
    C['e'] = np.deg2rad(C.longitude - lon0) * N * np.cos(la); C['n'] = np.deg2rad(C.latitude - lat0) * 6378137.0; C['u'] = C.altitude - C.altitude.mean()   # matches the dataset's x/y/z labels
    C['holdout'] = (C.i % 10 == 5); return C

def viewmat(e, n, u, heading, tilt):
    R = R_wc(heading, tilt); V = np.eye(4); V[:3, :3] = R.T; V[:3, 3] = -R.T @ np.array([e, n, u]); return V
def K_render(scale=1.0):
    """gsplat puts pixel centres at i+0.5; K0 is in integer-centre pixel coordinates of the full-resolution image."""
    K = K0.copy(); K[0, 0] *= scale; K[1, 1] *= scale; K[0, 2] = (K0[0, 2] + 0.5) * scale; K[1, 2] = (K0[1, 2] + 0.5) * scale; return K

def triangulate_init(tour, C):
    """Sparse points from the archived pass-1 SuperGlue matches of this tour's pairs, triangulated with the dataset poses."""
    P = pd.read_pickle(f'{SGD}/pairs.pkl'); P = P[P.tour == tour]; Ci = C.set_index('i'); Ki = np.linalg.inv(K0); pts, cols = [], []
    for r in P.itertuples():
        f = f'{SGD}/matches_rot/{r.manifest_index}.npz'
        if not os.path.exists(f) or r.a not in Ci.index or r.b not in Ci.index: continue
        d = np.load(f); m = (d['src'] >= 10) & (d['conf'] > 0.3); p0, p1 = d['p0'][m].astype(np.float64), d['p1'][m].astype(np.float64)
        if len(p0) < 20: continue
        a, b = Ci.loc[r.a], Ci.loc[r.b]; ca = np.array([a.e, a.n, a.u]); cb = np.array([b.e, b.n, b.u]); R, t = gt_relative(a.heading, a.tilt, b.heading, b.tilt, cb - ca)
        x0 = np.c_[p0, np.ones(len(p0))] @ Ki.T; x1 = np.c_[p1, np.ones(len(p1))] @ Ki.T
        X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], x0[:, :2].T.copy(), x1[:, :2].T.copy()); X = (X[:3] / X[3]).T; Xb = X @ R.T + t
        ra = np.linalg.norm(X[:, :2] / X[:, 2:3] - x0[:, :2], axis=1) * K0[0, 0]; rb = np.linalg.norm(Xb[:, :2] / Xb[:, 2:3] - x1[:, :2], axis=1) * K0[0, 0]
        ok = (X[:, 2] > 50) & (X[:, 2] < 600) & (Xb[:, 2] > 50) & (ra < 2) & (rb < 2)
        if ok.sum() < 10: continue
        im = cv2.imread(a.image, cv2.IMREAD_REDUCED_COLOR_2); px = np.clip((p0[ok] / 2).astype(int), 0, [im.shape[1] - 1, im.shape[0] - 1])
        pts.append(ca + X[ok] @ R_wc(a.heading, a.tilt).T); cols.append(im[px[:, 1], px[:, 0], ::-1])
    return (np.concatenate(pts), np.concatenate(cols)) if pts else (np.zeros((0, 3)), np.zeros((0, 3), np.uint8))

def _ssim(a, b, win):
    C1, C2 = 0.01**2, 0.03**2; p = win.shape[-1] // 2; f = lambda x: F.conv2d(x, win, padding=p, groups=3)
    mu1, mu2 = f(a), f(b); s1 = f(a * a) - mu1 * mu1; s2 = f(b * b) - mu2 * mu2; s12 = f(a * b) - mu1 * mu2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2))).mean()

def build(tour, iters=12000, cap=2500000, dev='cuda'):
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); C = load_cameras(tour); tr = C[~C.holdout].reset_index(drop=True)
    s = 0.5; ims = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_2), cv2.COLOR_BGR2RGB) for p in tr.image]; Hh, Wh = ims[0].shape[:2]
    gt = torch.from_numpy(np.stack(ims)).to(dev); V = torch.tensor(np.stack([viewmat(r.e, r.n, r.u, r.heading, r.tilt) for r in tr.itertuples()]), dtype=torch.float32, device=dev)
    Kh = K0.copy(); Kh[0, 0] *= s; Kh[1, 1] *= s; Kh[0, 2] = (K0[0, 2] - 0.5) * s + 0.5; Kh[1, 2] = (K0[1, 2] - 0.5) * s + 0.5; Kt = torch.tensor(Kh, dtype=torch.float32, device=dev)[None]   # 2x-reduced image: u_half = (u_full - 0.5) / 2
    # ---- initial points: triangulated matches + a ground grid at their median height, coloured from the first training image that sees each point
    pts, cols = triangulate_init(tour, C); zg = float(np.median(pts[:, 2])) if len(pts) > 100 else -212.0; print(f'[{tour}] triangulated {len(pts)} points, ground height {zg:.1f} m (camera mean = 0)', flush=True)
    g = np.arange(-330, 330.01, 0.6); gx, gy = np.meshgrid(g, g); grid = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, zg)]; gcol = np.zeros((len(grid), 3), np.uint8); seen = np.zeros(len(grid), bool); G = torch.tensor(grid, dtype=torch.float32, device=dev)
    for i in np.random.default_rng(0).permutation(len(tr)):
        Xc = G @ V[i, :3, :3].T + V[i, :3, 3]; u = Kh[0, 0] * Xc[:, 0] / Xc[:, 2] + Kh[0, 2] - 0.5; v = Kh[1, 1] * Xc[:, 1] / Xc[:, 2] + Kh[1, 2] - 0.5
        ok = ((Xc[:, 2] > 1) & (u >= 0) & (u < Wh - 1) & (v >= 0) & (v < Hh - 1)).cpu().numpy() & ~seen
        if ok.any(): ui, vi = u.cpu().numpy()[ok].astype(int), v.cpu().numpy()[ok].astype(int); gcol[ok] = ims[i][vi, ui]; seen |= ok
    P0 = np.concatenate([pts, grid[seen]]); C0 = np.concatenate([cols, gcol[seen]]).astype(np.float32) / 255; n0 = len(P0); print(f'[{tour}] init gaussians {n0}', flush=True)
    sc0 = np.full(n0, 0.35); sc0[:len(pts)] = 0.5
    params = torch.nn.ParameterDict(dict(means=torch.nn.Parameter(torch.tensor(P0, dtype=torch.float32)), scales=torch.nn.Parameter(torch.log(torch.tensor(sc0, dtype=torch.float32))[:, None].repeat(1, 3)),
        quats=torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0]).repeat(n0, 1)), opacities=torch.nn.Parameter(torch.logit(torch.full((n0,), 0.5))), colors=torch.nn.Parameter(torch.logit(torch.tensor(C0).clamp(0.02, 0.98))))).to(dev)
    lrs = dict(means=1.6e-4 * 30, scales=5e-3, quats=1e-3, opacities=5e-2, colors=2.5e-3); opt = {k: torch.optim.Adam([{'params': params[k], 'lr': lr}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opt['means'], gamma=0.01 ** (1.0 / iters)); strat = MCMCStrategy(cap_max=cap, refine_start_iter=300, refine_stop_iter=int(iters * 0.8), refine_every=100); strat.check_sanity(params, opt); state = strat.initialize_state()
    k1 = torch.tensor([math.exp(-(x - 5)**2 / (2 * 1.5**2)) for x in range(11)], device=dev); k1 = k1 / k1.sum(); win = (k1[:, None] * k1[None]).expand(3, 1, 11, 11).contiguous(); rng = np.random.default_rng(1)
    def render(Vm, Km, W, H):
        img, _, info = rasterization(params['means'], F.normalize(params['quats'], dim=-1), torch.exp(params['scales']), torch.sigmoid(params['opacities']), torch.sigmoid(params['colors']), Vm, Km, W, H,
                                     packed=False, rasterize_mode='antialiased', near_plane=10.0, far_plane=2000.0); return img, info
    for it in range(iters):
        i = int(rng.integers(len(tr))); img, info = render(V[i:i + 1], Kt, Wh, Hh); tgt = gt[i:i + 1].float() / 255
        a, b = img.permute(0, 3, 1, 2), tgt.permute(0, 3, 1, 2); loss = 0.8 * (a - b).abs().mean() + 0.2 * (1 - _ssim(a, b, win))
        loss = loss + 0.01 * torch.sigmoid(params['opacities']).mean() + 0.01 * torch.exp(params['scales']).mean()
        strat.step_pre_backward(params, opt, state, it, info); loss.backward(); strat.step_post_backward(params, opt, state, it, info, lr=sched.get_last_lr()[0])
        for o in opt.values(): o.step(); o.zero_grad(set_to_none=True)
        sched.step()
        if it % 1000 == 0 or it == iters - 1: print(f'[{tour}] it {it} loss {loss.item():.4f} gaussians {len(params["means"])} {time.time()-t0:.0f}s', flush=True)
    ck = {k: v.detach().cpu() for k, v in params.items()}; torch.save(ck, OUT / f'{tour}.pt')
    # ---- fidelity on the held-out images, rendered at FULL resolution (what the navigation loop sees)
    sc = Scene(tour, dev=dev); ps = []
    for r in C[C.holdout].itertuples():
        ren = sc.render(r.e, r.n, r.u, r.heading, r.tilt); real = cv2.imread(r.image); mse = np.mean((ren.astype(np.float32) - real.astype(np.float32))**2); ps.append(10 * math.log10(255**2 / max(mse, 1e-6)))
    res = dict(tour=tour, gaussians=int(len(ck['means'])), iters=iters, train_images=int(len(tr)), holdout_images=int(C.holdout.sum()), holdout_psnr_fullres=float(np.mean(ps)), ground_height=zg, seconds=time.time() - t0)
    json.dump(res, open(OUT / f'{tour}.json', 'w'), indent=1); print(res, flush=True); return res

class Scene:
    def __init__(self, tour, dev='cuda'):
        from gsplat import rasterization
        self.rast = rasterization; self.tour = tour; self.dev = dev; self.cams = load_cameras(tour); ck = torch.load(OUT / f'{tour}.pt', map_location=dev)
        self.means = ck['means']; self.quats = F.normalize(ck['quats'], dim=-1); self.scales = torch.exp(ck['scales']); self.opac = torch.sigmoid(ck['opacities']); self.colors = torch.sigmoid(ck['colors'])
        self.K = torch.tensor(K_render(1.0), dtype=torch.float32, device=dev)[None]
    @torch.inference_mode()
    def render(self, e, n, u, heading, tilt):
        """BGR uint8 image (3686x2073, canonical intrinsics) seen from ENU position (e, n, u) with the given heading / tilt in degrees."""
        V = torch.tensor(viewmat(e, n, u, heading, tilt), dtype=torch.float32, device=self.dev)[None]
        img, _, _ = self.rast(self.means, self.quats, self.scales, self.opac, self.colors, V, self.K, W0, H0, packed=False, rasterize_mode='antialiased', near_plane=10.0, far_plane=2000.0)
        return (img[0].clamp(0, 1) * 255).byte().cpu().numpy()[:, :, ::-1].copy()

if __name__ == '__main__':
    if sys.argv[1] == 'build':
        for t in sys.argv[2].split(','): build(t, *(int(x) for x in sys.argv[3:5]))
