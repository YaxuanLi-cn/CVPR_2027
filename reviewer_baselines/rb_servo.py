#!/usr/bin/env python3
"""Reviewer baselines, part 3: classical closed-loop servoing in the splat simulator of Table 5 (same 160 episodes, same
actuation noise; online/sim_nav.py). No learned pose model: features are SuperGlue matches (with the paper's rotation-aware
matching, otherwise half of the episodes start without a single feature), depth comes from the flight-height prior.

Policies
  ibvs      image-based visual servoing on point features (Chaumette & Hutchinson): e = s - s*, dq = -lambda * J^+ e with the
            point interaction matrix, depth of each feature from the ground plane at the flight height, robust (Cauchy) weights.
            The camera has 5 DoF (east, north, up, yaw, gimbal tilt); own tilt and heading are read from the vehicle, as for all policies.
  ibvs1     the same with lambda = 1 (one Gauss-Newton step per observation), 5 steps like the DreamNav policies
  geo       position-based servoing with the direct two-view pipeline: rotation-aware SuperGlue -> essential matrix -> scale
            from the flight-height prior, executed as a full step (no learned prior, MAP, zoom, imagination or fusion)
Usage: rb_servo.py run | rb_servo.py report
"""
import sys, os, json, time, numpy as np, pandas as pd, cv2
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/online', f'{ROOT}/reviewer_baselines']
from geom import R_wc
from pose_from_matches import K0, pose_features_multi
OUT = f'{ROOT}/outputs/reviewer_baselines'; SIM = f'{ROOT}/outputs/sim'; SGD = f'{ROOT}/outputs/pairuav5d_60k/superglue'
NOISE = dict(pos_mult=0.03, pos_add=0.05, heading=0.2, tilt=0.1); wrap = lambda a: (np.asarray(a) + 180) % 360 - 180
POL = dict(ibvs=dict(steps=20, lam=0.5), ibvs1=dict(steps=5, lam=1.0), geo=dict(steps=5)); MAXMOVE = 40.0


def height_prior():
    """Flight height above ground, a dataset constant fitted on TRAIN pairs (GT distance x triangulated height in baseline units)."""
    P = pd.read_pickle(f'{SGD}/pairs.pkl'); G = pd.read_pickle(f'{SGD}/geom_rot_full.pkl')
    D = P.merge(G[['manifest_index', 'sg_inl', 'sg_hgta_p50']], on='manifest_index'); D = D[(D.split == 'train') & (D.sg_inl >= 400)]
    return float(np.nanmedian(np.sqrt(D.x**2 + D.y**2 + D.z**2) * D.sg_hgta_p50))


class Matcher:
    """Rotation-aware SuperGlue exactly as in the coarse stage (pass 1): 4 rotations at 640x360, fine pass at 1280x720 on the best."""
    def __init__(self, dev='cuda'):
        import torch, run_superglue_rot as rs
        self.torch, self.rs, self.dev = torch, rs, dev; mk = lambda n: rs.SuperPoint({'nms_radius': 4 if n <= 1024 else 3, 'keypoint_threshold': 0.005, 'max_keypoints': n}).eval().to(dev)
        self.spc, self.spf = mk(768), mk(2048); self.sg = rs.SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(dev); self.sizes = [(640, 360), (1280, 720)]
    def prep(self, bgr):
        g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (1843, 1037), interpolation=cv2.INTER_AREA)
        return [(self.torch.from_numpy(cv2.resize(g, s, interpolation=cv2.INTER_AREA)).to(self.dev).float() / 255.0)[None, None] for s in self.sizes]
    def __call__(self, A, B):
        """A, B = prep(current), prep(goal) -> p0 (current), p1 (goal) in full-resolution pixels, conf, src (>=10: fine pass)"""
        torch, rs = self.torch, self.rs
        with torch.autocast('cuda', dtype=torch.float16):
            coarse, _ = rs.match_rot(self.spc, self.sg, A[0], B[0], [0, 1, 2, 3]); csum = [float(c[2].sum()) for c in coarse]; order = np.argsort(csum)[::-1]; kb = int(order[0])
            ks = [kb] + ([int(order[1])] if csum[order[1]] >= 0.5 * csum[kb] else []); fine, _ = rs.match_rot(self.spf, self.sg, A[1], B[1], ks)
        P0, P1, C, S = [], [], [], []
        for (grp, kk, size) in ((fine, ks, self.sizes[1]), (coarse, range(4), self.sizes[0])):
            sc = np.array([rs.W0 / size[0], rs.H0 / size[1]], dtype=np.float32)
            for k, (k0, k1, conf) in zip(kk, grp): P0.append(k0.float().cpu().numpy() * sc); P1.append(k1.float().cpu().numpy() * sc); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), (10 if grp is fine else 0) + k))
        return np.concatenate(P0).astype(np.float64), np.concatenate(P1).astype(np.float64), np.concatenate(C).astype(np.float64), np.concatenate(S)


def twist_map(heading, tilt):
    """6x5: camera-frame twist (v, w) per unit of (east, north, up [m], heading, tilt [deg])."""
    R = R_wc(heading, tilt); X = np.zeros((6, 5)); X[:3, :3] = R.T; d = 1e-3
    for j, (dh, dt) in enumerate(((d, 0), (0, d))):
        W = (R.T @ R_wc(heading + dh, tilt + dt) - np.eye(3)) / d; X[3:, 3 + j] = [W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]; X[3:, 3 + j] /= 2
    return X


def ibvs_step(p0, p1, conf, heading, tilt, Hc, lam, min_conf=0.3):
    """One IBVS update. Returns (dE, dN, dU, dheading, dtilt) scaled by lam, the unscaled translation length and the feature error (px)."""
    m = conf >= min_conf
    if m.sum() < 12: return np.zeros(5), np.nan, np.nan, int(m.sum())
    Ki = np.linalg.inv(K0); s = (np.c_[p0[m], np.ones(m.sum())] @ Ki.T)[:, :2]; sd = (np.c_[p1[m], np.ones(m.sum())] @ Ki.T)[:, :2]; x, y = s[:, 0], s[:, 1]
    t = np.deg2rad(tilt); Z = Hc / np.clip(y * np.sin(t) + np.cos(t), 0.2, None)             # ground plane seen from height Hc with the camera tilted by t from nadir
    Lx = np.c_[-1 / Z, np.zeros_like(x), x / Z, x * y, -(1 + x**2), y]; Ly = np.c_[np.zeros_like(x), -1 / Z, y / Z, 1 + y**2, -x * y, -x]
    J = np.concatenate([Lx, Ly]) @ twist_map(heading, tilt); e = np.concatenate([x - sd[:, 0], y - sd[:, 1]]); n = len(x); w = np.ones(2 * n); dq = np.zeros(5)
    for _ in range(8):                                                                        # IRLS: mismatches and off-plane points are down-weighted
        sw = np.sqrt(w); dq = -np.linalg.lstsq(J * sw[:, None], e * sw, rcond=None)[0]; r = e + J @ dq; rn = np.hypot(r[:n], r[n:]) * K0[0, 0]; c = max(2.0, 1.4826 * np.median(rn) * 2); w = np.tile(1 / (1 + (rn / c)**2), 2)
    return lam * dq, float(np.linalg.norm(dq[:3])), float(np.median(np.hypot(e[:n], e[n:])) * K0[0, 0]), int(n)


def geo_step(p0, p1, conf, src, Hc):
    F = pose_features_multi(p0, p1, conf, src)
    if F['sg_ok'] != 1 or not np.isfinite(F['sg_hgta_p50']) or F['sg_hgta_p50'] <= 0: return None
    d = Hc / F['sg_hgta_p50']; return np.array([d * F['sg_vx'], d * F['sg_vy'], d * F['sg_vz'], F['sg_dh'], F['sg_tb'] - F['sg_ta']]), float(d), F['sg_inl']


def run(max_eps=None, tours=None):
    from sim_scene import Scene
    H0 = height_prior(); print(f'flight-height prior {H0:.1f} m', flush=True); M = Matcher(); rng = np.random.default_rng(1); log = []; n_ep = 0
    for t in (tours or open(f'{SIM}/tours.txt').read().strip().split(',')):
        sc = Scene(t); C = sc.cams; goals = C[C.holdout].sample(min(10, int(C.holdout.sum())), random_state=0); ep = []   # identical episodes to online/sim_nav.py
        for g in goals.itertuples():
            for s in C[C.i != g.i].sample(2, random_state=int(g.i)).itertuples(): ep.append(dict(goal_image=g.image, g=np.array([g.e, g.n, g.u, g.heading, g.tilt]), s=np.array([s.e, s.n, s.u, s.heading, s.tilt])))
        if max_eps: ep = ep[:max_eps]
        G = np.stack([e['g'] for e in ep]); goal_feats = {}; t0 = time.time()
        for pol, cfg in POL.items():
            st = np.stack([e['s'] for e in ep]).copy(); Hc = np.full(len(ep), H0)
            for k in range(cfg['steps'] + 1):
                d = G - st
                for j in range(len(ep)): log.append(dict(tour=t, ep=n_ep + j, pol=pol, step=k, pos_err=float(np.linalg.norm(d[j, :3])), head_err=float(abs(wrap(d[j, 3]))), tilt_err=float(abs(d[j, 4]))))
                if k == cfg['steps']: break
                for j, e in enumerate(ep):
                    if e['goal_image'] not in goal_feats: goal_feats[e['goal_image']] = M.prep(cv2.imread(e['goal_image']))
                    p0, p1, conf, src = M(M.prep(sc.render(*st[j])), goal_feats[e['goal_image']]); rec = log[-len(ep) + j]; cmd = np.zeros(5); h = np.deg2rad(st[j, 3])
                    if pol == 'geo':
                        r = geo_step(p0, p1, conf, src, Hc[j])
                        if r is not None:
                            (lx, ly, z, dh, dt), dist, inl = r; cmd = np.array([np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly, z, dh, dt]); rec.update(pred_dist=dist, n=inl)
                    else:
                        f = src >= 10; cmd, dist, epx, n = ibvs_step(p0[f], p1[f], conf[f], st[j, 3], st[j, 4], Hc[j], cfg['lam'])
                        if n < 12: cmd, dist, epx, n = ibvs_step(p0, p1, conf, st[j, 3], st[j, 4], Hc[j], cfg['lam'])
                        rec.update(pred_dist=dist, feat_err_px=epx, n=n)
                    mv = cmd[:3].copy(); nm = np.linalg.norm(mv)
                    if nm > MAXMOVE: mv *= MAXMOVE / nm
                    if nm > 0: mv = mv * (1 + NOISE['pos_mult'] * rng.standard_normal(3)) + NOISE['pos_add'] * rng.standard_normal(3)
                    Hc[j] += mv[2]; st[j, :3] += mv; st[j, 3] = (st[j, 3] + cmd[3] + NOISE['heading'] * rng.standard_normal()) % 360; st[j, 4] = float(np.clip(st[j, 4] + cmd[4] + NOISE['tilt'] * rng.standard_normal(), 0, 47))
            print(f'[{t}] {pol}: median pos err start {np.median(np.linalg.norm((G - np.stack([e["s"] for e in ep]))[:, :3], axis=1)):.2f} -> final {np.median(np.linalg.norm((G - st)[:, :3], axis=1)):.2f} m ({time.time()-t0:.0f}s)', flush=True)
        n_ep += len(ep); del sc; pd.DataFrame(log).to_pickle(f'{OUT}/servo_log.pkl')
    json.dump(dict(noise=NOISE, H0=H0, policies=POL, episodes=n_ep), open(f'{OUT}/servo_config.json', 'w')); print('episodes', n_ep)


NAMES = dict(ibvs='IBVS, point features (λ=0.5, 20 steps)', ibvs1='IBVS, point features (λ=1, 5 steps)', geo='PBVS: rot.-aware SuperGlue + E + flight-height scale (5 steps)',
             s2p='DreamNav (full, Table 5)', s1='DreamNav Stage I only', mlp='MASt3R-stat MLP')
def report():
    L = pd.concat([pd.read_pickle(f'{OUT}/servo_log.pkl'), pd.read_pickle(f'{SIM}/nav_log.pkl')], ignore_index=True); L = L[L.pol.isin(NAMES)]; ne = L[L.pol == 'ibvs'].ep.nunique(); L = L[L.ep < ne]
    M = [f'# Closed-loop servoing baselines in the Table-5 simulator ({ne} episodes; same starts/goals and actuation noise)\n', '| Policy | start med. (m) | after 1 | after 2 | after 5 | final | SR@1m after 1 | after 2 | after 5 | final | heading err final (med.) | tilt err final |', '|---|' + '---:|' * 11]
    for p in NAMES:
        d = L[L.pol == p]
        if not len(d): continue
        T = int(d.step.max()); med = lambda k: d[d.step == min(k, T)].pos_err.median(); sr = lambda k: 100 * (d[d.step == min(k, T)].pos_err < 1).mean(); f = d[d.step == T]
        M.append(f'| {NAMES[p]} | {med(0):.2f} | {med(1):.2f} | {med(2):.2f} | {med(5):.2f} | {med(T):.2f} | {sr(1):.1f} | {sr(2):.1f} | {sr(5):.1f} | {sr(T):.1f} | {f.head_err.median():.2f} | {f.tilt_err.median():.2f} |')
    M += ['\n## Self-terminating flight (stop when the estimated remaining distance < τ; at most the policy\'s step budget)\n', '| Policy | τ (m) | stopped by itself | mean steps | pos. err at stop, median (m) | SR@1m |', '|---|---:|---:|---:|---:|---:|']
    for p in ('ibvs', 'ibvs1', 'geo', 's2p', 's1'):
        d = L[L.pol == p].sort_values(['ep', 'step'])
        if not len(d): continue
        T = int(d.step.max())
        for tau in (1.0, 2.0):
            fin, steps, auto = [], [], []
            for _, e in d.groupby('ep'):
                k = next((int(r.step) for r in e.itertuples() if r.step < T and r.pred_dist < tau), None); auto.append(k is not None); k = T if k is None else k; fin.append(float(e[e.step == k].pos_err.iloc[0])); steps.append(k)
            fin = np.array(fin); M.append(f'| {NAMES[p]} | {tau} | {100 * np.mean(auto):.1f}% | {np.mean(steps):.2f} | {np.median(fin):.2f} | {100 * (fin < 1).mean():.1f} |')
    # large vs small initial heading error (IBVS on Cartesian point features is known to break down for large rotations about the optical axis)
    M += ['\n## Final SR@1m by initial heading error\n', '| Policy | initial |Δθ| < 45° | 45–135° | > 135° |', '|---|---:|---:|---:|']
    for p in NAMES:
        d = L[L.pol == p]
        if not len(d): continue
        T = int(d.step.max()); h0 = d[d.step == 0].set_index('ep').head_err; f = d[d.step == T].set_index('ep').pos_err; cell = lambda m: f'{100 * (f[m] < 1).mean():.1f} (n={int(m.sum())})'
        M.append(f'| {NAMES[p]} | {cell(h0 < 45)} | {cell((h0 >= 45) & (h0 <= 135))} | {cell(h0 > 135)} |')
    open(f'{OUT}/servo_report.md', 'w').write('\n'.join(M) + '\n'); print('\n'.join(M))


if __name__ == '__main__':
    if sys.argv[1] == 'run': run(int(sys.argv[2]) if len(sys.argv) > 2 else None, sys.argv[3].split(',') if len(sys.argv) > 3 else None)
    else: report()
