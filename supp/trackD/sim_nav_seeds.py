#!/usr/bin/env python3
"""Closed-loop image-goal navigation with repeated actuation-noise draws (Track D, item 1).

Port of online/sim_nav.py. Identical: the simulator (one Gaussian-splat scene per site, online/sim_scene.py), the episodes
(sites of outputs/sim/tours.txt, 10 held-out goal images x 2 start cameras, same sampling seeds), the estimator
(online/engine.py), the commands of the policies and the noise model. Changed: only the seed of the actuation noise.

Several seeds are flown in ONE process (one engine, one scene in memory). Every seed owns its random stream, consumed in
exactly the order of the archived script: site after site, step after step, the five archived policies in their order
(s2p, s2, s1, s2w, mlp), episodes in order, 8 normal draws per (policy, episode). The draws that belonged to the two
policies not flown here (s2w, mlp) are generated and discarded, so seed s gives what online/sim_nav.py would have given
for s1 / s2 / s2p with `np.random.default_rng(s)`.

Policies
  s1   coarse stage only                      (Table 5, "Ours (Stage I)")
  s2   stage 1 + imagination-refined stage
  s2p  s2, and when its estimated distance is below 5 m and the ground-plane solver has >= 200 inliers the translation is
       taken from the ground-plane residual solver online/plane_servo.py   (Table 5, "Ours (Stage II)")
All three execute the estimated offset in the drone's own levelled frame (true heading and tilt are read from the airframe).

Usage: sim_nav_seeds.py --seeds 1,2,3 --out DIR [--tours 1407] [--steps 5] [--workers 4] [--loader-workers 3]
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ.setdefault(_k, '1')   # nproc reports 128 cores but the cgroup quota is ~30: without this every forked worker spawns 127 BLAS/OpenMP threads
import sys, os, json, shutil, time, argparse, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0, '/root/rasr-pairuav/online')
SIM = '/root/rasr-pairuav/outputs/sim'; LEGACY_POL = ['s2p', 's2', 's1', 's2w', 'mlp']; SWITCH = 5.0
NOISE = dict(pos_mult=0.03, pos_add=0.05, heading=0.2, tilt=0.1)   # 1 sigma per step: 3 % of the commanded move + 5 cm per axis, 0.2 deg yaw, 0.1 deg gimbal
wrap = lambda a: (np.asarray(a) + 180) % 360 - 180


def command(r, pol, heading_true, Hcam=None):
    """Estimated offset (east, north, up, d_heading, d_tilt), the policy's own distance estimate, and whether the plane solver was used."""
    h = np.deg2rad(heading_true); used = False
    if pol == 's2p':
        lx, ly, z, dist = r.s2_lx, r.s2_ly, r.s2_z, r.s2_dist
        if r.s2_dist < SWITCH and r.pl_inl >= 200 and np.isfinite(r.pl_wx): lx, ly, z = r.pl_wx * Hcam, r.pl_wy * Hcam, r.pl_wz * Hcam; dist = float(np.sqrt(lx**2 + ly**2 + z**2)); used = True
        return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly, z, r.s2_heading, r.s2_tilt, dist, used
    pre = {'s2': 's2_', 's1': 's1_'}[pol]; lx, ly, z = r[pre + 'lx'], r[pre + 'ly'], r[pre + 'z']
    return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly, z, r[pre + 'heading'], r[pre + 'tilt'], r[pre + 'dist'], used


def episodes(C, t, goals_per_tour=10, starts_per_goal=2):
    """Same sampling as online/sim_nav.py (goals: random_state=0; starts: random_state=goal index)."""
    goals = C[C.holdout].sample(min(goals_per_tour, int(C.holdout.sum())), random_state=0); ep = []
    for g in goals.itertuples():
        for s in C[C.i != g.i].sample(starts_per_goal, random_state=int(g.i)).itertuples():
            ep.append(dict(tour=t, goal=int(g.i), start=int(s.i), goal_image=g.image, start_image=s.image, g=np.array([g.e, g.n, g.u, g.heading, g.tilt]), s=np.array([s.e, s.n, s.u, s.heading, s.tilt])))
    return ep


def burn(rng): rng.standard_normal(3); rng.standard_normal(3); rng.standard_normal(); rng.standard_normal()


def run(a):
    from sim_scene import Scene, load_cameras
    from engine import Engine
    import torch
    tours_all = open(f'{SIM}/tours.txt').read().strip().split(','); tours = a.tours.split(',') if a.tours else tours_all; seeds = [int(s) for s in a.seeds.split(',')]; fly = a.policies.split(','); steps = a.steps
    assert all(p in ('s2p', 's2', 's1') for p in fly); rngs = {s: np.random.default_rng(s) for s in seeds}; os.makedirs(a.out, exist_ok=True); TMP = a.tmp; shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    eng = Engine(workers=a.workers, loader_workers=a.loader_workers, verbose=False); log = []; rec = {}; neps = 0; step0 = []; EP = []; t_all = time.time()
    last = max(tours_all.index(t) for t in tours)
    for ti, t in enumerate(tours_all[:last + 1]):
        if t not in tours:                                    # keep every seed's stream aligned with a full run
            n = len(episodes(load_cameras(t), t))
            for s in seeds:
                for _ in range(steps * len(LEGACY_POL) * n): burn(rngs[s])
            neps += n; continue
        sc = Scene(t); ep = episodes(sc.cams, t); n = len(ep); G = np.stack([e['g'] for e in ep]); t0 = time.time()
        for j, e in enumerate(ep): EP.append(dict(tour=t, ep=neps + j, goal=e['goal'], start=e['start'], **{f'g_{k}': v for k, v in zip('enuht', e['g'])}, **{f's_{k}': v for k, v in zip('enuht', e['s'])}))
        state = {(s, p): np.stack([e['s'] for e in ep]) for s in seeds for p in fly}; Hc = {s: np.full(n, np.nan) for s in seeds}
        for k in range(steps + 1):
            for s in seeds:
                for p in fly:
                    d = G - state[(s, p)]
                    for j in range(n):
                        r = dict(seed=s, tour=t, ep=neps + j, pol=p, step=k, pos_err=float(np.linalg.norm(d[j, :3])), head_err=float(abs(wrap(d[j, 3]))), tilt_err=float(abs(d[j, 4]))); log.append(r); rec[(s, p, j, k)] = r
            if k == steps: break
            files = []; tr = time.time()
            if k == 0:                                        # the first observation is the same for every seed and policy
                c0 = f'{a.step0_cache}/{t}.pkl' if a.step0_cache else None
                if c0 and os.path.exists(c0): R0 = pd.read_pickle(c0)      # the first estimate does not depend on the noise seed: computed once, reused by later runs
                else:
                    for j, e in enumerate(ep): f = f'{TMP}/{t}_start_{j}.jpg'; cv2.imwrite(f, sc.render(*e['s']), [cv2.IMWRITE_JPEG_QUALITY, 95]); files.append(f)
                    torch.cuda.empty_cache()                  # hand the rasteriser's cached blocks back before the estimator runs (shared GPU)
                    R0 = eng.predict([(f, e['goal_image']) for f, e in zip(files, ep)], plane=True, abs_heading=False)
                    if c0: os.makedirs(a.step0_cache, exist_ok=True); R0.to_pickle(c0)
                rows = {(s, p, j): R0.iloc[j] for s in seeds for p in fly for j in range(n)}
                q = R0.drop(columns=['image_a', 'image_b'], errors='ignore').copy(); q['tour'] = t; q['ep'] = neps + np.arange(n); step0.append(q)
            else:
                full, coarse = [], []
                for s in seeds:
                    for p in fly:
                        for j, e in enumerate(ep):
                            f = f'{TMP}/{t}_{s}_{p}_{j}_{k}.jpg'; cv2.imwrite(f, sc.render(*state[(s, p)][j]), [cv2.IMWRITE_JPEG_QUALITY, 95]); files.append(f); (coarse if p == 's1' else full).append(((s, p, j), f, e['goal_image']))
                rows = {}; torch.cuda.empty_cache(); print(f'   rendered {len(files)} views in {time.time() - tr:.0f}s', flush=True)
                if full:
                    Rf = eng.predict([(f, g) for _, f, g in full], plane=True, abs_heading=False); rows.update({key: Rf.iloc[i] for i, (key, _, _) in enumerate(full)})
                    print('   s/pair (stage 1+2):', {kk: round(v / len(full), 2) for kk, v in Rf.attrs['timing'].items()}, flush=True)
                if coarse:
                    R1 = eng.predict([(f, g) for _, f, g in coarse], stage2=False, abs_heading=False); rows.update({key: R1.iloc[i] for i, (key, _, _) in enumerate(coarse)})
            for s in seeds:
                rng = rngs[s]
                for p in LEGACY_POL:
                    for j in range(n):
                        if p not in fly: burn(rng); continue
                        r = rows[(s, p, j)]; st = state[(s, p)]
                        if p == 's2p' and k == 0: Hc[s][j] = r.g_H
                        de, dn, du, dh, dt, dist, used = command(r, p, st[j, 3], Hc[s][j]); mv = np.array([de, dn, du], float)
                        if p == 's2p': Hc[s][j] += du
                        mv = mv * (1 + NOISE['pos_mult'] * rng.standard_normal(3)) + NOISE['pos_add'] * rng.standard_normal(3)
                        rec[(s, p, j, k)].update(pred_dist=float(dist), pred_dh=float(dh), pred_dt=float(dt), plane_used=bool(used))
                        st[j, :3] += mv; st[j, 3] = (st[j, 3] + dh + NOISE['heading'] * rng.standard_normal()) % 360; st[j, 4] = float(np.clip(st[j, 4] + dt + NOISE['tilt'] * rng.standard_normal(), 0, 47))
            for f in files: os.remove(f)
            print(f'[{t}] step {k} done ({time.time()-t0:.0f}s) | median pos err now: ' + ' '.join(f'{s}/{p}={np.median(np.linalg.norm((G - state[(s, p)])[:, :3], axis=1)):.2f}' for s in seeds for p in fly), flush=True)
            pd.DataFrame(log).to_pickle(f'{a.out}/nav_log.pkl')
        neps += n; rec.clear(); del sc; torch.cuda.empty_cache()
        pd.concat(step0, ignore_index=True).to_pickle(f'{a.out}/step0_pred.pkl'); pd.DataFrame(EP).to_pickle(f'{a.out}/episodes.pkl')
    eng.close(); shutil.rmtree(TMP, ignore_errors=True); pd.DataFrame(log).to_pickle(f'{a.out}/nav_log.pkl')
    json.dump(dict(noise=NOISE, tours=tours, seeds=seeds, policies=fly, episodes=neps, steps=steps, seconds=time.time() - t_all, peak_gpu_mem_gb=float(torch.cuda.max_memory_allocated() / 2**30)), open(f'{a.out}/config.json', 'w'))
    print('episodes', neps, 'seconds', round(time.time() - t_all))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tours', default=None); ap.add_argument('--policies', default='s2p,s2,s1')
    ap.add_argument('--steps', type=int, default=5); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--loader-workers', type=int, default=3); ap.add_argument('--tmp', default=None); ap.add_argument('--step0-cache', default=None)
    a = ap.parse_args(); a.tmp = a.tmp or f'/dev/shm/simnav_trackD_{os.getpid()}'; run(a)
