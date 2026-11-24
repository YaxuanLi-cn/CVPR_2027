#!/usr/bin/env python3
"""Closed-loop image-goal navigation in the splat simulator (online/sim_scene.py).

Episode: the goal is a REAL dataset image that the simulator never saw (held-out), with known pose. The drone starts at the
pose of another camera of the same tour (so the initial offset follows the dataset distribution), observes a rendered
view, estimates the 5-D offset to the goal, executes it with actuation noise, and repeats for T steps.
Policies (each flies its own trajectory):
  s2      DreamNav, refined estimate (stage 1 + imagination), command executed in the drone's own levelled frame
  s1      DreamNav, coarse estimate only
  s2w     DreamNav refined, command executed in the WORLD frame (east/north from the learned absolute heading)
  mlp     MASt3R-statistics MLP (the RASR + sin/cos control), own-frame command
Every step's estimate and true error is logged, so stopping rules can be evaluated afterwards.
Usage: sim_nav.py run [goals_per_tour] [starts_per_goal] [steps]   |   sim_nav.py report
"""
import sys, os, json, shutil, time, numpy as np, pandas as pd, cv2
sys.path.insert(0, '/root/rasr-pairuav/online')
OUT = '/root/rasr-pairuav/outputs/sim'; TMP = '/dev/shm/simnav'; POL = ['s2p', 's2', 's1', 's2w', 'mlp']; SWITCH = 5.0   # s2p: below this estimated distance the translation comes from the ground-plane residual (plane_servo.py)
NOISE = dict(pos_mult=0.03, pos_add=0.05, heading=0.2, tilt=0.1)   # per-step actuation noise (1 sigma): 3 % of the commanded move + 5 cm per axis, 0.2 deg yaw, 0.1 deg gimbal
wrap = lambda a: (np.asarray(a) + 180) % 360 - 180

def command(r, pol, heading_true, Hcam=None):
    """Estimated offset (east, north, up, d_heading, d_tilt) the policy wants to execute, and its own distance estimate."""
    h = np.deg2rad(heading_true)
    if pol == 's2p':
        lx, ly, z, dist = r.s2_lx, r.s2_ly, r.s2_z, r.s2_dist
        if r.s2_dist < SWITCH and r.pl_inl >= 200 and np.isfinite(r.pl_wx): lx, ly, z = r.pl_wx * Hcam, r.pl_wy * Hcam, r.pl_wz * Hcam; dist = float(np.sqrt(lx**2 + ly**2 + z**2))
        return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly, z, r.s2_heading, r.s2_tilt, dist
    if pol == 's2w': return r.s2_x, r.s2_y, r.s2_zw, r.s2_heading, r.s2_tilt, r.s2_dist
    pre = {'s2': 's2_', 's1': 's1_', 'mlp': 'n_'}[pol]; lx, ly, z = r[pre + 'lx'], r[pre + 'ly'], r[pre + 'z']
    return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly, z, r[pre + 'heading'], r[pre + 'tilt'], (r[pre + 'dist'] if pre != 'n_' else float(np.sqrt(lx**2 + ly**2 + z**2)))

def run(goals_per_tour=10, starts_per_goal=2, steps=6):
    from sim_scene import Scene
    from engine import Engine
    tours = open(f'{OUT}/tours.txt').read().strip().split(','); rng = np.random.default_rng(0)
    os.makedirs(TMP, exist_ok=True); eng = Engine(verbose=False); log = []; eps = []; fid = []
    for t in tours:
        while not os.path.exists(f'{OUT}/scenes/{t}.json'): time.sleep(20)      # scene still being built
        sc = Scene(t); C = sc.cams; goals = C[C.holdout].sample(min(goals_per_tour, int(C.holdout.sum())), random_state=0)
        ep = []
        for g in goals.itertuples():
            for s in C[C.i != g.i].sample(starts_per_goal, random_state=int(g.i)).itertuples():
                ep.append(dict(tour=t, goal=g.i, start=s.i, goal_image=g.image, start_image=s.image, g=np.array([g.e, g.n, g.u, g.heading, g.tilt]), s=np.array([s.e, s.n, s.u, s.heading, s.tilt])))
        state = {p: np.stack([e['s'] for e in ep]) for p in POL}; G = np.stack([e['g'] for e in ep]); t0 = time.time()
        Hc = np.full(len(ep), np.nan)                                     # s2p: camera height above ground, estimated once at the first step and then dead-reckoned with the commanded climbs
        for k in range(steps + 1):
            # true errors before acting at step k
            for p in POL:
                d = G - state[p]
                for j, e in enumerate(ep): log.append(dict(tour=t, ep=len(eps) + j, pol=p, step=k, pos_err=float(np.linalg.norm(d[j, :3])), head_err=float(abs(wrap(d[j, 3]))), tilt_err=float(abs(d[j, 4]))))
            if k == steps: break
            pairs = []; meta = []
            for p in POL:
                for j, e in enumerate(ep):
                    if k == 0 and p != POL[0]: continue                       # all policies share the same first observation
                    f = f'{TMP}/{t}_{p}_{j}_{k}.jpg'; cv2.imwrite(f, sc.render(*state[p][j]), [cv2.IMWRITE_JPEG_QUALITY, 95]); pairs.append((f, e['goal_image'])); meta.append((p, j))
            if k == 0: pairs += [(e['start_image'], e['goal_image']) for e in ep]   # fidelity check: same pairs with the REAL start image
            mlp_rows = [i for i, (p, j) in enumerate(meta) if p == 'mlp'] if k else []; full_rows = [i for i in range(len(pairs)) if i not in set(mlp_rows)]
            R = [None] * len(pairs)
            Rf = eng.predict([pairs[i] for i in full_rows], plane=True)
            for a, i in enumerate(full_rows): R[i] = Rf.iloc[a]
            if mlp_rows:
                Rm = eng.predict([pairs[i] for i in mlp_rows], priors_only=True)
                for a, i in enumerate(mlp_rows): R[i] = Rm.iloc[a]
            if k == 0:
                n = len(ep)
                for j in range(n):
                    d = G[j] - ep[j]['s']; h = np.deg2rad(ep[j]['s'][3]); lx = np.cos(h) * d[0] - np.sin(h) * d[1]; ly = np.sin(h) * d[0] + np.cos(h) * d[1]
                    for nm, r in (('rendered', R[j]), ('real', R[n + j])):
                        fid.append(dict(tour=t, ep=len(eps) + j, obs=nm, e_lx=abs(r.s2_lx - lx), e_ly=abs(r.s2_ly - ly), e_z=abs(r.s2_z - d[2]), e_head=abs(wrap(r.s2_heading - d[3])), e_tilt=abs(r.s2_tilt - d[4]), e1_lx=abs(r.s1_lx - lx), e1_ly=abs(r.s1_ly - ly), zm_inl=r.zm_inl, im_inl=r.im_inl))
                meta = [(p, j) for p in POL for j in range(n)]; R = [R[j] for p in POL for j in range(n)]
            for (p, j), r in zip(meta, R):
                if p == 's2p' and k == 0: Hc[j] = r.g_H
                de, dn, du, dh, dt, dist = command(r, p, state[p][j, 3], Hc[j]); mv = np.array([de, dn, du], float)
                if p == 's2p': Hc[j] += du
                mv = mv * (1 + NOISE['pos_mult'] * rng.standard_normal(3)) + NOISE['pos_add'] * rng.standard_normal(3)
                rec = next(x for x in reversed(log) if x['pol'] == p and x['ep'] == len(eps) + j and x['step'] == k); rec.update(pred_dist=float(dist), pred_dh=float(dh), pred_dt=float(dt))
                state[p][j, :3] += mv; state[p][j, 3] = (state[p][j, 3] + dh + NOISE['heading'] * rng.standard_normal()) % 360; state[p][j, 4] = float(np.clip(state[p][j, 4] + dt + NOISE['tilt'] * rng.standard_normal(), 0, 47))
            for f, _ in pairs:
                if f.startswith(TMP): os.remove(f)
            print(f'[{t}] step {k} done ({time.time()-t0:.0f}s) | median pos err now: ' + ' '.join(f'{p}={np.median(np.linalg.norm((G - state[p])[:, :3], axis=1)):.2f}' for p in POL), flush=True)
            pd.DataFrame(log).to_pickle(f'{OUT}/nav_log.pkl'); pd.DataFrame(fid).to_pickle(f'{OUT}/nav_fidelity.pkl')
        eps += ep; del sc
    eng.close(); shutil.rmtree(TMP, ignore_errors=True); pd.DataFrame(log).to_pickle(f'{OUT}/nav_log.pkl'); pd.DataFrame(fid).to_pickle(f'{OUT}/nav_fidelity.pkl'); json.dump(dict(noise=NOISE, tours=tours, episodes=len(eps), steps=steps), open(f'{OUT}/nav_config.json', 'w')); print('episodes', len(eps))

if __name__ == '__main__':
    if sys.argv[1] == 'run': run(*(int(x) for x in sys.argv[2:5]))

NAMES = {'s2p': 'DreamNav + ground-plane short-range servo', 's2': 'DreamNav (stage 1 + imagination)', 's1': 'DreamNav stage 1 only', 's2w': 'DreamNav, world-frame command', 'mlp': 'MASt3R-stat MLP (RASR + sin/cos)'}
def report():
    L = pd.read_pickle(f'{OUT}/nav_log.pkl'); Fd = pd.read_pickle(f'{OUT}/nav_fidelity.pkl'); T = int(L.step.max()); ne = L.ep.nunique(); cfg = json.load(open(f'{OUT}/nav_config.json')) if os.path.exists(f'{OUT}/nav_config.json') else {}
    sc = [json.load(open(f'{OUT}/scenes/{t}.json')) for t in L.tour.unique() if os.path.exists(f'{OUT}/scenes/{t}.json')]
    M = [f'# Closed-loop navigation in simulation ({ne} episodes, {L.tour.nunique()} held-out test tours, {T} steps)\n',
         f'Simulator: one Gaussian-splat reconstruction per tour (135 posed dataset images each; 15 held out). Held-out views rendered at full resolution: PSNR {np.mean([s["holdout_psnr_fullres"] for s in sc]):.1f} dB on average.',
         f'Goal = real held-out image; start = pose of another camera in the tour; observation = rendered view. Actuation noise per step: {cfg.get("noise", NOISE)}.\n',
         '## Position error to the goal after each step (m)\n', '| Policy | start (median) | ' + ' | '.join(f'step {k}' for k in range(1, T + 1)) + f' | mean at step {T} |', '|---|---:|' + '---:|' * (T + 1)]
    for p in POL:
        d = L[L.pol == p]; M.append(f'| {NAMES[p]} | ' + ' | '.join(f'{d[d.step == k].pos_err.median():.2f}' for k in range(T + 1)) + f' | {d[d.step == T].pos_err.mean():.2f} |')
    M += ['\nMedian over episodes. Step 1 is the one-shot (open-loop) result.\n', f'## Success rate (%)\n', '| Policy | <1 m after 1 step | <1 m after 2 | <1 m final | <0.5 m final | <0.25 m final | <0.5 m & heading <0.5 deg & tilt <0.5 deg, final | heading err final (median, deg) | tilt err final |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in POL:
        d = L[L.pol == p]; f = d[d.step == T]; s = lambda k, th: 100 * (d[d.step == k].pos_err < th).mean()
        M.append(f'| {NAMES[p]} | {s(1, 1):.1f} | {s(2, 1):.1f} | {s(T, 1):.1f} | {s(T, .5):.1f} | {s(T, .25):.1f} | {100 * ((f.pos_err < .5) & (f.head_err < .5) & (f.tilt_err < .5)).mean():.1f} | {f.head_err.median():.3f} | {f.tilt_err.median():.3f} |')
    # self-terminating variant: stop the first time the policy's own distance estimate drops below tau
    M += ['\n## Self-terminating flight (stop when the estimated remaining distance < tau)\n', '| Policy | tau (m) | stopped by itself | mean steps flown | position error at stop: median (m) | <1 m | <0.5 m |', '|---|---:|---:|---:|---:|---:|---:|']
    for p in ('s2p', 's2', 's1'):
        d = L[L.pol == p].sort_values(['ep', 'step'])
        for tau in (0.5, 1.0, 2.0):
            fin, steps, auto = [], [], []
            for _, e in d.groupby('ep'):
                k = next((int(r.step) for r in e.itertuples() if r.step < T and r.pred_dist < tau), None); auto.append(k is not None); k = T if k is None else k; fin.append(float(e[e.step == k].pos_err.iloc[0])); steps.append(k)
            fin = np.array(fin); M.append(f'| {NAMES[p]} | {tau} | {100 * np.mean(auto):.1f}% | {np.mean(steps):.2f} | {np.median(fin):.2f} | {100 * (fin < 1).mean():.1f}% | {100 * (fin < .5).mean():.1f}% |')
    # does the estimate stay reliable close to the goal?
    M += ['\n## Estimated vs true remaining distance (all steps pooled)\n', '| Policy | true remaining distance (m) | observations | estimated distance: median (m) | median abs. difference (m) |', '|---|---|---:|---:|---:|']
    for p in ('s2', 's2p'):
        d = L[(L.pol == p) & (L.step < T)].copy(); d['bin'] = pd.cut(d.pos_err, [0, 0.25, 0.5, 1, 2, 5, 10, 20, 100])
        for b, e in d.groupby('bin', observed=True): M.append(f'| {NAMES[p]} | {b.left:g}-{b.right:g} | {len(e)} | {e.pred_dist.median():.2f} | {(e.pred_dist - e.pos_err).abs().median():.2f} |')
    r, q = Fd[Fd.obs == 'rendered'], Fd[Fd.obs == 'real']
    M += ['\n## Simulator fidelity: same start/goal pairs, rendered vs real start image (first step, DreamNav estimate error)\n', '| start image | lx median (m) | ly | z | heading (deg) | tilt (deg) | zoom inliers (median) | imagination inliers |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for nm, x in (('real photo from the dataset', q), ('simulator render at the same pose', r)): M.append(f'| {nm} | {x.e_lx.median():.3f} | {x.e_ly.median():.3f} | {x.e_z.median():.3f} | {x.e_head.median():.4f} | {x.e_tilt.median():.4f} | {x.zm_inl.median():.0f} | {x.im_inl.median():.0f} |')
    open(f'{OUT}/nav_report.md', 'w').write('\n'.join(M) + '\n'); print('\n'.join(M))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'; PAL = dict(s2p='#2a78d6', s2='#eb6834', s1='#1baf7a', s2w='#e87ba4', mlp='#eda100')
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e9e8e3', 'grid.linewidth': 0.8, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ks = np.arange(T + 1)
    for p in POL:
        d = L[L.pol == p]; med = [d[d.step == k].pos_err.median() for k in ks]; lo = [d[d.step == k].pos_err.quantile(.25) for k in ks]; hi = [d[d.step == k].pos_err.quantile(.75) for k in ks]
        ax[0].fill_between(ks, lo, hi, color=PAL[p], alpha=0.12, lw=0); ax[0].plot(ks, med, color=PAL[p], lw=2, marker='o', ms=5, label=NAMES[p])
        ax[1].plot(ks, [100 * (d[d.step == k].pos_err < 1).mean() for k in ks], color=PAL[p], lw=2, marker='o', ms=5, label=NAMES[p])
    ax[0].set_yscale('log'); ax[0].set_xlabel('control steps executed'); ax[0].set_ylabel('distance to goal (m), median and IQR'); ax[0].set_title('Remaining distance to the goal'); ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_ylim(0, 100); ax[1].set_xlabel('control steps executed'); ax[1].set_ylabel('% of episodes within 1 m of the goal'); ax[1].set_title('Success rate (< 1 m)'); fig.tight_layout(); fig.savefig(f'{OUT}/nav_convergence.png', dpi=170); plt.close(fig)

if __name__ == '__main__' and sys.argv[1] == 'report': report()
