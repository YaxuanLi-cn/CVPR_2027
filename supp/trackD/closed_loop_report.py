#!/usr/bin/env python3
"""Track D, item 1: Table 5 with repeated actuation-noise draws.

Inputs: the archived run (outputs/sim/nav_log.pkl, noise seed 0) and the runs of supp/trackD/sim_nav_seeds.py.
For every cell of Table 5 (one step; self-terminating flight with tau in {0.5, 1, 1.5, 2} m: mean number of steps, median
position error at the stop, SR@1 m) and the three policies: value per seed, mean +- std over seeds, 95 % CI from a
bootstrap over episodes (the 160 start/goal pairs, each carrying all its seeds) and over the 8 sites.
Usage: closed_loop_report.py <run_dir> [<run_dir> ...]      (run dirs contain nav_log.pkl with a `seed` column)
"""
import sys, os, json, numpy as np, pandas as pd
OUT = '/root/rasr-pairuav/outputs/supp/trackD'; SIM = '/root/rasr-pairuav/outputs/sim'; POL = ['s1', 's2', 's2p']; TAUS = [0.5, 1.0, 1.5, 2.0]; T = 5
NAMES = {'s1': 'Stage I', 's2': 'Stage I + imagination-refined stage', 's2p': 'Stage II of Table 5 (= s2 + short-range ground-plane solver)'}


def per_episode(L):
    """L: log of one seed and one policy. -> dict of per-episode arrays (episodes sorted): one-step error; per tau: error at stop, steps flown; error after each step."""
    L = L.sort_values(['ep', 'step']); eps = np.sort(L.ep.unique()); pos = L.pivot(index='ep', columns='step', values='pos_err').loc[eps].values; pdist = L.pivot(index='ep', columns='step', values='pred_dist').loc[eps].values
    out = {'ep': eps, 'pos': pos, 'head': L.pivot(index='ep', columns='step', values='head_err').loc[eps].values, 'tilt': L.pivot(index='ep', columns='step', values='tilt_err').loc[eps].values}
    for tau in TAUS:
        below = pdist[:, :T] < tau; k = np.where(below.any(1), below.argmax(1), T); out[f'k_{tau}'] = k; out[f'fin_{tau}'] = pos[np.arange(len(eps)), k]; out[f'auto_{tau}'] = below.any(1)
    return out


def cells(E, idx=None):
    """Table-5 cells from per-episode arrays (optionally on a resample of episode positions)."""
    i = np.arange(len(E['ep'])) if idx is None else idx; c = {'one_err': float(np.median(E['pos'][i, 1])), 'one_sr': float(100 * (E['pos'][i, 1] < 1).mean())}
    for tau in TAUS: c[f'steps_{tau}'] = float(E[f'k_{tau}'][i].mean()); c[f'err_{tau}'] = float(np.median(E[f'fin_{tau}'][i])); c[f'sr_{tau}'] = float(100 * (E[f'fin_{tau}'][i] < 1).mean()); c[f'auto_{tau}'] = float(100 * E[f'auto_{tau}'][i].mean()); c[f'sr05_{tau}'] = float(100 * (E[f'fin_{tau}'][i] < 0.5).mean())
    for k in range(1, T + 1): c[f'fixed_err_{k}'] = float(np.median(E['pos'][i, k])); c[f'fixed_sr_{k}'] = float(100 * (E['pos'][i, k] < 1).mean())
    c['final_head_med'] = float(np.median(E['head'][i, T])); c['final_tilt_med'] = float(np.median(E['tilt'][i, T])); return c


if __name__ == '__main__':
    A = pd.read_pickle(f'{SIM}/nav_log.pkl'); A = A[A.pol.isin(POL)].copy(); A['seed'] = 0; logs = [A]
    for d in sys.argv[1:]: logs.append(pd.read_pickle(f'{d}/nav_log.pkl'))
    L = pd.concat(logs, ignore_index=True); seeds = [int(s) for s in sorted(L.seed.unique())]; ep_site = L.drop_duplicates('ep').set_index('ep').tour.sort_index(); eps = ep_site.index.values; sites = ep_site.values; n = len(eps)
    for s in seeds: assert L[L.seed == s].ep.nunique() == n, f'seed {s} incomplete'
    E = {(s, p): per_episode(L[(L.seed == s) & (L.pol == p)]) for s in seeds for p in POL}; keys = list(cells(E[(seeds[0], POL[0])]).keys()); res = {'seeds': [int(s) for s in seeds], 'episodes': int(n), 'sites': sorted(set(sites)), 'steps': T, 'policies': NAMES}
    per_seed = {p: {int(s): cells(E[(s, p)]) for s in seeds} for p in POL}; res['per_seed'] = per_seed
    rng = np.random.default_rng(0); NB = 1000; boots_ep = [rng.integers(0, n, n) for _ in range(NB)]; us = np.unique(sites); by = {u: np.where(sites == u)[0] for u in us}; boots_site = [np.concatenate([by[u] for u in rng.choice(us, len(us))]) for _ in range(NB)]
    def mean_over_seeds(p, idx):
        cs = [cells(E[(s, p)], idx) for s in seeds]; return {k: float(np.mean([c[k] for c in cs])) for k in keys}
    BC = {(p, kind): pd.DataFrame([mean_over_seeds(p, i) for i in boots]) for p in POL for kind, boots in (('ep', boots_ep), ('site', boots_site))}     # the same resamples are used for every policy, so differences are paired
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]; summ = {}
    for p in POL:
        v = {k: np.array([per_seed[p][int(s)][k] for s in seeds]) for k in keys}
        summ[p] = {k: dict(mean=float(v[k].mean()), std=float(v[k].std(ddof=1)) if len(seeds) > 1 else 0.0, min=float(v[k].min()), max=float(v[k].max()), ci_episode=pc(BC[(p, 'ep')][k]), ci_site=pc(BC[(p, 'site')][k])) for k in keys}
    res['summary'] = summ
    # paired differences (same episodes, same seeds)
    pdiff = {}
    for a, b in (('s2p', 's1'), ('s2', 's1'), ('s2p', 's2')):
        per = {k: [float(per_seed[a][int(s)][k] - per_seed[b][int(s)][k]) for s in seeds] for k in keys}
        pdiff[f'{a} - {b}'] = {k: dict(delta=float(np.mean(per[k])), per_seed=per[k], ci_episode=pc(BC[(a, 'ep')][k] - BC[(b, 'ep')][k]), ci_site=pc(BC[(a, 'site')][k] - BC[(b, 'site')][k])) for k in keys}
    res['paired'] = pdiff; res['bootstrap_resamples'] = NB
    N = L[L.seed != 0]
    if len(N) and 'plane_used' in N: q = N[(N.pol == 's2p') & (N.step < T)]; res['plane_solver_use'] = dict(share_of_steps=float(q.plane_used.mean()), by_step={int(k): float(v) for k, v in q.groupby('step').plane_used.mean().items()}, episodes_using_it_at_least_once=float(q.groupby(['seed', 'ep']).plane_used.any().mean()))
    res['start_distance'] = dict(median=float(np.median(E[(seeds[0], 's1')]['pos'][:, 0])), mean=float(E[(seeds[0], 's1')]['pos'][:, 0].mean()))
    json.dump(res, open(f'{OUT}/closed_loop.json', 'w'), indent=1)
    ms = lambda d, f='.2f': f"{d['mean']:{f}} ± {d['std']:{f}}"; M = [f'## Table 5 with {len(seeds)} actuation-noise draws (seeds {seeds}; {n} episodes, {len(us)} sites, {T} steps)\n', 'Cells: mean ± std over seeds; SR additionally with the 95 % CI of an episode bootstrap [and of a site bootstrap].\n',
         '| Policy | one step: Err. (m) | SR (%) | ' + ' | '.join(f'tau={t:g}: #steps | Err. | SR (%)' for t in TAUS) + ' |', '|---|' + '---:|' * (2 + 3 * len(TAUS))]
    for p in POL:
        s = summ[p]; M.append(f"| {NAMES[p]} | {ms(s['one_err'])} | {ms(s['one_sr'], '.1f')} [{s['one_sr']['ci_episode'][0]:.1f}, {s['one_sr']['ci_episode'][1]:.1f}] | " + ' | '.join(f"{ms(s[f'steps_{t}'])} | {ms(s[f'err_{t}'])} | {ms(s[f'sr_{t}'], '.1f')} [{s[f'sr_{t}']['ci_episode'][0]:.1f}, {s[f'sr_{t}']['ci_episode'][1]:.1f}] [{s[f'sr_{t}']['ci_site'][0]:.1f}, {s[f'sr_{t}']['ci_site'][1]:.1f}]" for t in TAUS) + ' |')
    M += ['\n### Per seed (SR@1 m, %)\n', '| Policy | seed | one step | ' + ' | '.join(f'tau={t:g}' for t in TAUS) + ' | fixed 5 steps |', '|---|---:|' + '---:|' * (2 + len(TAUS))]
    for p in POL:
        for s in seeds: c = per_seed[p][int(s)]; M.append(f"| {p} | {s} | {c['one_sr']:.1f} | " + ' | '.join(f"{c[f'sr_{t}']:.1f}" for t in TAUS) + f" | {c['fixed_sr_5']:.1f} |")
    M += ['\n### Fixed number of steps (no stopping rule): SR@1 m after each step, mean ± std over seeds\n', '| Policy | ' + ' | '.join(f'after {k}' for k in range(1, T + 1)) + ' | median error after 5 steps (m) |', '|---|' + '---:|' * (T + 1)]
    for p in POL: s = summ[p]; M.append(f"| {NAMES[p]} | " + ' | '.join(ms(s[f'fixed_sr_{k}'], '.1f') for k in range(1, T + 1)) + f" | {ms(s['fixed_err_5'])} |")
    M += ['\n### Paired differences in SR@1 m (percentage points; same episodes and seeds), with episode-bootstrap [site-bootstrap] 95 % CIs\n', '| Difference | one step | ' + ' | '.join(f'tau={t:g}' for t in TAUS) + ' |', '|---|' + '---:|' * (1 + len(TAUS))]
    for k, v in pdiff.items(): M.append(f'| {k} | ' + ' | '.join(f"{v[c]['delta']:+.1f} [{v[c]['ci_episode'][0]:+.1f}, {v[c]['ci_episode'][1]:+.1f}] [{v[c]['ci_site'][0]:+.1f}, {v[c]['ci_site'][1]:+.1f}]" for c in ['one_sr'] + [f'sr_{t}' for t in TAUS]) + ' |')
    if 'plane_solver_use' in res: M.append(f"\nShort-range ground-plane solver inside s2p: used in {100 * res['plane_solver_use']['share_of_steps']:.1f} % of the decision steps (by step: " + ', '.join(f"{k}: {100 * v:.0f} %" for k, v in res['plane_solver_use']['by_step'].items()) + f"); {100 * res['plane_solver_use']['episodes_using_it_at_least_once']:.1f} % of the flights use it at least once.")
    open(f'{OUT}/closed_loop.md', 'w').write('\n'.join(M) + '\n'); print('\n'.join(M))
