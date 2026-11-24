#!/usr/bin/env python3
"""Classical visual servoing against DreamNav in the closed-loop simulator of Table 5 (same 160 episodes and actuation noise).
Re-scores the archived logs: outputs/reviewer_baselines/servo_log.pkl (IBVS, direct geometric servo) and outputs/sim/nav_log.pkl
(DreamNav). Success = true 3-D position error to the goal < 1 m. Intervals: bootstrap over episodes.
Output: outputs/supp/trackA/servo.json"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); from common import boot_indices, ci, OUT
S = pd.read_pickle('/root/rasr-pairuav/outputs/reviewer_baselines/servo_log.pkl'); L = pd.read_pickle('/root/rasr-pairuav/outputs/sim/nav_log.pkl'); cfg = json.load(open('/root/rasr-pairuav/outputs/reviewer_baselines/servo_config.json'))
POL = [('ibvs', S, 'IBVS, point features, gain 0.5', 20), ('ibvs1', S, 'IBVS, point features, gain 1', 5), ('geo', S, 'Direct geometric servo', 5), ('s1', L, 'DreamNav, coarse stage', 5), ('s2', L, 'DreamNav, refined stage', 5), ('s2p', L, 'DreamNav, refined + short-range solver', 5)]
R = {}
for p, D, name, T in POL:
    d = D[D.pol == p]; E = d.pivot(index='ep', columns='step', values='pos_err').sort_index(); Pd = d.pivot(index='ep', columns='step', values='pred_dist').sort_index(); n = len(E); T = min(T, int(E.columns.max())); bt = boot_indices(np.arange(n), 1000, 0)
    r = dict(name=name, episodes=n, budget=T, start_med=float(E[0].median()))
    for k in (1, 2, 5, T):
        if k in E.columns: e = E[k].values; r[f'med_after_{k}'] = float(np.median(e)); r[f'sr_after_{k}'] = float((e < 1).mean()); r[f'sr_after_{k}_ci'] = ci(lambda b: (e[b] < 1).mean(), bt)
    fin = E[T].values; r.update(final_med=float(np.median(fin)), final_mean=float(fin.mean()), final_p90=float(np.percentile(fin, 90)), final_sr=float((fin < 1).mean()), final_sr_ci=ci(lambda b: (fin[b] < 1).mean(), bt), final_lt05=float((fin < 0.5).mean()))
    mx = E.loc[:, 1:].max(1).values; r.update(excursion_gt50=float((mx > 50).mean()), farther_than_start_5m=float((mx > E[0].values + 5).mean()), ever_within_1m=float((E.loc[:, 1:] < 1).any(axis=1).mean()), max_dist_med=float(np.median(mx)))
    # self-terminating: stop the first time the policy's own estimate of the remaining distance is below tau
    for tau in (1.0, 2.0):
        st = []; fe = []
        for ep in E.index:
            k = next((s for s in range(T) if s in Pd.columns and np.isfinite(Pd.loc[ep, s]) and Pd.loc[ep, s] < tau), None); k = T if k is None else k; st.append(k); fe.append(E.loc[ep, k])
        fe = np.array(fe); r[f'stop_tau{tau:g}'] = dict(steps=float(np.mean(st)), med=float(np.median(fe)), sr=float((fe < 1).mean()), sr_ci=ci(lambda b: (fe[b] < 1).mean(), bt), stopped=float(np.mean(np.array(st) < T)))
    R[p] = r
json.dump(dict(noise=cfg['noise'], flight_height_prior=cfg['H0'], rows=R), open(f'{OUT}/trackA/servo.json', 'w'), indent=1)
for p, r in R.items(): print(f"{r['name']:42s} budget {r['budget']:2d} | SR after 1/2/5/final: {100*r.get('sr_after_1',0):5.1f} {100*r.get('sr_after_2',0):5.1f} {100*r.get('sr_after_5',0):5.1f} {100*r['final_sr']:5.1f} | final med {r['final_med']:.2f} mean {r['final_mean']:.2f} | >50m {100*r['excursion_gt50']:.1f}% | tau=1: {r['stop_tau1']['steps']:.2f} steps, SR {100*r['stop_tau1']['sr']:.1f}")
