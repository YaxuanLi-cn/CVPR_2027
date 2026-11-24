#!/usr/bin/env python3
"""Track B, part 7: the classical baseline on UNSCREENED pairs.

Every released PairUAV pair passed a SIFT + MAGSAC screen, so SIFT matches every test pair by construction. Track D sampled
4 directed pairs per test building uniformly from all 150 x 149 camera pairs (outputs/supp/trackD/cache/unfiltered_pairs.pkl,
1,160 pairs, labels rebuilt from the KML files). Here the SIFT + ratio test + E + flight-height baseline is run on them with
the settings, the inlier threshold and the flight-height constant fitted on the 300 calibration pairs (nothing is re-fitted),
optionally the steered DeDoDe matcher (single pass), and DreamNav Stage I / II from Track D's online-engine predictions on the
same pairs (as many as exist). Camera frame, supp/common.py, bootstrap over buildings.
Breakdowns use Track D's definitions: `inside` = all five pose differences inside the range of the released pairs,
`screen_pass` = Track D's re-implementation of the released screen (>= 20 SIFT inliers on >= 4 cells of a 4 x 4 grid),
`ovl` = fraction of the source image whose ground footprint falls inside the target image.
Output: outputs/supp/trackB/unscreened.json / .md
"""
import sys, os, json
import numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/supp', f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB'); sys.path.append(f'{ROOT}/supp/trackD')
import common as C
from fusion_lib import add_targets
from rb_recon import subset
from eval_pairuav import pick, unit, to_metric, fit_const, NAMES
OUT = f'{ROOT}/outputs/supp/trackB'; TD = f'{ROOT}/outputs/supp/trackD'; P = json.load(open(f'{OUT}/pairuav.json'))
U = pd.read_pickle(f'{TD}/cache/unfiltered_pairs.pkl').reset_index(drop=True); n = len(U); ids = U.index.values
# ---- Track D's descriptors of every pair
S = json.load(open(f'{TD}/dataset_stats.json'))['candidates_vs_selected']['selected_label_range']; inside = np.ones(n, bool)
for k in ('x', 'y', 'z', 'tilt', 'heading'): inside &= (U[k].values >= S[k][0]) & (U[k].values <= S[k][1])
U['inside'] = inside; Q = pd.read_pickle(f'{TD}/cache/unfiltered_sift.pkl').reset_index(drop=True); assert (Q.a.values == U.a.values).all() and (Q.tour.values == U.tour.values).all() and (Q.b.values == U.b.values).all()
U['td_sift_inliers'] = Q.sift_inliers.values; U['td_sift_grid_cells'] = Q.sift_grid_cells.values; U['screen_pass'] = (U.td_sift_inliers >= 20) & (U.td_sift_grid_cells >= 4)
from unfiltered import overlap
U['ovl'] = [overlap((r.e_a, r.n_a, r.u_a), r.heading_a, r.tilt_a, (r.e_b, r.n_b, r.u_b), r.heading_b, r.tilt_b) for r in U.itertuples()]
gt_all = C.gt_cam(U); Tc = add_targets(subset(300, split='train')); idc = Tc.manifest_index.values
# ---- geometric baselines (threshold and flight height from the calibration pairs)
M = {}
for m, tta, label in (('sift', False, 'SIFT + ratio test + E + flight-height prior'), ('dedode', False, 'DeDoDe-B + SO(2) steerer (C8), single pass + E + flight-height prior')):
    f = f'{OUT}/geom/{m}_unscreened.pkl'
    if not os.path.exists(f): continue
    thr = P['threshold_selection'][f'{m}|{"tta" if tta else "plain"}']['chosen']; G = pd.read_pickle(f); Gc = pd.read_pickle(f'{OUT}/geom/{m}_train.pkl'); mc = unit(pick(Gc, idc, thr, tta)); H, nfit = fit_const(mc[mc.have.values], Tc.dist.values[mc.have.values], 'height_p75ab')
    mt = unit(pick(G, ids, thr, tta)); p, ok = to_metric(mt, 'height_p75ab', H); M[label] = dict(pred=p, ok=ok, have=mt.have.values, thr=thr, H=H, inl=mt.inl.values, key=m)
# ---- DreamNav from Track D's engine run (rows = the first len(R) pairs of the table)
fr = f'{TD}/cache/unfiltered_pred.pkl'; D = {}
if os.path.exists(fr):
    R = pd.read_pickle(fr).reset_index(drop=True); nr = len(R); F = pd.concat([U.iloc[:nr].reset_index(drop=True), R.rename(columns={c: 'p_' + c[3:] for c in R.columns if c.startswith('s2_') and c[3:] in ('lx', 'ly', 'z', 'dist', 'tilt', 'heading')})], axis=1)
    for nm, pre in (('DreamNav Stage I', 's1_'), ('DreamNav Stage II', 'p_')):
        p = C.dreamnav(F, pre, 'cam'); have = np.zeros(n, bool); have[:nr] = True; D[nm] = dict(pred={k: np.r_[np.asarray(v, float), np.full(n - nr, np.nan)] for k, v in p.items() if k != 'abs_heading'}, ok=np.ones(n, bool), have=have)
    U['zm_inl'] = np.r_[R.zm_inl.values, np.full(n - nr, np.nan)]
ALL = {**M, **D}; res = dict(n_pairs=n, buildings=int(U.tour.nunique()), n_dreamnav=int(next(iter(D.values()))['have'].sum()) if D else 0, share_inside=float(U.inside.mean()), share_screen_pass=float(U.screen_pass.mean()), share_inside_and_pass=float((U.inside & U.screen_pass).mean()),
                           also_in_release=int(U.is_selected.sum()), settings={k: dict(thr_px=v['thr'], H=v['H']) for k, v in M.items()}, sets={}, paired={}, joint={}, by_overlap={})
def stats(name, sub):
    e = C.errors({k: np.asarray(v, float)[sub] for k, v in ALL[name]['pred'].items()}, {k: v[sub] for k, v in gt_all.items()}); s = C.summarize(e); b = C.boot_indices(U.tour.values[sub], 1000, 0)
    s['ci'] = {k: C.ci(f, b) for k, f in (('d_mae', lambda i: e['d'][i].mean()), ('sr_d_1m', lambda i: (e['d'][i] <= 1).mean()), ('sr_epe_1m', lambda i: (e['epe'][i] < 1).mean()), ('sr_epe_2m', lambda i: (e['epe'][i] < 2).mean()), ('epe_med', lambda i: np.median(e['epe'][i])))}
    s['fail'] = float(100 * (1 - ALL[name]['ok'][sub].mean())); return s, e
SUBS = {'all unscreened pairs': np.ones(n, bool), 'inside the pose-difference ranges of the release': U.inside.values, 'outside those ranges': ~U.inside.values, 'passing the screen proxy': U.screen_pass.values, 'failing the screen proxy': ~U.screen_pass.values,
        'inside the ranges and passing the screen proxy': (U.inside & U.screen_pass).values, 'inside the ranges, failing the screen proxy': (U.inside & ~U.screen_pass).values}
f1 = lambda v: f'{100 * v:.1f}'; L = [f'# Unscreened pairs ({n} directed pairs, 4 per test building, uniform over all camera pairs; Track D)\n',
     f"{f1(U.inside.mean())} % of them lie inside the pose-difference ranges of the released pairs, {f1(U.screen_pass.mean())} % pass Track D's proxy of the released SIFT screen, {f1((U.inside & U.screen_pass).mean())} % both; {int(U.is_selected.sum())} are released pairs. "
     + '; '.join(f"{k}: inlier threshold {v['thr']} px, H = {v['H']:.1f} m (both from the 300 calibration pairs, nothing re-fitted)" for k, v in M.items()) + f". DreamNav predictions (Track D, online engine) exist for {res['n_dreamnav']} of the {n} pairs; rows marked with their own n. 95% CI: bootstrap over buildings.\n"]
HEAD = ['| pairs | method | n | fail % | d MAE [CI] | SR_d@1m [CI] | EPE mean | EPE med. [CI] | SR_EPE@1m [CI] | SR_EPE@2m | e_theta MAE / med. | e_psi MAE / med. | dir. med. |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
L += ['## Same pairs for every method (pairs with DreamNav predictions)\n'] + HEAD if D else HEAD
common = np.ones(n, bool)
for v in ALL.values(): common &= v['have']
def emit(sname, sub, names, store):
    for nm in names:
        if not sub.sum() >= 5: continue
        s, e = stats(nm, sub); store.setdefault(sname, {})[nm] = s; c = s['ci']
        L.append(f"| {sname} | {nm} | {s['n']} | {s['fail']:.1f} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} [{c['epe_med'][0]:.2f}, {c['epe_med'][1]:.2f}] | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | {s['theta_mae']:.2f} / {s['theta_med']:.3f} | {s['psi_mae']:.2f} / {s['psi_med']:.3f} | {s['dir_med']:.1f} |")
res['sets_common'] = {}
for sname, sub in SUBS.items(): emit(sname, sub & common, list(ALL), res['sets_common'])
if D and not common.all():
    L += ['\n## Geometric baselines on all unscreened pairs they were run on\n'] + HEAD
    for sname, sub in SUBS.items():
        for nm in M: emit(sname, sub & M[nm]['have'], [nm], res['sets'])
# ---- paired differences and who succeeds where (pairs common to all methods)
if D and 'SIFT + ratio test + E + flight-height prior' in M:
    B = 'SIFT + ratio test + E + flight-height prior'
    L += ['\n## Paired differences, DreamNav minus the SIFT baseline (same pairs; 95% CI over buildings)\n', '| pairs | n | vs | SR_d@1m (points) | SR_EPE@1m (points) | SR_EPE@2m (points) | EPE median (m) | d MAE (m) |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for sname, sub in SUBS.items():
        sub = sub & common
        if sub.sum() < 20: continue
        _, eb = stats(B, sub); b = C.boot_indices(U.tour.values[sub], 1000, 0); m = int(sub.sum())
        for nm in D:
            _, e = stats(nm, sub); fns = dict(sr_d_points=lambda i: 100 * ((e['d'][i] <= 1).mean() - (eb['d'][i] <= 1).mean()), sr_epe1_points=lambda i: 100 * ((e['epe'][i] < 1).mean() - (eb['epe'][i] < 1).mean()), sr_epe2_points=lambda i: 100 * ((e['epe'][i] < 2).mean() - (eb['epe'][i] < 2).mean()),
                                              epe_med=lambda i: np.median(e['epe'][i]) - np.median(eb['epe'][i]), d_mae=lambda i: e['d'][i].mean() - eb['d'][i].mean())
            r = {k: dict(delta=float(f(np.arange(m))), ci=C.ci(f, b)) for k, f in fns.items()}; res['paired'].setdefault(sname, {})[nm] = dict(n=m, **r)
            L.append(f'| {sname} | {m} | {nm} | ' + ' | '.join(f"{r[k]['delta']:+.2f} [{r[k]['ci'][0]:+.2f}, {r[k]['ci'][1]:+.2f}]" for k in ('sr_d_points', 'sr_epe1_points', 'sr_epe2_points', 'epe_med', 'd_mae')) + ' |')
            if nm == 'DreamNav Stage II':
                a, c2 = eb['epe'] < 1, e['epe'] < 1; res['joint'][sname] = dict(n=m, both=float((a & c2).mean()), sift_only=float((a & ~c2).mean()), dreamnav_only=float((~a & c2).mean()), neither=float((~a & ~c2).mean()))
    L += ['\n## Who succeeds (EPE < 1 m): SIFT baseline vs DreamNav Stage II\n', '| pairs | n | both | SIFT baseline only | DreamNav Stage II only | neither |', '|---|---:|---:|---:|---:|---:|']
    for sname, j in res['joint'].items(): L.append(f"| {sname} | {j['n']} | {f1(j['both'])} | {f1(j['sift_only'])} | {f1(j['dreamnav_only'])} | {f1(j['neither'])} |")
# ---- by ground-footprint overlap and by the screen statistics
bins = [0, 0.02, 0.2, 0.4, 0.6, 1.0001]; lab = ['< 0.02', '0.02-0.2', '0.2-0.4', '0.4-0.6', '>= 0.6']
L += ['\n## By ground-footprint overlap (pairs common to all methods)\n', '| overlap | n | share | passing the screen proxy | ' + ' | '.join(f'{nm}: fail % / SR_d@1m / SR_EPE@1m / EPE med.' for nm in ALL) + ' |', '|---|---:|---:|---:|' + '---|' * len(ALL)]
for lo, hi, l in zip(bins[:-1], bins[1:], lab):
    sub = (U.ovl.values >= lo) & (U.ovl.values < hi) & common
    if sub.sum() < 5: continue
    row = dict(n=int(sub.sum()), share=float(sub.sum() / common.sum()), screen_pass=float(U.screen_pass.values[sub].mean())); cells = []
    for nm in ALL: s, e = stats(nm, sub); row[nm] = dict(fail=s['fail'], sr_d_1m=s['sr_d_1m'], sr_epe_1m=s['sr_epe_1m'], epe_med=s['epe_med']); cells.append(f"{s['fail']:.1f} / {f1(s['sr_d_1m'])} / {f1(s['sr_epe_1m'])} / {s['epe_med']:.2f}")
    res['by_overlap'][l] = row; L.append(f"| {l} | {row['n']} | {f1(row['share'])} | {f1(row['screen_pass'])} | " + ' | '.join(cells) + ' |')
res['by_sift_inliers_of_screen_proxy'] = {}
L += ['\n## By the number of SIFT inliers of the screen proxy (pairs common to all methods)\n', '| SIFT inliers (Track D proxy) | n | ' + ' | '.join(f'{nm}: fail % / SR_d@1m / SR_EPE@1m / EPE med.' for nm in ALL) + ' |', '|---|---:|' + '---|' * len(ALL)]
for lo, hi, l in ((0, 8, '< 8'), (8, 20, '8-19'), (20, 50, '20-49'), (50, 200, '50-199'), (200, 1e9, '>= 200')):
    sub = (U.td_sift_inliers.values >= lo) & (U.td_sift_inliers.values < hi) & common
    if sub.sum() < 5: continue
    row = dict(n=int(sub.sum())); cells = []
    for nm in ALL: s, e = stats(nm, sub); row[nm] = dict(fail=s['fail'], sr_d_1m=s['sr_d_1m'], sr_epe_1m=s['sr_epe_1m'], epe_med=s['epe_med']); cells.append(f"{s['fail']:.1f} / {f1(s['sr_d_1m'])} / {f1(s['sr_epe_1m'])} / {s['epe_med']:.2f}")
    res['by_sift_inliers_of_screen_proxy'][l] = row; L.append(f"| {l} | {row['n']} | " + ' | '.join(cells) + ' |')
json.dump(res, open(f'{OUT}/unscreened.json', 'w'), indent=1); open(f'{OUT}/unscreened.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
