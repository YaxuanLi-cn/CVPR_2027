#!/usr/bin/env python3
"""Track B, part 6: the classical baseline (SIFT + ratio test + five-point MAGSAC++ + flight-height prior) on the FULL test set of
Table 2 (6,364 pairs; or the random subset processed so far), so that its row can sit next to the rows of Table 2.

Fitted on the 300 calibration pairs only: the inlier threshold (already chosen, pairuav.json) and the flight height H.
Reported in the camera frame (no heading involved) and in the east-north-up frame of Table 2, where the absolute heading of the
source camera comes from the paper's heading network combined with the baseline's own heading change (Eq. 4), or from the true heading.
Output: outputs/supp/trackB/sift_full.json / .md
"""
import sys, os, json
import numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/supp', f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
import common as C
from fusion_lib import add_targets
from rb_recon import subset
from eval_pairuav import pick, unit, to_metric, fit_const
OUT = f'{ROOT}/outputs/supp/trackB'; P = json.load(open(f'{OUT}/pairuav.json')); thr = P['threshold_selection']['sift|plain']['chosen']
T6 = C.load_test(); Tc = add_targets(subset(300, split='train')); idc = Tc.manifest_index.values
G = pd.read_pickle(f'{OUT}/geom/sift_test_full.pkl'); Gc = pd.read_pickle(f'{OUT}/geom/sift_train.pkl'); mc = unit(pick(Gc, idc, thr, False)); H, nfit = fit_const(mc, Tc.dist.values, 'height_p75ab')
mt = unit(pick(G, T6.manifest_index.values, thr, False)); sub = mt.have.values; T = T6[sub].reset_index(drop=True); mt = mt[sub].reset_index(drop=True); n = len(T); print('pairs', n, 'of', len(T6), '| thr', thr, '| H', round(H, 2))
p, ok = to_metric(mt, 'height_p75ab', H); boots = C.boot_indices(T.tour.values, 1000, 0); rows = {}
def add(name, pred, gt, frame):
    e = C.errors(pred, gt); s = C.summarize(e); s['frame'] = frame
    s['ci'] = {k: C.ci(f, boots) for k, f in (('d_mae', lambda b: e['d'][b].mean()), ('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('epe_med', lambda b: np.median(e['epe'][b])), ('x_mae', lambda b: e['x'][b].mean()), ('y_mae', lambda b: e['y'][b].mean()), ('z_mae', lambda b: e['z'][b].mean()),
                                              ('theta_mae', lambda b: e['theta'][b].mean()), ('psi_mae', lambda b: e['psi'][b].mean()), ('sr_5dof_1m_1deg', lambda b: ((e['epe'][b] < 1) & (e['theta'][b] < 1) & (e['psi'][b] < 1)).mean()))}
    rows[name] = s; return e
gc, gw = C.gt_cam(T), C.gt_world(T); B = 'SIFT + ratio test + E (MAGSAC++) + flight-height prior'
eb = add(f'{B} (camera frame)', p, gc, 'cam')
for tag, h in (('ENU, heading network of the paper', C.fuse_heading(T, C.BV + C.V2, p['heading'])[0]), ('ENU, true heading', T.heading_a.values)):
    x, y = C.world(h, p['x'], p['y']); add(f'{B} ({tag})', dict(x=x, y=y, z=p['z'], tilt=p['tilt'], heading=p['heading']), gw, 'world')
E = {}
for nm, pre in (('DreamNav Stage I', 's1_'), ('DreamNav Stage II', 'p_')):
    E[nm] = add(f'{nm} (camera frame)', C.dreamnav(T, pre, 'cam'), gc, 'cam'); add(f'{nm} (ENU, learned heading)', C.dreamnav(T, pre, 'net'), gw, 'world'); add(f'{nm} (ENU, true heading)', C.dreamnav(T, pre, 'true'), gw, 'world')
paired = {}
for nm, e in E.items():
    fns = dict(sr_d_points=lambda b: 100 * ((eb['d'][b] <= 1).mean() - (e['d'][b] <= 1).mean()), sr_epe1_points=lambda b: 100 * ((eb['epe'][b] < 1).mean() - (e['epe'][b] < 1).mean()), d_mae=lambda b: eb['d'][b].mean() - e['d'][b].mean(),
               epe_mean=lambda b: eb['epe'][b].mean() - e['epe'][b].mean(), epe_med=lambda b: np.median(eb['epe'][b]) - np.median(e['epe'][b]), theta_mae=lambda b: eb['theta'][b].mean() - e['theta'][b].mean(), psi_mae=lambda b: eb['psi'][b].mean() - e['psi'][b].mean())
    paired[nm] = {k: dict(delta=float(f(np.arange(n))), ci=C.ci(f, boots)) for k, f in fns.items()}
# who succeeds where the other fails, and the comparison as a function of the SIFT inliers recorded by the dataset screen
e2 = E['DreamNav Stage II']; e1 = E['DreamNav Stage I']; joint = {}
for lab, a, b2 in (('EPE < 1 m', eb['epe'] < 1, e2['epe'] < 1), ('range error d <= 1 m', eb['d'] <= 1, e2['d'] <= 1)):
    joint[lab] = dict(both=float((a & b2).mean()), baseline_only=float((a & ~b2).mean()), dreamnav2_only=float((~a & b2).mean()), neither=float((~a & ~b2).mean()), oracle_best_of_two=float((a | b2).mean()),
                      ci_baseline_only=C.ci(lambda i: (a[i] & ~b2[i]).mean(), boots), ci_dreamnav2_only=C.ci(lambda i: (~a[i] & b2[i]).mean(), boots))
bins = {}
for cn, v, edges, labs in (('SIFT inliers recorded by the dataset screen', T.sift_inliers.values, [0, 50, 100, 200, 400, 800, 1e9], ['20-49', '50-99', '100-199', '200-399', '400-799', '>= 800']), ('true distance (m)', T.dist.values, [0, 10, 20, 30, 1e9], ['< 10', '10-20', '20-30', '>= 30']),
                           ('|tilt change| (deg)', np.abs(T.tilt.values), [0, 10, 20, 30, 46], ['0-10', '10-20', '20-30', '30-45']), ('|heading change| (deg)', np.abs(T.heading.values), [0, 45, 90, 135, 181], ['0-45', '45-90', '90-135', '135-180'])):
    bins[cn] = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labs):
        k = (v >= lo) & (v < hi)
        if k.sum() < 10: continue
        f = lambda e: dict(sr_d_1m=float((e['d'][k] <= 1).mean()), sr_epe_1m=float((e['epe'][k] < 1).mean()), epe_med=float(np.median(e['epe'][k])))
        idx = np.where(k)[0]; bb = C.boot_indices(T.tour.values[idx], 500, 0); dd = lambda i: 100 * ((e2['epe'][idx][i] < 1).mean() - (eb['epe'][idx][i] < 1).mean())
        bins[cn].append(dict(bin=lab, n=int(k.sum()), baseline=f(eb), stage1=f(e1), stage2=f(e2), stage2_minus_baseline_sr_epe_points=dict(delta=float(dd(np.arange(len(idx)))), ci=C.ci(dd, bb))))
json.dump(dict(n=n, n_full=len(T6), thr_px=thr, H=H, H_fitted_on=nfit, fail_pct=float(100 * (1 - ok.mean())), rows=rows, paired_baseline_minus_dreamnav_camera_frame=paired, joint_success_vs_stage2=joint, bins=bins), open(f'{OUT}/sift_full.json', 'w'), indent=1)
f1 = lambda v: f'{100 * v:.1f}'
L = [f'# Classical baseline on the test set of Table 2: {n} of {len(T6)} test pairs (fixed random order), {T.tour.nunique()} buildings\n', f'Inlier threshold {thr} px and flight height H = {H:.1f} m, both from the 300 calibration pairs. Pairs without a pose: {100 * (1 - ok.mean()):.2f}% (zero motion). 95% CI: bootstrap over buildings.\n',
     '| Method | dx MAE | dy MAE | dz MAE | d MAE [CI] | e_theta MAE | e_psi MAE | SR_d@1m [CI] | EPE mean | EPE med. | SR_EPE@1m [CI] | 5-DoF (1 m, 1 deg) | AUC@5 / 10 / 20 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for k, s in rows.items():
    c = s['ci']; L.append(f"| {k} | {s['x_mae']:.2f} | {s['y_mae']:.2f} | {s['z_mae']:.2f} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {s['theta_mae']:.3f} | {s['psi_mae']:.3f} | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_5dof_1m_1deg'])} | {f1(s['auc_5'])} / {f1(s['auc_10'])} / {f1(s['auc_20'])} |")
L += ['\nIn the camera frame, dx / dy are the right / forward components. Paired differences, baseline minus DreamNav (camera frame):\n']
for nm, r in paired.items(): L.append(f'- vs {nm}: ' + '; '.join(f"{k} {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]" for k, v in r.items()))
L += ['\nWho succeeds (baseline vs DreamNav Stage II, camera frame):\n', '| criterion | both | baseline only [CI] | DreamNav Stage II only [CI] | neither | best of the two |', '|---|---:|---:|---:|---:|---:|']
for lab, j in joint.items(): L.append(f"| {lab} | {f1(j['both'])} | {f1(j['baseline_only'])} [{f1(j['ci_baseline_only'][0])}, {f1(j['ci_baseline_only'][1])}] | {f1(j['dreamnav2_only'])} [{f1(j['ci_dreamnav2_only'][0])}, {f1(j['ci_dreamnav2_only'][1])}] | {f1(j['neither'])} | {f1(j['oracle_best_of_two'])} |")
for cn, rr in bins.items():
    L += [f'\nBy {cn} (SR_d@1m / SR_EPE@1m / EPE median, camera frame):\n', '| bin | pairs | SIFT baseline | DreamNav Stage I | DreamNav Stage II | Stage II minus baseline, SR_EPE@1m points [CI] |', '|---|---:|---|---|---|---:|']
    for r in rr: L.append(f"| {r['bin']} | {r['n']} | " + ' | '.join(f"{f1(r[k]['sr_d_1m'])} / {f1(r[k]['sr_epe_1m'])} / {r[k]['epe_med']:.2f}" for k in ('baseline', 'stage1', 'stage2')) + f" | {r['stage2_minus_baseline_sr_epe_points']['delta']:+.1f} [{r['stage2_minus_baseline_sr_epe_points']['ci'][0]:+.1f}, {r['stage2_minus_baseline_sr_epe_points']['ci'][1]:+.1f}] |")
open(f'{OUT}/sift_full.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
