#!/usr/bin/env python3
"""Track B, part 5: paired comparison of one geometric baseline (default: SIFT + E + flight-height prior) with DreamNav on the same
PairUAV test pairs: who wins per pair, error tails, and success by difficulty bin. Output: outputs/supp/trackB/paired_<matcher>.json / .md
Usage: compare_best.py [matcher] [variant: plain|tta]"""
import sys, os, json
import numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/supp', f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
import common as C
from fusion_lib import add_targets
from rb_recon import subset
from eval_pairuav import pick, unit, to_metric, fit_const, NAMES
OUT = f'{ROOT}/outputs/supp/trackB'; m = sys.argv[1] if len(sys.argv) > 1 else 'sift'; tta = (sys.argv[2] == 'tta') if len(sys.argv) > 2 else False
P = json.load(open(f'{OUT}/pairuav.json')); thr = P['threshold_selection'][f'{m}|{"tta" if tta else "plain"}']['chosen']
T = add_targets(subset(1000, split='test')); Tc = add_targets(subset(300, split='train')); ids, idc = T.manifest_index.values, Tc.manifest_index.values
T6 = C.load_test().set_index('manifest_index').loc[ids].reset_index()
Gt, Gc = pd.read_pickle(f'{OUT}/geom/{m}_test.pkl'), pd.read_pickle(f'{OUT}/geom/{m}_train.pkl'); mt, mc = unit(pick(Gt, ids, thr, tta)), unit(pick(Gc, idc, thr, tta))
H, _ = fit_const(mc[mc.have.values], Tc.dist.values[mc.have.values], 'height_p75ab'); pb, ok = to_metric(mt, 'height_p75ab', H); sub = mt.have.values
Ts = T6[sub].reset_index(drop=True); gt = C.gt_cam(Ts); eb = C.errors({k: v[sub] for k, v in pb.items()}, gt); e1 = C.errors({k: np.asarray(v)[sub] for k, v in C.dreamnav(T6, 's1_', 'cam').items()}, gt); e2 = C.errors({k: np.asarray(v)[sub] for k, v in C.dreamnav(T6, 'p_', 'cam').items()}, gt)
n = int(sub.sum()); boots = C.boot_indices(Ts.tour.values, 1000, 0); name = f'{NAMES[m]} + E, {"rot. TTA" if tta else "plain"}, flight-height prior'
res = dict(baseline=name, n=n, thr_px=thr, H=H); L = [f'# Paired comparison on the same {n} PairUAV test pairs: {name} vs DreamNav\n']
def tail(e): return {k: dict(mean=float(e[k].mean()), median=float(np.median(e[k])), p75=float(np.percentile(e[k], 75)), p90=float(np.percentile(e[k], 90)), p95=float(np.percentile(e[k], 95)), p99=float(np.percentile(e[k], 99))) for k in ('d', 'epe', 'theta', 'psi', 'dir')}
res['tails'] = {'baseline': tail(eb), 'DreamNav Stage I': tail(e1), 'DreamNav Stage II': tail(e2)}
L += ['## Error distribution\n', '| method | metric | mean | median | p75 | p90 | p95 | p99 |', '|---|---|---:|---:|---:|---:|---:|---:|']
for nm, t in res['tails'].items():
    for k, lab in (('d', 'range error d (m)'), ('epe', 'EPE (m)'), ('theta', 'e_theta (deg)'), ('psi', 'e_psi (deg)'), ('dir', 'direction (deg)')): L.append(f"| {nm} | {lab} | {t[k]['mean']:.3f} | {t[k]['median']:.3f} | {t[k]['p75']:.3f} | {t[k]['p90']:.3f} | {t[k]['p95']:.3f} | {t[k]['p99']:.3f} |")
# paired differences (baseline - DreamNav), CI over buildings
res['paired'] = {}
L += ['\n## Paired differences, baseline minus DreamNav (95% CI over buildings)\n', '| vs | SR_d@1m (points) | SR_EPE@1m (points) | SR_EPE@2m (points) | d MAE (m) | EPE mean (m) | EPE median (m) | share of pairs where the baseline has the smaller EPE |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for nm, e in (('DreamNav Stage I', e1), ('DreamNav Stage II', e2)):
    fns = dict(sr_d=lambda b: 100 * ((eb['d'][b] <= 1).mean() - (e['d'][b] <= 1).mean()), sr_epe1=lambda b: 100 * ((eb['epe'][b] < 1).mean() - (e['epe'][b] < 1).mean()), sr_epe2=lambda b: 100 * ((eb['epe'][b] < 2).mean() - (e['epe'][b] < 2).mean()),
               d_mae=lambda b: eb['d'][b].mean() - e['d'][b].mean(), epe_mean=lambda b: eb['epe'][b].mean() - e['epe'][b].mean(), epe_med=lambda b: np.median(eb['epe'][b]) - np.median(e['epe'][b]), win=lambda b: 100 * (eb['epe'][b] < e['epe'][b]).mean())
    r = {k: dict(delta=float(f(np.arange(n))), ci=C.ci(f, boots)) for k, f in fns.items()}; res['paired'][nm] = r
    L.append(f'| {nm} | ' + ' | '.join(f"{r[k]['delta']:+.2f} [{r[k]['ci'][0]:+.2f}, {r[k]['ci'][1]:+.2f}]" for k in ('sr_d', 'sr_epe1', 'sr_epe2', 'd_mae', 'epe_mean', 'epe_med')) + f" | {r['win']['delta']:.1f}% [{r['win']['ci'][0]:.1f}, {r['win']['ci'][1]:.1f}] |")
# diagnostic hybrids: the baseline's direction and rotation with another source of metric scale (not baselines; they show where the error is)
u = mt[['ux', 'uy', 'uz']].values.astype(float)[sub]; okb = np.asarray(ok)[sub]; res['hybrids'] = {}
L += ['\n## Diagnostic: the baseline direction with another source of scale\n', '| scale source | d MAE | SR_d@1m | EPE mean | EPE med. | SR_EPE@1m | SR_EPE@2m |', '|---|---:|---:|---:|---:|---:|---:|']
for lab, dist in (('flight-height prior (the baseline itself)', None), ('true distance (oracle)', Ts.dist.values), ('DreamNav Stage I fused distance', Ts.s1_dist.values), ('DreamNav Stage II fused distance', Ts.p_dist.values)):
    if dist is None: e = eb
    else: t = np.where(okb[:, None], u * dist[:, None], 0.0); e = C.errors(dict(x=t[:, 0], y=t[:, 1], z=t[:, 2], tilt=pb['tilt'][sub], heading=pb['heading'][sub]), gt)
    r = dict(d_mae=float(e['d'].mean()), sr_d=float(100 * (e['d'] <= 1).mean()), epe_mean=float(e['epe'].mean()), epe_med=float(np.median(e['epe'])), sr_epe1=float(100 * (e['epe'] < 1).mean()), sr_epe2=float(100 * (e['epe'] < 2).mean())); res['hybrids'][lab] = r
    L.append(f"| {lab} | {r['d_mae']:.2f} | {r['sr_d']:.1f} | {r['epe_mean']:.2f} | {r['epe_med']:.2f} | {r['sr_epe1']:.1f} | {r['sr_epe2']:.1f} |")
for nm, e in (('DreamNav Stage I (its own direction and distance)', e1), ('DreamNav Stage II (its own direction and distance)', e2)):
    L.append(f"| {nm} | {e['d'].mean():.2f} | {100 * (e['d'] <= 1).mean():.1f} | {e['epe'].mean():.2f} | {np.median(e['epe']):.2f} | {100 * (e['epe'] < 1).mean():.1f} | {100 * (e['epe'] < 2).mean():.1f} |")
# joint success table
a, b2 = eb['epe'] < 1, e2['epe'] < 1; res['joint_epe_1m_vs_stage2'] = dict(both=float((a & b2).mean()), baseline_only=float((a & ~b2).mean()), dreamnav_only=float((~a & b2).mean()), neither=float((~a & ~b2).mean()))
L += ['\n## Who succeeds (EPE < 1 m), baseline vs DreamNav Stage II\n', f"both {100 * (a & b2).mean():.1f}% | baseline only {100 * (a & ~b2).mean():.1f}% | DreamNav only {100 * (~a & b2).mean():.1f}% | neither {100 * (~a & ~b2).mean():.1f}%  (an oracle that picks the better of the two per pair would reach {100 * (a | b2).mean():.1f}%)\n"]
# difficulty bins
gd = Ts.dist.values; cols = {'true distance (m)': (gd, [0, 5, 10, 20, 30, 1e9], ['0-5', '5-10', '10-20', '20-30', '>=30']), '|heading change| (deg)': (np.abs(Ts.heading.values), [0, 45, 90, 135, 181], ['0-45', '45-90', '90-135', '135-180']),
        '|tilt change| (deg)': (np.abs(Ts.tilt.values), [0, 10, 20, 30, 46], ['0-10', '10-20', '20-30', '30-45']), '|altitude change| (m)': (np.abs(Ts.z.values), [0, 5, 15, 25, 1e9], ['0-5', '5-15', '15-25', '>=25']),
        'SIFT inliers recorded by the dataset screen': (T.sift_inliers.values[sub], [0, 50, 100, 200, 400, 1e9], ['<50', '50-100', '100-200', '200-400', '>=400']), 'DreamNav coarse-stage zoom inliers': (Ts.zm_inl.fillna(0).values, [0, 100, 200, 400, 800, 1e9], ['<100', '100-200', '200-400', '400-800', '>=800'])}
res['bins'] = {}
for cn, (v, bins, labs) in cols.items():
    L += [f'\n**By {cn}**\n', '| bin | pairs | baseline SR_d / SR_EPE@1m / EPE med. | Stage I SR_d / SR_EPE@1m / EPE med. | Stage II SR_d / SR_EPE@1m / EPE med. |', '|---|---:|---|---|---|']; res['bins'][cn] = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labs):
        k = (v >= lo) & (v < hi)
        if k.sum() < 5: continue
        f = lambda e: (float(100 * (e['d'][k] <= 1).mean()), float(100 * (e['epe'][k] < 1).mean()), float(np.median(e['epe'][k]))); r = dict(bin=lab, n=int(k.sum()), baseline=f(eb), stage1=f(e1), stage2=f(e2)); res['bins'][cn].append(r)
        L.append(f"| {lab} | {r['n']} | " + ' | '.join(f'{x[0]:.1f} / {x[1]:.1f} / {x[2]:.2f}' for x in (r['baseline'], r['stage1'], r['stage2'])) + ' |')
json.dump(res, open(f'{OUT}/paired_{m}{"_tta" if tta else ""}.json', 'w'), indent=1); open(f'{OUT}/paired_{m}{"_tta" if tta else ""}.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
