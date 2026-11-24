#!/usr/bin/env python3
"""Real photographs (Mezieres, DJI Phantom 4 RTK, 60 m above ground): Table 4 of the paper re-scored with supp/common.py,
with uncertainty, for the pairs inside and beyond the training range, with and without the flight height.
Camera frame of the first image (no absolute heading). Intervals: bootstrap over source images.
Output: outputs/supp/trackA/real.json"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); from common import *
REAL = '/root/realdata/mezieres'; D = pd.read_pickle(f'{REAL}/pred.pkl'); K = 60.0 / 220.18; res = {}
def rows(d, sub):
    gt = dict(x=d.lx.values, y=d.ly.values, z=d.z.values, tilt=d.tilt.values, heading=d.heading.values); gtw = dict(x=d.x.values, y=d.y.values, z=d.z.values, tilt=d.tilt.values, heading=d.heading.values)
    bt = boot_indices(d.a.values, 1000, 0); out = {}
    def add(name, pred, g=gt):
        e = errors(pred, g); s = summarize(e)
        s['ci'] = {k: ci(fn, bt) for k, fn in (('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_d_2m', lambda b: (e['d'][b] <= 2).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('sr_epe_2m', lambda b: (e['epe'][b] < 2).mean()), ('epe_med', lambda b: np.median(e['epe'][b])), ('d_mae', lambda b: e['d'][b].mean()))}
        out[name] = s; return e
    P = lambda pre, sc: dict(x=d[pre + 'lx'].values * sc, y=d[pre + 'ly'].values * sc, z=d[pre + 'z'].values * sc, tilt=d[pre + 'tilt'].values, heading=d[pre + 'heading'].values)
    e1 = add('DreamNav Stage I', P('s1_', K)); e2 = add('DreamNav Stage II', P('s2_', K)); add('DreamNav Stage I, no flight height', P('s1_', 1.0)); add('DreamNav Stage II, no flight height', P('s2_', 1.0))
    add('MASt3R-statistics MLP', P('n_', 1.0)); add('MASt3R-statistics MLP, flight height', P('n_', K)); z0 = np.zeros(len(d)); add('Constant (zero motion)', dict(x=z0, y=z0, z=z0, tilt=z0, heading=z0))
    # ENU output of DreamNav with the learned absolute heading (scaled by the flight height)
    for pre, nm in (('s1', 'DreamNav Stage I, ENU (learned heading)'), ('s2', 'DreamNav Stage II, ENU (learned heading)')):
        x, y = world(d[f'{pre}_absh'].values, d[f'{pre}_lx'].values * K, d[f'{pre}_ly'].values * K); add(nm, dict(x=x, y=y, z=d[f'{pre}_z'].values * K, tilt=d[f'{pre}_tilt'].values, heading=d[f'{pre}_heading'].values), gtw)
        out[nm]['abs_heading_err_med'] = float(np.median(np.abs(wrap(d[f'{pre}_absh'].values - d.heading_a.values)))); out[nm]['abs_heading_err_mean'] = float(np.mean(np.abs(wrap(d[f'{pre}_absh'].values - d.heading_a.values))))
    if sub == 'in':   # policy baselines of Table 4 (ENU outputs; zero-shot, no flight height), available for the in-range pairs only
        for name, f in (('ViNT', 'vint'), ('NoMaD', 'nomad'), ('pi0-FAST', 'pi0_fast')):
            p = np.load(f'{REAL}/baselines/{f}_predictions.npz'); assert np.array_equal(p['pair_index'], d.index.values); q = p['pred_xyz_tilt_heading']
            add(name, dict(x=q[:, 0], y=q[:, 1], z=q[:, 2], tilt=q[:, 3], heading=q[:, 4]), gtw)
    out['_paired_II_minus_I'] = {k: dict(delta=float(fn(np.arange(len(d)))), ci=ci(fn, bt)) for k, fn in (('sr_d_1m', lambda b: (e2['d'][b] <= 1).mean() - (e1['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e2['epe'][b] < 1).mean() - (e1['epe'][b] < 1).mean()), ('epe_med', lambda b: np.median(e2['epe'][b]) - np.median(e1['epe'][b])))}
    out['_info'] = dict(n=int(len(d)), images=int(pd.unique(np.r_[d.a.values, d.b.values]).size), dist_mean=float(d.dist.mean()), dist_med=float(d.dist.median()), dist_max=float(d.dist.max()), dist_min=float(d.dist.min()), abs_heading_change_med=float(d.heading.abs().median()), tilt_a_mean=float(d.tilt_a.mean()), tilt_a_max=float(d.tilt_a.max()), same_flight=float(d.same_flight.mean()))
    return out
res['in_range'] = rows(D[D.dist_equiv <= 38], 'in'); res['beyond_range'] = rows(D[D.dist_equiv > 38], 'out'); res['all'] = rows(D, 'all'); res['flight_height_m'] = 60.0; res['training_height_m'] = 220.18; res['range_limit_m'] = 38 * K
json.dump(res, open(f'{OUT}/trackA/real.json', 'w'), indent=1)
for k in ('in_range', 'beyond_range'):
    print('==', k, res[k]['_info'])
    for nm, s in res[k].items():
        if nm.startswith('_'): continue
        print(f"{nm:42s} EPE med {s['epe_med']:6.2f} mean {s['epe_mae']:6.2f} | SR_d 1m/2m {100*s['sr_d_1m']:5.1f} {100*s['sr_d_2m']:5.1f} | SR_EPE 1m/2m {100*s['sr_epe_1m']:5.1f} [{100*s['ci']['sr_epe_1m'][0]:.1f},{100*s['ci']['sr_epe_1m'][1]:.1f}] {100*s['sr_epe_2m']:5.1f} | th {s['theta_mae']:6.2f}/{s['theta_med']:.2f} psi {s['psi_mae']:5.2f} dir {s['dir_med']:5.1f}" + (f" | absH med {s['abs_heading_err_med']:.1f}" if 'abs_heading_err_med' in s else ''))
    print('  paired II-I:', {a: (round(v['delta'], 3), [round(c, 3) for c in v['ci']]) for a, v in res[k]['_paired_II_minus_I'].items()})
