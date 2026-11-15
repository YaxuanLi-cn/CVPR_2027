#!/usr/bin/env python3
"""Final assembly: fused (heading, tilt, distance, z, level-frame xy) + single-image absolute heading -> world x,y.
Writes the six headline MAEs on the full 12,313-pair test split."""
import sys, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='fusion'); ap.add_argument('--abs', default='cnx_t'); ap.add_argument('--out', default=None); a = ap.parse_args()
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); F = pd.read_pickle(f'{SG}/{a.tag}_pred.pkl'); A = None
for tag in a.abs.split(','):   # several absolute-heading models are combined by summing von Mises natural parameters
    B = pd.read_pickle(f'{SG}/absheading/pred_{tag}.pkl').set_index('image'); B['s'] = B.ah_kappa * np.sin(np.deg2rad(B.ah_pred)); B['c'] = B.ah_kappa * np.cos(np.deg2rad(B.ah_pred))
    if A is None: A = B[['part', 's', 'c']].copy()
    else:
        A = A.loc[A.index.intersection(B.index)].copy(); A['s'] += B.s.reindex(A.index); A['c'] += B.c.reindex(A.index); A['part'] = B.part.reindex(A.index)
A['ah_pred'] = np.rad2deg(np.arctan2(A.s, A.c)); A['ah_kappa'] = np.hypot(A.s, A.c)
D = P.merge(F.drop(columns=['split', 'tour']), on='manifest_index')
for s in 'ab':
    D[f'ah_{s}'] = A.ah_pred.reindex(D[f'image_{s}']).values; D[f'ak_{s}'] = A.ah_kappa.reindex(D[f'image_{s}']).values
D['part'] = A.part.reindex(D.image_a).values
# absolute heading of camera a: combine image a's estimate with image b's estimate minus the fused relative heading (von Mises natural parameters)
def combine(D, use_b=True):
    ta = np.deg2rad(D.ah_a); tb = np.deg2rad(D.ah_b - D.p_heading); kb = D.ak_b if use_b else 0.0
    s = D.ak_a * np.sin(ta) + kb * np.sin(tb); c = D.ak_a * np.cos(ta) + kb * np.cos(tb); return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)
def world_xy(D, ha_deg, lx, ly, rho=1.0):
    h = np.deg2rad(ha_deg); return rho * (np.cos(h) * lx + np.sin(h) * ly), rho * (-np.sin(h) * lx + np.cos(h) * ly)
te = D[D.split == 'test']; va = D[D.part == 'val']; res = {}
for nm, d in (('val', va), ('test', te)):
    for use_b in (False, True):
        h, k = combine(d, use_b); e = np.abs(wrap(h - d.heading_a)); print(f'abs-heading {nm} use_b={use_b}: MAE {e.mean():.2f} median {np.median(e):.2f} <15deg {(e<15).mean():.3f} <45deg {(e<45).mean():.3f}')
        res[f'abs_heading_{nm}_{"ab" if use_b else "a"}'] = float(e.mean())
# shrink factor rho(kappa): fitted on validation tours (never seen by the CNN) to minimise L1 of x,y; kappa-binned
h_va, k_va = combine(va); bins = np.quantile(k_va, np.linspace(0, 1, 9)); bins[0] = -np.inf; bins[-1] = np.inf; rhos = []
for i in range(8):
    m = (k_va > bins[i]) & (k_va <= bins[i + 1]); d = va[m]; best = min(np.linspace(0, 1, 41), key=lambda r: np.mean(np.abs(np.r_[world_xy(d, h_va[m], d.p_lx, d.p_ly, r)[0] - d.x, world_xy(d, h_va[m], d.p_lx, d.p_ly, r)[1] - d.y]))); rhos.append(best)
print('rho per kappa-bin (val):', np.round(rhos, 2)); res['rho_bins'] = [float(r) for r in rhos]
def variants(d):
    h, k = combine(d); out = {}
    x, y = world_xy(d, h, d.p_lx, d.p_ly); out['fusion + abs-heading (no shrink)'] = (x, y, d.p_z)
    n = np.sqrt(x**2 + y**2 + d.p_z**2).clip(lower=1e-6); sc = d.p_dist / n; out['fusion + abs-heading, norm rescaled to fused distance'] = (x * sc, y * sc, d.p_z * sc)
    for al in (0.25, 0.5, 0.75): out[f'fusion + abs-heading, norm rescaled with exponent {al}'] = (x * sc**al, y * sc**al, d.p_z * sc**al)
    hr = np.sqrt(x**2 + y**2).clip(lower=1e-6); sh = np.sqrt(np.clip(d.p_dist**2 - d.p_z**2, 0, None)) / hr; out['fusion + abs-heading, xy rescaled so |xyz| = fused distance (z kept)'] = (x * sh, y * sh, d.p_z)
    rho = np.array(rhos)[np.clip(np.searchsorted(bins, k) - 1, 0, 7)]; xs, ys = world_xy(d, h, d.p_lx, d.p_ly, rho); out['fusion + abs-heading, kappa-shrunk xy'] = (xs, ys, d.p_z)
    ha = d.heading_a; xo, yo = world_xy(d, ha, d.p_lx, d.p_ly); out['[oracle abs-heading upper bound]'] = (xo, yo, d.p_z)
    out['fusion, xy from MASt3R MLP (no abs-heading)'] = (d.n_x, d.n_y, d.p_z)
    return out
for nm, d in (('val', va), ('test', te)):
    print(f'--- {nm} ({len(d)} pairs)')
    res[nm] = {'MASt3R-only MLP (control)': report(d, d.n_x.values, d.n_y.values, d.n_z.values, d.n_tilt.values, d.n_heading.values, f'{"MASt3R-only MLP (control)":72s}')}
    for k, (x, y, z) in variants(d).items(): res[nm][k] = report(d, np.asarray(x), np.asarray(y), np.asarray(z), d.p_tilt.values, d.p_heading.values, f'{k:72s}')
    res[nm]['aux'] = dict(MAE_dist_direct=float(np.mean(np.abs(d.p_dist - d.dist))), MAE_lx=float(np.mean(np.abs(d.p_lx - d.lx))), MAE_ly=float(np.mean(np.abs(d.p_ly - d.ly))), rows=int(len(d)))
# headline = the norm-rescale exponent with the lowest sum of the four metre-valued MAEs on the VALIDATION tours (never tuned on test)
fam = ['fusion + abs-heading (no shrink)'] + [f'fusion + abs-heading, norm rescaled with exponent {al}' for al in (0.25, 0.5, 0.75)] + ['fusion + abs-heading, norm rescaled to fused distance']
score = lambda r: r['MAE_x_m'] + r['MAE_y_m'] + r['MAE_z_m'] + r['MAE_distance_from_xyz_m']
best = min(fam, key=lambda k: score(res['val'][k])); res['headline_variant'] = best; res['headline_test'] = res['test'][best]
print(f'\nHEADLINE (chosen on val): {best}'); report(te, *[np.asarray(v) for v in variants(te)[best]], te.p_tilt.values, te.p_heading.values, 'HEADLINE test'.ljust(72))
sub = te[te.has_sg]; res['subset_original_superglue_coverage'] = {'rows': int(len(sub)),
    'control_mlp': report(sub, sub.n_x.values, sub.n_y.values, sub.n_z.values, sub.n_tilt.values, sub.n_heading.values, f'[orig-SG-coverage subset, {len(sub)} pairs] MLP control'.ljust(72)),
    'headline': report(sub, *[np.asarray(v) for v in variants(sub)[best]], sub.p_tilt.values, sub.p_heading.values, f'[orig-SG-coverage subset, {len(sub)} pairs] headline'.ljust(72))}
json.dump(res, open(a.out or f'{SG}/final_metrics_{a.tag}.json', 'w'), indent=2)
