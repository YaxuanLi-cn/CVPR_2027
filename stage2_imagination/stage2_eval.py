import sys, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='s2'); ap.add_argument('--bev', default='bev_s'); ap.add_argument('--out', default=None); a = ap.parse_args(); S2 = f'{SG}/stage2'
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); F = pd.read_pickle(f'{S2}/{a.tag}_pred.pkl'); D = P.merge(F.drop(columns=['split', 'tour']), on='manifest_index')
A = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image')
for s in 'ab': D[f'v2h_{s}'] = A.ah_pred.reindex(D[f'image_{s}']).values; D[f'v2k_{s}'] = A.ah_kappa.reindex(D[f'image_{s}']).values
D['part'] = A.part.reindex(D.image_a).values; have_bev = False
try:
    B = pd.read_pickle(f'{S2}/pred_{a.bev}.pkl')
    for s in 'ab':
        b = B[B.side == s].set_index('manifest_index'); D[f'bvh_{s}'] = b.ah_est.reindex(D.manifest_index).values; D[f'bvk_{s}'] = b.ak_est.reindex(D.manifest_index).values
    have_bev = True
except FileNotFoundError: print('no BEV predictions yet -> module B rows skipped')
D = D[D.part.isin(['val', 'test'])].copy(); print('rows', D.part.value_counts().to_dict())

def combine(d, hk, dh):
    """hk: list of (heading_a, kappa_a, heading_b, kappa_b) sources; dh: relative heading used to bring b's estimate to camera a."""
    s = 0; c = 0
    for ha, ka, hb, kb in hk:
        ta = np.deg2rad(d[ha]); tb = np.deg2rad(d[hb] - dh); s = s + d[ka] * np.sin(ta) + d[kb] * np.sin(tb); c = c + d[ka] * np.cos(ta) + d[kb] * np.cos(tb)
    return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)
def world_xy(h_deg, lx, ly): h = np.deg2rad(h_deg); return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly
V2 = [('v2h_a', 'v2k_a', 'v2h_b', 'v2k_b')]; BV = [('bvh_a', 'bvk_a', 'bvh_b', 'bvk_b')]
def rows(d):
    out = {}
    def add(name, pre, src):
        h, k = combine(d, src, d[pre + 'heading']); x, y = world_xy(h, d[pre + 'lx'], d[pre + 'ly']); e = np.abs(wrap(h - d.heading_a))
        out[name] = dict(xyz=(x.values, y.values, d[pre + 'z'].values), tilt=d[pre + 'tilt'].values, heading=d[pre + 'heading'].values, dist=d[pre + 'dist'].values, abs_heading_mae=float(e.mean()), abs_heading_median=float(np.median(e)))
    add('stage 1 (SuperGlue fusion + abs-heading CNN v2)', 's1_', V2); add('stage 2: + module A (imagined target view + SuperGlue re-matching)', 'p_', V2)
    if have_bev:
        add('stage 2: + module B (imagined BEV abs-heading)', 's1_', BV); add('stage 2: A + B', 'p_', BV); add('stage 2: A + B, abs-heading = BEV + CNN v2 ensemble', 'p_', BV + V2)
    for nm, pre in (('[upper bound] stage 1 + true abs-heading', 's1_'), ('[upper bound] stage 2 A + true abs-heading', 'p_')):
        x, y = world_xy(d.heading_a, d[pre + 'lx'], d[pre + 'ly']); out[nm] = dict(xyz=(x.values, y.values, d[pre + 'z'].values), tilt=d[pre + 'tilt'].values, heading=d[pre + 'heading'].values, dist=d[pre + 'dist'].values, abs_heading_mae=0.0, abs_heading_median=0.0)
    return out
def rescale(v, al):
    x, y, z = v['xyz']; n = np.sqrt(x**2 + y**2 + z**2).clip(1e-6); sc = (v['dist'] / n) ** al; return x * sc, y * sc, z * sc
res = {}; va = D[D.part == 'val']; te = D[D.part == 'test']; Rv = rows(va); Rt = rows(te)
print(f'--- test ({len(te)} pairs); val ({len(va)} pairs, OOF predictions) is only used to pick the norm-rescaling exponent')
for k in Rt:
    r = report(te, *Rt[k]['xyz'], Rt[k]['tilt'], Rt[k]['heading'], f'{k:72s}'); r['abs_heading_mae'] = Rt[k]['abs_heading_mae']; r['abs_heading_median'] = Rt[k]['abs_heading_median']; res[k] = {'no_rescale': r}
    score = lambda al: sum(report(va, *rescale(Rv[k], al), Rv[k]['tilt'], Rv[k]['heading'])[m] for m in ('MAE_x_m', 'MAE_y_m', 'MAE_z_m', 'MAE_distance_from_xyz_m'))
    al = min((0.0, 0.25, 0.5, 0.75, 1.0), key=score); res[k]['rescale_exponent_chosen_on_val'] = al
    res[k]['rescaled'] = report(te, *rescale(Rt[k], al), Rt[k]['tilt'], Rt[k]['heading'], f'    norm rescaled to fused distance, exponent {al} (chosen on val)'.ljust(72))
    print(f'    abs-heading MAE {Rt[k]["abs_heading_mae"]:.2f} deg, median {Rt[k]["abs_heading_median"]:.2f} deg')
# which absolute-heading source to use with module A is decided on the validation tours (x+y MAE), never on test
if have_bev:
    cand = ['stage 2: + module A (imagined target view + SuperGlue re-matching)', 'stage 2: A + B', 'stage 2: A + B, abs-heading = BEV + CNN v2 ensemble']
    vs = {k: (lambda r: r['MAE_x_m'] + r['MAE_y_m'])(report(va, *Rv[k]['xyz'], Rv[k]['tilt'], Rv[k]['heading'])) for k in cand}; best = min(vs, key=vs.get)
    print('val x+y MAE per abs-heading source:', {k: round(v, 4) for k, v in vs.items()}, '| val abs-heading MAE:', {k: round(Rv[k]['abs_heading_mae'], 2) for k in cand}); print('HEADLINE (chosen on val):', best)
    res['headline_variant'] = best; res['headline_test'] = res[best]; res['val_xy_sum'] = vs
res['rows'] = dict(test=int(len(te)), val=int(len(va))); json.dump(res, open(a.out or f'{S2}/final_{a.tag}_{a.bev}.json', 'w'), indent=2)
