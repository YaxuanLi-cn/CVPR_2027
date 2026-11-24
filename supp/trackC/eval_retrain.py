#!/usr/bin/env python3
"""Track C, item 3: site-disjoint re-training of the absolute-heading network(s), evaluated on the test buildings of the
held-out sites, on exactly the same images / pairs as the archived networks.

  A  archived network            (trained on 1,010 buildings; has seen OTHER buildings of the held-out sites)
  C  size-matched control        (243 training buildings removed at random from the sites that are NOT held out)
  B  site-disjoint network       (the 243 training buildings of the held-out sites removed: those sites are never seen)
A vs C = effect of training on fewer buildings; C vs B = effect of never having seen the site, at equal training-set size.
Pose geometry is unchanged throughout (archived Stage II); only the heading used to rotate into ENU changes.
Output: outputs/supp/trackC/part_retrain.json
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp')
from common import *
O = f'{OUT}/trackC'; T = load_test(); n = len(T); ST = pd.read_pickle(f'{O}/sites.pkl').set_index('tour')
for c in ('site', 'heldout', 'km_to_nearest_kept_train', 'site_n_train'): T[c] = ST[c].reindex(T.tour).values
gw = gt_world(T); gc = gt_cam(T); tours = T.tour.values; R = {}
def circ(d): d = np.deg2rad(np.asarray(d, float)); s, c = np.sin(d).mean(), np.cos(d).mean(); return float(np.rad2deg(np.arctan2(s, c))), float(np.hypot(s, c))
def hstats(e): e = np.asarray(e, float); return dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)), p75=float(np.percentile(e, 75)), p90=float(np.percentile(e, 90)), p95=float(np.percentile(e, 95)), gt10=float((e > 10).mean()), gt20=float((e > 20).mean()), gt45=float((e > 45).mean()), gt90=float((e > 90).mean()))

# ---- per-image predictions of each raw-view model, mapped onto the pair sides
RAW = {'A archived': None}
for tag, nm in (('raw_control', 'C size-matched control'), ('raw_sitedisjoint', 'B site-disjoint')):
    f = f'{O}/pred_{tag}.pkl'
    if os.path.exists(f): RAW[nm] = pd.read_pickle(f).set_index('image')
cols = {}
for nm, P in RAW.items():
    k = nm[0]
    for s in 'ab':
        if P is None: T[f'{k}h_{s}'] = T[f'v2h_{s}'].values; T[f'{k}k_{s}'] = T[f'v2k_{s}'].values
        else: T[f'{k}h_{s}'] = P.ah_pred.reindex(T[f'image_{s}']).values; T[f'{k}k_{s}'] = P.ah_kappa.reindex(T[f'image_{s}']).values
    assert T[[f'{k}h_a', f'{k}h_b']].notna().all().all(), nm; cols[nm] = [(f'{k}h_a', f'{k}k_a', f'{k}h_b', f'{k}k_b')]
# nadir-rectified networks: archived (bv*) and, when trained, site-disjoint
BEV = {'A archived': BV}
f = f'{O}/pred_bev_sitedisjoint.pkl'
if os.path.exists(f):
    Bp = pd.read_pickle(f)
    for s in 'ab':
        b = Bp[Bp.side == s].set_index('manifest_index'); T[f'Bbh_{s}'] = b.ah_est.reindex(T.manifest_index).values; T[f'Bbk_{s}'] = b.ak_est.reindex(T.manifest_index).values
    BEV['B site-disjoint'] = [('Bbh_a', 'Bbk_a', 'Bbh_b', 'Bbk_b')]

SUB = {'held-out sites': T.heldout.values, 'other sites': ~T.heldout.values, 'all test pairs': np.ones(n, bool)}
def pose(h): e = errors(dreamnav(T, 'p_', h), gw); return e
ecam = errors(dreamnav(T, 'p_', 'cam'), gc)
def block(name, src):
    """src: list of fuse sources -> per-pair fused heading error + Stage II ENU errors."""
    h, k = fuse_heading(T, src, T.p_heading.values); sg = wrap(h - T.heading_a.values); return dict(h=h, k=k, sg=sg, err=np.abs(sg), pose=pose(h))
EST = {f'raw-view | {nm}': block(nm, c) for nm, c in cols.items()}
for nm in BEV:
    if nm in cols: EST[f'ensemble | {nm}'] = block(nm, BEV[nm] + cols[nm])
    EST[f'nadir-rectified | {nm}'] = block(nm, BEV[nm])
out = {}
for sn, m in SUB.items():
    idx = np.where(m)[0]; bt = boot_indices(tours[m], 1000, 0); o = {'buildings': int(len(set(tours[m]))), 'pairs': int(m.sum()), 'camera_frame': dict(epe_mean=float(ecam['epe'][m].mean()), epe_med=float(np.median(ecam['epe'][m])), sr_epe_1m=float((ecam['epe'][m] < 1).mean()))}
    for en, b in EST.items():
        e = b['pose']; mu, Rl = circ(b['sg'][m])
        o[en] = dict(fused_heading=hstats(b['err'][m]), fused_heading_mean_ci=ci(lambda q: b['err'][idx[q]].mean(), bt), fused_heading_median_ci=ci(lambda q: np.median(b['err'][idx[q]]), bt), kappa_median=float(np.median(b['k'][m])),
                     x_mae=float(e['x'][m].mean()), y_mae=float(e['y'][m].mean()), epe_mean=float(e['epe'][m].mean()), epe_med=float(np.median(e['epe'][m])), sr_epe_1m=float((e['epe'][m] < 1).mean()), sr_epe_1m_ci=ci(lambda q: (e['epe'][idx[q]] < 1).mean(), bt),
                     sr_epe_2m=float((e['epe'][m] < 2).mean()), sr_5dof_1m_1deg=float(((e['epe'][m] < 1) & (e['theta'][m] < 1) & (e['psi'][m] < 1)).mean()), epe_med_ci=ci(lambda q: np.median(e['epe'][idx[q]]), bt))
    # single image (pair sides) per raw-view model
    for nm in cols:
        k = nm[0]; es = np.r_[np.abs(wrap(T[f'{k}h_a'] - T.heading_a))[m], np.abs(wrap(T[f'{k}h_b'] - T.heading_b))[m]]; o[f'single image raw-view | {nm}'] = hstats(es)
    if 'B site-disjoint' in BEV: o['single image nadir-rectified | B site-disjoint'] = hstats(np.r_[np.abs(wrap(T.Bbh_a - T.heading_a))[m], np.abs(wrap(T.Bbh_b - T.heading_b))[m]])
    o['single image nadir-rectified | A archived'] = hstats(np.r_[np.abs(wrap(T.bvh_a - T.heading_a))[m], np.abs(wrap(T.bvh_b - T.heading_b))[m]])
    # paired differences (same pairs, building bootstrap)
    def pd_(a_, b_, f): return dict(delta=float(f(EST[b_], idx) - f(EST[a_], idx)), ci=ci(lambda q: f(EST[b_], idx[q]) - f(EST[a_], idx[q]), bt))
    fm = {'heading_mean': lambda b, i: b['err'][i].mean(), 'heading_median': lambda b, i: np.median(b['err'][i]), 'sr_epe_1m': lambda b, i: (b['pose']['epe'][i] < 1).mean(), 'epe_med': lambda b, i: np.median(b['pose']['epe'][i]), 'epe_mean': lambda b, i: b['pose']['epe'][i].mean()}
    pairs_ = [('raw-view | A archived', 'raw-view | B site-disjoint'), ('raw-view | C size-matched control', 'raw-view | B site-disjoint'), ('raw-view | A archived', 'raw-view | C size-matched control'), ('ensemble | A archived', 'ensemble | B site-disjoint')]
    o['paired'] = {f'{b_} minus {a_}': {k: pd_(a_, b_, f) for k, f in fm.items()} for a_, b_ in pairs_ if a_ in EST and b_ in EST}
    out[sn] = o
R['subsets'] = out
# ---- inside the held-out sites: per site (constant offset?), and by distance to the nearest remaining training building
hm = T.heldout.values; per = []
for s in sorted(set(T.site.values[hm])):
    m = hm & (T.site.values == s); r = dict(site=int(s), lat=float(ST[ST.site == s].lat.mean()), lon=float(ST[ST.site == s].lon.mean()), buildings=int(len(set(tours[m]))), pairs=int(m.sum()), km_to_nearest_kept_train=float(T.km_to_nearest_kept_train.values[m].min()))
    for en in ('raw-view | A archived', 'raw-view | C size-matched control', 'raw-view | B site-disjoint', 'ensemble | B site-disjoint'):
        if en not in EST: continue
        b = EST[en]; mu, Rl = circ(b['sg'][m]); bb = pd.Series(b['sg'][m]).groupby(tours[m]).apply(lambda v: circ(v)[0]).values
        r[en] = dict(heading_mean=float(b['err'][m].mean()), heading_median=float(np.median(b['err'][m])), signed_circular_mean=mu, resultant_length=Rl, mae_after_removing_site_offset=float(np.abs(wrap(b['sg'][m] - mu)).mean()),
                     building_bias_min=float(bb.min()), building_bias_max=float(bb.max()), building_bias_abs_median=float(np.median(np.abs(bb))), sr_epe_1m=float((b['pose']['epe'][m] < 1).mean()), kappa_median=float(np.median(b['k'][m])))
    per.append(r)
R['heldout_per_site'] = per
if 'raw-view | B site-disjoint' in EST:
    b = EST['raw-view | B site-disjoint']; a_ = EST['raw-view | A archived']; R['heldout_by_distance_to_nearest_kept_training_building'] = []
    for lo, hi in ((0, 30), (30, 1e9)):
        m = hm & (T.km_to_nearest_kept_train.values >= lo) & (T.km_to_nearest_kept_train.values < hi)
        if m.any(): R['heldout_by_distance_to_nearest_kept_training_building'].append(dict(bin=f'{lo}-{hi if hi < 1e8 else "inf"} km', buildings=int(len(set(tours[m]))), pairs=int(m.sum()), A=hstats(a_['err'][m]), B=hstats(b['err'][m]), B_sr_epe_1m=float((b['pose']['epe'][m] < 1).mean()), A_sr_epe_1m=float((a_['pose']['epe'][m] < 1).mean())))
    # building-level decomposition on the held-out sites for the site-disjoint model
    bias = pd.Series(b['sg'][hm]).groupby(tours[hm]).apply(lambda v: circ(v)[0]); bi = bias.reindex(tours[hm]).values; rs = wrap(b['sg'][hm] - bi); ab = np.abs(bias.values)
    R['heldout_bias_vs_noise_B'] = dict(buildings=int(len(bias)), abs_building_bias=dict(mean=float(ab.mean()), median=float(np.median(ab)), gt10=float((ab > 10).mean()), gt20=float((ab > 20).mean()), gt45=float((ab > 45).mean())),
                                        heading_mae=dict(full=float(b['err'][hm].mean()), building_bias_only=float(np.abs(bi).mean()), image_noise_only=float(np.abs(rs).mean())))
    # is the failure detectable? risk-coverage by kappa on the held-out sites
    k = b['k'][hm]; e = b['err'][hm]; ep = b['pose']['epe'][hm]; o_ = np.argsort(-k); R['heldout_risk_coverage_B'] = [dict(coverage=c, heading_mae=float(e[o_[:int(round(c * len(o_)))]].mean()), heading_median=float(np.median(e[o_[:int(round(c * len(o_)))]])), sr_epe_1m=float((ep[o_[:int(round(c * len(o_)))]] < 1).mean())) for c in (1.0, 0.9, 0.8, 0.5, 0.25)]
    from scipy.stats import spearmanr
    R['heldout_spearman_kappa_vs_error'] = dict(A=float(spearmanr(a_['k'][hm], a_['err'][hm]).statistic), B=float(spearmanr(k, e).statistic))
# ---- validation buildings of the held-out sites (never trained on by any model): single-image error, extra sample
Aimg = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image'); im = pd.read_pickle(f'{SG}/absheading/images.pkl').set_index('image'); V = {}
for nm, P in RAW.items():
    Pp = Aimg if P is None else P; v = Pp[Pp.part == 'val']; t = im.tour.reindex(v.index).values; held = ST.heldout.reindex(t).values.astype(bool); e = np.abs(wrap(v.ah_pred.values - im.heading.reindex(v.index).values))
    V[nm] = dict(heldout=dict(buildings=int(len(set(t[held]))), **hstats(e[held])), other=dict(buildings=int(len(set(t[~held]))), **hstats(e[~held])))
    # all test images (not only those in the evaluated pairs)
    v = Pp[Pp.part == 'test']; t = im.tour.reindex(v.index).values; held = ST.heldout.reindex(t).values.astype(bool); e = np.abs(wrap(v.ah_pred.values - im.heading.reindex(v.index).values))
    V[nm]['test_images_heldout'] = hstats(e[held]); V[nm]['test_images_other'] = hstats(e[~held])
R['single_image_val_and_all_test_images'] = V
# ---- image-level covariates on the held-out sites (single test images, raw-view networks): tilt and two crude illumination proxies
if 'B site-disjoint' in RAW and RAW['B site-disjoint'] is not None:
    Pb = RAW['B site-disjoint']; v = Pb[(Pb.part == 'test') & Pb.heldout.astype(bool)].copy(); v['errB'] = np.abs(wrap(v.ah_pred.values - v.heading.values)); v['errA'] = np.abs(wrap(Aimg.ah_pred.reindex(v.index).values - v.heading.values))
    if 'C size-matched control' in RAW: v['errC'] = np.abs(wrap(RAW['C size-matched control'].ah_pred.reindex(v.index).values - v.heading.values))
    I1 = np.load(f'{SG}/absheading/images_448x256.npy', mmap_mode='r'); rowi = pd.Series(np.arange(len(im)), index=im.index); con, deep = [], []
    for i in rowi.reindex(v.index).values:
        x = I1[i].astype(np.float32); Lm = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]; con.append(float(Lm.std())); deep.append(float(np.percentile(Lm, 5) / max(np.median(Lm), 1.0)))
    v['contrast'] = con; v['dark_ratio'] = deep; COV = {}
    def bins(col, edges, labels, name):
        rows = []
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            m = (v[col].values >= lo) & (v[col].values < hi)
            if m.any(): rows.append(dict(bin=lab, n=int(m.sum()), mean=float(v.errB.values[m].mean()), median=float(np.median(v.errB.values[m])), gt45=float((v.errB.values[m] > 45).mean()), A_mean=float(v.errA.values[m].mean()), A_median=float(np.median(v.errA.values[m]))))
        COV[name] = rows
    bins('tilt', [0, 15, 30, 46], ['0-15 deg', '15-30 deg', '30-45 deg'], 'camera tilt from nadir')
    for col, name in (('contrast', 'luminance contrast (std of luma), quartile'), ('dark_ratio', 'P5 / median of luma (low = deep shadows), quartile')):
        q = np.quantile(v[col].values, [0, .25, .5, .75, 1]); q[-1] += 1e-6; bins(col, q, [f'Q1 (lowest, {q[0]:.2f}-{q[1]:.2f})', f'Q2 ({q[1]:.2f}-{q[2]:.2f})', f'Q3 ({q[2]:.2f}-{q[3]:.2f})', f'Q4 (highest, {q[3]:.2f}-{q[4]:.2f})'], name)
    from scipy.stats import spearmanr as _sp
    R['heldout_image_covariates_B'] = COV; R['heldout_image_covariates_spearman_B'] = {c: float(_sp(v[c].values, v.errB.values).statistic) for c in ('tilt', 'contrast', 'dark_ratio')}; R['heldout_image_covariates_n_images'] = int(len(v))
    print('covariates', json.dumps(COV, indent=1), R['heldout_image_covariates_spearman_B'])
# ---- size-matched control: stopped after 3 epochs (GPU budget of the shared machine). Epoch-matched monitoring numbers of the two runs
#      (same schedule, same number of training images; single centre view, all cached test images), parsed from the training logs.
import re
def _mon(f):
    out = []
    for l in open(f).read().splitlines():
        m = re.match(r"== epoch (\d+) loss ([-\d.]+) \| val: MAE ([\d.]+) med ([\d.]+) .*?\| test: MAE ([\d.]+) med ([\d.]+) .*?\| test/heldout: MAE ([\d.]+) med ([\d.]+) .*?\| test/other: MAE ([\d.]+) med ([\d.]+)", l)
        if m: out.append(dict(epoch=int(m.group(1)), loss=float(m.group(2)), val=[float(m.group(3)), float(m.group(4))], test=[float(m.group(5)), float(m.group(6))], test_heldout=[float(m.group(7)), float(m.group(8))], test_other=[float(m.group(9)), float(m.group(10))]))
    return out
if os.path.exists(f'{O}/logs/train_raw_control_stopped.log'):
    mb, mc = _mon(f'{O}/logs/train_raw_sitedisjoint.log'), _mon(f'{O}/logs/train_raw_control_stopped.log'); ne = min(len(mb), len(mc))
    R['control_partial'] = dict(note='size-matched control stopped after its last completed epoch listed here; single centre view, no test-time averaging, all cached test images (5,184 in held-out sites, 12,627 elsewhere)', epochs=ne, site_disjoint=mb[:ne], control=mc[:ne], site_disjoint_all_epochs=mb)
    print('control (partial):', json.dumps(R['control_partial']['control'])); print('site-disjoint same epochs:', json.dumps(mb[:ne]))
R['training'] = {t: json.load(open(f'{O}/train_{t}.json')) for t in ('raw_sitedisjoint', 'raw_control', 'bev_sitedisjoint') if os.path.exists(f'{O}/train_{t}.json')}
json.dump(R, open(f'{O}/part_retrain.json', 'w'), indent=1)
np.savez_compressed(f'{O}/retrain_arrays.npz', heldout=hm, tour=tours.astype(str), site=T.site.values, **{f'err|{k}': v['err'] for k, v in EST.items()}, **{f'sg|{k}': v['sg'] for k, v in EST.items()}, **{f'epe|{k}': v['pose']['epe'] for k, v in EST.items()}, **{f'kappa|{k}': v['k'] for k, v in EST.items()}, epe_cam=ecam['epe'])
for sn, o in out.items():
    print(f"== {sn}: {o['buildings']} buildings, {o['pairs']} pairs | camera frame: EPE med {o['camera_frame']['epe_med']:.2f} SR {100*o['camera_frame']['sr_epe_1m']:.1f}")
    for en in EST:
        r = o[en]; h = r['fused_heading']; print(f"   {en:44s} heading mean {h['mean']:6.2f} [{r['fused_heading_mean_ci'][0]:.2f},{r['fused_heading_mean_ci'][1]:.2f}] med {h['median']:5.2f} >10 {100*h['gt10']:4.1f}% >45 {100*h['gt45']:4.1f}% | x/y {r['x_mae']:.2f}/{r['y_mae']:.2f} EPE mean {r['epe_mean']:.2f} med {r['epe_med']:.2f} SR_EPE {100*r['sr_epe_1m']:.1f} [{100*r['sr_epe_1m_ci'][0]:.1f},{100*r['sr_epe_1m_ci'][1]:.1f}] | kappa {r['kappa_median']:.0f}")
    for k, v in o['paired'].items(): print('   paired', k, {q: (round(w['delta'], 3), [round(x, 3) for x in w['ci']]) for q, w in v.items()})
    for k in o:
        if k.startswith('single image'): print(f"   {k:52s} mean {o[k]['mean']:6.2f} med {o[k]['median']:5.2f} >45 {100*o[k]['gt45']:.1f}%")
print('per held-out site'); [print('  ', json.dumps(r)) for r in per]
for k in ('heldout_by_distance_to_nearest_kept_training_building', 'heldout_bias_vs_noise_B', 'heldout_risk_coverage_B', 'heldout_spearman_kappa_vs_error', 'single_image_val_and_all_test_images'):
    if k in R: print(k, json.dumps(R[k], indent=1))
