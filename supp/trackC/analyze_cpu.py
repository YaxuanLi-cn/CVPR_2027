#!/usr/bin/env python3
"""Track C, CPU part: error statistics of the absolute-heading estimate on the 6,364 test pairs, where the error comes
from (building bias vs image noise, site familiarity, hemisphere, confidence) and what it costs in the pose output.
Uses only archived predictions. Output: outputs/supp/trackC/part_cpu.json
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp')
from common import *
O = f'{OUT}/trackC'; T = load_test(); n = len(T); ST = pd.read_pickle(f'{O}/sites.pkl').set_index('tour'); boots = boot_indices(T.tour.values, 1000, 0)
for c in ('site', 'site_n_train', 'heldout', 'lat', 'km_to_nearest_kept_train'): T[c] = ST[c].reindex(T.tour).values
gw = gt_world(T); gc = gt_cam(T); R = {'n_pairs': int(n), 'n_buildings': int(T.tour.nunique())}

def stats(e, groups=None, bt=None):
    e = np.asarray(e, float); s = dict(n=int(len(e)), mean=float(e.mean()), median=float(np.median(e)), **{f'p{q}': float(np.percentile(e, q)) for q in (75, 90, 95, 99)}, **{f'gt{t}': float((e > t).mean()) for t in (5, 10, 20, 45, 90)})
    if bt is not None: s['mean_ci'] = ci(lambda b: e[b].mean(), bt); s['median_ci'] = ci(lambda b: np.median(e[b]), bt)
    return s
def circ_mean(d): d = np.deg2rad(np.asarray(d, float)); return float(np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean())))

# ------------------------------------------------------------------ 1. error statistics
E1 = {}
side = lambda net: (np.r_[np.abs(wrap(T[f'{net}h_a'] - T.heading_a)), np.abs(wrap(T[f'{net}h_b'] - T.heading_b))], np.r_[T.tour.values, T.tour.values])
for net, nm in (('v2', 'raw-view network, single image'), ('bv', 'nadir-rectified network, single image')):
    e, g = side(net); E1[nm + ' (12,728 pair sides)'] = stats(e, bt=boot_indices(g, 1000, 0))
# unique test images, raw-view network (its prediction does not depend on the pair)
A = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl'); im = pd.read_pickle(f'{SG}/absheading/images.pkl').set_index('image'); A = A[A.part == 'test'].copy(); A['heading'] = im.heading.reindex(A.image).values; A['tilt'] = im.tilt.reindex(A.image).values; A['tour'] = im.tour.reindex(A.image).values
A['err'] = np.abs(wrap(A.ah_pred - A.heading)); used = set(T.image_a) | set(T.image_b); Au = A[A.image.isin(used)]
E1[f'raw-view network, unique images of the test pairs ({len(Au)})'] = stats(Au.err, bt=boot_indices(Au.tour.values, 1000, 0)); E1[f'raw-view network, all cached test images ({len(A)})'] = stats(A.err, bt=boot_indices(A.tour.values, 1000, 0))
H = {}     # fused estimates of the heading of camera a
for nm, src, dh in (('raw-view, fused a+b (Stage I)', V2, T.s1_heading.values), ('nadir-rectified, fused a+b', BV, T.p_heading.values), ('raw-view, fused a+b (with refined rotation)', V2, T.p_heading.values), ('two-network ensemble, fused a+b (Stage II)', BV + V2, T.p_heading.values)):
    h, k = fuse_heading(T, src, dh); H[nm] = (h, k); E1[nm] = stats(np.abs(wrap(h - T.heading_a.values)), bt=boots)
R['error_statistics'] = E1
hS2, kS2 = H['two-network ensemble, fused a+b (Stage II)']; hS1, kS1 = H['raw-view, fused a+b (Stage I)']; sgn2 = wrap(hS2 - T.heading_a.values); err2 = np.abs(sgn2); sgn1 = wrap(hS1 - T.heading_a.values)
# gain of fusing the two views with the relative rotation (single image a vs fused)
R['fusion_gain'] = {'raw-view, image a only': stats(np.abs(wrap(T.v2h_a - T.heading_a)), bt=boots), 'raw-view, fused a+b': E1['raw-view, fused a+b (Stage I)']}

# ------------------------------------------------------------------ 6. consequences for the pose output
def pose_row(pred, gt): e = errors(pred, gt); return dict(x_mae=float(e['x'].mean()), y_mae=float(e['y'].mean()), z_mae=float(e['z'].mean()), epe_mean=float(e['epe'].mean()), epe_med=float(np.median(e['epe'])), sr_epe_1m=float((e['epe'] < 1).mean()), sr_epe_2m=float((e['epe'] < 2).mean()),
                                                         sr_5dof_1m_1deg=float(((e['epe'] < 1) & (e['theta'] < 1) & (e['psi'] < 1)).mean()), sr_d_1m=float((e['d'] <= 1).mean()), d_mae=float(e['d'].mean())), e
C = {}; rng = np.random.default_rng(0)
for nm, pre in (('Stage I', 's1_'), ('Stage II', 'p_')):
    c = {}; c['camera frame (no heading needed)'], ecam = pose_row(dreamnav(T, pre, 'cam'), gc); c['ENU, learned heading'], enet = pose_row(dreamnav(T, pre, 'net'), gw)
    for k in ('epe_mean', 'epe_med', 'sr_epe_1m', 'sr_5dof_1m_1deg', 'x_mae', 'y_mae'):
        f = {'epe_mean': lambda e: e['epe'].mean(), 'epe_med': lambda e: np.median(e['epe']), 'sr_epe_1m': lambda e: (e['epe'] < 1).mean(), 'sr_5dof_1m_1deg': lambda e: ((e['epe'] < 1) & (e['theta'] < 1) & (e['psi'] < 1)).mean(), 'x_mae': lambda e: e['x'].mean(), 'y_mae': lambda e: e['y'].mean()}[k]
        c['ENU, learned heading'][k + '_ci'] = ci(lambda b: f({q: v[b] for q, v in enet.items()}), boots); c['camera frame (no heading needed)'][k + '_ci'] = ci(lambda b: f({q: v[b] for q, v in ecam.items()}), boots)
    for sg in (0, 1, 2, 5, 10):
        rows = [pose_row(dreamnav(T, pre, T.heading_a.values + sg * rng.standard_normal(n)), gw)[0] for _ in range(1 if sg == 0 else 20)]
        c[f'ENU, compass (true heading + N(0, {sg} deg))'] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}; c[f'ENU, compass (true heading + N(0, {sg} deg))']['draws'] = len(rows)
    et = pose_row(dreamnav(T, pre, 'true'), gw)[1]; fail = enet['epe'] >= 1
    c['failure_attribution'] = dict(failures_enu_learned=int(fail.sum()), fail_rate=float(fail.mean()), fixed_by_true_heading=float((et['epe'][fail] < 1).mean()), still_failing_with_true_heading=float((et['epe'][fail] >= 1).mean()),
                                    failures_camera_frame=int((et['epe'] >= 1).sum()))
    C[nm] = c
R['consequences'] = C
# horizontal displacement magnitude: a heading error delta moves the endpoint by 2 * rho * sin(delta / 2)
rho = np.hypot(T.x.values, T.y.values); R['horizontal_displacement_m'] = dict(mean=float(rho.mean()), median=float(np.median(rho)), p90=float(np.percentile(rho, 90)), max=float(rho.max()))
e2net = errors(dreamnav(T, 'p_', 'net'), gw); e2cam = errors(dreamnav(T, 'p_', 'cam'), gc)
# ENU endpoint error by absolute-heading error (Stage II)
B = []
for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 181)):
    m = (err2 >= lo) & (err2 < hi); B.append(dict(bin=f'{lo}-{hi if hi < 181 else 180}', n=int(m.sum()), frac=float(m.mean()), epe_enu_mean=float(e2net['epe'][m].mean()), epe_enu_med=float(np.median(e2net['epe'][m])), epe_cam_mean=float(e2cam['epe'][m].mean()), epe_cam_med=float(np.median(e2cam['epe'][m])),
                                              sr_epe_enu=float((e2net['epe'][m] < 1).mean()), sr_epe_cam=float((e2cam['epe'][m] < 1).mean()), share_of_excess_epe=float((e2net['epe'][m] - e2cam['epe'][m]).sum() / (e2net['epe'] - e2cam['epe']).sum())))
R['by_heading_error'] = B

# ------------------------------------------------------------------ 2a. building-level bias vs image-level noise
tours = T.tour.values; bias = pd.Series(sgn2).groupby(tours).apply(circ_mean); b_i = bias.reindex(tours).values; resid = wrap(sgn2 - b_i)
def enu_with(herr): return pose_row(dreamnav(T, 'p_', T.heading_a.values + herr), gw)[0]
full, bonly, nonly, none = enu_with(sgn2), enu_with(b_i), enu_with(resid), enu_with(np.zeros(n))
ab = np.abs(bias.values); cnt = pd.Series(err2).groupby(tours).mean().sort_values(ascending=False); tot = err2.sum()
R['bias_vs_noise'] = dict(buildings=int(len(bias)), pairs_per_building_median=float(pd.Series(tours).value_counts().median()),
    abs_building_bias=dict(mean=float(ab.mean()), median=float(np.median(ab)), p90=float(np.percentile(ab, 90)), gt5=float((ab > 5).mean()), gt10=float((ab > 10).mean()), gt20=float((ab > 20).mean()), gt45=float((ab > 45).mean())),
    heading_mae=dict(full=float(err2.mean()), building_bias_only=float(np.abs(b_i).mean()), image_noise_only=float(np.abs(resid).mean())), heading_median=dict(full=float(np.median(err2)), building_bias_only=float(np.median(np.abs(b_i))), image_noise_only=float(np.median(np.abs(resid)))),
    enu=dict(full=full, building_bias_only=bonly, image_noise_only=nonly, true_heading=none),
    share_of_total_abs_error_from_worst_buildings={f'top{k}': float(pd.Series(err2).groupby(tours).sum().sort_values(ascending=False).iloc[:k].sum() / tot) for k in (5, 10, 15, 29)},
    buildings_with_mean_error_gt20=int((cnt > 20).sum()), buildings_with_mean_error_gt45=int((cnt > 45).sum()))
# same decomposition at the image level for the raw-view network (all cached test images, ~61 per building)
A['sgn'] = wrap(A.ah_pred - A.heading); bA = A.groupby('tour').sgn.apply(circ_mean); rA = wrap(A.sgn - bA.reindex(A.tour).values)
R['bias_vs_noise_single_image_raw_view'] = dict(images=int(len(A)), buildings=int(A.tour.nunique()), mae_full=float(A.err.mean()), mae_building_bias_only=float(np.abs(bA.reindex(A.tour).values).mean()), mae_image_noise_only=float(np.abs(rA).mean()),
                                                median_full=float(A.err.median()), median_image_noise_only=float(np.median(np.abs(rA))), abs_bias_median=float(np.median(np.abs(bA))), abs_bias_gt20=float((np.abs(bA) > 20).mean()))

# single-image error by camera tilt (archived networks)
Bv = pd.read_pickle(f'{S2}/pred_bev_s.pkl'); Bv = Bv[Bv.part == 'test']; Bv = Bv.assign(err=np.abs(wrap(Bv.ah_est - Bv.heading_gt))); TL = []
for lo, hi in ((0, 15), (15, 30), (30, 46)):
    ma = (A.tilt >= lo) & (A.tilt < hi); mb = (Bv.tilt_gt >= lo) & (Bv.tilt_gt < hi)
    TL.append(dict(bin=f'{lo}-{min(hi, 45)} deg', raw_view=dict(n=int(ma.sum()), mean=float(A.err[ma].mean()), median=float(A.err[ma].median()), gt45=float((A.err[ma] > 45).mean())), nadir_rectified=dict(n=int(mb.sum()), mean=float(Bv.err[mb].mean()), median=float(Bv.err[mb].median()), gt45=float((Bv.err[mb] > 45).mean()))))
R['single_image_by_tilt'] = TL; print('BY TILT', json.dumps(TL))
# ------------------------------------------------------------------ 2b. site familiarity; 2c. hemisphere
def group_table(key, labels):
    out = []
    for lab in labels:
        m = (key == lab)
        if not m.any(): continue
        bt = boot_indices(tours[m], 1000, 0); idx = np.where(m)[0]
        out.append(dict(bin=str(lab), buildings=int(len(set(tours[m]))), pairs=int(m.sum()), heading_s2=stats(err2[m], bt=bt), heading_s1=stats(np.abs(sgn1[m]), bt=bt),
                        epe_enu_mean=float(e2net['epe'][m].mean()), epe_enu_med=float(np.median(e2net['epe'][m])), sr_epe_enu=float((e2net['epe'][m] < 1).mean()), sr_epe_enu_ci=ci(lambda b: (e2net['epe'][idx[b]] < 1).mean(), bt),
                        epe_cam_mean=float(e2cam['epe'][m].mean()), epe_cam_med=float(np.median(e2cam['epe'][m])), sr_epe_cam=float((e2cam['epe'][m] < 1).mean()), sr_epe_cam_ci=ci(lambda b: (e2cam['epe'][idx[b]] < 1).mean(), bt),
                        x_mae=float(e2net['x'][m].mean()), y_mae=float(e2net['y'][m].mean())))
    return out
fam = pd.cut(T.site_n_train, [-1, 0, 5, 20, 1e9], labels=['0', '1-5', '6-20', '>20']).astype(str).values; R['by_site_training_buildings'] = group_table(fam, ['0', '1-5', '6-20', '>20'])
# distance to the nearest building the heading networks were trained on (their views cover a few hundred metres of ground, so close neighbours share scene content)
Sall = pd.read_pickle(f'{O}/sites.pkl'); la, lo = np.deg2rad(Sall.lat.values), np.deg2rad(Sall.lon.values); X = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * 6371.0e3
tr = (Sall.part == 'train').values; dmin = np.sqrt(((X[:, None, :] - X[None, tr, :])**2).sum(-1)).min(1); Sall['m_to_nearest_train'] = dmin; T['m_to_nearest_train'] = Sall.set_index('tour').m_to_nearest_train.reindex(T.tour).values
lab = pd.cut(T.m_to_nearest_train, [-1, 100, 200, 400, 1000, 5000, 1e12], labels=['<100 m', '100-200 m', '200-400 m', '0.4-1 km', '1-5 km', '>5 km']).astype(str).values
R['by_distance_to_nearest_training_building'] = group_table(lab, ['<100 m', '100-200 m', '200-400 m', '0.4-1 km', '1-5 km', '>5 km'])
tb = T.groupby('tour').agg(n_train=('site_n_train', 'first'), m=('m_to_nearest_train', 'first')).assign(err=pd.Series(err2).groupby(tours).mean())
from scipy.stats import spearmanr
R['building_level_correlation'] = dict(spearman_meanerr_vs_site_n_train=float(spearmanr(tb.err, tb.n_train).statistic), spearman_meanerr_vs_distance=float(spearmanr(tb.err, tb.m).statistic), buildings=int(len(tb)),
                                       m_to_nearest_train=dict(median=float(tb.m.median()), p10=float(tb.m.quantile(.1)), p90=float(tb.m.quantile(.9))))
# replication on the validation buildings (10 % of the training buildings, never used to train the networks): single image, raw-view network
Av = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl'); Av = Av[Av.part == 'val'].copy(); Av['heading'] = im.heading.reindex(Av.image).values; Av['tour'] = im.tour.reindex(Av.image).values; Av['err'] = np.abs(wrap(Av.ah_pred - Av.heading))
Sx = Sall.set_index('tour'); Av['m'] = Sx.m_to_nearest_train.reindex(Av.tour).values; Av['lat'] = Sx.lat.reindex(Av.tour).values; A['m'] = Sx.m_to_nearest_train.reindex(A.tour).values; A['lat'] = Sx.lat.reindex(A.tour).values; REP = []
for lo, hi, lab_ in ((-1, 100, '<100 m'), (100, 200, '100-200 m'), (200, 400, '200-400 m'), (400, 1000, '0.4-1 km'), (1000, 5000, '1-5 km'), (5000, 1e12, '>5 km')):
    r = dict(bin=lab_)
    for nm, Q in (('test', A), ('val', Av)):
        q = Q[(Q.m > lo) & (Q.m <= hi) & (Q.lat > 0)]; r[nm] = dict(buildings=int(q.tour.nunique()), images=int(len(q)), mean=float(q.err.mean()) if len(q) else None, median=float(q.err.median()) if len(q) else None, gt45=float((q.err > 45).mean()) if len(q) else None)
    REP.append(r)
R['single_image_raw_view_by_distance_test_and_val_northern'] = REP; print('REPLICATION', json.dumps(REP))
hem = np.where(T.lat.values < 0, 'south', 'north'); R['by_hemisphere'] = group_table(hem, ['north', 'south'])
south = hem == 'south'; R['southern_signed_error_circular_mean'] = circ_mean(sgn2[south]) if south.any() else None
ex = ~south; R['excluding_southern_hemisphere'] = dict(pairs=int(ex.sum()), heading=stats(err2[ex]), epe_enu_mean=float(e2net['epe'][ex].mean()), epe_enu_med=float(np.median(e2net['epe'][ex])), sr_epe_enu=float((e2net['epe'][ex] < 1).mean()), x_mae=float(e2net['x'][ex].mean()), y_mae=float(e2net['y'][ex].mean()))

# ------------------------------------------------------------------ 2d. confidence (kappa): risk-coverage
o = np.argsort(-kS2); RC = []
for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25):
    k = o[:int(round(cov * n))]; RC.append(dict(coverage=cov, pairs=int(len(k)), kappa_min=float(kS2[k].min()), heading_mae=float(err2[k].mean()), heading_median=float(np.median(err2[k])), heading_gt10=float((err2[k] > 10).mean()), heading_gt45=float((err2[k] > 45).mean()),
                                                epe_enu_mean=float(e2net['epe'][k].mean()), epe_enu_med=float(np.median(e2net['epe'][k])), sr_epe_enu=float((e2net['epe'][k] < 1).mean()), sr_epe_cam_same_pairs=float((e2cam['epe'][k] < 1).mean())))
oo = np.argsort(err2); orc = [dict(coverage=c, heading_mae=float(err2[oo[:int(round(c * n))]].mean())) for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25)]
R['risk_coverage'] = dict(by_kappa=RC, oracle=orc, spearman_kappa_vs_error=float(spearmanr(kS2, err2).statistic))
q = np.quantile(kS2, np.linspace(0, 1, 9)); bi = np.clip(np.searchsorted(q, kS2, side='right') - 1, 0, 7); R['kappa_octiles'] = [dict(octile=int(i + 1), kappa_mean=float(kS2[bi == i].mean()), heading_mae=float(err2[bi == i].mean()), heading_median=float(np.median(err2[bi == i]))) for i in range(8)]
# hybrid that switches to a compass only when the network is unsure would need the compass anyway; report the selective SR instead (above)
json.dump(R, open(f'{O}/part_cpu.json', 'w'), indent=1)
np.savez_compressed(f'{O}/cpu_arrays.npz', err_s2=err2, sgn_s2=sgn2, kappa_s2=kS2, err_s1=np.abs(sgn1), epe_enu=e2net['epe'], epe_cam=e2cam['epe'], site_n_train=T.site_n_train.values, m_to_nearest_train=T.m_to_nearest_train.values, tour=T.tour.values.astype(str), lat=T.lat.values,
                    err_single_raw=np.abs(wrap(T.v2h_a - T.heading_a)), err_single_bev=np.abs(wrap(T.bvh_a - T.heading_a)), err_img_raw=A.err.values)
p = lambda d: json.dumps(d, indent=1)
print('ERROR STATS'); [print(f"  {k:60s} mean {v['mean']:.2f} [{v.get('mean_ci', [0, 0])[0]:.2f},{v.get('mean_ci', [0, 0])[1]:.2f}] med {v['median']:.2f} p90 {v['p90']:.1f} p95 {v['p95']:.1f} p99 {v['p99']:.1f} >10 {100*v['gt10']:.1f}% >45 {100*v['gt45']:.1f}% >90 {100*v['gt90']:.1f}%") for k, v in E1.items()]
print('CONSEQUENCES'); [print(f"  {s:8s} {k:48s} x {v['x_mae']:.2f} y {v['y_mae']:.2f} EPE mean {v['epe_mean']:.2f} med {v['epe_med']:.2f} SR_EPE {100*v['sr_epe_1m']:.1f} 5DoF {100*v['sr_5dof_1m_1deg']:.1f}") for s, c in C.items() for k, v in c.items() if k != 'failure_attribution']
print('  failure attribution', p({s: c['failure_attribution'] for s, c in C.items()}))
print('BIAS VS NOISE', p(R['bias_vs_noise'])); print(p(R['bias_vs_noise_single_image_raw_view']))
for key in ('by_site_training_buildings', 'by_distance_to_nearest_training_building', 'by_hemisphere'):
    print(key.upper()); [print(f"  {r['bin']:10s} bld {r['buildings']:4d} pairs {r['pairs']:5d} | heading S2 mean {r['heading_s2']['mean']:.2f} [{r['heading_s2']['mean_ci'][0]:.2f},{r['heading_s2']['mean_ci'][1]:.2f}] med {r['heading_s2']['median']:.2f} >45 {100*r['heading_s2']['gt45']:.1f}% | EPE ENU mean {r['epe_enu_mean']:.2f} med {r['epe_enu_med']:.2f} SR {100*r['sr_epe_enu']:.1f} | cam EPE med {r['epe_cam_med']:.2f} SR {100*r['sr_epe_cam']:.1f}") for r in R[key]]
print('corr', p(R['building_level_correlation'])); print('south signed', R['southern_signed_error_circular_mean'], 'excl south', p(R['excluding_southern_hemisphere']))
print('RISK-COVERAGE'); [print(f"  cov {r['coverage']:.2f} head mae {r['heading_mae']:.2f} med {r['heading_median']:.2f} >45 {100*r['heading_gt45']:.2f}% | EPE ENU med {r['epe_enu_med']:.2f} SR {100*r['sr_epe_enu']:.1f} (cam SR same pairs {100*r['sr_epe_cam_same_pairs']:.1f})") for r in RC]
print('by heading error'); [print('  ', b) for b in B]
