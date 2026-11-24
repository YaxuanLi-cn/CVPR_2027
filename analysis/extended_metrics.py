#!/usr/bin/env python3
"""Extended metrics on the archived result-set predictions (test = 6,364 pairs, tour-disjoint).

Beyond the six MAEs: medians / tail percentiles, 3-D position error, joint success rates, pose AUC, the original
PairUAV relative errors, per-difficulty breakdowns, tour-bootstrap confidence intervals, error CDF figures and
(when the refit stores exist) reliability diagnostics of the fusion's predicted error.
Outputs: outputs/analysis/extended_metrics.{json,md}, cdf_*.png, reliability.png
"""
import sys, json, os, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import SG, add_targets, wrap
S2 = f'{SG}/stage2'; OUT = '/root/rasr-pairuav/outputs/analysis'; os.makedirs(OUT, exist_ok=True); BL = '/root/baseline'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'; PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']

# ------------------------------------------------------------------ assemble predictions (same recipe as stage2_eval.py)
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); F = pd.read_pickle(f'{S2}/s2_pred.pkl'); Z = pd.read_pickle(f'{SG}/half_zoom_pred.pkl')
D = P.merge(F.drop(columns=['split', 'tour']), on='manifest_index').merge(Z[['manifest_index', 'n_x', 'n_y', 'n_z', 'n_tilt', 'n_heading']], on='manifest_index')
D = D.merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl')[['manifest_index', 'zm_inl']], on='manifest_index', how='left')
A = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image'); B = pd.read_pickle(f'{S2}/pred_bev_s.pkl')
for s in 'ab':
    D[f'v2h_{s}'] = A.ah_pred.reindex(D[f'image_{s}']).values; D[f'v2k_{s}'] = A.ah_kappa.reindex(D[f'image_{s}']).values
    b = B[B.side == s].set_index('manifest_index'); D[f'bvh_{s}'] = b.ah_est.reindex(D.manifest_index).values; D[f'bvk_{s}'] = b.ak_est.reindex(D.manifest_index).values
T = D[D.split == 'test'].reset_index(drop=True); print('test rows', len(T))

def combine(d, hk, dh):
    s = 0; c = 0
    for ha, ka, hb, kb in hk:
        ta = np.deg2rad(d[ha]); tb = np.deg2rad(d[hb] - dh); s = s + d[ka] * np.sin(ta) + d[kb] * np.sin(tb); c = c + d[ka] * np.cos(ta) + d[kb] * np.cos(tb)
    return np.rad2deg(np.arctan2(s, c))
def world(h, lx, ly): h = np.deg2rad(h); return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly
V2 = [('v2h_a', 'v2k_a', 'v2h_b', 'v2k_b')]; BV = [('bvh_a', 'bvk_a', 'bvh_b', 'bvk_b')]
def ours(d, pre, src, al):
    h = d.heading_a.values if src is None else combine(d, src, d[pre + 'heading']); x, y = world(h, d[pre + 'lx'].values, d[pre + 'ly'].values); z = d[pre + 'z'].values
    n = np.sqrt(x**2 + y**2 + z**2).clip(1e-6); sc = (d[pre + 'dist'].values / n) ** al
    return dict(x=x * sc, y=y * sc, z=z * sc, tilt=d[pre + 'tilt'].values, heading=d[pre + 'heading'].values, lx=d[pre + 'lx'].values, ly=d[pre + 'ly'].values)
M = {'DreamNav stage 2 (final)': ours(T, 'p_', BV + V2, 0.5), 'DreamNav stage 1 (coarse)': ours(T, 's1_', V2, 0.75),
     'DreamNav stage 2, true abs. heading': ours(T, 'p_', None, 0.5),
     'MASt3R-stat MLP (RASR + sin/cos)': dict(x=T.n_x.values, y=T.n_y.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values)}
M['Constant (zero motion)'] = dict(x=np.zeros(len(T)), y=np.zeros(len(T)), z=np.zeros(len(T)), tilt=np.zeros(len(T)), heading=np.zeros(len(T)))
# external baselines: 12,313 rows in test-manifest order -> restrict to the result-set rows (checked against the labels)
Pte = P[P.split == 'test'].sort_values('manifest_index').reset_index(drop=True); pos = pd.Series(np.arange(len(Pte)), index=Pte.manifest_index.values).reindex(T.manifest_index).values
for name, f, kp, kt in (('DINOv3 ViT-7B', 'dinov3/outputs_xyz7b_mast3r_split/test_predictions.npz', 'pred_xyz_tilt_heading', 'true_xyz_tilt_heading'), ('DINOv3 ViT-L', 'dinov3/outputs_xyz_mast3r_split/test_predictions.npz', 'pred_xyz_tilt_heading', 'true_xyz_tilt_heading'),
                         ('Sample4Geo', 'sample4geo/outputs_xyz_mast3r_split_rerun/train_test_predictions.npz', 'pred_xyz_tilt_heading', 'true_xyz_tilt_heading'), ('DINO-WM', 'gaoyuezhou-dino_wm/outputs_pairuav5d/test_predictions.npz', 'pred_xyz_tilt_heading', 'true_xyz_tilt_heading'),
                         ('Visual Foresight', 'visualnav-transformer/outputs_pairuav5d_visual_foresight_v2/test_predictions.npz', 'pred_xyz_tilt_heading', 'true_xyz_tilt_heading'), ('AI2-THOR', 'ai2thor/checkpoints_xyz_mast3r_split_supervised/test_predictions.npz', 'pred', 'true')):
    try:
        d = np.load(f'{BL}/{f}'); p, t = d[kp], d[kt]
        ok = np.abs(t[pos, 0] - T.x.values).max() < 1e-2 and np.abs(wrap(t[pos, 4] - T.heading.values)).max() < 1e-2
        if not ok:   # fall back to matching on the labels themselves
            key = lambda a: [tuple(np.round(r, 3)) for r in a]; lut = {k: i for i, k in enumerate(key(t[:, :3]))}; ix = [lut.get(k, -1) for k in key(T[['x', 'y', 'z']].values)]
            if min(ix) < 0: print('skip (rows do not match):', name); continue
            p = p[ix]
        else: p = p[pos]
        M[name] = dict(x=p[:, 0], y=p[:, 1], z=p[:, 2], tilt=p[:, 3], heading=p[:, 4])
    except Exception as e: print('skip', name, e)

# ------------------------------------------------------------------ metrics
gt = dict(x=T.x.values, y=T.y.values, z=T.z.values, tilt=T.tilt.values, heading=T.heading.values); gd = np.sqrt(gt['x']**2 + gt['y']**2 + gt['z']**2)
def errs(m):
    e = dict(x=np.abs(m['x'] - gt['x']), y=np.abs(m['y'] - gt['y']), z=np.abs(m['z'] - gt['z']), tilt=np.abs(m['tilt'] - gt['tilt']), heading=np.abs(wrap(m['heading'] - gt['heading'])))
    pd_ = np.sqrt(m['x']**2 + m['y']**2 + m['z']**2); e['distance'] = np.abs(pd_ - gd); e['pos3d'] = np.sqrt(e['x']**2 + e['y']**2 + e['z']**2); e['pos_h'] = np.hypot(e['x'], e['y'])
    cos = (m['x'] * gt['x'] + m['y'] * gt['y'] + m['z'] * gt['z']) / (pd_.clip(1e-9) * gd.clip(1e-9)); e['tdir'] = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
    e['dist_rel'] = e['distance'] / np.maximum(gd, 0.5); e['pos_rel'] = e['pos3d'] / np.maximum(gd, 0.5)
    e['pose_ang'] = np.maximum(e['heading'], e['tilt'])                              # rotation-only pose error: max(heading error, tilt error); the translation direction (e['tdir']) is not part of it
    if 'lx' in m: e['lx'] = np.abs(m['lx'] - T.lx.values); e['ly'] = np.abs(m['ly'] - T.ly.values)
    return e
def auc(e, thr):
    """Area under the recall-vs-threshold curve up to thr, normalised to [0,1] (the usual pose-AUC definition)."""
    e = np.sort(e); r = (np.arange(len(e)) + 1) / len(e); i = np.searchsorted(e, thr); x = np.r_[0.0, e[:i], thr]; y = np.r_[0.0, r[:i], r[i - 1] if i else 0.0]
    return float(np.trapezoid(y, x) / thr)
def summary(e):
    s = {}
    for k in ('x', 'y', 'z', 'distance', 'heading', 'tilt', 'pos3d', 'pos_h', 'tdir') + (('lx', 'ly') if 'lx' in e else ()):
        s[k] = dict(mae=float(e[k].mean()), median=float(np.median(e[k])), p90=float(np.percentile(e[k], 90)), p95=float(np.percentile(e[k], 95)))
    s['rel'] = dict(distance_rel=float(e['dist_rel'].mean()), position_rel=float(e['pos_rel'].mean()), heading_rel=float((e['heading'] / 180).mean()))
    s['success'] = {f'pos<{t}m': float((e['pos3d'] < t).mean()) for t in (0.25, 0.5, 1, 2, 5)}
    s['success'].update({'dist<1m (benchmark_results.md)': float((e['distance'] <= 1.0).mean())})
    s['success'].update({f'pos<{a}m & head<{b}deg & tilt<{b}deg': float(((e['pos3d'] < a) & (e['heading'] < b) & (e['tilt'] < b)).mean()) for a, b in ((0.5, 0.5), (1, 1), (2, 2), (5, 5))})
    s['auc'] = {**{f'pose_auc@{t}deg': auc(e['pose_ang'], t) for t in (1, 5, 10, 20)}, **{f'pos_auc@{t}m': auc(e['pos3d'], t) for t in (1, 2, 5)}}
    return s
E = {k: errs(m) for k, m in M.items()}; R = {k: summary(e) for k, e in E.items()}

# tour-level bootstrap for the two DreamNav rows and their difference
tours = T.tour.values; ut = np.unique(tours); idx_by = {t: np.where(tours == t)[0] for t in ut}; rng = np.random.default_rng(0); boots = [np.concatenate([idx_by[t] for t in rng.choice(ut, len(ut))]) for _ in range(1000)]
def ci(f): v = np.array([f(b) for b in boots]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
CI = {}
for name in ('DreamNav stage 2 (final)', 'DreamNav stage 1 (coarse)'):
    e = E[name]; CI[name] = {'pos3d_mae': ci(lambda b: e['pos3d'][b].mean()), 'pos<1m': ci(lambda b: (e['pos3d'][b] < 1).mean()), 'joint(1m,1deg)': ci(lambda b: ((e['pos3d'][b] < 1) & (e['heading'][b] < 1) & (e['tilt'][b] < 1)).mean()), 'pose_auc@5deg': ci(lambda b: auc(e['pose_ang'][b], 5))}
e2, e1 = E['DreamNav stage 2 (final)'], E['DreamNav stage 1 (coarse)']
CI['stage2 - stage1'] = {'pos<1m': ci(lambda b: (e2['pos3d'][b] < 1).mean() - (e1['pos3d'][b] < 1).mean()), 'pos3d_mae': ci(lambda b: e2['pos3d'][b].mean() - e1['pos3d'][b].mean())}

# breakdowns for the final model
def breakdown(col, bins, labels):
    out = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        m = (col >= lo) & (col < hi)
        if m.sum(): out.append(dict(bin=lab, n=int(m.sum()), **{f'{nm}|{k}': float(E[full][k][m].mean()) for nm, full in (('s1', 'DreamNav stage 1 (coarse)'), ('s2', 'DreamNav stage 2 (final)')) for k in ('pos3d', 'heading', 'tilt')},
                                    s2_success_1m=float((e2['pos3d'][m] < 1).mean())))
    return out
BR = {'true distance (m)': breakdown(gd, [0, 5, 10, 20, 30, 1e9], ['0-5', '5-10', '10-20', '20-30', '>=30']),
      '|heading change| (deg)': breakdown(np.abs(gt['heading']), [0, 30, 60, 90, 135, 181], ['0-30', '30-60', '60-90', '90-135', '135-180']),
      '|tilt change| (deg)': breakdown(np.abs(gt['tilt']), [0, 10, 20, 30, 46], ['0-10', '10-20', '20-30', '30-45']),
      '|altitude change| (m)': breakdown(np.abs(gt['z']), [0, 5, 15, 25, 1e9], ['0-5', '5-15', '15-25', '>=25']),
      'stage-1 zoom inliers': breakdown(T.zm_inl.fillna(0).values, [0, 100, 200, 400, 800, 1e9], ['<100', '100-200', '200-400', '400-800', '>=800'])}

# reliability of the fusion's own predicted error (needs the refit stores; rows identical to the archived runs)
REL = {}
try:
    Rf = pd.read_pickle('/root/rasr-pairuav/outputs/online_models/stage2_refit_pred.pkl').merge(P, on='manifest_index'); Rf = Rf[Rf.split_x == 'test']
    from scipy.stats import spearmanr
    for k, act in (('lx', np.abs(Rf.p_lx - Rf.lx)), ('ly', np.abs(Rf.p_ly - Rf.ly)), ('z', np.abs(Rf.p_z - Rf.z)), ('heading', np.abs(wrap(Rf.p_heading - Rf.heading))), ('tilt', np.abs(Rf.p_tilt - Rf.tilt)), ('logd', np.abs(np.log(Rf.p_dist) - Rf.logd))):
        pr = Rf['r_' + k].values; act = np.asarray(act); o = np.argsort(pr); n = len(o); keep = np.linspace(0.1, 1.0, 19)
        spars = [float(act[o[:int(n * f)]].mean()) for f in keep]; oo = np.sort(act); orac = [float(oo[:int(n * f)].mean()) for f in keep]
        q = np.quantile(pr, np.linspace(0, 1, 11)); bi = np.clip(np.searchsorted(q, pr, side='right') - 1, 0, 9)
        REL[k] = dict(spearman=float(spearmanr(pr, act).statistic), mae_all=float(act.mean()), mae_most_confident_50pct=spars[int(np.argmin(np.abs(keep - 0.5)))], mae_most_confident_25pct=spars[int(np.argmin(np.abs(keep - 0.25)))],
                      ause=float(np.mean(np.array(spars) - np.array(orac)) / act.mean()), keep=keep.tolist(), sparsification=spars, oracle=orac,
                      bins_pred=[float(pr[bi == b].mean()) for b in range(10)], bins_actual=[float(act[bi == b].mean()) for b in range(10)])
except FileNotFoundError: print('refit stores not found -> reliability section skipped')

json.dump(dict(rows=int(len(T)), methods=R, bootstrap_ci95=CI, breakdowns=BR, reliability=REL), open(f'{OUT}/extended_metrics.json', 'w'), indent=1)

# ------------------------------------------------------------------ markdown
L = [f'# Extended metrics (test: {len(T):,} pairs, {len(ut)} tours)\n', '## Accuracy beyond MAE\n',
     '| Method | pos. err mean (m) | median | P90 | heading median (deg) | P90 | tilt median (deg) | P90 | transl. dir. median (deg) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for k, r in R.items(): L.append(f"| {k} | {r['pos3d']['mae']:.3f} | {r['pos3d']['median']:.3f} | {r['pos3d']['p90']:.3f} | {r['heading']['median']:.3f} | {r['heading']['p90']:.3f} | {r['tilt']['median']:.3f} | {r['tilt']['p90']:.3f} | {r['tdir']['median']:.2f} |")
L += ['\n## Success rate and AUC\n', '| Method | pos<0.5 m | pos<1 m | pos<2 m | pos<5 m | (1 m, 1 deg) | (2 m, 2 deg) | (5 m, 5 deg) | pose AUC@5 deg | @10 | @20 | pos AUC@1 m | @5 m |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for k, r in R.items():
    s, a = r['success'], r['auc']
    L.append(f"| {k} | {100*s['pos<0.5m']:.1f} | {100*s['pos<1m']:.1f} | {100*s['pos<2m']:.1f} | {100*s['pos<5m']:.1f} | {100*s['pos<1m & head<1deg & tilt<1deg']:.1f} | {100*s['pos<2m & head<2deg & tilt<2deg']:.1f} | {100*s['pos<5m & head<5deg & tilt<5deg']:.1f} | {100*a['pose_auc@5deg']:.1f} | {100*a['pose_auc@10deg']:.1f} | {100*a['pose_auc@20deg']:.1f} | {100*a['pos_auc@1m']:.1f} | {100*a['pos_auc@5m']:.1f} |")
L += ['\nSuccess = 3-D position error below the threshold; the joint columns also require heading-change and tilt-change error below the angle. Pose AUC uses the rotation only: max(heading error, tilt error). All in %.\n',
      '## Original PairUAV-style relative errors\n', '| Method | distance rel. | position rel. | heading / 180 |', '|---|---:|---:|---:|']
for k, r in R.items(): L.append(f"| {k} | {r['rel']['distance_rel']:.4f} | {r['rel']['position_rel']:.4f} | {r['rel']['heading_rel']:.5f} |")
L += ['\n## 95% confidence intervals (bootstrap over tours, 1000 resamples)\n']
for k, v in CI.items(): L.append(f'- **{k}**: ' + '; '.join(f'{m} [{a:.3f}, {b:.3f}]' for m, (a, b) in v.items()))
L += ['\n## Where the error is (stage 1 -> stage 2, mean 3-D position error in m)\n']
for nm, rows in BR.items():
    L += [f'\n**By {nm}**\n', '| bin | pairs | pos. err s1 | s2 | heading s1 | s2 | tilt s1 | s2 | s2 success <1 m |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows: L.append(f"| {r['bin']} | {r['n']} | {r['s1|pos3d']:.3f} | {r['s2|pos3d']:.3f} | {r['s1|heading']:.3f} | {r['s2|heading']:.3f} | {r['s1|tilt']:.3f} | {r['s2|tilt']:.3f} | {100*r['s2_success_1m']:.1f}% |")
if REL:
    L += ['\n## Does the fusion know when it is wrong? (stage-2 predicted error vs actual error, test pairs)\n', '| target | Spearman | MAE all | MAE of most-confident 50% | most-confident 25% | AUSE (normalised) |', '|---|---:|---:|---:|---:|---:|']
    for k, r in REL.items(): L.append(f"| {k} | {r['spearman']:.3f} | {r['mae_all']:.4f} | {r['mae_most_confident_50pct']:.4f} | {r['mae_most_confident_25pct']:.4f} | {r['ause']:.3f} |")
open(f'{OUT}/extended_metrics.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

# ------------------------------------------------------------------ figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e9e8e3', 'grid.linewidth': 0.8, 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
show = ['DreamNav stage 2 (final)', 'DreamNav stage 1 (coarse)', 'MASt3R-stat MLP (RASR + sin/cos)'] + (['DINOv3 ViT-7B'] if 'DINOv3 ViT-7B' in E else [])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
for a, (k, lab, xmax, logx) in zip(ax, (('pos3d', '3-D position error (m)', 20, True), ('heading', 'heading-change error (deg)', 20, True), ('tilt', 'tilt-change error (deg)', 10, True))):
    for c, nm in zip(PAL, show):
        e = np.sort(E[nm][k]); a.plot(np.maximum(e, 1e-3), (np.arange(len(e)) + 1) / len(e) * 100, color=c, lw=2, label=nm)
    a.set_xscale('log'); a.set_xlim(0.01 if k != 'pos3d' else 0.05, xmax * 5); a.set_ylim(0, 100); a.set_xlabel(lab); a.set_title(f'Pairs within error ({lab.split(" (")[0]})'); a.set_ylabel('% of test pairs')
ax[0].legend(frameon=False, fontsize=9, loc='upper left'); fig.tight_layout(); fig.savefig(f'{OUT}/cdf_errors.png', dpi=170); plt.close(fig)
if REL:
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.9)); ks = ['lx', 'z', 'heading', 'tilt']
    for c, k in zip(PAL, ks):
        r = REL[k]; ax[0].plot(np.array(r['keep']) * 100, np.array(r['sparsification']) / r['mae_all'], color=c, lw=2, label=k)
        ax[1].plot(r['bins_pred'], r['bins_actual'], color=c, lw=2, marker='o', ms=5, label=k)
    ax[0].set_xlabel('% of pairs kept (most confident first)'); ax[0].set_ylabel('MAE of kept pairs / MAE of all'); ax[0].set_title('Sparsification'); ax[0].set_ylim(0, 1.05); ax[0].legend(frameon=False, fontsize=9)
    lim = [1e-3, 30]; ax[1].plot(lim, lim, color='#c9c8c2', lw=1, ls='--'); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('predicted |error| (decile mean; m or deg)'); ax[1].set_ylabel('actual |error|'); ax[1].set_title('Calibration of predicted error'); ax[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/reliability.png', dpi=170); plt.close(fig)
print('wrote', OUT)
