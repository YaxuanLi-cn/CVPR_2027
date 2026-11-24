#!/usr/bin/env python3
"""Track D, item 4: DreamNav on UNFILTERED pairs.

sample : 4 directed pairs per test building drawn uniformly at random from ALL 150 x 149 camera pairs (nothing excluded),
         labels rebuilt from the KML files (supp/trackD/labels.py; identical to the released labels to < 1e-6 m)
run    : the full online engine (Stage I + II + absolute heading), no re-training
report : metrics (supp/common.py) next to the archived predictions on the selected test pairs of the same buildings;
         how the unfiltered pairs differ (pose differences, baseline, ground-footprint overlap, position inside the
         label ranges of the selected pairs); SR as a function of overlap for both sets
Usage: unfiltered.py sample | run | report
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ.setdefault(_k, '1')   # nproc reports 128 cores but the cgroup quota is ~30: without this every forked worker spawns 127 BLAS/OpenMP threads
import sys, os, json, time, numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'
for p in (f'{ROOT}/supp', f'{ROOT}/supp/trackD', f'{ROOT}/online', f'{ROOT}/superglue_fusion'): sys.path.insert(0, p)
from labels import DATA, cameras, pair_labels
OUT = f'{ROOT}/outputs/supp/trackD'; C_ = f'{OUT}/cache'; PER = 4; SEED = 2026; W0, H0 = 3686, 2073


def sample():
    from common import load_test
    T = load_test(); tours = sorted(T.tour.unique()); sel = pd.read_pickle(f'{C_}/pairs_selected.pkl'); selset = set(zip(sel.tour, sel.a, sel.b)); rows = []
    for t in tours:
        C = cameras(t); n = len(C); rng = np.random.default_rng([SEED, int(t)]); idx = rng.choice(n * (n - 1), PER, replace=False); ia = idx // (n - 1); ib = idx % (n - 1); ib = ib + (ib >= ia)   # uniform over ordered pairs a != b
        a = C.i.values[ia]; b = C.i.values[ib]; L = pair_labels(C, a, b); Ci = C.set_index('i')
        for k in range(PER):
            rows.append(dict(tour=t, a=int(a[k]), b=int(b[k]), image_a=Ci.image[a[k]], image_b=Ci.image[b[k]], **{key: float(v[k]) for key, v in L.items()}, is_selected=(t, int(a[k]), int(b[k])) in selset,
                             **{f'{c}_{s}': float(Ci[c][i]) for s, i in (('a', a[k]), ('b', b[k])) for c in ('e', 'n', 'u', 'hag')}))
    U = pd.DataFrame(rows); h = np.deg2rad(U.heading_a); U['lx'] = np.cos(h) * U.x - np.sin(h) * U.y; U['ly'] = np.sin(h) * U.x + np.cos(h) * U.y; U.to_pickle(f'{C_}/unfiltered_pairs.pkl')
    print(len(U), 'pairs from', U.tour.nunique(), 'buildings | also among the selected pairs:', int(U.is_selected.sum()), '| baseline quantiles', U.dist.quantile([.05, .5, .95]).round(1).tolist())


def run(chunk=80):
    from engine import Engine
    U = pd.read_pickle(f'{C_}/unfiltered_pairs.pkl'); f = f'{C_}/unfiltered_pred.pkl'; done = pd.read_pickle(f) if os.path.exists(f) else None; start = 0 if done is None else len(done); parts = [] if done is None else [done]
    eng = Engine(workers=4, loader_workers=3, verbose=False); t0 = time.time(); T = {}
    for i in range(start, len(U), chunk):
        R = eng.predict(U.iloc[i:i + chunk][['image_a', 'image_b']]); parts.append(R.drop(columns=['manifest_index', 'image_a', 'image_b']))
        for k, v in R.attrs['timing'].items(): T[k] = T.get(k, 0.0) + v
        pd.concat(parts, ignore_index=True).to_pickle(f); print(f'{min(i + chunk, len(U))}/{len(U)} pairs, {(time.time() - t0) / (min(i + chunk, len(U)) - start):.2f} s/pair', flush=True)
    eng.close(); json.dump(dict(seconds=time.time() - t0, pairs=len(U) - start, phases=T), open(f'{OUT}/logs/unfiltered_timing.json', 'w')); print('done')


def overlap(pa, ha, ta, pb, hb, tb, nx=48, ny=27):
    """Fraction of the source image (grid of pixels) whose ground footprint lies inside the target image.
    Ground = horizontal plane at the terrain altitude of the building anchor (u = 0 in the anchor frame); p = (e, n, u)."""
    from geom import R_wc
    from pose_from_matches import K0
    u, v = np.meshgrid(np.linspace(0, W0 - 1, nx), np.linspace(0, H0 - 1, ny)); ray = np.c_[(u.ravel() - K0[0, 2]) / K0[0, 0], (v.ravel() - K0[1, 2]) / K0[1, 1], np.ones(u.size)] @ R_wc(ha, ta).T
    ok = ray[:, 2] < -1e-6; s = np.where(ok, -pa[2] / np.where(ok, ray[:, 2], -1.0), np.nan); X = np.asarray(pa)[None] + s[:, None] * ray; Xc = (X - np.asarray(pb)[None]) @ R_wc(hb, tb)
    with np.errstate(all='ignore'): px = K0[0, 0] * Xc[:, 0] / Xc[:, 2] + K0[0, 2]; py = K0[1, 1] * Xc[:, 1] / Xc[:, 2] + K0[1, 2]
    return float(np.mean(ok & (Xc[:, 2] > 0) & (px >= -0.5) & (px <= W0 - 0.5) & (py >= -0.5) & (py <= H0 - 0.5)))


def report():
    from common import errors, summarize, boot_indices, ci, load_test, dreamnav, gt_cam, gt_world
    U = pd.read_pickle(f'{C_}/unfiltered_pairs.pkl'); R = pd.read_pickle(f'{C_}/unfiltered_pred.pkl'); partial = len(R) < len(U)
    if partial: assert os.environ.get('UNF_PARTIAL'), 'predictions incomplete'; U = U.iloc[:len(R)]; print('PARTIAL report on', len(R), 'pairs (debug only)')
    T = load_test(); res = {}
    F = pd.concat([U.reset_index(drop=True), R.rename(columns={c: 'p_' + c[3:] for c in R.columns if c.startswith('s2_') and c[3:] in ('lx', 'ly', 'z', 'dist', 'tilt', 'heading')})], axis=1)
    S = json.load(open(f'{OUT}/dataset_stats.json'))['candidates_vs_selected']['selected_label_range']; inside = np.ones(len(F), bool); per = {}
    for k in ('x', 'y', 'z', 'tilt', 'heading'): m = (F[k].values >= S[k][0]) & (F[k].values <= S[k][1]); per[k] = float(m.mean()); inside &= m
    F['inside'] = inside
    # proxy of the released visual screen (supp/trackD/sift_screen.py): >= 20 MAGSAC inliers of SIFT mutual-ratio matches, covering >= 4 cells of a 4 x 4 grid
    sf = f'{C_}/unfiltered_sift.pkl'; have_screen = os.path.exists(sf)
    if have_screen: Q = pd.read_pickle(sf).iloc[:len(F)]; assert (Q.a.values == F.a.values).all() and (Q.tour.values == F.tour.values).all(); F['sift_inliers'] = Q.sift_inliers.values; F['sift_grid_cells'] = Q.sift_grid_cells.values; F['screen_pass'] = (F.sift_inliers >= 20) & (F.sift_grid_cells >= 4)
    # ---- ground-footprint overlap for both sets
    F['ovl'] = [overlap((r.e_a, r.n_a, r.u_a), r.heading_a, r.tilt_a, (r.e_b, r.n_b, r.u_b), r.heading_b, r.tilt_b) for r in F.itertuples()]
    cam = {t: cameras(t).set_index('i') for t in sorted(T.tour.unique())}
    T['ovl'] = [overlap(tuple(cam[r.tour].loc[r.a, ['e', 'n', 'u']]), r.heading_a, r.tilt_a, tuple(cam[r.tour].loc[r.b, ['e', 'n', 'u']]), r.heading_b, r.tilt_b) for r in T.itertuples()]
    def block(D, name, groups):
        out = {}; b = boot_indices(groups, 1000, 0)
        for nm, pre in (('Stage I', 's1_'), ('Stage II', 'p_')):
            for fr, h, g in (('camera frame', 'cam', gt_cam(D)), ('ENU, learned heading', 'net', gt_world(D))):
                e = errors(dreamnav(D, pre, h), g); s = summarize(e)
                s['ci'] = dict(d_mae=ci(lambda i: e['d'][i].mean(), b), sr_d_1m=ci(lambda i: (e['d'][i] <= 1).mean(), b), epe_med=ci(lambda i: np.median(e['epe'][i]), b), epe_mae=ci(lambda i: e['epe'][i].mean(), b), sr_epe_1m=ci(lambda i: (e['epe'][i] < 1).mean(), b),
                               theta_mae=ci(lambda i: e['theta'][i].mean(), b), psi_mae=ci(lambda i: e['psi'][i].mean(), b)); out[f'{nm} | {fr}'] = s; D[f'_epe_{pre}{h}'] = e['epe']; D[f'_d_{pre}{h}'] = e['d']
        res[name] = dict(n=int(len(D)), buildings=int(pd.Series(groups).nunique()), methods=out); return out
    block(T, 'selected test pairs (archived predictions)', T.tour.values); block(F, 'unfiltered pairs (all)', F.tour.values)
    Fi = F[F.inside].reset_index(drop=True); Fo = F[~F.inside].reset_index(drop=True); block(Fi, 'unfiltered pairs inside the label ranges of the selected pairs', Fi.tour.values); block(Fo, 'unfiltered pairs outside those ranges', Fo.tour.values)
    if have_screen:
        for nm, m in (('unfiltered pairs passing the screen proxy', F.screen_pass), ('unfiltered pairs failing the screen proxy', ~F.screen_pass), ('unfiltered pairs inside the label ranges and passing the screen proxy', F.inside & F.screen_pass)):
            d = F[m].reset_index(drop=True); block(d, nm, d.tour.values)
        K = pd.read_pickle(f'{C_}/sift_control.pkl'); kp = (K.sift_inliers >= 20) & (K.sift_grid_cells >= 4); li = np.log((K.sift_inliers + 1) / (K.stored_inliers + 1))
        res['screen_proxy'] = dict(config=json.load(open(f'{C_}/sift_calib.json'))['best'], note=json.load(open(f'{C_}/sift_calib.json'))['note'], criterion='inliers >= 20 and grid_cells >= 4 (the minima stored with the selected pairs)',
                                   control_selected_pairs=dict(n=int(len(K)), pass_rate=float(kp.mean()), median_ratio_my_over_stored_inliers=float(np.exp(li.median())), mad_log_ratio=float((li - li.median()).abs().median()), corr_log_inliers=float(np.corrcoef(np.log1p(K.sift_inliers), np.log1p(K.stored_inliers))[0, 1])),
                                   unfiltered=dict(n=int(len(F)), pass_rate=float(F.screen_pass.mean()), pass_rate_threshold_16=float(((F.sift_inliers >= 16) & (F.sift_grid_cells >= 4)).mean()), pass_rate_threshold_25=float(((F.sift_inliers >= 25) & (F.sift_grid_cells >= 4)).mean()),
                                                   fewer_than_20_inliers=float((F.sift_inliers < 20).mean()), fewer_than_4_cells_given_20_inliers=float(((F.sift_inliers >= 20) & (F.sift_grid_cells < 4)).mean()), pass_rate_inside_ranges=float(F.screen_pass[F.inside].mean()), pass_rate_outside_ranges=float(F.screen_pass[~F.inside].mean()),
                                                   inside_ranges_and_pass=float((F.inside & F.screen_pass).mean()), pass_rate_corrected_for_proxy_misses=float(F.screen_pass.mean() / kp.mean()), pass_rate_of_pairs_that_are_in_the_released_selection=float(F.screen_pass[F.is_selected].mean()), n_in_released_selection=int(F.is_selected.sum())),
                                   median_overlap=dict(pass_=float(F.ovl[F.screen_pass].median()), fail=float(F.ovl[~F.screen_pass].median())))
    # ---- how the two sets differ
    q = lambda v: {f'p{p}': float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)} | {'min': float(np.min(v)), 'max': float(np.max(v)), 'mean': float(np.mean(v))}
    res['distributions'] = {k: dict(selected=q(T[k].abs() if k != 'dist' else T[k]), unfiltered=q(F[k].abs() if k != 'dist' else F[k])) for k in ('x', 'y', 'z', 'tilt', 'heading', 'dist')} | {'overlap': dict(selected=q(T.ovl), unfiltered=q(F.ovl))}
    res['range_check'] = dict(selected_label_range=S, fraction_unfiltered_inside_per_component=per, fraction_unfiltered_inside_all_five=float(inside.mean()), fraction_unfiltered_outside=float(1 - inside.mean()), unfiltered_also_selected=int(F.is_selected.sum()))
    bins = [0, 0.02, 0.2, 0.4, 0.6, 1.0001]; lab = ['< 0.02', '0.02-0.2', '0.2-0.4', '0.4-0.6', '>= 0.6']; ob = {}
    for nm, D in (('selected', T), ('unfiltered', F), ('unfiltered inside ranges', F[F.inside])):
        rows = []
        for lo, hi, l in zip(bins[:-1], bins[1:], lab):
            m = (D.ovl >= lo) & (D.ovl < hi)
            rows.append(dict(bin=l, n=int(m.sum()), share=float(m.mean()), **({} if not m.sum() else dict(sr_d_1m=float((D.loc[m, '_d_p_cam'] <= 1).mean()), sr_epe_1m_cam=float((D.loc[m, '_epe_p_cam'] < 1).mean()), epe_med_cam=float(D.loc[m, '_epe_p_cam'].median()),
                                                                                                         sr_epe_1m_stage1_cam=float((D.loc[m, '_epe_s1_cam'] < 1).mean()), zoom_inliers_median=float(D.loc[m, 'zm_inl'].median())))))
        ob[nm] = rows
    res['by_overlap'] = ob
    db = {}
    for nm, D in (('selected', T), ('unfiltered', F)):
        rows = []
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 60)):
            m = (D.dist >= lo) & (D.dist < hi)
            if m.sum(): rows.append(dict(bin=f'{lo}-{hi}', n=int(m.sum()), sr_d_1m=float((D.loc[m, '_d_p_cam'] <= 1).mean()), sr_epe_1m_cam=float((D.loc[m, '_epe_p_cam'] < 1).mean()), epe_med_cam=float(D.loc[m, '_epe_p_cam'].median())))
        db[nm] = rows
    res['by_baseline'] = db
    res['diagnostics'] = dict(unfiltered=dict(zoom_inliers_median=float(F.zm_inl.median()), zoom_inliers_lt100=float((F.zm_inl.fillna(0) < 100).mean()), imag_inliers_median=float(F.im_inl.median()), pass1_ok=float(F.sg_ok.mean())),
                              selected=dict(zoom_inliers_median=float(T.zm_inl.median()), zoom_inliers_lt100=float((T.zm_inl.fillna(0) < 100).mean())))
    sfx = '_partial' if partial else ''; F.drop(columns=[c for c in F.columns if c.startswith('_')]).to_pickle(f'{C_}/unfiltered_scored{sfx}.pkl'); json.dump(res, open(f'{OUT}/unfiltered{sfx}.json', 'w'), indent=1)
    f1 = lambda v: f'{100 * v:.1f}'; L = ['## Unfiltered pairs (uniform over all camera pairs of the test buildings) vs the selected test pairs\n', '| Set | pairs | method / frame | d MAE (m) | SR_d@1m | EPE mean | EPE median | SR_EPE@1m | e_theta MAE | e_psi MAE |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, blk in res.items():
        if not isinstance(blk, dict) or 'methods' not in blk: continue
        for m, s in blk['methods'].items():
            c = s['ci']; L.append(f"| {name} | {blk['n']} | {m.replace(' | ', ', ')} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} [{c['epe_med'][0]:.2f}, {c['epe_med'][1]:.2f}] | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {s['theta_mae']:.3f} | {s['psi_mae']:.3f} |")
    L += ['\n### Stage II (camera frame) by ground-footprint overlap\n', '| Set | overlap | pairs | share | SR_d@1m | SR_EPE@1m | EPE median (m) | Stage I SR_EPE@1m | zoom inliers (median) |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for nm, rows in ob.items():
        for r in rows: L.append(f"| {nm} | {r['bin']} | {r['n']} | {f1(r['share'])} | " + (f"{f1(r['sr_d_1m'])} | {f1(r['sr_epe_1m_cam'])} | {r['epe_med_cam']:.2f} | {f1(r['sr_epe_1m_stage1_cam'])} | {r['zoom_inliers_median']:.0f} |" if r['n'] else '- | - | - | - | - |'))
    L += ['\n### Stage II (camera frame) by baseline length\n', '| Set | baseline (m) | pairs | SR_d@1m | SR_EPE@1m | EPE median (m) |', '|---|---|---:|---:|---:|---:|']
    for nm, rows in db.items():
        for r in rows: L.append(f"| {nm} | {r['bin']} | {r['n']} | {f1(r['sr_d_1m'])} | {f1(r['sr_epe_1m_cam'])} | {r['epe_med_cam']:.2f} |")
    L.append(f"\nUnfiltered pairs inside the label ranges of the selected pairs on all five components: {f1(inside.mean())} % (per component: " + ', '.join(f'{k} {f1(v)}' for k, v in per.items()) + ').')
    if have_screen:
        sp = res['screen_proxy']; L.append(f"\nScreen proxy (re-implementation calibrated on stored statistics; not the original code): passes {f1(sp['control_selected_pairs']['pass_rate'])} % of {sp['control_selected_pairs']['n']} selected test pairs and {f1(sp['unfiltered']['pass_rate'])} % of the unfiltered pairs "
                                           f"({f1(sp['unfiltered']['fewer_than_20_inliers'])} % have fewer than 20 inliers); {f1(sp['unfiltered']['inside_ranges_and_pass'])} % of the unfiltered pairs are inside the label ranges and pass.")
    open(f'{OUT}/unfiltered{sfx}.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L)); print(res['diagnostics'])


if __name__ == '__main__':
    {'sample': sample, 'run': run, 'report': report}[sys.argv[1]]()
