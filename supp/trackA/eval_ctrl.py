#!/usr/bin/env python3
"""Score the control variants of the refinement stage on the 3,000 test pairs of the 9k ablation subset
(all fusions fitted on the same 6,000 training pairs). Camera-frame metrics from supp/common.py.
Output: outputs/supp/trackA/controls.json, controls.md
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from common import *
from fusion_lib import add_targets
O = f'{OUT}/trackA'; AL = 0.5
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); ids = np.load(f'{O}/ids_small_test.npy'); T = P[P.manifest_index.isin(ids)].sort_values('manifest_index').reset_index(drop=True); gc = gt_cam(T); n = len(T)
boots = boot_indices(T.tour.values, 1000, 0); print('test pairs', n, 'tours', T.tour.nunique())

def vec(lx, ly, z, dist, tilt, heading, al=AL):
    lx, ly, z, dist = (np.asarray(v, float) for v in (lx, ly, z, dist)); nn = np.sqrt(lx**2 + ly**2 + z**2).clip(1e-6); sc = (dist / nn) ** al
    return dict(x=lx * sc, y=ly * sc, z=z * sc, tilt=np.asarray(tilt, float), heading=np.asarray(heading, float))
def load(tag):
    f = f'{O}/{tag}_pred.pkl'
    if not os.path.exists(f): return None
    return pd.read_pickle(f).set_index('manifest_index').reindex(T.manifest_index)

ROWS = []   # (label, description, prediction dict)
F = load('full'); assert F is not None
ROWS.append(('Coarse stage (Stage I)', 'input of every variant below', vec(F.s1_lx, F.s1_ly, F.s1_z, F.s1_dist, F.s1_tilt, F.s1_heading, 0.75)))
def add(tag, label, desc, pre='p_'):
    d = load(tag)
    if d is None: print('missing', tag); return
    ROWS.append((label, desc, vec(d[pre + 'lx'], d[pre + 'ly'], d[pre + 'z'], d[pre + 'dist'], d[pre + 'tilt'], d[pre + 'heading'])))
add('fusion_only', 'Second fusion stage only', 'no re-matching: same learner, coarse-stage candidates and features only')
add('rot_tiles', 'Tile-wise re-matching, rotation-only alignment', 'compute-matched second matching pass (<= 3x3 native tiles) without the plane-induced warp; same fusion')
add('plane_1tile', 'Plane-induced warp, single tile', 'warp without extra keypoint budget (1 tile of 1280x720); same fusion')
add('plane_512', 'Plane-induced warp at 512x288', 'warp rendered at the resolution of a latent-diffusion image, single tile; same fusion')
# full re-matching, different ways of using it
def nofuse(F, pre):   # pose re-estimated from the imagined matches used directly, coarse distance; coarse estimate where the re-estimation failed
    ok = (F[('imap' if pre == 'imap_' else 'im') + '_ok'].values == 1) & np.isfinite(F[pre + 'vx'].values); w = lambda a, b: np.where(ok, a, b)
    return vec(w(F.s1_dist * F[pre + 'vx'], F.s1_lx), w(F.s1_dist * F[pre + 'vy'], F.s1_ly), w(F.s1_dist * F[pre + 'vz'], F.s1_z), F.s1_dist, w(F[pre + 'dtilt'], F.s1_tilt), w(F[pre + 'dh'], F.s1_heading)), float(ok.mean())
p, okf = nofuse(F, 'imap_'); ROWS.append(('Warp + tiles, re-estimated pose used directly', f'no fusion: MAP pose from the imagined matches replaces the coarse pose (valid for {100 * okf:.1f}% of pairs), coarse distance', p))
add('uniform_nores', 'Warp + tiles, uniform candidate weights', 'equal weights over valid candidates, no residual regressor')
add('learned_nores', 'Warp + tiles, predicted-error weights', 'reliability-weighted blend, no residual regressor')
add('no_ver', 'Warp + tiles, fusion without verification scores', 'full fusion minus the hypothesis-verification features')
add('no_scale', 'Warp + tiles, fusion with the coarse scale', 'distance kept from the coarse stage; no re-triangulated scale cue or feature')
ROWS.append(('Full refinement (Stage II)', 'plane-induced warp, <= 3x3 native tiles, verification, reliability-weighted fusion', vec(F.p_lx, F.p_ly, F.p_z, F.p_dist, F.p_tilt, F.p_heading)))
A9 = pd.read_pickle(f'{S2}/s2_9k_pred.pkl').set_index('manifest_index').reindex(T.manifest_index)   # archived fits with more training pairs: sensitivity to the size of the fusion training set
ROWS.append(('Stage II, fusion fitted on 6,000 pairs', 'archived fit', vec(A9.p_lx, A9.p_ly, A9.p_z, A9.p_dist, A9.p_tilt, A9.p_heading)))
A12 = pd.read_pickle(f'{S2}/s2_pred.pkl').set_index('manifest_index').reindex(T.manifest_index)
ROWS.append(('Stage II, fusion fitted on 12,000 pairs (main paper)', 'archived fit', vec(A12.p_lx, A12.p_ly, A12.p_z, A12.p_dist, A12.p_tilt, A12.p_heading)))

E = {lab: errors(p, gc) for lab, _, p in ROWS}; ref = E['Full refinement (Stage II)']; base = E['Coarse stage (Stage I)']; R = []
for lab, desc, p in ROWS:
    e = E[lab]; s = summarize(e); s['label'] = lab; s['desc'] = desc
    s['ci'] = {k: ci(fn, boots) for k, fn in (('d_mae', lambda b: e['d'][b].mean()), ('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('epe_mae', lambda b: e['epe'][b].mean()))}
    s['vs_coarse'] = {k: dict(delta=float(fn(np.arange(n))), ci=ci(fn, boots)) for k, fn in (('sr_d_1m', lambda b: (e['d'][b] <= 1).mean() - (base['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean() - (base['epe'][b] < 1).mean()), ('d_mae', lambda b: e['d'][b].mean() - base['d'][b].mean()))}
    s['vs_full'] = {k: dict(delta=float(fn(np.arange(n))), ci=ci(fn, boots)) for k, fn in (('sr_d_1m', lambda b: (e['d'][b] <= 1).mean() - (ref['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean() - (ref['epe'][b] < 1).mean()), ('d_mae', lambda b: e['d'][b].mean() - ref['d'][b].mean()))}
    R.append(s)
full_gain = {k: next(r for r in R if r['label'] == 'Full refinement (Stage II)')['vs_coarse'][k]['delta'] for k in ('sr_d_1m', 'sr_epe_1m', 'd_mae')}
for s in R: s['share_of_full_gain'] = {k: (s['vs_coarse'][k]['delta'] / full_gain[k] if abs(full_gain[k]) > 1e-9 else float('nan')) for k in full_gain}

# ---- raw two-view geometry of each matching pass (no learning): what the re-matching itself delivers
def raw(G, pre_e, pre_m, okc_e, okc_m, extra=None, sub=None):
    global T
    T_all = T
    if sub is not None: T = T_all[T_all.manifest_index.isin(sub)].reset_index(drop=True)
    try: return _raw(G, pre_e, pre_m, okc_e, okc_m, extra)
    finally: T = T_all
def _raw(G, pre_e, pre_m, okc_e, okc_m, extra=None):
    G = G.set_index('manifest_index').reindex(T.manifest_index); out = {'n': int(len(T))}
    for nm, pre, okc in (('E', pre_e, okc_e), ('MAP', pre_m, okc_m)):
        ok = (G[okc].fillna(0).values == 1); v = G[[pre + 'vx', pre + 'vy', pre + 'vz']].values.astype(float); u = np.c_[T.ux.values, T.uy.values, T.uz.values]
        with np.errstate(all='ignore'): ang = np.rad2deg(np.arccos(np.clip((v * u).sum(1) / np.linalg.norm(v, axis=1).clip(1e-9), -1, 1)))
        eh = np.abs(wrap(G[pre + 'dh'].values - T.heading.values)); et = np.abs(G[pre + 'dtilt'].values - T.tilt.values)
        # failed pairs count as the worst case of each quantity so that rows with different failure rates stay comparable
        ang = np.where(ok & np.isfinite(ang), ang, 90.0); eh = np.where(ok & np.isfinite(eh), eh, 90.0); et = np.where(ok & np.isfinite(et), et, 20.0)
        out[nm] = dict(solved=float(ok.mean()), dir_med=float(np.median(ang)), dir_mean=float(ang.mean()), dir_lt2=float((ang < 2).mean()), dir_lt5=float((ang < 5).mean()), theta_med=float(np.median(eh)), theta_mean=float(eh.mean()), psi_med=float(np.median(et)), psi_mean=float(et.mean()))
    if extra: out.update(extra(G))
    return out
im_extra = lambda G: dict(inliers_med=float(G.im_inl.median()), inl_lt100=float((G.im_inl.fillna(0) < 100).mean()), resid_px_med=float(G.im_disp_med.median()), tiles_mean=float(G.im_ntiles.mean()))
RAW = {}
Gz = pd.read_pickle(f'{SG}/geom_zoom_half.pkl'); RAW['Coarse stage, zoom pass'] = raw(Gz, 'zm_', 'zmap_', 'zm_ok', 'zmap_ok', lambda G: dict(inliers_med=float(G.zm_inl.median()), inl_lt100=float((G.zm_inl.fillna(0) < 100).mean())))
for tag, lab in (('rot_tiles', 'Rotation-only alignment, <= 3x3 tiles'), ('plane_1tile', 'Plane-induced warp, 1 tile'), ('plane_512', 'Plane-induced warp, 512x288'), (None, 'Plane-induced warp, <= 3x3 tiles (ours)')):
    f = f'{S2}/geom_imag_9k.pkl' if tag is None else f'{O}/geom_{tag}.pkl'
    if os.path.exists(f): RAW[lab] = raw(pd.read_pickle(f), 'im_', 'imap_', 'im_ok', 'imap_ok', im_extra)
if os.path.exists(f'{O}/geom_oracle.pkl'):      # oracle guide: run on a subset of the test pairs; our warp is re-scored on the same subset
    Go = pd.read_pickle(f'{O}/geom_oracle.pkl'); sub = set(Go.manifest_index.tolist()) & set(T.manifest_index.tolist())
    RAW['Plane-induced warp, <= 3x3 tiles (ours), oracle subset'] = raw(pd.read_pickle(f'{S2}/geom_imag_9k.pkl'), 'im_', 'imap_', 'im_ok', 'imap_ok', im_extra, sub=sub)
    RAW['Plane-induced warp under the true pose, <= 3x3 tiles'] = raw(Go, 'im_', 'imap_', 'im_ok', 'imap_ok', im_extra, sub=sub)
    if os.path.exists(f'{O}/geom_rot90_tiles.pkl'):   # no pose-guided alignment: only run on (a subset of) the oracle pairs
        G90 = pd.read_pickle(f'{O}/geom_rot90_tiles.pkl'); sub90 = set(G90.manifest_index.tolist()) & sub
        RAW['No pose-guided alignment, <= 3x3 tiles, oracle subset'] = raw(G90, 'im_', 'imap_', 'im_ok', 'imap_ok', im_extra, sub=sub90)
        RAW['Rotation-only alignment, <= 3x3 tiles, oracle subset'] = raw(pd.read_pickle(f'{O}/geom_rot_tiles.pkl'), 'im_', 'imap_', 'im_ok', 'imap_ok', im_extra, sub=sub90)
    Gz_sub = raw(Gz, 'zm_', 'zmap_', 'zm_ok', 'zmap_ok', lambda G: dict(inliers_med=float(G.zm_inl.median()), inl_lt100=float((G.zm_inl.fillna(0) < 100).mean())), sub=sub); RAW['Coarse stage, zoom pass, oracle subset'] = Gz_sub
json.dump(dict(n=n, tours=int(T.tour.nunique()), train_rows=int(len(np.load(f'{O}/ids_ctrl_train.npy'))), rows=R, raw_geometry=RAW), open(f'{O}/controls.json', 'w'), indent=1)
f1 = lambda v: f'{100 * v:.1f}'
L = [f'# Controls for the refinement stage ({n} test pairs, {T.tour.nunique()} buildings; every fusion fitted on the same 3,000 training pairs)\n',
     '| Variant | d MAE | SR_d@1m | EPE mean | EPE med. | SR_EPE@1m | e_theta | e_psi | dir. med. | dSR_d vs coarse | dSR_EPE vs coarse | share of full gain (SR_d / SR_EPE) |', '|---|' + '---:|' * 11]
for s in R:
    c = s['ci']; L.append(f"| {s['label']} | {s['d_mae']:.3f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.3f} | {s['epe_med']:.3f} | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {s['theta_mae']:.4f} | {s['psi_mae']:.4f} | {s['dir_med']:.2f} | "
                     f"{100 * s['vs_coarse']['sr_d_1m']['delta']:+.1f} [{100 * s['vs_coarse']['sr_d_1m']['ci'][0]:+.1f}, {100 * s['vs_coarse']['sr_d_1m']['ci'][1]:+.1f}] | {100 * s['vs_coarse']['sr_epe_1m']['delta']:+.1f} [{100 * s['vs_coarse']['sr_epe_1m']['ci'][0]:+.1f}, {100 * s['vs_coarse']['sr_epe_1m']['ci'][1]:+.1f}] | {f1(s['share_of_full_gain']['sr_d_1m'])}% / {f1(s['share_of_full_gain']['sr_epe_1m'])}% |")
L += ['\n## Raw two-view geometry of each matching pass (no learning; failed pairs counted as 90 deg / 20 deg)\n', '| Matching pass | pose solved | inliers med. | <100 inliers | residual px med. | tiles | dir. med. (E / MAP) | dir. mean (E / MAP) | dir. < 2 deg (MAP) | e_theta mean (E / MAP) | e_psi mean (E / MAP) |', '|---|' + '---:|' * 10]
for k, r in RAW.items():
    L.append(f"| {k} | {f1(r['E']['solved'])} / {f1(r['MAP']['solved'])} | {r.get('inliers_med', float('nan')):.0f} | {f1(r.get('inl_lt100', float('nan')))} | {r.get('resid_px_med', float('nan')):.1f} | {r.get('tiles_mean', float('nan')):.2f} | {r['E']['dir_med']:.2f} / {r['MAP']['dir_med']:.2f} | {r['E']['dir_mean']:.2f} / {r['MAP']['dir_mean']:.2f} | {f1(r['MAP']['dir_lt2'])} | {r['E']['theta_mean']:.3f} / {r['MAP']['theta_mean']:.3f} | {r['E']['psi_mean']:.3f} / {r['MAP']['psi_mean']:.3f} |")
open(f'{O}/controls.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
