#!/usr/bin/env python3
"""Component inventory behind Table 3, recomputed on the 6,364 test pairs with explicit definitions (supp/common.py).

Rows go from the learned priors alone, through pure two-view geometry (no learning) at each matching pass, to the two fused stages.
Every row is scored in the camera frame (no absolute heading involved) and, where a world-frame output exists, in ENU.
Pure-geometry rows take the translation direction and the two angles from the pose solver and the length either from the
flight-height prior (baseline = H0 / height of the camera above the triangulated scene in baseline units; H0 fitted on training pairs)
or from the learned distance head of the prior MLP. Pairs whose pose cannot be solved predict zero motion.
Output: outputs/supp/trackA/ablation_full.json, ablation_full.md
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from common import *
from fusion_lib import add_targets
O = f'{OUT}/trackA'; T = load_test(); n = len(T); gc = gt_cam(T); gw = gt_world(T); boots = boot_indices(T.tour.values, 1000, 0); ids = T.manifest_index.values
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')).set_index('manifest_index'); N0 = pd.read_pickle(f'{SG}/stage0_mlp.pkl').set_index('manifest_index')
TR = pd.read_pickle(f'{SG}/half_zoom_pred.pkl'); TR = TR[TR.split == 'train'].manifest_index.values       # training pairs of the result set (24,909)

def geom_rows(G, pre, okc, hgt_cols, name):
    """Pure geometry from one matching pass. Returns {scale variant: prediction dict}, failure rate, H0."""
    G = G.set_index('manifest_index'); h = sum(G[c] for c in hgt_cols) / len(hgt_cols); ok = (G[okc].fillna(0) == 1) & np.isfinite(G[pre + 'vx']) & np.isfinite(h) & (h > 1e-3)
    tr = ok.reindex(TR).fillna(False).values.astype(bool); H0 = float(np.median((P.dist.reindex(TR).values * h.reindex(TR).values)[tr]))
    g = G.reindex(ids); okt = ok.reindex(ids).fillna(False).values.astype(bool); v = g[[pre + 'vx', pre + 'vy', pre + 'vz']].values.astype(float); out = {}
    for tag, dist in (('flight-height prior', H0 / h.reindex(ids).values), ('learned distance head', N0.n_dist.reindex(ids).values)):
        d = np.where(okt, dist, 0.0); d = np.where(np.isfinite(d) & (d < 500), d, 0.0); t = np.where(okt[:, None], v, 0.0) * d[:, None]
        out[tag] = dict(x=t[:, 0], y=t[:, 1], z=t[:, 2], tilt=np.where(okt, g[pre + 'dtilt'].values, 0.0), heading=np.where(okt, g[pre + 'dh'].values, 0.0))
    return out, float(1 - okt.mean()), H0

def fused(F, pre, al):
    F = F.set_index('manifest_index').reindex(ids); lx, ly, z = F[pre + 'lx'].values, F[pre + 'ly'].values, F[pre + 'z'].values; nn = np.sqrt(lx**2 + ly**2 + z**2).clip(1e-6); sc = (F[pre + 'dist'].values / nn) ** al
    return dict(x=lx * sc, y=ly * sc, z=z * sc, tilt=F[pre + 'tilt'].values, heading=F[pre + 'heading'].values)

ROWS = []   # (id, label, components, camera-frame prediction, world-frame prediction or None, failure rate)
z0 = np.zeros(n)
ROWS.append(('P', 'Learned priors only (MLP on pooled frozen-MASt3R statistics)', 'priors', dict(x=T.n_lx.values, y=T.n_ly.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values),
             dict(x=T.n_x.values, y=T.n_y.values, z=T.n_z.values, tilt=T.n_tilt.values, heading=T.n_heading.values), 0.0))
H = {}
f = f'{O}/geom_sg_plain_test.pkl'
if os.path.exists(f):
    G = pd.read_pickle(f); Gt = pd.read_pickle(f'{SG}/geom_half_k2.pkl')    # H0 for the plain matcher: the constant of the rotation-aware pass (same height statistic)
    _, _, H0 = geom_rows(Gt, 'sg_', 'sg_ok', ['sg_hgta_p75', 'sg_hgtb_p75'], 'rot'); G = G.set_index('manifest_index').reindex(ids); h = (G.sg_hgta_p75 + G.sg_hgtb_p75) / 2
    ok = ((G.sg_ok.fillna(0) == 1) & np.isfinite(G.sg_vx) & np.isfinite(h) & (h > 1e-3)).values; v = G[['sg_vx', 'sg_vy', 'sg_vz']].values.astype(float)
    for tag, dist in (('flight-height prior', H0 / h.values), ('learned distance head', N0.n_dist.reindex(ids).values)):
        d = np.where(ok, dist, 0.0); d = np.where(np.isfinite(d) & (d < 500), d, 0.0); t = np.where(ok[:, None], v, 0.0) * d[:, None]
        ROWS.append(('G0', f'Frozen SuperGlue, single pass + E; {tag}', 'matcher (1 pass) + E', dict(x=t[:, 0], y=t[:, 1], z=t[:, 2], tilt=np.where(ok, G.sg_dtilt.values, 0.0), heading=np.where(ok, G.sg_dh.values, 0.0)), None, float(1 - ok.mean())))
for gid, lab, comp, fpk, pre, okc, hc in (('G1', 'Rotation-aware matching + E', 'rot. search + fine pass + E', 'geom_half_k2.pkl', 'sg_', 'sg_ok', ['sg_hgta_p75', 'sg_hgtb_p75']),
                                         ('G2', 'Rotation-aware matching + MAP (Eq. 1)', '+ angle priors in the pose refinement', 'map_half_k2.pkl', 'map_', 'map_ok', ['map_hgta', 'map_hgtb']),
                                         ('G3', 'Zoom re-matching + E', '+ co-visible crop', 'geom_zoom_half.pkl', 'zm_', 'zm_ok', ['zm_hgta_p75', 'zm_hgtb_p75']),
                                         ('G4', 'Zoom re-matching + MAP', '+ co-visible crop, priors', 'geom_zoom_half.pkl', 'zmap_', 'zmap_ok', ['zmap_hgta', 'zmap_hgtb']),
                                         ('G5', 'Imagination re-matching + E', '+ plane-induced warp, tiles', f'stage2/geom_imag.pkl', 'im_', 'im_ok', ['im_hgta_p75', 'im_hgtb_p75']),
                                         ('G6', 'Imagination re-matching + MAP', '+ plane-induced warp, tiles, priors', f'stage2/geom_imag.pkl', 'imap_', 'imap_ok', ['imap_hgta', 'imap_hgtb'])):
    G = pd.read_pickle(f'{SG}/{fpk}'); out, fail, H0 = geom_rows(G, pre, okc, hc, lab); H[lab] = H0
    for tag, p in out.items(): ROWS.append((gid, f'{lab}; {tag}', comp, p, None, fail))
K = pd.read_pickle(f'{SG}/half_k2_pred.pkl'); Z = pd.read_pickle(f'{SG}/half_zoom_pred.pkl'); S = pd.read_pickle(f'{S2}/s2_pred.pkl')
Tk = T.copy(); kk = K.set_index('manifest_index').reindex(ids)
for c in ('lx', 'ly', 'z', 'dist', 'tilt', 'heading'): Tk['k_' + c] = kk['p_' + c].values
ROWS.append(('F1', 'Fused, rotation-aware pass only (no zoom)', 'priors + rot.-aware matching + E/MAP + scale cues + fusion', fused(K, 'p_', 0.75), dreamnav(Tk, 'k_', fuse_heading(Tk, V2, Tk.k_heading.values)[0], 0.75), 0.0))
ROWS.append(('F2', 'Coarse stage (Stage I): + zoom re-matching', '+ zoom candidates', dreamnav(T, 's1_', 'cam'), dreamnav(T, 's1_', 'net'), 0.0))
ROWS.append(('F3', 'Refined stage (Stage II): + geometric imagination', '+ imagination candidates, verification, second fusion', dreamnav(T, 'p_', 'cam'), dreamnav(T, 'p_', 'net'), 0.0))
# the row of Table 3 that has no absolute heading: world-frame x/y cannot be formed from the geometry and fall back to the direct regression of the prior MLP
t3 = dict(x=T.n_x.values, y=T.n_y.values, z=fused(K, 'p_', 0.0)['z'], tilt=kk.p_tilt.values, heading=kk.p_heading.values)

R = []
for gid, lab, comp, pc, pw, fail in ROWS:
    e = errors(pc, gc); s = summarize(e); s.update(id=gid, label=lab, components=comp, fail=fail)
    s['ci'] = {k: ci(fn, boots) for k, fn in (('sr_d_1m', lambda b: (e['d'][b] <= 1).mean()), ('sr_epe_1m', lambda b: (e['epe'][b] < 1).mean()), ('d_mae', lambda b: e['d'][b].mean()))}
    if pw is not None:
        ew = errors(pw, gw); s['enu'] = dict(x_mae=float(ew['x'].mean()), y_mae=float(ew['y'].mean()), d_mae=float(ew['d'].mean()), sr_d_1m=float((ew['d'] <= 1).mean()), epe_mae=float(ew['epe'].mean()), epe_med=float(np.median(ew['epe'])), sr_epe_1m=float((ew['epe'] < 1).mean()))
    R.append(s)
e3 = errors(t3, gw); extra = {'table3_row2_as_printed (z, angles fused; x/y from the world-frame regression)': dict(d_mae=float(e3['d'].mean()), sr_d_1m=float((e3['d'] <= 1).mean()), x_mae=float(e3['x'].mean()), y_mae=float(e3['y'].mean()), z_mae=float(e3['z'].mean()), theta=float(e3['theta'].mean()), psi=float(e3['psi'].mean()), sr_epe_1m=float((e3['epe'] < 1).mean()))}
# which statistic of the no-zoom + heading configuration could correspond to "2.25 m / 35.1 %" printed in Table 3?
ew = errors(dreamnav(Tk, 'k_', fuse_heading(Tk, V2, Tk.k_heading.values)[0], 0.75), gw); ew0 = errors(dreamnav(Tk, 'k_', fuse_heading(Tk, V2, Tk.k_heading.values)[0], 0.0), gw)
extra['no_zoom_plus_heading_statistics'] = {k: float(v) for k, v in dict(d_mae=ew['d'].mean(), d_mae_norescale=ew0['d'].mean(), sr_d_1m=(ew['d'] <= 1).mean(), sr_epe_1m=(ew['epe'] < 1).mean(), sr_epe_1m_norescale=(ew0['epe'] < 1).mean(), epe_mae=ew['epe'].mean(), epe_med=np.median(ew['epe']),
                                           epe_h_mae=ew['epe_h'].mean(), epe_h_med=np.median(ew['epe_h']), x_mae=ew['x'].mean(), y_mae=ew['y'].mean(), z_mae=ew['z'].mean(), xy_mean=(ew['x'].mean() + ew['y'].mean()) / 2, d_p75=np.percentile(ew['d'], 75), d_p90=np.percentile(ew['d'], 90)).items()}
json.dump(dict(n=n, rows=R, H0=H, extra=extra), open(f'{O}/ablation_full.json', 'w'), indent=1)
f1 = lambda v: f'{100 * v:.1f}'
L = [f'# Component inventory ({n} test pairs)\n', '| id | Configuration | fail % | d MAE | SR_d@1m | EPE cam mean / med. | SR_EPE cam | e_theta MAE / med. | e_psi MAE / med. | dir. med. | ENU: x / y MAE | ENU EPE med. | ENU SR_EPE |', '|---|---|' + '---:|' * 11]
for s in R:
    en = s.get('enu'); L.append(f"| {s['id']} | {s['label']} | {f1(s['fail'])} | {s['d_mae']:.2f} | {f1(s['sr_d_1m'])} [{f1(s['ci']['sr_d_1m'][0])}, {f1(s['ci']['sr_d_1m'][1])}] | {s['epe_mae']:.2f} / {s['epe_med']:.2f} | {f1(s['sr_epe_1m'])} | {s['theta_mae']:.3f} / {s['theta_med']:.3f} | {s['psi_mae']:.3f} / {s['psi_med']:.3f} | {s['dir_med']:.2f} | "
                              + (f"{en['x_mae']:.2f} / {en['y_mae']:.2f} | {en['epe_med']:.2f} | {f1(en['sr_epe_1m'])} |" if en else '- | - | - |'))
L.append('\nFlight-height constants H0 (m) fitted on training pairs: ' + '; '.join(f'{k}: {v:.1f}' for k, v in H.items())); L.append('\n' + json.dumps(extra, indent=1))
open(f'{O}/ablation_full.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
