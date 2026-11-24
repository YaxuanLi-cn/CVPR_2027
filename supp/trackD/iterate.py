#!/usr/bin/env python3
"""Track D, item 2: repeating the imagination-refined stage INTERNALLY, on a fixed image pair, without any motion.

Round 1 = the archived Stage II. Round k+1: the round-k fused estimate becomes the guide pose (same recipe as
stage2_imagination/build_guide2.py and the g_* block of online/engine.py), the target is re-imagined and re-matched
(same settings as the paper: <= 3x3 tiles, native resolution, 2048 keypoints per tile), poses are re-estimated and the
FITTED Stage II fusion (outputs/online_models/stage2.pkl, predict mode, no re-training) is applied again.
  protocol A: the fusion keeps the Stage I estimate as its "stage1" candidate; only the imagination candidates / features are new.
  protocol B: the round-k estimate replaces the Stage I estimate as the "stage1" candidate (true recursion; outside the
              distribution the fusion was fitted on, where "stage1" is always the coarse estimate).
Steps (each idempotent):  guide <round> <proto>   -> guide pickle for the matcher
                          geom  <round> <proto>   -> pose features of the new matches
                          fuse  <round> <proto>   -> fused estimate of the round
                          report
The GPU step in between is the unmodified stage2_imagination/imagine_match.py (see run_iterate.sh).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ.setdefault(_k, '1')   # nproc reports 128 cores but the cgroup quota is ~30: without this every forked worker spawns 127 BLAS/OpenMP threads
import sys, os, json, glob, numpy as np, pandas as pd
ROOT = '/root/rasr-pairuav'
for p in (f'{ROOT}/superglue_fusion', f'{ROOT}/stage2_imagination', f'{ROOT}/online', f'{ROOT}/reviewer_baselines', f'{ROOT}/supp'): sys.path.insert(0, p)
from fusion_lib import SG, add_targets, load_mast3r
S2 = f'{SG}/stage2'; OUT = f'{ROOT}/outputs/supp/trackD'; W = f'{OUT}/iter'; os.makedirs(W, exist_ok=True); MODELS = f'{ROOT}/outputs/online_models'
KEYS = ['heading', 'tilt', 'dist', 'z', 'lx', 'ly']


def ids():
    from rb_recon import subset
    return subset(1000, split='test').manifest_index.values


def base_frame():
    """Stage-1 side of the fusion input for the 1000 pairs, assembled exactly like online/train_stores.py::stage2."""
    f = f'{W}/base.pkl'
    if os.path.exists(f): return pd.read_pickle(f)
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); P = P[P.manifest_index.isin(ids())]; M = load_mast3r(); S0 = pd.read_pickle(f'{SG}/stage0.pkl').merge(pd.read_pickle(f'{SG}/stage0_mlp.pkl'), on='manifest_index')
    D = P.merge(S0, on='manifest_index').merge(M.drop(columns='split'), on='manifest_index').merge(pd.read_pickle(f'{SG}/geom_half.pkl'), on='manifest_index', how='left') \
         .merge(pd.read_pickle(f'{SG}/map_half.pkl'), on='manifest_index', how='left').merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl'), on='manifest_index', how='left')
    D.to_pickle(f); return D


def stage1_est():
    F = pd.read_pickle(f'{SG}/half_zoom_pred.pkl'); F = F[F.manifest_index.isin(ids())]
    return F[['manifest_index'] + ['p_' + k for k in KEYS]].rename(columns=lambda c: c.replace('p_', 's1_'))


def est_path(rnd, proto): return f'{W}/est_r{rnd}{proto}.pkl'
def geom_path(rnd, proto): return f'{W}/geom_r{rnd}{proto}.pkl'
def guide_path(rnd, proto): return f'{W}/guide_r{rnd}{proto}.pkl'
def match_dir(rnd, proto): return f'{W}/matches_r{rnd}{"" if rnd == 2 else proto}'      # the round-2 matches are shared by both protocols (same guide: the round-1 estimate)


def load_round(rnd, proto):
    """(estimate p_*, imagination geometry, guide) of a finished round. Round 1 = archived files."""
    if rnd == 1:
        E = pd.read_pickle(f'{S2}/s2_pred.pkl'); E = E[E.manifest_index.isin(ids())][['manifest_index'] + ['p_' + k for k in KEYS]]
        I = pd.read_pickle(f'{S2}/geom_imag.pkl'); I = I[I.manifest_index.isin(ids())]; G = pd.read_pickle(f'{S2}/guide.pkl'); G = G[G.manifest_index.isin(ids())]
        return E.reset_index(drop=True), I.reset_index(drop=True), G.reset_index(drop=True)
    return pd.read_pickle(est_path(rnd, proto))[['manifest_index'] + ['p_' + k for k in KEYS]], pd.read_pickle(geom_path(rnd, proto)), pd.read_pickle(guide_path(rnd, proto))


def guide(rnd, proto):
    """Guide pose for round `rnd` from the round rnd-1 estimate."""
    import pickle
    H0 = pickle.load(open(f'{MODELS}/stage2.pkl', 'rb'))['s2/H0']; E, I, G = load_round(rnd - 1, proto); P = pd.read_pickle(f'{SG}/pairs.pkl')[['manifest_index', 'split', 'tour', 'image_a', 'image_b']]
    D = P.merge(E, on='manifest_index').merge(I[['manifest_index', 'imap_ok', 'imap_ta', 'imap_tb', 'imap_hgta', 'im_inl']], on='manifest_index').merge(G[['manifest_index', 'g_ta', 'g_tb', 'g_H']].rename(columns=lambda c: c.replace('g_', 'o_')), on='manifest_index')
    fa = np.where(D.imap_ok == 1, D.imap_ta, D.o_ta); fb = np.where(D.imap_ok == 1, D.imap_tb, D.o_tb); r = (D.p_tilt.values - (fb - fa)) / 2
    D['g_ta'] = np.clip(fa - r, 0, 47); D['g_tb'] = np.clip(fb + r, 0, 47); D['g_dh'] = D.p_heading; H = D.p_dist * D.imap_hgta
    D['g_H'] = np.where(np.isfinite(H) & (D.imap_ok == 1) & (D.im_inl > 100), np.clip(H, 0.75 * H0, 1.3 * H0), D.o_H); D['zm_inl'] = D.im_inl
    D[['manifest_index', 'split', 'tour', 'image_a', 'image_b', 'g_dh', 'g_ta', 'g_tb', 'p_lx', 'p_ly', 'p_z', 'p_dist', 'g_H', 'zm_inl']].to_pickle(guide_path(rnd, proto)); print('guide', rnd, proto, len(D), 'H', D.g_H.describe().round(1).to_dict())


def geom(rnd, proto, procs=5):
    import cv2; from multiprocessing import Pool; import imagine_geom
    cv2.setNumThreads(1); N = pd.read_pickle(f'{SG}/stage0_mlp.pkl').set_index('manifest_index'); G1 = pd.read_pickle(f'{SG}/geom_half.pkl').set_index('manifest_index'); M1 = pd.read_pickle(f'{SG}/map_half.pkl').set_index('manifest_index')
    Z = pd.read_pickle(f'{SG}/geom_zoom_half.pkl').set_index('manifest_index'); U = (pd.read_pickle(f'{S2}/guide.pkl') if proto == 'A' else pd.read_pickle(guide_path(rnd, proto))).set_index('manifest_index'); jobs = []   # "s1" verification candidate: Stage I pose (A) or the pose that guided this round (B)
    for f in sorted(glob.glob(f'{match_dir(rnd, proto)}/*.npz')):
        i = int(os.path.basename(f)[:-4]); n = N.loc[i]; g = G1.loc[i]; m = M1.loc[i]; z = Z.loc[i]; u = U.loc[i]; fu = np.array([u.p_lx, u.p_ly, u.p_z]); fu = fu / (np.linalg.norm(fu) + 1e-9)
        cands = dict(sg=(g.sg_dh, g.sg_ta, g.sg_tb, (g.sg_vx, g.sg_vy, g.sg_vz)), map=(m.map_dh, m.map_ta, m.map_tb, (m.map_vx, m.map_vy, m.map_vz)),
                     zm=(z.zm_dh, z.zm_ta, z.zm_tb, (z.zm_vx, z.zm_vy, z.zm_vz)), zmap=(z.zmap_dh, z.zmap_ta, z.zmap_tb, (z.zmap_vx, z.zmap_vy, z.zmap_vz)), s1=(u.g_dh, u.g_ta, u.g_tb, tuple(fu)))
        jobs.append((f, dict(h=float(n.n_heading), ta=float(np.clip(n.n_tilt_a, 0, 47)), tb=float(np.clip(n.n_tilt_b, 0, 47)), u=np.array([n.n_ux, n.n_uy, n.n_uz], float)), cands))
    with Pool(procs) as p: rows = p.map(imagine_geom.work, jobs, chunksize=8)
    R = pd.DataFrame(rows); R.to_pickle(geom_path(rnd, proto)); print('geom', rnd, proto, len(R), 'im_ok', R.im_ok.mean(), 'imap_ok', R.imap_ok.mean(), 'median inliers', R.im_inl.median(), 'median residual px', R.im_disp_med.median())


def fuse(rnd, proto, I=None, tag=None):
    import fusion_core as fc
    st = fc.Store(f'{MODELS}/stage2.pkl'); D = base_frame(); I = pd.read_pickle(geom_path(rnd, proto)) if I is None else I
    s1 = stage1_est() if proto == 'A' else load_round(rnd - 1, proto)[0].rename(columns=lambda c: c.replace('p_', 's1_'))
    D2 = D.merge(s1, on='manifest_index').merge(I, on='manifest_index'); out = fc.stage2(D2, st, log=lambda *a, **k: None); out = out.merge(I[['manifest_index', 'imap_ta', 'imap_tb', 'imap_ok', 'im_inl']], on='manifest_index')
    out.to_pickle(tag or est_path(rnd, proto)); print('fused', rnd, proto, len(out)); return out


def report():
    from common import errors, summarize, boot_indices, ci, dreamnav, gt_cam
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); P = P[P.manifest_index.isin(ids())].sort_values('manifest_index').reset_index(drop=True); boots = boot_indices(P.tour.values, 1000, 0); gt = gt_cam(P); res = {}; E = {}
    def score(est, name, I=None):
        f = P.merge(est[['manifest_index'] + ['p_' + k for k in KEYS]], on='manifest_index').sort_values('manifest_index').reset_index(drop=True); assert len(f) == len(P)
        e = errors(dreamnav(f, 'p_', 'cam'), gt); s = summarize(e); E[name] = e
        s['ci'] = dict(d_mae=ci(lambda i: e['d'][i].mean(), boots), sr_d_1m=ci(lambda i: (e['d'][i] <= 1).mean(), boots), epe_mae=ci(lambda i: e['epe'][i].mean(), boots), epe_med=ci(lambda i: np.median(e['epe'][i]), boots), sr_epe_1m=ci(lambda i: (e['epe'][i] < 1).mean(), boots))
        s['raw'] = {k: float(np.mean(np.abs(f['p_' + k] - f[k]))) for k in ('lx', 'ly', 'z', 'dist')}
        if I is not None:
            g = P.merge(I, on='manifest_index'); ok = (g.imap_ok == 1).values; eh = np.abs((g.imap_dh - g.heading + 180) % 360 - 180)[ok]; et = np.abs(g.imap_dtilt - g.tilt)[ok]; cs = (g.imap_vx * g.ux + g.imap_vy * g.uy + g.imap_vz * g.uz).clip(-1, 1)[ok]; ed = np.rad2deg(np.arccos(cs))
            s['raw_map_pose'] = dict(valid=float(ok.mean()), heading_err_med=float(np.median(eh)), heading_err_mean=float(eh.mean()), tilt_err_med=float(np.median(et)), tilt_err_mean=float(et.mean()), dir_err_med=float(np.median(ed)), dir_err_mean=float(ed.mean()))
        if I is not None: s['imagination'] = dict(residual_px_median=float(I.im_disp_med.median()), residual_px_mean=float(I.im_disp_med.mean()), share_matches_lt8px_median=float(I.im_disp_lt8.median()), inliers_median=float(I.im_inl.median()), tiles_mean=float(I.im_ntiles.mean()), imap_ok=float(I.imap_ok.mean()))
        res[name] = s
    s1 = stage1_est().rename(columns=lambda c: c.replace('s1_', 'p_'))
    f = P.merge(s1, on='manifest_index').sort_values('manifest_index').reset_index(drop=True); e = errors(dreamnav(f, 'p_', 'cam', al=0.75), gt); E['Stage I'] = e; res['Stage I'] = summarize(e)
    E1, I1, _ = load_round(1, 'A'); score(E1, 'Round 1 (archived Stage II)', I1)
    if os.path.exists(f'{W}/est_r1_store.pkl'): score(pd.read_pickle(f'{W}/est_r1_store.pkl'), 'Round 1 (re-applied fitted fusion)', I1)
    for proto in 'AB':
        for rnd in (2, 3, 4):
            if os.path.exists(est_path(rnd, proto)): score(pd.read_pickle(est_path(rnd, proto)), f'Round {rnd}, protocol {proto}', pd.read_pickle(geom_path(rnd, proto)))
    ref = 'Round 1 (re-applied fitted fusion)' if 'Round 1 (re-applied fitted fusion)' in E else 'Round 1 (archived Stage II)'; pd_ = {}
    for k, e in E.items():
        if k.startswith('Round') and k != ref and 'Round 1' not in k:
            r = E[ref]; pd_[k] = {m: dict(delta=float(fn(np.arange(len(P)))), ci=ci(fn, boots)) for m, fn in (('sr_d_1m', lambda i: (e['d'][i] <= 1).mean() - (r['d'][i] <= 1).mean()), ('sr_epe_1m', lambda i: (e['epe'][i] < 1).mean() - (r['epe'][i] < 1).mean()),
                                                                                                             ('d_mae', lambda i: e['d'][i].mean() - r['d'][i].mean()), ('epe_mae', lambda i: e['epe'][i].mean() - r['epe'][i].mean()), ('epe_med', lambda i: np.median(e['epe'][i]) - np.median(r['epe'][i])),
                                                                                                             ('theta_mae', lambda i: e['theta'][i].mean() - r['theta'][i].mean()), ('psi_mae', lambda i: e['psi'][i].mean() - r['psi'][i].mean()))}
    res['_paired_vs_round1'] = dict(reference=ref, deltas=pd_); res['_n'] = int(len(P)); res['_tours'] = int(P.tour.nunique()); json.dump(res, open(f'{OUT}/iteration.json', 'w'), indent=1)
    f1 = lambda v: f'{100 * v:.1f}'; L = [f'## Internal iteration on fixed pairs ({len(P)} test pairs, {P.tour.nunique()} buildings; camera frame; 95 % CI = bootstrap over buildings)\n',
         '| Round | d MAE (m) | SR_d@1m | EPE mean (m) | EPE median (m) | SR_EPE@1m | e_theta MAE (deg) | e_psi MAE (deg) | imagined-vs-real residual, median (px) | inliers (median) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k, s in res.items():
        if k.startswith('_'): continue
        c = s.get('ci'); im = s.get('imagination', {})
        L.append(f"| {k} | {s['d_mae']:.3f}" + (f" [{c['d_mae'][0]:.3f}, {c['d_mae'][1]:.3f}]" if c else '') + f" | {f1(s['sr_d_1m'])}" + (f" [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}]" if c else '') + f" | {s['epe_mae']:.3f} | {s['epe_med']:.3f} | {f1(s['sr_epe_1m'])}" + (f" [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}]" if c else '')
                 + f" | {s['theta_mae']:.4f} | {s['psi_mae']:.4f} | {im.get('residual_px_median', float('nan')):.2f} | {im.get('inliers_median', float('nan')):.0f} |")
    L += ['\nPose re-estimated from the imagined matches alone (MAP solver, before fusion):\n', '| Round | valid | heading error median / mean (deg) | tilt-change error median / mean (deg) | translation-direction error median / mean (deg) |', '|---|---:|---:|---:|---:|']
    for k, s in res.items():
        if not k.startswith('_') and 'raw_map_pose' in s: r = s['raw_map_pose']; L.append(f"| {k} | {f1(r['valid'])} | {r['heading_err_med']:.4f} / {r['heading_err_mean']:.3f} | {r['tilt_err_med']:.4f} / {r['tilt_err_mean']:.3f} | {r['dir_err_med']:.3f} / {r['dir_err_mean']:.2f} |")
    L.append(f'\nPaired differences with respect to "{ref}" (negative error / positive SR = better):')
    for k, v in pd_.items(): L.append(f'- {k}: ' + '; '.join(f"{m} {x['delta']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}]" for m, x in v.items()))
    open(f'{OUT}/iteration.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'report': report()
    elif cmd == 'check':     # the fitted fusion applied to the archived round-1 features must reproduce the archived Stage II
        _, I1, _ = load_round(1, 'A'); fuse(1, 'A', I=I1, tag=f'{W}/est_r1_store.pkl')
    else: {'guide': guide, 'geom': geom, 'fuse': fuse}[cmd](int(sys.argv[2]), sys.argv[3])
