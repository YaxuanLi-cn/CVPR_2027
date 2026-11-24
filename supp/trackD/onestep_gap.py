#!/usr/bin/env python3
"""Track D, item 3: why Table 2 (SR 83.3 %) and the one-step column of Table 5 (38.8 %) differ.

collect : for the 160 closed-loop episodes (start camera -> held-out goal image) run the estimator once on
          (a) the real start photograph and (b) the simulator render at the start pose; full prediction rows are saved.
report  : the chain  Table-2 definition on the test pairs -> same definition on the 160 start/goal pairs (real photo)
          -> endpoint definition -> rendered observation -> executed command without / with actuation noise.
Usage: onestep_gap.py collect | report
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ.setdefault(_k, '1')   # nproc reports 128 cores but the cgroup quota is ~30: without this every forked worker spawns 127 BLAS/OpenMP threads
import sys, os, json, shutil, time, numpy as np, pandas as pd, cv2
cv2.setNumThreads(1)
sys.path.insert(0, '/root/rasr-pairuav/online'); sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/supp/trackD')
OUT = '/root/rasr-pairuav/outputs/supp/trackD'; SIM = '/root/rasr-pairuav/outputs/sim'; TMP = f'/dev/shm/onestep_trackD_{os.getpid()}'
NOISE = dict(pos_mult=0.03, pos_add=0.05, heading=0.2, tilt=0.1); SWITCH = 5.0


def collect():
    from sim_scene import Scene
    from engine import Engine
    from sim_nav_seeds import episodes
    import torch
    tours = open(f'{SIM}/tours.txt').read().strip().split(','); os.makedirs(TMP, exist_ok=True); eng = Engine(workers=4, loader_workers=3, verbose=False); rows = []; neps = 0; t0 = time.time()
    for t in tours:
        sc = Scene(t); ep = episodes(sc.cams, t); n = len(ep); files = []
        for j, e in enumerate(ep): f = f'{TMP}/{t}_{j}.jpg'; cv2.imwrite(f, sc.render(*e['s']), [cv2.IMWRITE_JPEG_QUALITY, 95]); files.append(f)
        R = eng.predict([(f, e['goal_image']) for f, e in zip(files, ep)] + [(e['start_image'], e['goal_image']) for e in ep], plane=True, abs_heading=True)
        for i in range(2 * n):
            j = i % n; e = ep[j]; d = e['g'] - e['s']; h = np.deg2rad(e['s'][3]); r = R.iloc[i].drop(['image_a', 'image_b', 'manifest_index']).to_dict()
            r.update(tour=t, ep=neps + j, goal=e['goal'], start=e['start'], obs='rendered' if i < n else 'real', x=d[0], y=d[1], z=d[2], heading=float((d[3] + 180) % 360 - 180), tilt=d[4], heading_a=e['s'][3], tilt_a=e['s'][4],
                     lx=np.cos(h) * d[0] - np.sin(h) * d[1], ly=np.sin(h) * d[0] + np.cos(h) * d[1], dist=float(np.linalg.norm(d[:3]))); rows.append(r)
        for f in files: os.remove(f)
        neps += n; del sc; torch.cuda.empty_cache(); print(f'[{t}] {n} episodes done ({time.time()-t0:.0f}s) | GPU peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB allocated, {torch.cuda.max_memory_reserved()/2**30:.2f} GB reserved', flush=True)
        pd.DataFrame(rows).to_pickle(f'{OUT}/cache/onestep_pred.pkl')
    eng.close(); shutil.rmtree(TMP, ignore_errors=True); print('saved', len(rows), 'rows')


def report():
    from common import errors, summarize, boot_indices, ci, load_test, dreamnav, gt_cam, wrap
    D = pd.read_pickle(f'{OUT}/cache/onestep_pred.pkl'); T = load_test(); res = {}; L = []
    def frame(d):                                               # columns that common.dreamnav expects
        f = d.rename(columns={c: 'p_' + c[3:] for c in d.columns if c.startswith('s2_') and c[3:] in ('lx', 'ly', 'z', 'dist', 'tilt', 'heading')}).reset_index(drop=True); return f
    def score(d, pre, al):
        f = frame(d); e = errors(dreamnav(f, pre, 'cam', al=al), gt_cam(f)); s = summarize(e); b = boot_indices(f.ep.values, 1000, 0)
        s['ci'] = dict(sr_d_1m=ci(lambda i: (e['d'][i] <= 1).mean(), b), sr_epe_1m=ci(lambda i: (e['epe'][i] < 1).mean(), b), epe_med=ci(lambda i: np.median(e['epe'][i]), b), d_mae=ci(lambda i: e['d'][i].mean(), b)); return s, e
    # ---- reference: the test pairs of Table 2, and the test pairs of the 8 simulator sites
    sites = sorted(D.tour.unique()); ref = {}
    for nm, pre in (('Stage I', 's1_'), ('Stage II', 'p_')):
        e = errors(dreamnav(T, pre, 'cam'), gt_cam(T)); m = T.tour.isin(sites).values; b = boot_indices(T.tour.values, 1000, 0)
        ref[nm] = dict(all_test=summarize(e), sim_sites=summarize({k: v[m] for k, v in e.items()}), n_sim_site_pairs=int(m.sum()), ci_all=dict(sr_d_1m=ci(lambda i: (e['d'][i] <= 1).mean(), b), sr_epe_1m=ci(lambda i: (e['epe'][i] < 1).mean(), b)))
    res['test_pairs'] = ref
    # ---- the 160 episodes: paper protocol (norm rescaled to the fused distance with the exponent of the paper) and the raw vector the simulator executes
    ep = {}
    for obs in ('real', 'rendered'):
        d = D[D.obs == obs].sort_values('ep')
        for nm, pre, al in (('Stage I', 's1_', 0.75), ('Stage II', 'p_', 0.5)):
            s_paper, _ = score(d, pre, al); s_raw, e_raw = score(d, pre, 0.0); ep[f'{nm} | {obs}'] = dict(paper_protocol=s_paper, raw_vector=s_raw)
    res['episodes_first_step'] = ep
    # ---- executed outcome: command of each policy from the rendered observation; position error after the move without / with actuation noise (Monte Carlo)
    d = D[D.obs == 'rendered'].sort_values('ep').reset_index(drop=True); gt = d[['x', 'y', 'z']].values; rng = np.random.default_rng(0); ex = {}; hh = np.deg2rad(d.heading_a.values)
    to_w = lambda v: np.c_[np.cos(hh) * v[:, 0] + np.sin(hh) * v[:, 1], -np.sin(hh) * v[:, 0] + np.cos(hh) * v[:, 1], v[:, 2]]      # the simulator executes (and perturbs) the command in east / north / up
    use_pl = (d.s2_dist < SWITCH) & (d.pl_inl >= 200) & np.isfinite(d.pl_wx); cmd = {'s1': d[['s1_lx', 's1_ly', 's1_z']].values.astype(float), 's2': d[['s2_lx', 's2_ly', 's2_z']].values.astype(float)}
    c = cmd['s2'].copy(); c[use_pl.values] = (d.loc[use_pl, ['pl_wx', 'pl_wy', 'pl_wz']].values * d.loc[use_pl, ['g_H']].values); cmd['s2p'] = c; cmd = {k: to_w(v) for k, v in cmd.items()}
    A = pd.read_pickle(f'{SIM}/nav_log.pkl'); boots = boot_indices(d.ep.values, 1000, 0)
    for p, v in cmd.items():
        e0 = np.linalg.norm(v - gt, axis=1); nd = 2000; mv = v[None] * (1 + NOISE['pos_mult'] * rng.standard_normal((nd, len(v), 3))) + NOISE['pos_add'] * rng.standard_normal((nd, len(v), 3)); en = np.linalg.norm(mv - gt[None], axis=2)
        a1 = A[(A.pol == p) & (A.step == 1)].sort_values('ep').pos_err.values
        ex[p] = dict(no_noise=dict(sr_1m=float((e0 < 1).mean()), median=float(np.median(e0)), ci_sr=ci(lambda i: (e0[i] < 1).mean(), boots)),
                     with_noise=dict(sr_1m_expected=float((en < 1).mean()), sr_1m_std_over_draws=float((en < 1).mean(1).std()), sr_1m_range_95=[float(np.percentile((en < 1).mean(1), 2.5)), float(np.percentile((en < 1).mean(1), 97.5))], median_expected=float(np.median(en, axis=1).mean()),
                                     per_episode_success_probability_mean=float((en < 1).mean(0).mean())),
                     archived_single_draw=dict(sr_1m=float((a1 < 1).mean()), median=float(np.median(a1))), plane_solver_used_at_first_step=int(use_pl.sum()),
                     noise_only_displacement_m=dict(median=float(np.median(np.linalg.norm(mv - v[None], axis=2))), p90=float(np.percentile(np.linalg.norm(mv - v[None], axis=2), 90))))
    res['executed_first_step'] = ex
    # ---- start distances and label ranges
    qs = [5, 25, 50, 75, 95]; S = json.load(open(f'{OUT}/dataset_stats.json'))['candidates_vs_selected']['selected_label_range']
    inside = np.ones(len(d), bool)
    for k in ('x', 'y', 'z', 'tilt', 'heading'): inside &= (d[k].values >= S[k][0]) & (d[k].values <= S[k][1])
    res['distances'] = dict(episodes={f'p{q}': float(np.percentile(d.dist, q)) for q in qs}, test_pairs={f'p{q}': float(np.percentile(T.dist, q)) for q in qs}, episodes_mean=float(d.dist.mean()), test_mean=float(T.dist.mean()),
                            episodes_inside_selected_label_ranges=float(inside.mean()), episodes_abs_dx_or_dy_gt_15m=float(((d.x.abs() > 15) | (d.y.abs() > 15)).mean()))
    io = {}
    for obs in ('real', 'rendered'):
        dd = D[D.obs == obs].sort_values('ep').reset_index(drop=True); f2 = frame(dd)
        for nm, pre, al in (('Stage I', 's1_', 0.75), ('Stage II', 'p_', 0.5)):
            e2 = errors(dreamnav(f2, pre, 'cam', al=al), gt_cam(f2))
            io[f'{nm} | {obs}'] = {('inside' if k else 'outside'): dict(n=int((inside == k).sum()), sr_d_1m=float((e2['d'][inside == k] <= 1).mean()), sr_epe_1m=float((e2['epe'][inside == k] < 1).mean()), epe_med=float(np.median(e2['epe'][inside == k])), d_mae=float(e2['d'][inside == k].mean())) for k in (True, False)}
    res['inside_vs_outside_label_range'] = io
    json.dump(res, open(f'{OUT}/onestep_gap.json', 'w'), indent=1)
    f1 = lambda v: f'{100 * v:.1f}'
    L += ['## From Table 2 to the one-step column of Table 5 (Stage II with 95 % bootstrap CI over buildings / episodes; Stage I in parentheses)\n', '| Step of the chain | pairs | SR_d@1m (Table 2 definition) | SR_EPE@1m (camera frame) | EPE median (m) |', '|---|---:|---:|---:|---:|']
    a, b = ref['Stage II'], ref['Stage I']
    L.append(f"| Test pairs, archived predictions | {a['all_test']['n']} | {f1(a['all_test']['sr_d_1m'])} [{f1(a['ci_all']['sr_d_1m'][0])}, {f1(a['ci_all']['sr_d_1m'][1])}] ({f1(b['all_test']['sr_d_1m'])}) | {f1(a['all_test']['sr_epe_1m'])} [{f1(a['ci_all']['sr_epe_1m'][0])}, {f1(a['ci_all']['sr_epe_1m'][1])}] ({f1(b['all_test']['sr_epe_1m'])}) | {a['all_test']['epe_med']:.2f} ({b['all_test']['epe_med']:.2f}) |")
    L.append(f"| Test pairs of the 8 simulator sites | {a['n_sim_site_pairs']} | {f1(a['sim_sites']['sr_d_1m'])} ({f1(b['sim_sites']['sr_d_1m'])}) | {f1(a['sim_sites']['sr_epe_1m'])} ({f1(b['sim_sites']['sr_epe_1m'])}) | {a['sim_sites']['epe_med']:.2f} ({b['sim_sites']['epe_med']:.2f}) |")
    for obs, lab in (('real', 'Episode start/goal pairs, real start photo'), ('rendered', 'Same pairs, start view rendered by the simulator')):
        a, b = ep[f'Stage II | {obs}']['paper_protocol'], ep[f'Stage I | {obs}']['paper_protocol']
        L.append(f"| {lab} | {a['n']} | {f1(a['sr_d_1m'])} [{f1(a['ci']['sr_d_1m'][0])}, {f1(a['ci']['sr_d_1m'][1])}] ({f1(b['sr_d_1m'])}) | {f1(a['sr_epe_1m'])} [{f1(a['ci']['sr_epe_1m'][0])}, {f1(a['ci']['sr_epe_1m'][1])}] ({f1(b['sr_epe_1m'])}) | {a['epe_med']:.2f} ({b['epe_med']:.2f}) |")
    L += ['\n| Executed first step (rendered observation) | policy | SR@1m, no actuation noise | SR@1m, with actuation noise (mean over 2000 draws, 95 % range of a single draw) | archived single draw |', '|---|---|---:|---:|---:|']
    for p in ('s1', 's2', 's2p'): x = ex[p]; L.append(f"| position error after the move | {p} | {f1(x['no_noise']['sr_1m'])} | {f1(x['with_noise']['sr_1m_expected'])} [{f1(x['with_noise']['sr_1m_range_95'][0])}, {f1(x['with_noise']['sr_1m_range_95'][1])}] | {f1(x['archived_single_draw']['sr_1m'])} |")
    L.append(f"\nStart distance (m), percentiles 5/25/50/75/95: episodes {' / '.join(f'{v:.1f}' for v in res['distances']['episodes'].values())}; test pairs {' / '.join(f'{v:.1f}' for v in res['distances']['test_pairs'].values())}. "
             f"Episodes whose five pose differences all lie inside the label ranges of the selected pairs: {f1(res['distances']['episodes_inside_selected_label_ranges'])} %.")
    L += ['\n| Episode pairs split by the label ranges of the selected pairs (paper protocol, camera frame) | pairs | SR_d@1m | SR_EPE@1m | EPE median (m) |', '|---|---:|---:|---:|---:|']
    for k, v in io.items():
        for w in ('inside', 'outside'): L.append(f"| {k.replace(' | ', ', ')}, {w} | {v[w]['n']} | {f1(v[w]['sr_d_1m'])} | {f1(v[w]['sr_epe_1m'])} | {v[w]['epe_med']:.2f} |")
    open(f'{OUT}/onestep_gap.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    {'collect': collect, 'report': report}[sys.argv[1]]()
