"""Every table number of experiment.tex that is recomputed on the common 6,364-pair test set.
Reuses the assembly code of analysis/extended_metrics.py; writes tables.json, errs.npz, T.pkl next to this file."""
import sys, json, os, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
src = open('/root/rasr-pairuav/analysis/extended_metrics.py').read()
head = src.split('# tour-level bootstrap')[0]
ns = {}; exec(compile(head, 'ext_head', 'exec'), ns)
T, M, errs, summary, ours, V2, BV, SG, S2, P, wrap, BL, pos = [ns[k] for k in ('T','M','errs','summary','ours','V2','BV','SG','S2','P','wrap','BL','pos')]
world = ns['world']; combine = ns['combine']
print('methods from ext:', list(M))
# ---- extra baselines aligned on the same rows
def add_npz(name, f, kp='pred_xyz_tilt_heading', kt='true_xyz_tilt_heading'):
    from scipy.spatial import cKDTree
    d = np.load(f'{BL}/{f}'); p, t = d[kp], d[kt]
    if np.abs(t[pos, 0] - T.x.values).max() < 1e-2: ix = pos
    else:
        q = T[['x', 'y', 'z', 'tilt']].values; dist, ix = cKDTree(t[:, :4]).query(q); print(name, 'matched by labels: max label distance', dist.max(), 'unique', len(set(ix)), 'tilt range in file', t[:, 3].min(), t[:, 3].max(), 'heading range', t[:, 4].min(), t[:, 4].max())
        assert dist.max() < 1e-2 and len(set(ix)) == len(ix), name
    p = p[ix]; print(name, 'heading label check', np.abs(wrap(t[ix, 4] - T.heading.values)).max())
    M[name] = dict(x=p[:, 0], y=p[:, 1], z=p[:, 2], tilt=p[:, 3], heading=p[:, 4])
add_npz('Sample4Geo', 'sample4geo/outputs_xyz_mast3r_split_rerun/train_test_predictions.npz')
add_npz('AI2-THOR', 'ai2thor/checkpoints_xyz_mast3r_split_supervised/test_predictions.npz', 'pred', 'true')
add_npz('ViNT', 'visualnav-transformer/pairuav5d_v2/runs/vint_native/test_predictions.npz')
add_npz('NoMaD', 'visualnav-transformer/pairuav5d_v2/runs/nomad_native/test_predictions.npz')
HEAD = {}                                           # ViNT / NoMaD also predict the distance with their own official distance head (benchmark_results.md reports that one)
for name, f in (('ViNT', 'vint_native'), ('NoMaD', 'nomad_native')):
    from scipy.spatial import cKDTree
    dd = np.load(f'{BL}/visualnav-transformer/pairuav5d_v2/runs/{f}/test_predictions.npz'); _, ix = cKDTree(dd['true_xyz_tilt_heading'][:, :4]).query(T[['x', 'y', 'z', 'tilt']].values); HEAD[name] = dd['pred_distance_head_m'][ix]
add_npz('PI-0-FAST', 'openpi/pairuav_v2/results/pi0_fast/test_predictions.npz')   # pi0-FAST, LoRA, 1 epoch, greedy decoding
add_npz('PI-0.5', 'openpi/pairuav_v2/results/pi05/test_predictions.npz')   # pi0.5 retrained without the crop/rotation augmentation (LoRA, 1 epoch); the earlier checkpoint's outputs are in results/pi05_old/
# ---- original RASR heads on the same rows
sys.path.insert(0, '/root/rasr-pairuav'); from models_src.distance_head import DistanceHead
F = pd.read_csv('/root/rasr-pairuav/outputs/pairuav5d_60k/features/test_xyz.csv').set_index('manifest_index') if 'manifest_index' in pd.read_csv('/root/rasr-pairuav/outputs/pairuav5d_60k/features/test_xyz.csv', nrows=1).columns else None
MD = '/root/rasr-pairuav/outputs/pairuav5d_60k/models'
def head(path, X):
    ck = torch.load(path, map_location='cpu', weights_only=False); m = DistanceHead(ck['feature_dim'], ck['hidden_dim'], ck['dropout']); m.load_state_dict(ck['model_state_dict']); m.eval()
    x = X[ck['feature_names']].values.astype(np.float32); x = (x - np.asarray(ck['feature_mean'], np.float32)) / np.maximum(np.asarray(ck['feature_std'], np.float32), 1e-6)
    with torch.inference_mode(): return m(torch.as_tensor(x)).numpy().reshape(-1) * float(ck['target_scale'])
if F is not None:
    X = F.loc[T.manifest_index.values]; assert np.abs(X.x_num.values - T.x.values).max() < 1e-2
    r = {a: head(f'{MD}/coordinate_head_{a}.pt', X) for a in 'xyz'}; r['tilt'] = head(f'{MD}/tilt_head.pt', X)
    h = json.load(open(f'{MD}/heading_baseline.json')); hx = (X[h['feature_names']].values.astype(np.float64) - np.asarray(h['mean'])) / np.where(np.asarray(h['std']) < 1e-8, 1.0, np.asarray(h['std'])); hx = np.c_[hx, np.ones(len(hx))]
    r['heading'] = np.rad2deg(np.arctan2(hx @ np.asarray(h['sin_weights']), hx @ np.asarray(h['cos_weights']))); M['RASR (original)'] = r
    h = json.load(open(f'{MD}/heading_raw.json')); hx = (X[h['feature_names']].values.astype(np.float64) - np.asarray(h['mean'])) / np.where(np.asarray(h['std']) < 1e-8, 1.0, np.asarray(h['std'])); hx = np.c_[hx, np.ones(len(hx))]
    r2 = dict(r); r2['heading'] = hx @ np.asarray(h['weights']); M['RASR (raw-degree heading)'] = r2
# ---- stage variants
M['S2 module A only (v2 heading)'] = ours(T, 'p_', V2, 0.5)
M['S1 + module B heading'] = ours(T, 's1_', BV + V2, 0.75)
M['S1, true abs heading'] = ours(T, 's1_', None, 0.75)
K = pd.read_pickle(f'{SG}/half_k2_pred.pkl').set_index('manifest_index').loc[T.manifest_index.values].reset_index()
Tk = T.copy()
for c in ('lx', 'ly', 'z', 'dist', 'tilt', 'heading'): Tk['k_' + c] = K['p_' + c].values
M['rot-aware fusion, no zoom (v2 heading)'] = ours(Tk, 'k_', V2, 0.75)
mm = ours(Tk, 'k_', V2, 0.0); M['rot-aware fusion, no zoom, x/y from MLP'] = dict(x=T.n_x.values, y=T.n_y.values, z=mm['z'], tilt=mm['tilt'], heading=mm['heading'])
E = {k: errs(m) for k, m in M.items()}; R = {}
for k, e in E.items():
    if np.isnan(e['heading']).any():
        s = {}
        for kk in ('x', 'y', 'z', 'distance', 'pos3d'): s[kk] = dict(mae=float(e[kk].mean()), median=float(np.median(e[kk])))
        s['success'] = {f'pos<{t}m': float((e['pos3d'] < t).mean()) for t in (0.5, 1, 2, 5)}; R[k] = s
    else: R[k] = summary(e)
json.dump(R, open(os.path.join(HERE, 'tables.json'), 'w'), indent=1)
for k, e in E.items(): R[k]['success']['SR: |norm(pred) - norm(gt)| <= 1 m'] = float((e['distance'] <= 1.0).mean())     # the success rate reported in the paper
json.dump(R, open(os.path.join(HERE, 'tables.json'), 'w'), indent=1)
print(f"{'method':42s} {'x':>7s} {'y':>7s} {'z':>7s} {'dist':>7s} {'head':>8s} {'tilt':>7s} | SR(dist<=1m) pos<1m pos<5m | AUC5 AUC10 AUC20 | pos med")
for k, s in R.items():
    g = lambda a, b='mae': s[a][b] if a in s else float('nan'); au = s.get('auc', {})
    print(f"{k:42s} {g('x'):7.4f} {g('y'):7.4f} {g('z'):7.4f} {g('distance'):7.4f} {g('heading'):8.4f} {g('tilt'):7.4f} | {100*s['success']['SR: |norm(pred) - norm(gt)| <= 1 m']:5.1f} {100*s['success']['pos<1m']:5.1f} {100*s['success']['pos<5m']:5.1f} | {100*au.get('pose_auc@5deg', float('nan')):4.1f} {100*au.get('pose_auc@10deg', float('nan')):5.1f} {100*au.get('pose_auc@20deg', float('nan')):5.1f} | {s['pos3d']['median']:.3f}")
np.savez(os.path.join(HERE, 'errs.npz'), **{f'{k}|{kk}': v for k, e in E.items() for kk, v in e.items()})
T[['manifest_index', 'tour', 'x', 'y', 'z', 'tilt', 'heading', 'dist', 'zm_inl', 'heading_a']].to_pickle(os.path.join(HERE, 'T.pkl'))
gdist = np.sqrt(T.x.values**2 + T.y.values**2 + T.z.values**2)
for name, q in HEAD.items():
    e = np.abs(q - gdist); R[name]['distance_head'] = dict(mae=float(e.mean()), SR=float((e <= 1.0).mean())); print(f'{name}: official distance head MAE {e.mean():.2f} m, SR {100 * (e <= 1.0).mean():.1f}% (from the xyz norm: {R[name]["distance"]["mae"]:.2f} m, {100 * R[name]["success"]["SR: |norm(pred) - norm(gt)| <= 1 m"]:.1f}%)')
json.dump(R, open(os.path.join(HERE, 'tables.json'), 'w'), indent=1)
# ---- failure attribution quoted in the Limitations paragraph (stage II, success radius 1 m)
s2, orc = E['DreamNav stage 2 (final)']['pos3d'], E['DreamNav stage 2, true abs. heading']['pos3d']; fail = s2 >= 1; a = fail & (orc < 1); low = fail & ~a & (T.zm_inl.values < 200); n = fail.sum()
print(f'failures at 1 m: {n} | fixed by the true absolute heading {a.sum() / n:.1%} | fewer than 200 coarse inliers {low.sum() / n:.1%} | rest {(n - a.sum() - low.sum()) / n:.1%}')
