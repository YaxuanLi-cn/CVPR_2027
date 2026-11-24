#!/usr/bin/env python3
"""Fit and persist everything the online engine needs, reproducing the archived result-set runs:
  stage0  : MASt3R-statistics priors (14 LightGBM heads m_*, 14-way MLP n_*) trained on all train rows of the 60k split
  stage1  : fusion store (same rows/config as `half_zoom`)
  stage2  : fusion store (same rows/config as `s2`)
Each step checks its test-row predictions against the archived prediction files.
Usage: train_stores.py stage0|stage1|stage2
"""
import sys, pickle, json, numpy as np, pandas as pd, torch
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion'); sys.path.insert(0, '/root/rasr-pairuav/online')
from fusion_lib import *
import fusion_core as fc
OUT = '/root/rasr-pairuav/outputs/online_models'; os.makedirs(OUT, exist_ok=True)
T0 = {'x': 'l1', 'y': 'l1', 'z': 'l1', 'tilt': 'l1', 'hsin': 'l2', 'hcos': 'l2', 'logd': 'l1', 'tilt_a': 'l1', 'tilt_b': 'l1', 'lx': 'l1', 'ly': 'l1', 'ux': 'l2', 'uy': 'l2', 'uz': 'l2'}
T1 = ['x', 'y', 'z', 'tilt', 'dist', 'hsin', 'hcos', 'tilt_a', 'tilt_b', 'lx', 'ly', 'ux', 'uy', 'uz']

def mae(a, b, circ=False): d = np.asarray(a) - np.asarray(b); d = wrap(d) if circ else d; return float(np.mean(np.abs(d)))

def stage0():
    import lightgbm as lgb, torch.nn.functional as F
    from stage0_mlp import Heads
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); D = P.merge(M.drop(columns='split'), on='manifest_index')
    feats = [c for c in M.columns if c not in ('manifest_index', 'split')]; tr, te = D[D.split == 'train'].reset_index(drop=True), D[D.split == 'test'].reset_index(drop=True)
    Xtr, Xte = tr[feats].values.astype(np.float32), te[feats].values.astype(np.float32)
    def run(t, obj):
        m = lgb.LGBMRegressor(**{**LGB, 'objective': obj, 'n_jobs': 6}); m.fit(Xtr, tr[t].values.astype(np.float64)); return t, m.booster_.model_to_string(), m.predict(Xte)
    res = Parallel(n_jobs=4)(delayed(run)(t, o) for t, o in T0.items()); store = {'feats': feats, 'lgb': {t: s for t, s, _ in res}}
    old = pd.read_pickle(f'{SG}/stage0.pkl').set_index('manifest_index').loc[te.manifest_index]
    print('stage0 LightGBM: |new - archived| on test rows:', {t: round(mae(p, old['m_' + t]), 4) for t, _, p in res}, flush=True)
    # MLP: same recipe/seed as stage0_mlp.py's full-train fit
    dev = 'cuda'; X = torch.tensor(Xtr, device=dev); Y = torch.tensor(tr[T1].values, dtype=torch.float32, device=dev)
    torch.manual_seed(61); mu = X.mean(0); sd = X.std(0); sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd); ys = Y.abs().max(0).values.clamp(min=1.0); Xn = (X - mu) / sd; Yn = Y / ys
    net = Heads(Y.shape[1], X.shape[1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); n = len(Xn)
    for ep in range(30):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            ix = perm[i:i + 256]; loss = F.smooth_l1_loss(net(Xn[ix]), Yn[ix], reduction='none').mean(0).sum(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.inference_mode(): pte = (net((torch.tensor(Xte, device=dev) - mu) / sd) * ys).cpu().numpy()
    oldn = pd.read_pickle(f'{SG}/stage0_mlp.pkl').set_index('manifest_index').loc[te.manifest_index]
    print('stage0 MLP: |new - archived| on test rows:', {t: round(mae(pte[:, j], oldn['n_' + t]), 4) for j, t in enumerate(T1)})
    print('stage0 MLP test MAE new vs archived:', {t: (round(mae(pte[:, j], te[t]), 4), round(mae(oldn['n_' + t], te[t]), 4)) for j, t in enumerate(T1) if t in te})
    store['mlp'] = dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mu=mu.cpu(), sd=sd.cpu(), ys=ys.cpu(), targets=T1)
    pickle.dump(store, open(f'{OUT}/stage0.pkl', 'wb')); print('saved stage0')

def frame_s1():
    P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); S0 = pd.read_pickle(f'{SG}/stage0.pkl').merge(pd.read_pickle(f'{SG}/stage0_mlp.pkl'), on='manifest_index')
    D = P.merge(S0, on='manifest_index').merge(M.drop(columns='split'), on='manifest_index').merge(pd.read_pickle(f'{SG}/geom_half_k2.pkl'), on='manifest_index', how='inner')
    D = D.merge(pd.read_pickle(f'{SG}/map_half_k2.pkl'), on='manifest_index', how='left').merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl'), on='manifest_index', how='left')
    D.attrs['mfeats'] = [c for c in M.columns if c not in ('manifest_index', 'split')]; D.attrs['s0cols'] = list(S0.columns); return D

def stage1():
    D = frame_s1(); st = fc.Store(fitting=True, jobs=4); st.set_rows(D.split == 'train', D.tour.values); print('rows', len(D), flush=True)
    out = fc.stage1(D, st); st.save(f'{OUT}/stage1.pkl'); out['split'] = D.split.values; out.to_pickle(f'{OUT}/stage1_refit_pred.pkl')
    old = pd.read_pickle(f'{SG}/half_zoom_pred.pkl').set_index('manifest_index').loc[out.manifest_index]; te = (D.split == 'test').values
    for k in ('heading', 'tilt', 'dist', 'z', 'lx', 'ly'):
        y = D['dist' if k == 'dist' else k].values
        print(f'{k:8s} test MAE refit {mae(out["p_" + k][te], y[te], k == "heading"):.4f} archived {mae(old["p_" + k].values[te], y[te], k == "heading"):.4f} | mean |refit-archived| {mae(out["p_" + k][te], old["p_" + k].values[te], k == "heading"):.4f}')

def stage2():
    S2 = f'{SG}/stage2'; P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); S0 = pd.read_pickle(f'{SG}/stage0.pkl').merge(pd.read_pickle(f'{SG}/stage0_mlp.pkl'), on='manifest_index')
    I = pd.read_pickle(f'{S2}/geom_imag.pkl'); F1 = pd.read_pickle(f'{SG}/half_zoom_pred.pkl')[['manifest_index', 'p_heading', 'p_tilt', 'p_dist', 'p_z', 'p_lx', 'p_ly']].rename(columns=lambda c: c.replace('p_', 's1_'))
    D = P.merge(I, on='manifest_index').merge(F1, on='manifest_index').merge(S0, on='manifest_index').merge(M.drop(columns='split'), on='manifest_index') \
         .merge(pd.read_pickle(f'{SG}/geom_half.pkl'), on='manifest_index', how='left').merge(pd.read_pickle(f'{SG}/map_half.pkl'), on='manifest_index', how='left').merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl'), on='manifest_index', how='left')
    D.attrs['mfeats'] = [c for c in M.columns if c not in ('manifest_index', 'split')]; D.attrs['s0cols'] = list(S0.columns)
    st = fc.Store(fitting=True, jobs=4); st.set_rows(D.split == 'train', D.tour.values); print('rows', len(D), D.split.value_counts().to_dict(), flush=True)
    out = fc.stage2(D, st)
    # camera-height constant used to render the imagined view (build_guide.py)
    Fz = pd.read_pickle(f'{SG}/half_zoom_pred.pkl').merge(pd.read_pickle(f'{SG}/geom_zoom_half.pkl')[['manifest_index', 'zmap_hgta', 'zm_inl']], on='manifest_index', how='left')
    H = Fz.p_dist * Fz.zmap_hgta; st.d['s2/H0'] = float(np.median(H[(Fz.split == 'train') & (Fz.zm_inl > 400) & np.isfinite(H)])); print('guide H0', st.d['s2/H0'])
    st.save(f'{OUT}/stage2.pkl'); out['split'] = D.split.values; out.to_pickle(f'{OUT}/stage2_refit_pred.pkl')
    old = pd.read_pickle(f'{S2}/s2_pred.pkl').set_index('manifest_index').loc[out.manifest_index]; te = (D.split == 'test').values
    for k in ('heading', 'tilt', 'dist', 'z', 'lx', 'ly'):
        y = D['dist' if k == 'dist' else k].values
        print(f'{k:8s} test MAE refit {mae(out["p_" + k][te], y[te], k == "heading"):.4f} archived {mae(old["p_" + k].values[te], y[te], k == "heading"):.4f} | mean |refit-archived| {mae(out["p_" + k][te], old["p_" + k].values[te], k == "heading"):.4f}')

if __name__ == '__main__':
    {'stage0': stage0, 'stage1': stage1, 'stage2': stage2}[sys.argv[1]]()
