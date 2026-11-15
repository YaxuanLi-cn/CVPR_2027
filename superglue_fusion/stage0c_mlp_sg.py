#!/usr/bin/env python3
"""Stage 0c: neural fusion candidate. Same MLP head recipe as the baseline, but fed MASt3R features + SuperGlue pose features
(E-matrix + MAP). Tour-grouped OOF; its predictions become extra candidates in stage 1.
Usage: stage0c_mlp_sg.py <geom.pkl> <map.pkl> <out.pkl> [--partial]"""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
from stage0_mlp import train_predict, dev
from sklearn.model_selection import GroupKFold
geom, mp, dst = sys.argv[1:4]; partial = '--partial' in sys.argv
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); M = load_mast3r(); S = pd.read_pickle(f'{SG}/stage0_mlp.pkl')
D = P.merge(M.drop(columns='split'), on='manifest_index').merge(S, on='manifest_index').merge(pd.read_pickle(geom), on='manifest_index', how='inner' if partial else 'left').merge(pd.read_pickle(mp), on='manifest_index', how='left')
mfeats = [c for c in M.columns if c not in ('manifest_index', 'split')]
D['f_logpar'] = np.log(D.sg_par.clip(lower=1e-6)); D['f_maplogpar'] = np.log(D.map_par.clip(lower=1e-6)); D['f_loginl'] = np.log1p(D.sg_inl.fillna(0)); D['f_logn'] = np.log1p(D.sg_n.fillna(0))
for c in ('sg_hgta_p50', 'sg_hgta_p75', 'sg_hgtb_p50', 'sg_hgtb_p75', 'map_hgta', 'map_hgtb'): D['l_' + c] = -np.log(D[c].clip(lower=1e-3))
D['map_dh_sin'] = np.sin(np.deg2rad(D.map_dh)); D['map_dh_cos'] = np.cos(np.deg2rad(D.map_dh))
gf = ['sg_ok', 'sg_dh_sin', 'sg_dh_cos', 'sg_ta', 'sg_tb', 'sg_dtilt', 'sg_tsum', 'sg_vx', 'sg_vy', 'sg_vz', 'sg_inl_ratio', 'sg_resid', 'sg_tinfo', 'sg_cheir', 'sg_zspread', 'sg_conf_mean', 'sg_prim_fine',
      'sg_flow_mx', 'sg_flow_my', 'sg_flow_sx', 'sg_flow_sy', 'sg_cov0', 'sg_cov1', 'alt_ok', 'alt_dh_sin', 'alt_dh_cos', 'alt_ta', 'alt_tb', 'alt_vx', 'alt_vy', 'alt_vz', 'sg_alt_dh_diff', 'sg_alt_dir_cos',
      'map_ok', 'map_dh_sin', 'map_dh_cos', 'map_ta', 'map_tb', 'map_dtilt', 'map_vx', 'map_vy', 'map_vz', 'map_shift_t', 'map_shift_dir', 'map_resid', 'map_from_prior',
      'f_logpar', 'f_maplogpar', 'f_loginl', 'f_logn'] + [c for c in D.columns if c.startswith('l_')] + [f'sg_R{i}{j}' for i in range(3) for j in range(3)] + ['sg_t0', 'sg_t1', 'sg_t2']
gf = [c for c in gf if c in D.columns]
G = D[gf].astype(np.float64).copy(); tr = (D.split == 'train').values
lo, hi = G[tr].quantile(0.005), G[tr].quantile(0.995); G = G.clip(lo, hi, axis=1); miss = G.isna().astype(np.float32).add_prefix('isna_'); miss = miss.loc[:, miss[tr].mean() > 0.002]
G = G.fillna(G[tr].median()).fillna(0.0)
X = np.column_stack([D[mfeats].values, G.values, miss.values]).astype(np.float32); print('rows', len(D), 'features', X.shape[1], flush=True)
targets = ['z', 'tilt', 'dist', 'hsin', 'hcos', 'lx', 'ly', 'tilt_a', 'tilt_b', 'ux', 'uy', 'uz']
Xtr = torch.tensor(X[tr], device=dev); Xte = torch.tensor(X[~tr], device=dev); Ytr = torch.tensor(D.loc[tr, targets].values, dtype=torch.float32, device=dev); oof = np.zeros((tr.sum(), len(targets)))
for f, (a, b) in enumerate(GroupKFold(5).split(X[tr], groups=D.tour.values[tr])):
    oof[b] = train_predict(Xtr[torch.tensor(a, device=dev)], Ytr[torch.tensor(a, device=dev)], [Xtr[torch.tensor(b, device=dev)]], seed=200 + f)[0]
pte = train_predict(Xtr, Ytr, [Xte], seed=71)[0]
out = pd.DataFrame(np.zeros((len(D), len(targets))), columns=['q_' + t for t in targets]); out.loc[tr, :] = oof; out.loc[~tr, :] = pte; out['manifest_index'] = D.manifest_index.values
out['q_heading'] = np.rad2deg(np.arctan2(out.q_hsin, out.q_hcos)); out['q_logd'] = np.log(out.q_dist.clip(lower=0.3)); out.to_pickle(dst)
for nm, m in (('OOF ', tr), ('TEST', ~tr)):
    d = D[m]; o = out[m]
    print(nm, ' '.join(f'{t}={np.mean(np.abs(o["q_"+t].values - d[t].values)):.4f}' for t in ('z', 'tilt', 'dist', 'lx', 'ly', 'tilt_a', 'tilt_b')), f'heading={np.mean(np.abs(wrap(o.q_heading.values - d.heading.values))):.4f}', flush=True)
