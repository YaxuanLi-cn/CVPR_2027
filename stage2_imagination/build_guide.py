#!/usr/bin/env python3
"""Stage-1 pose per pair, in the form needed to imagine the target view: relative heading, per-camera tilt, metric displacement in
cam-a's levelled frame and the height of camera a above the ground. For train rows the stage-1 values are tour-grouped OOF predictions."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
OUT = f'{SG}/stage2'; import os; os.makedirs(OUT, exist_ok=True)
P = pd.read_pickle(f'{SG}/pairs.pkl'); Fz = pd.read_pickle(f'{SG}/half_zoom_pred.pkl'); G = pd.read_pickle(f'{SG}/geom_zoom_half.pkl'); M = pd.read_pickle(f'{SG}/map_half.pkl'); N = pd.read_pickle(f'{SG}/stage0_mlp.pkl')
D = P[['manifest_index', 'split', 'tour', 'image_a', 'image_b']].merge(Fz[['manifest_index', 'p_tilt', 'p_heading', 'p_dist', 'p_lx', 'p_ly', 'p_z']], on='manifest_index') \
     .merge(G[['manifest_index', 'zmap_ok', 'zmap_ta', 'zmap_tb', 'zmap_hgta', 'zm_inl']], on='manifest_index', how='left') \
     .merge(M[['manifest_index', 'map_ok', 'map_ta', 'map_tb']], on='manifest_index', how='left').merge(N[['manifest_index', 'n_tilt_a', 'n_tilt_b']], on='manifest_index', how='left')
fa = np.where(D.zmap_ok == 1, D.zmap_ta, np.where(D.map_ok == 1, D.map_ta, D.n_tilt_a)); fb = np.where(D.zmap_ok == 1, D.zmap_tb, np.where(D.map_ok == 1, D.map_tb, D.n_tilt_b))
r = (D.p_tilt.values - (fb - fa)) / 2; D['g_ta'] = np.clip(fa - r, 0, 47); D['g_tb'] = np.clip(fb + r, 0, 47); D['g_dh'] = D.p_heading
H = D.p_dist * D.zmap_hgta; tr = (D.split == 'train') & (D.zm_inl > 400) & np.isfinite(H); H0 = float(np.median(H[tr])); print('median camera height on well-matched TRAIN pairs:', H0)
D['g_H'] = np.where(np.isfinite(H) & (D.zmap_ok == 1) & (D.zm_inl > 100), np.clip(H, 0.75 * H0, 1.3 * H0), H0)
D[['manifest_index', 'split', 'tour', 'image_a', 'image_b', 'g_dh', 'g_ta', 'g_tb', 'p_lx', 'p_ly', 'p_z', 'p_dist', 'g_H', 'zm_inl']].to_pickle(f'{OUT}/guide.pkl'); print(len(D), D.g_H.describe().round(1).to_dict())
