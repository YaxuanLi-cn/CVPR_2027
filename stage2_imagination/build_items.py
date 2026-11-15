#!/usr/bin/env python3
"""Per-(pair, side) table for stage-2 inference on the val/test pairs of the stage-1 result set.
Per-camera tilt estimate = SuperGlue geometry (zoom-MAP, else pass-1 MAP, else MLP prior), then made consistent with the fused relative tilt."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
OUT = f'{SG}/stage2'; import os; os.makedirs(OUT, exist_ok=True)
P = pd.read_pickle(f'{SG}/pairs.pkl'); Fz = pd.read_pickle(f'{SG}/half_zoom_pred.pkl'); G = pd.read_pickle(f'{SG}/geom_zoom_half.pkl'); M = pd.read_pickle(f'{SG}/map_half.pkl'); N = pd.read_pickle(f'{SG}/stage0_mlp.pkl')
A = pd.read_pickle(f'{SG}/absheading/pred_cnx_s_d8.pkl').set_index('image')
D = P.merge(Fz[['manifest_index', 'p_tilt', 'p_heading']], on='manifest_index').merge(G[['manifest_index', 'zmap_ok', 'zmap_ta', 'zmap_tb']], on='manifest_index', how='left') \
     .merge(M[['manifest_index', 'map_ok', 'map_ta', 'map_tb']], on='manifest_index', how='left').merge(N[['manifest_index', 'n_tilt_a', 'n_tilt_b']], on='manifest_index', how='left')
D['part'] = A.part.reindex(D.image_a).values; D = D[D.part.isin(['val', 'test'])].copy()
fa = np.where(D.zmap_ok == 1, D.zmap_ta, np.where(D.map_ok == 1, D.map_ta, D.n_tilt_a)); fb = np.where(D.zmap_ok == 1, D.zmap_tb, np.where(D.map_ok == 1, D.map_tb, D.n_tilt_b))
r = (D.p_tilt.values - (fb - fa)) / 2; D['te_a'] = np.clip(fa - r, 0, 47); D['te_b'] = np.clip(fb + r, 0, 47)
rows = []
for s in 'ab':
    rows.append(pd.DataFrame(dict(manifest_index=D.manifest_index.values, side=s, part=D.part.values, tour=D.tour.values, image=D[f'image_{s}'].values,
                                  tilt_est=D[f'te_{s}'].values, tilt_gt=D[f'tilt_{s}'].values, heading_gt=D[f'heading_{s}'].values)))
it = pd.concat(rows, ignore_index=True); it.to_pickle(f'{OUT}/bev_items.pkl')
print(len(it), it.part.value_counts().to_dict(), 'tilt MAE (est vs gt, for reporting only):', it.groupby('part').apply(lambda d: np.abs(d.tilt_est - d.tilt_gt).mean()).to_dict())
