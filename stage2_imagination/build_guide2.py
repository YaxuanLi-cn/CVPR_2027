#!/usr/bin/env python3
"""Guide poses for a SECOND imagination iteration: same as build_guide.py but taken from the stage-2 prediction (OOF for train rows).
Usage: build_guide2.py <stage2_pred_tag> <geom_imag.pkl> <out_guide.pkl>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
tag, gi, dst = sys.argv[1:4]; S2 = f'{SG}/stage2'
P = pd.read_pickle(f'{SG}/pairs.pkl'); F = pd.read_pickle(f'{S2}/{tag}_pred.pkl'); I = pd.read_pickle(gi)[['manifest_index', 'imap_hgta', 'im_inl']]; G1 = pd.read_pickle(f'{S2}/guide.pkl')[['manifest_index', 'g_ta', 'g_tb', 'g_H']].rename(columns=lambda c: c.replace('g_', 'o_'))
D = P[['manifest_index', 'split', 'tour', 'image_a', 'image_b']].merge(F[['manifest_index', 'p_heading', 'p_tilt', 'p_dist', 'p_lx', 'p_ly', 'p_z', 'imap_ta', 'imap_tb', 'imap_ok']], on='manifest_index').merge(I, on='manifest_index').merge(G1, on='manifest_index')
fa = np.where(D.imap_ok == 1, D.imap_ta, D.o_ta); fb = np.where(D.imap_ok == 1, D.imap_tb, D.o_tb); r = (D.p_tilt.values - (fb - fa)) / 2
D['g_ta'] = np.clip(fa - r, 0, 47); D['g_tb'] = np.clip(fb + r, 0, 47); D['g_dh'] = D.p_heading; H = D.p_dist * D.imap_hgta
D['g_H'] = np.where(np.isfinite(H) & (D.imap_ok == 1) & (D.im_inl > 100), np.clip(H, 165, 286), D.o_H); D['zm_inl'] = D.im_inl
D[['manifest_index', 'split', 'tour', 'image_a', 'image_b', 'g_dh', 'g_ta', 'g_tb', 'p_lx', 'p_ly', 'p_z', 'p_dist', 'g_H', 'zm_inl']].to_pickle(dst); print(len(D))
