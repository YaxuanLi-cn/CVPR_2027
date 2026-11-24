#!/usr/bin/env python3
"""Accuracy of every matcher + essential matrix on the 300 calibration pairs, per inlier threshold (used to choose the threshold)."""
import sys, numpy as np, pandas as pd, os
ROOT='/root/rasr-pairuav'; sys.path[:0]=[f'{ROOT}/supp',f'{ROOT}/superglue_fusion',f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
from eval_pairuav import pick, unit, dir_err, NAMES, HAS_TTA
from fusion_lib import add_targets, wrap
from rb_recon import subset
Tc=add_targets(subset(300,split='train')); idc=Tc.manifest_index.values; only = sys.argv[1].split(',') if len(sys.argv) > 1 else list(NAMES)
for m in only:
    f=f'{ROOT}/outputs/supp/trackB/geom/{m}_train.pkl'
    if not os.path.exists(f): continue
    G=pd.read_pickle(f)
    for tta in ((False,True) if HAS_TTA[m] else (False,)):
        for t in sorted(G.thr.unique()):
            u=unit(pick(G,idc,t,tta)); h=u.have.values; de=dir_err(u,Tc)[h]; ok=(u.ok.values==1)[h]; T=Tc[h]
            eh=np.where(ok,np.abs(wrap(u.dh.values[h]-T.heading.values)),180); et=np.where(ok,np.abs(u.tb.values[h]-u.ta.values[h]-T.tilt.values),45)
            print(f"{m:8s} {'tta' if tta else 'plain':5s} thr {t:4.1f} n {h.sum():3d} | solved {ok.mean():.3f} | dir<5deg {np.mean(de<5):.3f} dir med {np.median(de):6.2f} | dh med {np.nanmedian(eh):7.3f} <1deg {np.mean(eh<1):.3f} | dtilt med {np.nanmedian(et):6.3f} | inl med {np.nanmedian(u.inl.values[h]):.0f} | rot hist {np.bincount(u.k.values[h][np.isfinite(u.k.values[h])].astype(int),minlength=4).tolist()}")
