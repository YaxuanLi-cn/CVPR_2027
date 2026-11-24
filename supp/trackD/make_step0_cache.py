#!/usr/bin/env python3
"""Step-0 estimates of the 160 episodes (simulator render at the start pose -> goal image) were already computed by onestep_gap.py collect.
The estimator is deterministic (checked: identical rows to the verification run on site 1407), so the closed-loop runs reuse them."""
import sys, numpy as np, pandas as pd, os
OUT = '/root/rasr-pairuav/outputs/supp/trackD'; D = pd.read_pickle(f'{OUT}/cache/onestep_pred.pkl'); D = D[D.obs == 'rendered']; V = pd.read_pickle(f'{OUT}/sim_verify_seed0/step0_pred.pkl')
cols = [c for c in V.columns if c in D.columns and c not in ('tour', 'ep')]; a = D[D.tour == '1407'].sort_values('ep')[cols].reset_index(drop=True); b = V.sort_values('ep')[cols].reset_index(drop=True)
num = [c for c in cols if np.issubdtype(a[c].dtype, np.number)]; diff = (a[num].astype(float) - b[num].astype(float)).abs().max().max(); print('columns compared', len(num), '| max abs difference between collect and verify step-0 rows (site 1407):', diff)
assert diff == 0 or np.isnan(diff) or diff < 1e-12, diff
nan_equal = (a[num].isna() == b[num].isna()).all().all(); print('NaN pattern identical:', bool(nan_equal)); assert nan_equal
os.makedirs(f'{OUT}/cache/step0', exist_ok=True)
for t, d in D.groupby('tour'):
    d = d.sort_values('ep').reset_index(drop=True); d.drop(columns=['tour', 'ep', 'goal', 'start', 'obs', 'x', 'y', 'z', 'heading', 'tilt', 'heading_a', 'tilt_a', 'lx', 'ly', 'dist']).to_pickle(f'{OUT}/cache/step0/{t}.pkl'); print(t, len(d))
