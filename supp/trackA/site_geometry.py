#!/usr/bin/env python3
"""Does the camera-frame accuracy depend on how many TRAINING buildings share the campus of a test building?
Sites: single-linkage clusters (5 km) of the buildings (outputs/supp/trackC/sites.pkl). Output: outputs/supp/trackA/site_geometry.json"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp'); from common import *
T = load_test(); S = pd.read_pickle(f'{OUT}/trackC/sites.pkl').set_index('tour'); T['site_n_train'] = T.tour.map(S.site_n_train).values; T['site'] = T.tour.map(S.site).values
e1 = errors(dreamnav(T, 's1_', 'cam'), gt_cam(T)); e2 = errors(dreamnav(T, 'p_', 'cam'), gt_cam(T)); R = []
for lo, hi, lab in ((0, 0, '0'), (1, 5, '1-5'), (6, 20, '6-20'), (21, 10**6, '>20')):
    m = ((T.site_n_train >= lo) & (T.site_n_train <= hi)).values; bt = boot_indices(T.tour.values[m], 1000, 0); r = dict(bin=lab, pairs=int(m.sum()), buildings=int(T.tour[m].nunique()), sites=int(T.site[m].nunique()))
    for tag, e in (('s1', e1), ('s2', e2)):
        d, ep, th, ps = e['d'][m], e['epe'][m], e['theta'][m], e['psi'][m]
        r[tag] = dict(d_mae=float(d.mean()), sr_d=float((d <= 1).mean()), sr_d_ci=ci(lambda b: (d[b] <= 1).mean(), bt), epe_med=float(np.median(ep)), sr_epe=float((ep < 1).mean()), sr_epe_ci=ci(lambda b: (ep[b] < 1).mean(), bt), theta_mae=float(th.mean()), psi_mae=float(ps.mean()))
    R.append(r)
from scipy.stats import spearmanr
bld = T.assign(ok=(e2['epe'] < 1).astype(float)).groupby('tour').agg(ok=('ok', 'mean'), n=('site_n_train', 'first')); sp = spearmanr(bld.n, bld.ok)
json.dump(dict(rows=R, spearman_building_success_vs_site_n_train=dict(rho=float(sp.statistic), p=float(sp.pvalue), buildings=int(len(bld)))), open(f'{OUT}/trackA/site_geometry.json', 'w'), indent=1)
for r in R: print(r['bin'], r['pairs'], r['buildings'], r['sites'], '| S1 SR_d %.1f SR_EPE %.1f | S2 SR_d %.1f [%.1f, %.1f] SR_EPE %.1f [%.1f, %.1f] d %.2f' % (100*r['s1']['sr_d'], 100*r['s1']['sr_epe'], 100*r['s2']['sr_d'], 100*r['s2']['sr_d_ci'][0], 100*r['s2']['sr_d_ci'][1], 100*r['s2']['sr_epe'], 100*r['s2']['sr_epe_ci'][0], 100*r['s2']['sr_epe_ci'][1], r['s2']['d_mae']))
print('spearman', sp)
