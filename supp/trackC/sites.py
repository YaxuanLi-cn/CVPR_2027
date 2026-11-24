#!/usr/bin/env python3
"""Track C, step 0: geographic sites of the 1,412 buildings (tours) and the held-out-site split for the site-disjoint test.

Site = single-linkage cluster (5 km) of the buildings' mean camera positions. `part` is the role of a building for the
absolute-heading networks: train / val (10 % of the training buildings, never used for training the networks) / test.
Held-out sites: sites are visited in a fixed random order (seed 0) and added while the number of TEST buildings they
contain stays <= 35 % of all test buildings; we stop once it is >= 25 %. All training buildings of those sites are
removed from the training set of the site-disjoint networks.
Also writes the size-matched control: the same number of training buildings removed, drawn at random (seed 0) from the
sites that are NOT held out.
"""
import json, numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
SG = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; O = '/root/rasr-pairuav/outputs/supp/trackC'
P = pd.read_pickle(f'{SG}/pairs.pkl'); m1 = pd.read_pickle(f'{SG}/absheading/images.pkl')
T = P.groupby('tour').agg(lat=('latitude_a', 'mean'), lon=('longitude_a', 'mean'), split=('split', 'first')).reset_index()
la, lo = np.deg2rad(T.lat.values), np.deg2rad(T.lon.values); X = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * 6371.0
T['site'] = fcluster(linkage(X, 'single'), 5.0, 'distance')
# role of each building for the heading networks (identical to abs_heading_train2.py / bev_heading.py)
rng = np.random.default_rng(0); tr_tours = np.array(sorted(m1[m1.split == 'train'].tour.unique())); val_tours = set(rng.choice(tr_tours, len(tr_tours) // 10, replace=False))
T['part'] = np.where(T.split == 'test', 'test', np.where(T.tour.isin(val_tours), 'val', 'train'))
S = T.groupby('site').agg(n=('tour', 'size'), n_train=('part', lambda s: int((s == 'train').sum())), n_val=('part', lambda s: int((s == 'val').sum())), n_test=('part', lambda s: int((s == 'test').sum())), lat=('lat', 'mean'), lon=('lon', 'mean')).reset_index()
T = T.merge(S[['site', 'n_train']].rename(columns={'n_train': 'site_n_train'}), on='site')
ntest = int((T.part == 'test').sum()); order = np.random.default_rng(0).permutation(S.site.values); held = []; cnt = 0
for s in order:
    k = int(S.loc[S.site == s, 'n_test'].iloc[0])
    if k == 0: continue                       # a site without test buildings adds nothing to the evaluation
    if cnt + k > 0.35 * ntest: continue
    held.append(int(s)); cnt += k
    if cnt >= 0.25 * ntest: break
T['heldout'] = T.site.isin(held)
rm = sorted(T[(T.part == 'train') & T.heldout].tour.tolist())                      # training buildings removed for the site-disjoint networks
pool = sorted(T[(T.part == 'train') & ~T.heldout].tour.tolist()); ctrl = sorted(np.random.default_rng(0).choice(pool, len(rm), replace=False).tolist())   # size-matched control
# distance from every building to the nearest training building that REMAINS in the site-disjoint training set
keep = T[(T.part == 'train') & ~T.heldout]; Xk = X[keep.index.values]; d = np.sqrt(((X[:, None, :] - Xk[None, :, :])**2).sum(-1)); T['km_to_nearest_kept_train'] = d.min(1)
T.to_pickle(f'{O}/sites.pkl'); S['heldout'] = S.site.isin(held); S.to_csv(f'{O}/sites.csv', index=False)
info = dict(n_sites=int(len(S)), linkage_km=5.0, heldout_sites=held, heldout_site_table=S[S.heldout][['site', 'lat', 'lon', 'n', 'n_train', 'n_val', 'n_test']].round(3).to_dict('records'),
            test_buildings=ntest, test_buildings_heldout=int(((T.part == 'test') & T.heldout).sum()), val_buildings_heldout=int(((T.part == 'val') & T.heldout).sum()),
            train_buildings=int((T.part == 'train').sum()), train_buildings_removed=len(rm), control_buildings_removed=len(ctrl),
            test_buildings_by_site_n_train={k: int(v) for k, v in pd.cut(T[T.part == 'test'].site_n_train, [-1, 0, 5, 20, 1e9], labels=['0', '1-5', '6-20', '>20']).value_counts().sort_index().items()},
            southern_hemisphere=dict(train=int(((T.part == 'train') & (T.lat < 0)).sum()), val=int(((T.part == 'val') & (T.lat < 0)).sum()), test=int(((T.part == 'test') & (T.lat < 0)).sum())))
json.dump(info, open(f'{O}/sites.json', 'w'), indent=1); open(f'{O}/exclude_sitedisjoint.txt', 'w').write('\n'.join(rm)); open(f'{O}/exclude_control.txt', 'w').write('\n'.join(ctrl))
print(json.dumps(info, indent=1)); h = T[(T.part == 'test') & T.heldout]; print('held-out test buildings: km to nearest kept training building', h.km_to_nearest_kept_train.describe().round(1).to_dict())
