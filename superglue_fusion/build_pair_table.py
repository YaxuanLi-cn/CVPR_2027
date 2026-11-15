#!/usr/bin/env python3
"""Build a table of the 60k split pairs: labels + per-camera KML pose (analysis/validation only)."""
import json, re, sys
from pathlib import Path
import pandas as pd
DATA = Path('/root/pairUAV-5D-temp'); OUT = Path('/root/rasr-pairuav/outputs/pairuav5d_60k')
pat = {k: re.compile(rf'<{k}>([-\d.eE+]+)</{k}>') for k in ('longitude','latitude','altitude','heading','tilt','roll')}
fov_pat = re.compile(r'<gx:horizFov>([-\d.]+)</gx:horizFov>')
kml_cache = {}
def kml(rel):
    if rel not in kml_cache:
        s = (DATA/rel).read_text()
        d = {k: float(p.search(s).group(1)) for k, p in pat.items()}
        d['hfov'] = float(fov_pat.search(s).group(1))
        kml_cache[rel] = d
    return kml_cache[rel]
rows = []
for split in ('train', 'test'):
    handles = {}
    for line in (OUT/f'{split}_manifest.txt').read_text().splitlines():
        idx, ref = line.split('\t'); f, off = ref.rsplit('#', 1)
        if f not in handles: handles[f] = open(DATA/f, 'rb')
        h = handles[f]; h.seek(int(off)); p = json.loads(h.readline())
        r = dict(manifest_index=int(idx), split=split, pair_id=p['pair_id'], tour=p['tour_id'], a=p['a'], b=p['b'],
                 image_a=p['image_a'], image_b=p['image_b'],
                 x=p['delta']['x_m'], y=p['delta']['y_m'], z=p['delta']['z_m'], tilt=p['delta']['tilt_deg'], heading=p['delta']['heading_deg'],
                 sift_inliers=p['inliers'], sift_matches=p['matches'])
        t = p['tour_id']
        r['image_a'] = f"images/{t}/{Path(p['image_a']).name}"; r['image_b'] = f"images/{t}/{Path(p['image_b']).name}"
        for s, key in (('a', 'kml_a'), ('b', 'kml_b')):
            for k, v in kml(f"kml/{t}/{Path(p[key]).name}").items(): r[f'{k}_{s}'] = v
        rows.append(r)
df = pd.DataFrame(rows)
df['has_sg'] = [(DATA/'matches_superglue'/t/f'{pid}.npz').exists() for t, pid in zip(df.tour, df.pair_id)]
df.to_pickle(OUT/'superglue/pairs.pkl')
print(df.groupby('split').agg(n=('x','size'), has_sg=('has_sg','sum'), tours=('tour','nunique')))
print(df.describe().T.to_string())
