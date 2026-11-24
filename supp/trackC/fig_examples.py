#!/usr/bin/env python3
"""Track C: qualitative examples of the single-image absolute-heading estimate (archived raw-view network, 24 views) on
near-nadir test images: true north (white) and predicted north (orange) drawn at the image centre.
For a near-nadir view the top of the image points along the camera heading h (clockwise from north), so north is the
image-up direction rotated counter-clockwise by h."""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/root/rasr-pairuav/supp'); from common import wrap, SG, OUT
O = f'{OUT}/trackC'; D = f'{SG}/absheading'; im = pd.read_pickle(f'{D}/images.pkl'); I1 = np.load(f'{D}/images_448x256.npy', mmap_mode='r'); im['row'] = np.arange(len(im))
A = pd.read_pickle(f'{D}/pred_cnx_s_d8.pkl'); A = A[A.part == 'test'].merge(im[['image', 'heading', 'tilt', 'tour', 'lat', 'row']], on='image'); A['sg'] = wrap(A.ah_pred - A.heading); A['err'] = A.sg.abs()
ST = pd.read_pickle(f'{O}/sites.pkl').set_index('tour'); A['ntr'] = ST.site_n_train.reindex(A.tour).values; N = A[A.tilt < 12].copy(); rng = np.random.default_rng(3)
def pick(m, lo, hi):
    c = N[m & (N.err >= lo) & (N.err < hi)]; return c.iloc[int(rng.integers(len(c)))] if len(c) else None
sel = [('site with > 20 training buildings', pick(N.ntr > 20, 0.5, 1.5)), ('site with > 20 training buildings', pick(N.ntr > 20, 1.5, 3.0)), ('site with 1-5 training buildings', pick((N.ntr >= 1) & (N.ntr <= 5) & (N.lat > 0), 3, 12)),
       ('site without training buildings', pick((N.ntr == 0) & (N.lat > 0), 8, 40)), ('southern hemisphere', pick(N.lat < 0, 90, 181)), ('low confidence', N[(N.lat > 0)].sort_values('ah_kappa').iloc[5])]
sel = [(t, r) for t, r in sel if r is not None]; fig, ax = plt.subplots(2, 3, figsize=(6.9, 2.75)); ax = ax.ravel(); rec = []
for a, (t, r) in zip(ax, sel):
    x = I1[int(r.row)]; a.imshow(x); a.set_xticks([]); a.set_yticks([]); cx, cy, L = 224, 128, 78
    for h, col, lw, L in ((r.heading, 'white', 2.6, 92), (r.ah_pred, '#ff7f0e', 1.5, 74)):
        hr = np.deg2rad(h); dx, dy = -np.sin(hr) * L, -np.cos(hr) * L; a.annotate('', xy=(cx + dx, cy + dy), xytext=(cx, cy), arrowprops=dict(arrowstyle='-|>', color=col, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9))
    a.set_title(f'{t}\nerror {r.sg:+.1f}$^\\circ$, $\\kappa$ = {r.ah_kappa:.0f}', fontsize=6.5, pad=2); rec.append(dict(title=t, image=r.image, heading=float(r.heading), pred=float(r.ah_pred), signed_error=float(r.sg), kappa=float(r.ah_kappa), tilt=float(r.tilt), site_n_train=int(r.ntr)))
for a in ax[len(sel):]: a.axis('off')
fig.subplots_adjust(wspace=0.04, hspace=0.32, left=0.005, right=0.995, top=0.9, bottom=0.01); fig.savefig(f'{O}/fig_heading_examples.pdf', dpi=300); fig.savefig(f'{O}/fig_heading_examples.png', dpi=300); json.dump(rec, open(f'{O}/fig_heading_examples.json', 'w'), indent=1); print(json.dumps(rec, indent=1))
