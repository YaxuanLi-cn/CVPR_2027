#!/usr/bin/env python3
"""Fig. S: camera and pair statistics of PairUAV (from the caches of supp/trackD/dataset_stats.py)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp/figs'); import style; style.use(); plt = style.plt
C = '/root/rasr-pairuav/outputs/supp/trackD/cache'; cam = pd.read_pickle(f'{C}/cameras.pkl'); h = np.load(f'{C}/candidate_hist.npz')
fig, ax = plt.subplots(2, 3, figsize=(style.COL, 2.15)); k = 1e3
def hist(a, v, bins, xl, ticks=None):
    c, e = np.histogram(v, bins=bins); a.stairs(c / k, e, fill=True, color=style.BLUE, lw=0); a.set_xlabel(xl); a.set_xlim(e[0], e[-1])
    if ticks is not None: a.set_xticks(ticks)
hist(ax[0, 0], cam.hag.values, np.arange(202, 238.01, 1.0), 'height above terrain (m)', [205, 220, 235]); ax[0, 0].set_ylabel(r'cameras ($\times10^3$)')
hist(ax[0, 1], cam.tilt.values, np.arange(0, 45.01, 1.5), r'tilt $\psi$ ($^\circ$)', [0, 15, 30, 45])
hist(ax[0, 2], cam.heading.values, np.arange(0, 360.01, 4.0), r'heading $\theta$ ($^\circ$)', [0, 120, 240, 360])
for a in ax[0]: a.set_yticks([0, 5])
def both(a, key, xl, ticks=None, fold=False):
    e = h[f'bins_{key}']; al = h[f'all_{key}'].astype(float); se = h[f'sel_{key}'].astype(float); w = np.diff(e)
    al = al / (al.sum() * w); se = se / (se.sum() * w)                                   # densities
    a.stairs(al, e, fill=True, color=style.GREY, lw=0, label='all camera pairs'); a.stairs(se, e, color=style.BLUE, lw=0.9, label='selected pairs'); a.set_xlabel(xl); a.set_xlim(e[0], e[-1]); a.set_yticks([])
    if ticks is not None: a.set_xticks(ticks)
both(ax[1, 0], 'dist', r'baseline $\|\Delta\mathbf{p}\|$ (m)', [0, 20, 40]); ax[1, 0].set_ylabel('density'); ax[1, 0].set_xlim(0, 55)
both(ax[1, 1], 'x', r'$\Delta x$ (m)', [-30, 0, 30]); both(ax[1, 2], 'tilt', r'$\Delta\psi$ ($^\circ$)', [-45, 0, 45])
for a in ax[1]: a.spines['left'].set_visible(False); a.grid(False)
hd, lb = ax[1, 0].get_legend_handles_labels(); fig.legend(hd, lb, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.045), columnspacing=1.5)
fig.subplots_adjust(left=0.115, right=0.985, top=0.985, bottom=0.2, wspace=0.32, hspace=0.72); style.save(fig, 'fig_dataset')
