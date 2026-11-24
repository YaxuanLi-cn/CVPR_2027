#!/usr/bin/env python3
"""Re-draws fig_dataset_pose_diff from the cached histograms (dataset_stats.py writes cache/candidate_hist.npz)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/root/rasr-pairuav/outputs/supp/trackD'; Z = np.load(f'{OUT}/cache/candidate_hist.npz')
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'xtick.major.size': 2.5})
lab = dict(x=r'$\Delta x$ (m)', y=r'$\Delta y$ (m)', z=r'$\Delta z$ (m)', tilt=r'$\Delta\psi$ ($^\circ$)', heading=r'$\Delta\theta$ ($^\circ$)', dist='baseline (m)'); agg = dict(x=4, y=4, z=4, tilt=4, heading=10, dist=4)
ticks = dict(x=[-30, -15, 0, 15, 30], y=[-30, -15, 0, 15, 30], z=[-25, 0, 25], tilt=[-30, 0, 30], heading=[-120, 0, 120], dist=[0, 20, 40]); lim = dict(x=(-31, 31), y=(-31, 31), z=(-35, 35), tilt=(-47, 47), heading=(-180, 180), dist=(0, 54))
fig, ax = plt.subplots(2, 3, figsize=(3.3, 2.05))
for a, k in zip(ax.ravel(), ('x', 'y', 'z', 'tilt', 'heading', 'dist')):
    b = Z[f'bins_{k}']; g = agg[k]; n = (len(b) - 1) // g * g; ha = Z[f'all_{k}'][:n].reshape(-1, g).sum(1).astype(float); hs = Z[f'sel_{k}'][:n].reshape(-1, g).sum(1).astype(float); e = b[:n + 1:g]; w = e[1] - e[0]; ha /= ha.sum() * w; hs /= hs.sum() * w
    h1 = a.stairs(ha, e, fill=True, color='#c9c8c2', lw=0); h2 = a.stairs(hs, e, color='#eb6834', lw=0.9); a.set_xlabel(lab[k], labelpad=1.5); a.set_yticks([]); a.set_ylim(0, 1.12 * max(ha.max(), hs.max())); a.set_xlim(*lim[k]); a.set_xticks(ticks[k]); a.spines['left'].set_visible(False)
fig.legend([h1, h2], ['all 150 x 149 camera pairs', 'selected pairs'], loc='upper center', ncol=2, frameon=False, handlelength=1.4, columnspacing=1.5, bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.14, wspace=0.14, hspace=0.62); fig.savefig(f'{OUT}/fig_dataset_pose_diff.pdf'); fig.savefig(f'{OUT}/fig_dataset_pose_diff.png', dpi=300); print('saved')
