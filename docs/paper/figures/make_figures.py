"""Paper figures for experiment.tex (run calc_tables.py first).
  fig_cdf.pdf          cumulative error curves on the PairUAV test pairs
  fig_breakdown.pdf    where the refinement helps (stage I -> stage II by difficulty)
  fig_reliability.pdf  sparsification and calibration of the predicted error
  fig_real.pdf         zero-shot transfer to real drone photographs
  fig_nav.pdf          closed-loop convergence (only when outputs/sim/nav_config.json exists)
Usage: make_figures.py [cdf breakdown reliability real nav]
"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = '/root/rasr-pairuav'
INK, INK2, GRID, AX = '#0b0b0b', '#52514e', '#e4e3de', '#b9b8b2'
C = dict(s2='#2a78d6', s1='#eb6834', rasr='#1baf7a', ext='#eda100')            # validated categorical slots 1-4 (fixed order)
LS = dict(s2='-', s1=(0, (5, 1.5)), rasr=(0, (3, 1, 1, 1)), ext=(0, (1, 1)))   # secondary encoding for greyscale print
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'TeX Gyre Termes', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.edgecolor': AX, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 0, 'ytick.minor.size': 0,
                     'legend.frameon': False, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'axes.titlesize': 8, 'axes.titlelocation': 'left', 'axes.titlepad': 4})
FULL, COL = 6.9, 3.3
def save(fig, name):
    fig.savefig(f'{HERE}/{name}.pdf'); fig.savefig(f'{HERE}/{name}.png', dpi=220); plt.close(fig); print('wrote', name)
NAMES = dict(s2='DreamNav (stage II)', s1='DreamNav (stage I)', rasr='RASR + sin/cos', ext='DINOv3 ViT-7B')

def cdf():
    E = np.load(f'{HERE}/errs.npz'); key = dict(s2='DreamNav stage 2 (final)', s1='DreamNav stage 1 (coarse)', rasr='MASt3R-stat MLP (RASR + sin/cos)', ext='DINOv3 ViT-7B')
    fig, ax = plt.subplots(1, 3, figsize=(FULL, 1.95), sharey=True)
    spec = [('pos3d', '3-D position error (m)', (0.05, 60), {'s2': (0.36, 30), 's1': (1.9, 56), 'rasr': (5.2, 22), 'ext': (27, 44)}),
            ('heading', 'heading-change error (deg)', (0.003, 180), {'s2': (0.0034, 62), 's1': (0.03, 30), 'rasr': (0.75, 60), 'ext': (9, 20)}),
            ('tilt', 'tilt-change error (deg)', (0.003, 60), {'s2': (0.0034, 30), 's1': (0.07, 16), 'rasr': (0.25, 40), 'ext': (4.6, 22)})]
    for a, (k, xl, lim, lab) in zip(ax, spec):
        for s in ('ext', 'rasr', 's1', 's2'):
            e = np.sort(E[f'{key[s]}|{k}']); a.plot(np.maximum(e, lim[0] * 0.5), 100 * (np.arange(len(e)) + 1) / len(e), color=C[s], lw=1.4, ls=LS[s], solid_capstyle='round')
        a.set_xscale('log'); a.set_xlim(*lim); a.set_ylim(0, 100); a.set_xlabel(xl); a.set_yticks([0, 25, 50, 75, 100])
    ax[0].set_ylabel('test pairs within error (%)')
    h = [plt.Line2D([], [], color=C[s], lw=1.4, ls=LS[s]) for s in ('s2', 's1', 'rasr', 'ext')]
    fig.legend(h, [NAMES[s] for s in ('s2', 's1', 'rasr', 'ext')], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.08); save(fig, 'fig_cdf')

def breakdown():
    B = json.load(open(f'{ROOT}/outputs/analysis/extended_metrics.json'))['breakdowns']
    spec = [('stage-1 zoom inliers', 'stage-I inlier matches'), ('|tilt change| (deg)', r'$|\Delta\phi|$ (deg)'), ('|heading change| (deg)', r'$|\Delta\theta|$ (deg)'), ('true distance (m)', 'distance to goal (m)')]
    fig, ax = plt.subplots(1, 4, figsize=(FULL, 1.7), sharey=True, gridspec_kw=dict(width_ratios=[5, 4, 5, 5]))
    for a, (k, xl) in zip(ax, spec):
        rows = B[k]; x = np.arange(len(rows))
        s1 = np.array([r['s1|pos3d'] for r in rows], float); s2 = np.array([r['s2|pos3d'] for r in rows], float)
        for xi, a1, a2 in zip(x, s1, s2): a.plot([xi, xi], [a1, a2], color=AX, lw=1.6, solid_capstyle='round', zorder=1)
        a.scatter(x, s1, s=22, color=C['s1'], zorder=3, edgecolor='white', linewidth=0.6, marker='s'); a.scatter(x, s2, s=24, color=C['s2'], zorder=3, edgecolor='white', linewidth=0.6)
        a.set_xticks(x); a.set_xticklabels([str(r['bin']).replace('>=', '≥').replace('-', '–') for r in rows], fontsize=6, rotation=30, ha='right', rotation_mode='anchor'); a.set_xlabel(xl); a.set_xlim(-0.6, len(rows) - 0.4); a.grid(axis='x', visible=False)
        for xi, r in zip(x, rows): a.text(xi, 0.985, f"{r['n']}", ha='center', va='top', fontsize=5.6, color=INK2, transform=a.get_xaxis_transform())
    ax[0].set_ylabel('mean 3-D position error (m)'); ax[0].set_ylim(0, 9.8); ax[0].set_yticks([0, 2, 4, 6, 8]); ax[3].text(1.0, 1.01, 'top row: number of pairs', fontsize=5.6, color=INK2, transform=ax[3].transAxes, ha='right', va='bottom')
    h = [plt.Line2D([], [], color=C['s1'], marker='s', ls='', ms=4.5), plt.Line2D([], [], color=C['s2'], marker='o', ls='', ms=4.8)]
    fig.legend(h, [NAMES['s1'], NAMES['s2']], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), columnspacing=1.6, handletextpad=0.3)
    fig.subplots_adjust(wspace=0.08); save(fig, 'fig_breakdown')

def reliability():
    R = json.load(open(f'{ROOT}/outputs/analysis/extended_metrics.json'))['reliability']
    tg = [('lx', r'$\ell_x$ (m)', C['s2'], '-', 'o'), ('z', r'$\Delta z$ (m)', C['s1'], (0, (5, 1.5)), 's'), ('heading', r'$\Delta\theta$ (deg)', C['rasr'], (0, (3, 1, 1, 1)), '^'), ('tilt', r'$\Delta\phi$ (deg)', C['ext'], (0, (1, 1)), 'D')]
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.85), gridspec_kw=dict(width_ratios=[1.0, 1]))
    for k, lab, c, ls, mk in tg:
        r = R[k]; ax[0].plot(100 * np.array(r['keep']), np.array(r['sparsification']) / r['mae_all'], color=c, lw=1.3, ls=ls, label=lab)
        ax[1].plot(r['bins_pred'], r['bins_actual'], color=c, lw=0.9, marker=mk, ms=2.6, mec='white', mew=0.3, label=lab)
    ax[0].set_xlabel('pairs kept (%), most confident first', fontsize=7); ax[0].set_ylabel('MAE of kept pairs / MAE of all', fontsize=7); ax[0].set_xlim(10, 100); ax[0].set_ylim(0, 1.02); ax[0].set_xticks([20, 40, 60, 80, 100])
    lo, hi = 8e-4, 4; ax[1].plot([lo, hi], [lo, hi], color=INK2, lw=0.5, ls=(0, (2, 2))); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
    ax[1].set_xlabel('predicted |error|', fontsize=7); ax[1].set_ylabel('actual |error|', fontsize=7); ax[1].set_aspect('equal', adjustable='box'); ax[1].set_xticks([1e-3, 1e-2, 1e-1, 1]); ax[1].set_yticks([1e-3, 1e-2, 1e-1, 1])
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.06), handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.5); save(fig, 'fig_reliability')

def real():
    D = pd.read_pickle('/root/realdata/mezieres/pred.pkl'); k = 60.0 / 220.18
    def pe(pre, sc): return np.sqrt((D[pre + 'lx'] * sc - D.lx)**2 + (D[pre + 'ly'] * sc - D.ly)**2 + (D[pre + 'z'] * sc - D.z)**2).values
    M = dict(s2=pe('s2_', k), s1=pe('s1_', k), rasr=pe('n_', k)); inr = (D.dist_equiv <= 38).values
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True)
    for a, m, t in ((ax[0], inr, 'within training range'), (ax[1], ~inr, 'beyond training range')):
        for s in ('rasr', 's1', 's2'):
            e = np.sort(M[s][m]); a.plot(e, 100 * (np.arange(len(e)) + 1) / len(e), color=C[s], lw=1.4, ls=LS[s])
        a.set_xscale('log'); a.set_xlim(0.1, 60); a.set_ylim(0, 100); a.set_yticks([0, 25, 50, 75, 100]); a.set_xlabel('3-D position error (m)'); a.set_title(t, fontsize=7)
    ax[0].set_ylabel('real pairs within error (%)')
    h = [plt.Line2D([], [], color=C[s], lw=1.4, ls=LS[s]) for s in ('s2', 's1', 'rasr')]
    fig.legend(h, ['DreamNav (stage II)', 'DreamNav (stage I)', 'RASR + sin/cos (rescaled)'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.22), handlelength=2.6, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.1); save(fig, 'fig_real')

def nav():
    """Self-terminating closed-loop flight: the policy stops at the first step (>= 1) where its own estimated distance to the goal is below tau."""
    f = f'{ROOT}/outputs/sim/nav_config.json'
    if not os.path.exists(f): print('closed-loop run not finished -> fig_nav skipped'); return
    L = pd.read_pickle(f'{ROOT}/outputs/sim/nav_log.pkl'); T = int(L.step.max()); taus = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5])
    def stop(p, tau):
        fin, steps = [], []
        for _, e in L[L.pol == p].sort_values(['ep', 'step']).groupby('ep'):
            k = next((int(r.step) for r in e.itertuples() if 1 <= r.step < T and r.pred_dist < tau), T); fin.append(float(e[e.step == k].pos_err.iloc[0])); steps.append(k)
        return np.array(fin), np.array(steps)
    fig, ax = plt.subplots(1, 2, figsize=(COL, 1.7))
    for p, s, mk in (('s2p', 's2', 'o'), ('s1', 's1', 's')):     # s2p = stage II with its short-range mode (the closed-loop configuration of the paper)
        r = [stop(p, t) for t in taus]; one = 100 * (L[(L.pol == p) & (L.step == 1)].pos_err < 1).mean()
        ax[0].plot(taus, [100 * (a < 1).mean() for a, _ in r], color=C[s], lw=1.3, ls=LS[s], marker=mk, ms=2.6, mec='white', mew=0.3, label=NAMES[s]); ax[0].axhline(one, color=C[s], lw=0.6, ls=(0, (1, 2)))
        ax[1].plot(taus, [b.mean() for _, b in r], color=C[s], lw=1.3, ls=LS[s], marker=mk, ms=2.6, mec='white', mew=0.3)
        print(p, 'one step SR', round(one, 1), {float(t): (round(100 * (a < 1).mean(), 1), round(float(np.median(a)), 2), round(float(b.mean()), 2)) for t, (a, b) in zip(taus, r) if t in (0.5, 1, 1.5, 2)})
    ax[0].text(5.0, 33, 'one step', fontsize=6, color=INK2, ha='right', va='top')
    ax[0].set_xlabel(r'stopping threshold $\tau$ (m)'); ax[0].set_ylabel('SR at 1 m (%)'); ax[0].set_ylim(0, 100); ax[0].set_yticks([0, 25, 50, 75, 100]); ax[1].set_xlabel(r'stopping threshold $\tau$ (m)'); ax[1].set_ylabel('mean steps flown'); ax[1].set_ylim(0, 5.2)
    for a in ax: a.set_xticks([1, 2, 3, 4, 5]); a.set_xlim(0.3, 5.2)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), handlelength=2.6, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.42); save(fig, 'fig_nav')

if __name__ == '__main__':
    todo = sys.argv[1:] or ['cdf', 'breakdown', 'reliability', 'real', 'nav']
    for t in todo: globals()[t]()
