#!/usr/bin/env python3
"""Track C figures (single-column width, PDF + PNG). Reads the arrays / JSON parts written by the analysis scripts."""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = '/root/rasr-pairuav/outputs/supp/trackC'; W = 3.3
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6,
                     'lines.linewidth': 1.4, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
C = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']     # Okabe-Ito
def save(fig, name): fig.savefig(f'{O}/{name}.pdf'); fig.savefig(f'{O}/{name}.png', dpi=300); plt.close(fig); print('wrote', name)
def cdf(ax, e, **kw): e = np.sort(np.asarray(e, float)); ax.plot(np.maximum(e, 1e-2), 100 * (np.arange(len(e)) + 1) / len(e), **kw)
A = np.load(f'{O}/cpu_arrays.npz', allow_pickle=True); P = json.load(open(f'{O}/part_cpu.json'))

# 1. error CDF on the test pairs
fig, ax = plt.subplots(figsize=(W, 2.3))
cdf(ax, A['err_single_raw'], color=C[5], ls='--', label='raw-view net, image $a$ only'); cdf(ax, A['err_single_bev'], color=C[4], ls='--', label='nadir-rectified net, image $a$ only')
cdf(ax, A['err_s1'], color=C[0], label='raw-view, fused $a{+}b$ (Stage I)'); cdf(ax, A['err_s2'], color=C[1], label='ensemble, fused $a{+}b$ (Stage II)')
ax.set_xscale('log'); ax.set_xlim(0.05, 180); ax.set_ylim(0, 100); ax.set_xlabel('absolute-heading error (deg)'); ax.set_ylabel('% of test pairs'); ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(['0.1', '1', '10', '100']); ax.legend(frameon=False, loc='lower right'); save(fig, 'fig_heading_cdf')

# 2. error vs proximity of training buildings (Stage II ensemble); bars = median, markers = mean with building-bootstrap CI
for key, fname, xl in (('by_distance_to_nearest_training_building', 'fig_heading_vs_distance', 'distance to the nearest training building'), ('by_site_training_buildings', 'fig_heading_vs_site_buildings', 'training buildings in the same site')):
    rows = P[key]; x = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.bar(x, [r['heading_s2']['median'] for r in rows], 0.6, color=C[5], label='median', zorder=2)
    m = np.array([r['heading_s2']['mean'] for r in rows]); lo = np.array([r['heading_s2']['mean_ci'][0] for r in rows]); hi = np.array([r['heading_s2']['mean_ci'][1] for r in rows])
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt='o', ms=3.5, color=C[1], capsize=2, lw=1, label='mean (95% CI)', zorder=3)
    ax.set_yscale('log'); ax.set_ylim(0.5, 120); ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(['1', '3', '10', '30', '100']); short = {'<100 m': '<0.1', '100-200 m': '0.1-0.2', '200-400 m': '0.2-0.4', '0.4-1 km': '0.4-1', '1-5 km': '1-5', '>5 km': '>5'}
    ax.set_xticks(x); ax.set_xticklabels([f"{short.get(r['bin'], r['bin'])}\n({r['buildings']})" for r in rows]); ax.set_xlabel(xl + ('; km' if key.startswith('by_distance') else '') + ' (number of test buildings)'); ax.set_ylabel('absolute-heading error (deg)')
    ax2 = ax.twinx(); ax2.plot(x, [100 * r['sr_epe_enu'] for r in rows], 's-', ms=3, color=C[2], lw=1, label='SR$_{EPE}$@1 m, ENU'); ax2.plot(x, [100 * r['sr_epe_cam'] for r in rows], '^--', ms=3, color=C[6], lw=1, label='SR$_{EPE}$@1 m, camera frame')
    ax2.set_ylim(0, 100); ax2.set_ylabel('success rate (%)'); ax2.grid(False); ax2.spines['right'].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.22), columnspacing=0.8, handletextpad=0.4); save(fig, fname)

# 3. risk-coverage: reject the pairs with the lowest fused concentration
e = A['err_s2']; k = A['kappa_s2']; epe = A['epe_enu']; ec = A['epe_cam']; o = np.argsort(-k); n = len(e); cov = np.linspace(0.2, 1.0, 81); idx = [o[:int(round(c * n))] for c in cov]; oo = np.argsort(e)
fig, ax = plt.subplots(figsize=(W, 2.3)); ax.plot(100 * cov, [e[i].mean() for i in idx], color=C[1], label='heading MAE, by confidence $\\kappa$'); ax.plot(100 * cov, [e[oo[:int(round(c * n))]].mean() for c in cov], color=C[1], ls=':', label='heading MAE, oracle ranking')
ax.set_xlabel('% of test pairs kept (most confident first)'); ax.set_ylabel('absolute-heading MAE (deg)'); ax.set_ylim(0, 6); ax2 = ax.twinx(); ax2.plot(100 * cov, [100 * (epe[i] < 1).mean() for i in idx], color=C[2], label='SR$_{EPE}$@1 m, ENU'); ax2.plot(100 * cov, [100 * (ec[i] < 1).mean() for i in idx], color=C[6], ls='--', label='SR$_{EPE}$@1 m, camera frame')
ax2.set_ylim(50, 90); ax2.set_ylabel('success rate (%)'); ax2.grid(False); ax2.spines['right'].set_visible(True); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.24), columnspacing=0.8, handletextpad=0.4); save(fig, 'fig_heading_risk_coverage')

# 4. site-disjoint re-training: error CDF on the held-out sites
if os.path.exists(f'{O}/retrain_arrays.npz'):
    B = np.load(f'{O}/retrain_arrays.npz', allow_pickle=True); hm = B['heldout']; fig, ax = plt.subplots(figsize=(W, 2.3))
    for key, lab, c, ls in (('err|raw-view | A archived', 'archived (other buildings of the site seen)', C[0], '-'), ('err|raw-view | C size-matched control', 'size-matched control (site seen)', C[2], '--'), ('err|raw-view | B site-disjoint', 'site-disjoint (site never seen)', C[1], '-')):
        if key in B: cdf(ax, B[key][hm], color=c, ls=ls, label=lab)
    ax.set_xscale('log'); ax.set_xlim(0.05, 180); ax.set_ylim(0, 100); ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(['0.1', '1', '10', '100']); ax.set_xlabel('absolute-heading error (deg), fused $a{+}b$'); ax.set_ylabel('% of pairs, held-out sites'); ax.legend(frameon=False, loc='lower right'); save(fig, 'fig_heading_sitedisjoint_cdf')
    # ENU endpoint error on the held-out sites
    fig, ax = plt.subplots(figsize=(W, 2.3)); cdf(ax, B['epe_cam'][hm], color=C[6], ls=':', label='camera frame (no heading)')
    for key, lab, c, ls in (('epe|raw-view | A archived', 'ENU, archived heading net', C[0], '-'), ('epe|raw-view | C size-matched control', 'ENU, size-matched control', C[2], '--'), ('epe|raw-view | B site-disjoint', 'ENU, site-disjoint heading net', C[1], '-')):
        if key in B: cdf(ax, B[key][hm], color=c, ls=ls, label=lab)
    ax.set_xscale('log'); ax.set_xlim(0.05, 60); ax.set_ylim(0, 100); ax.axvline(1.0, color='#999999', lw=0.6); ax.set_xticks([0.1, 1, 10]); ax.set_xticklabels(['0.1', '1', '10']); ax.set_xlabel('3-D endpoint error (m), Stage II'); ax.set_ylabel('% of pairs, held-out sites'); ax.legend(frameon=False, loc='lower right'); save(fig, 'fig_epe_sitedisjoint_cdf')

# 5. held-out sites: per-building signed heading bias, archived vs site-disjoint (one dot per test building)
if os.path.exists(f'{O}/retrain_arrays.npz'):
    B = np.load(f'{O}/retrain_arrays.npz', allow_pickle=True)
    if 'sg|raw-view | B site-disjoint' in B:
        hm = B['heldout']; tours = B['tour'][hm]; site = B['site'][hm]; ut = np.unique(tours); rng = np.random.default_rng(0)
        def cm(v): v = np.deg2rad(v); return np.rad2deg(np.arctan2(np.sin(v).mean(), np.cos(v).mean()))
        sites_ = sorted(set(site), key=lambda s: -len(set(tours[site == s]))); fig, ax = plt.subplots(figsize=(W, 2.3))
        for j, s in enumerate(sites_):
            tb = np.unique(tours[site == s])
            for key, c, dx, lab in (('sg|raw-view | A archived', C[0], -0.17, 'archived'), ('sg|raw-view | B site-disjoint', C[1], 0.17, 'site-disjoint')):
                y = np.array([cm(B[key][hm][tours == t]) for t in tb]); ax.scatter(j + dx + rng.uniform(-0.09, 0.09, len(y)), y, s=7, color=c, alpha=0.75, lw=0, label=lab if j == 0 else None, zorder=3)
        ax.axhline(0, color='#999999', lw=0.6); ax.set_xticks(range(len(sites_))); ax.set_xticklabels([f'{chr(65 + j)}\n({len(np.unique(tours[site == s]))})' for j, s in enumerate(sites_)]); ax.set_xlabel('held-out site (number of test buildings)'); ax.set_ylabel('signed heading error per building (deg)')
        ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.14), handletextpad=0.2, columnspacing=1.0); save(fig, 'fig_heading_sitedisjoint_per_building'); json.dump({chr(65 + j): int(s) for j, s in enumerate(sites_)}, open(f'{O}/fig_heading_sitedisjoint_per_building_sites.json', 'w'))
