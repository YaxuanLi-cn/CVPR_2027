#!/usr/bin/env python3
"""Fig. S: the absolute-heading prior depends on having seen the site.
(a) cumulative distribution of the fused heading error on the six held-out sites, for the archived raw-view network (which saw other
    buildings of these sites) and for the same network re-trained without them;
(b) endpoint success of the refined stage by distance from the test building to the nearest training building, in ENU with the learned
    heading (archived networks) and in the camera frame, which needs no heading.
Data: outputs/supp/trackC/retrain_arrays.npz, heading.json. Output: docs/paper/supp/figures/fig_heading.{pdf,png}"""
import sys, json, numpy as np
sys.path.insert(0, '/root/rasr-pairuav/supp/figs'); import style as S
S.use(); plt = S.plt
O = '/root/rasr-pairuav/outputs/supp/trackC'; B = np.load(f'{O}/retrain_arrays.npz', allow_pickle=True); H = json.load(open(f'{O}/heading.json')); hm = B['heldout']
fig, ax = plt.subplots(1, 2, figsize=(2 * S.COL + 0.3, 1.9), gridspec_kw=dict(wspace=0.28))
# (a) CDF, one axis, direct labels
a = ax[0]
for key, lab, c, xy in (('err|raw-view | A archived', 'site seen through\nother buildings', S.BLUE, (0.062, 78)), ('err|raw-view | B site-disjoint', 'site never seen', S.ORANGE, (15, 28))):
    e = np.sort(np.maximum(B[key][hm], 2e-2)); a.plot(e, 100 * (np.arange(len(e)) + 1) / len(e), color=c, lw=1.5); a.text(xy[0], xy[1], lab, color=S.INK, ha='left', va='center', fontsize=6.8, linespacing=1.0)
    med = float(np.median(B[key][hm])); a.plot([med], [50], 'o', ms=3.6, mfc=c, mec='white', mew=0.8, zorder=5)
a.set_xscale('log'); a.set_xlim(0.05, 180); a.set_ylim(0, 100); a.set_xticks([0.1, 1, 10, 100]); a.set_xticklabels(['0.1', '1', '10', '100']); a.minorticks_off()
a.set_xlabel('heading error of the prior ($^\\circ$)'); a.set_ylabel('pairs of the held-out sites (%)'); a.set_title('(a) six sites removed from training', loc='left', fontsize=7.5)
# (b) success by distance to the nearest training building, one axis
b = ax[1]; D = H['test_set']['by_distance_to_nearest_training_building']; x = np.arange(len(D)); lab = [d['bin'].replace(' m', '').replace(' km', '').replace('<100', '<0.1').replace('100-200', '0.1-0.2').replace('200-400', '0.2-0.4') for d in D]
cam = [100 * d['sr_epe_cam'] for d in D]; enu = [100 * d['sr_epe_enu'] for d in D]
b.plot(x, cam, color=S.INK2, lw=1.2, ls=(0, (4, 2)), marker='o', ms=3.2, mfc=S.INK2, mec='white', mew=0.7); b.plot(x, enu, color=S.BLUE, lw=1.5, marker='o', ms=3.6, mfc=S.BLUE, mec='white', mew=0.8)
b.text(2.5, 92, 'camera frame (no heading needed)', color=S.INK, ha='center', va='center', fontsize=6.8); b.text(-0.25, 11, 'ENU, learned heading', color=S.INK, ha='left', va='center', fontsize=6.8)
b.set_xticks(x); b.set_xticklabels([f'{l}\n({d["buildings"]})' for l, d in zip(lab, D)]); b.set_ylim(0, 100); b.set_xlim(-0.4, len(D) - 0.6)
b.set_xlabel('distance to the nearest training building (km)'); b.set_ylabel('SR$_{\\mathrm{EPE}}$@1 m (%)'); b.set_title('(b) networks of the paper, all test buildings', loc='left', fontsize=7.5)
for k in ax: k.grid(axis='x', visible=False)
S.save(fig, 'fig_heading')
json.dump(dict(median_seen=float(np.median(B['err|raw-view | A archived'][hm])), median_unseen=float(np.median(B['err|raw-view | B site-disjoint'][hm])), bins=[dict(bin=d['bin'], buildings=d['buildings'], cam=c, enu=e) for d, c, e in zip(D, cam, enu)]), open(f'{O}/../trackA/fig_heading.json', 'w'), indent=1)
