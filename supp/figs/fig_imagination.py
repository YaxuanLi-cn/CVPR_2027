#!/usr/bin/env python3
"""Fig. S: what the plane-induced warp does and does not align. For a low-relief and a high-relief test pair:
source crop | target aligned by the rotation only | target warped with the ground-plane homography (the imagined view) | checkerboard of source and imagined view.
Poses are the coarse-stage estimates used by the method (guide.pkl); nothing uses ground truth."""
import sys, numpy as np, pandas as pd, cv2
sys.path.insert(0, '/root/rasr-pairuav/supp/figs'); sys.path.insert(0, '/root/rasr-pairuav/stage2_imagination'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
import style; style.use(); plt = style.plt
import imagine_match as im
DATA = '/root/pairUAV-5D-temp'; O = '/root/rasr-pairuav/outputs/supp/trackA'; S2 = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/stage2'
P = pd.read_pickle(f'{O}/planarity_pairs.pkl'); G = pd.read_pickle(f'{S2}/guide.pkl').set_index('manifest_index')
ok = (P.n_tri >= 1500) & (P.e1_epe < 0.6) & P.dist.between(18, 26) & (P.tilt_max < 30) & (np.abs(P.heading) < 150)
lo = P[ok & P.relief.between(0.05, 0.07)].sort_values('im_inl', ascending=False).iloc[0]; hi = P[ok & (P.relief > 0.28)].sort_values('relief', ascending=False).iloc[0]
print('low relief:', int(lo.manifest_index), round(lo.relief, 3), round(lo.struct, 1), round(lo.dist, 1), '| high relief:', int(hi.manifest_index), round(hi.relief, 3), round(hi.struct, 1), round(hi.dist, 1))
W0, H0 = 3686, 2073; CW, CH = 1100, 700
fig, ax = plt.subplots(2, 4, figsize=(2 * style.COL + 0.3, 2.5)); info = []
for row, r in enumerate((lo, hi)):
    g = G.loc[int(r.manifest_index)]; ia = cv2.cvtColor(cv2.imread(f'{DATA}/{g.image_a}'), cv2.COLOR_BGR2RGB); ib = cv2.cvtColor(cv2.imread(f'{DATA}/{g.image_b}'), cv2.COLOR_BGR2RGB)
    Hp = im.homography_b2a(g, True); Hr = im.homography_b2a(g, False); wp = cv2.warpPerspective(ib, Hp, (W0, H0)); wr = cv2.warpPerspective(ib, Hr, (W0, H0)); mask = cv2.warpPerspective(np.full(ib.shape[:2], 255, np.uint8), Hp, (W0, H0)) > 0
    ys, xs = np.nonzero(mask); cx = int(np.clip(np.median(xs), CW // 2, W0 - CW // 2)); cy = int(np.clip(np.median(ys), CH // 2, H0 - CH // 2)); sl = (slice(cy - CH // 2, cy + CH // 2), slice(cx - CW // 2, cx + CW // 2))
    a, p, q = ia[sl], wp[sl], wr[sl]; cb = a.copy(); t = 100
    for yy in range(0, CH, t):
        for xx in range(0, CW, t):
            if ((yy // t) + (xx // t)) % 2: cb[yy:yy + t, xx:xx + t] = p[yy:yy + t, xx:xx + t]
    for col, (img, ttl) in enumerate(((a, 'source'), (q, 'target, rotation only'), (p, 'target, plane warp'), (cb, 'source / plane warp'))):
        ax[row, col].imshow(img); ax[row, col].set_xticks([]); ax[row, col].set_yticks([]); ax[row, col].grid(False)
        for s in ax[row, col].spines.values(): s.set_visible(False)
        if row == 0: ax[row, col].set_title(ttl)
    ax[row, 0].set_ylabel(f'relief {100 * r.relief:.0f}%\n(structures {r.struct:.0f} m)'); info.append(dict(manifest_index=int(r.manifest_index), relief=float(r.relief), struct=float(r.struct), dist=float(r.dist), dz=float(r.z), dheading=float(r.heading), resid_px=float(r.im_disp_med), epe_coarse=float(r.e1_epe), epe_refined=float(r.e2_epe), inliers=float(r.im_inl)))
fig.subplots_adjust(left=0.06, right=0.998, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04); style.save(fig, 'fig_imagination'); import json; json.dump(info, open(f'{O}/fig_imagination.json', 'w'), indent=1); print(info)
