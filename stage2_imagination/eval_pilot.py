import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from fusion_lib import *
P = add_targets(pd.read_pickle(f'{SG}/pairs.pkl')); Z = pd.read_pickle(f'{SG}/geom_zoom_half.pkl'); U = pd.read_pickle(f'{SG}/stage2/guide.pkl')
def stats(D, pre, ok):
    m = (D[ok] == 1).values; d = D[m]; eh = np.abs(wrap(d[f'{pre}_dh'] - d.heading)); et = np.abs(d[f'{pre}_dtilt'] - d.tilt)
    cs = (d[f'{pre}_vx'] * d.ux + d[f'{pre}_vy'] * d.uy + d[f'{pre}_vz'] * d.uz).clip(-1, 1); ed = np.rad2deg(np.arccos(cs))
    return f'ok {m.mean():.3f} | heading med/mean {np.median(eh):.3f}/{eh.mean():.3f} | tilt {np.median(et):.3f}/{et.mean():.3f} | dir {np.median(ed):.2f}/{ed.mean():.2f}', eh, et, ed, m
for f in sys.argv[1:]:
    I = pd.read_pickle(f); D = P.merge(I, on='manifest_index').merge(Z, on='manifest_index').merge(U[['manifest_index', 'p_dist', 'g_H']], on='manifest_index')
    print(f'=== {f}  ({len(D)} pairs)  tiles/pair {D.im_ntiles.mean():.1f}  zoom {D.im_zoom.median():.2f}  matches {D.im_n.median():.0f}  inliers median: zoom-pass {D.zm_inl.median():.0f} -> imagination {D.im_inl.median():.0f}; <100 inliers: {(D.zm_inl.fillna(0)<100).mean():.3f} -> {(D.im_inl.fillna(0)<100).mean():.3f}')
    print(f'    imagination-vs-reality displacement (orig px): median {D.im_disp_med.median():.1f}, share of matches <8px: {D.im_disp_lt8.median():.2f}')
    for nm, pre, ok in (('zoom E   ', 'zm', 'zm_ok'), ('zoom MAP ', 'zmap', 'zmap_ok'), ('imag E   ', 'im', 'im_ok'), ('imag MAP ', 'imap', 'imap_ok')): print('   ', nm, stats(D, pre, ok)[0])
    # scale: distance from the camera-height prior, H0 / (ground depth in baseline units); offset calibrated by the median on these rows (pilot only)
    for nm, c in (('zoom MAP', 'zmap_hgta'), ('imag MAP', 'imap_hgta')):
        s = -np.log(D[c].clip(lower=1e-3)); okc = np.isfinite(s); off = np.nanmedian((D.logd - s)[okc]); e = np.abs(D.logd - s - off)[okc]; print(f'    height-prior log-distance error {nm}: median {e.median():.4f} mean {e.mean():.4f} (n={okc.sum()})')
    g = D.groupby(pd.cut(D.zm_inl.fillna(0), [-1, 100, 200, 400, 800, 1e5]))
    for k, d in g:
        if len(d) < 5: continue
        a = stats(d, 'zmap', 'zmap_ok'); b = stats(d, 'imap', 'imap_ok'); print(f'    zoom-inl {str(k):18s} n={len(d):4d} | zoom MAP dir {np.median(a[3]):.2f}/{a[3].mean():.2f} tilt {a[2].mean():.3f} head {a[1].mean():.3f} | imag MAP dir {np.median(b[3]):.2f}/{b[3].mean():.2f} tilt {b[2].mean():.3f} head {b[1].mean():.3f} | inl {d.zm_inl.median():.0f}->{d.im_inl.median():.0f}')
