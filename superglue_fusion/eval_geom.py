import sys, pandas as pd, numpy as np
g = pd.read_pickle(sys.argv[1]).merge(pd.read_pickle('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/pairs.pkl'), on='manifest_index')
g['e_h'] = np.abs((g.sg_dh - g.heading + 180) % 360 - 180); g['e_t'] = np.abs(g.sg_dtilt - g.tilt)
d = np.sqrt(g.x**2 + g.y**2 + g.z**2); ha = np.deg2rad(g.heading_a)
lx = np.cos(ha) * g.x - np.sin(ha) * g.y; ly = np.sin(ha) * g.x + np.cos(ha) * g.y
g['e_ang'] = np.rad2deg(np.arccos(np.clip((g.sg_vx * lx + g.sg_vy * ly + g.sg_vz * g.z) / d, -1, 1))); g['e_vz'] = np.abs(g.sg_vz - g.z / d)
g['e_ta'] = np.abs(g.sg_ta - g.tilt_a)
print(f'n={len(g)} ok={g.sg_ok.mean():.3f}  inl>=50: {(g.sg_inl>=50).mean():.3f}  inl>=150: {(g.sg_inl>=150).mean():.3f}  median inl {g.sg_inl.median():.0f}')
q = g[g.sg_ok == 1]
print(f'ALL ok: heading med/mean {q.e_h.median():.3f}/{q.e_h.mean():.3f}  tilt {q.e_t.median():.3f}/{q.e_t.mean():.3f}  dir {q.e_ang.median():.2f}/{q.e_ang.mean():.2f}  vz {q.e_vz.median():.3f}/{q.e_vz.mean():.3f}  tilt_a {q.e_ta.median():.2f}/{q.e_ta.mean():.2f}')
for lo, hi in ((0, 20), (20, 50), (50, 150), (150, 400), (400, 5000)):
    q = g[(g.sg_inl >= lo) & (g.sg_inl < hi) & (g.sg_ok == 1)]
    if len(q): print(f' inl[{lo},{hi}) n={len(q)} heading {q.e_h.median():.3f}/{q.e_h.mean():.3f}  tilt {q.e_t.median():.3f}/{q.e_t.mean():.3f}  dir {q.e_ang.median():.2f}/{q.e_ang.mean():.2f} vz {q.e_vz.median():.3f}/{q.e_vz.mean():.3f}')
g['ah'] = g.heading.abs()
for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 150), (150, 180)):
    q = g[(g.ah >= lo) & (g.ah < hi)]; print(f' |dh| [{lo},{hi}) n={len(q)} median inl {q.sg_inl.median():.0f}  inl>=50 {(q.sg_inl>=50).mean():.2f}  heading-err>5deg {(~(q.e_h<5)).mean():.3f}')
