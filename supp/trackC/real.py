#!/usr/bin/env python3
"""Track C, item 5: the archived absolute-heading networks on the real Mezieres photographs (DJI Phantom 4 RTK, 60 m above
ground, about 30 deg off nadir, two flights of one survey => essentially one sun azimuth). True heading from the
bundle-adjusted poses. Also: fused per-pair heading and the ENU pose error of the 600 evaluated pairs (archived pred.pkl).
Output: outputs/supp/trackC/part_real.json      Usage: real.py [--dev cuda]
"""
import sys, json, argparse, numpy as np, pandas as pd, cv2
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/supp/trackC')
from common import wrap, world, OUT
from nets import Heading, combine
ap = argparse.ArgumentParser(); ap.add_argument('--dev', default='cuda'); ap.add_argument('--bs', type=int, default=48); ap.add_argument('--limit', type=int, default=None); a = ap.parse_args()
O = f'{OUT}/trackC'; REAL = '/root/realdata/mezieres'; P = pd.read_pickle(f'{REAL}/cameras.pkl')
if a.limit: P = P.iloc[:a.limit].copy()
def load(p): im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4); return cv2.cvtColor(cv2.resize(im, (448, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)    # same as online/engine.py::_small
X = np.stack([load(p) for p in P.path]); H = Heading(dev=a.dev, bs=a.bs); gt = P.heading.values
def circ(d): d = np.deg2rad(np.asarray(d, float)); s, c = np.sin(d).mean(), np.cos(d).mean(); return float(np.rad2deg(np.arctan2(s, c))), float(np.hypot(s, c))
def st(h, m=None):
    m = np.ones(len(h), bool) if m is None else m; sg = wrap(h - gt)[m]; e = np.abs(sg); mu, R = circ(sg)
    return dict(n=int(m.sum()), mae=float(e.mean()), median=float(np.median(e)), lt10=float((e < 10).mean()), lt30=float((e < 30).mean()), lt90=float((e < 90).mean()), signed_circular_mean=mu, resultant_length=R,
                mae_after_removing_constant_offset=float(np.abs(wrap(sg - mu)).mean()), median_after_removing_constant_offset=float(np.median(np.abs(wrap(sg - mu)))))
hr, kr = H.raw_view(X); hb, kb = H.nadir(X, P.tilt.values); he, ke = combine((hr, kr), (hb, kb)); R = {'photos': int(len(P)), 'flights': P.flight.value_counts().to_dict(), 'tilt_deg': dict(mean=float(P.tilt.mean()), min=float(P.tilt.min()), max=float(P.tilt.max())),
     'heading_range_deg': [float(gt.min()), float(gt.max())], 'chance_level': dict(mae=90.0, median=90.0), 'single_image': {}}
for nm, h, k in (('raw-view', hr, kr), ('nadir-rectified', hb, kb), ('ensemble', he, ke)):
    R['single_image'][nm] = dict(all=st(h), **{f'flight {f}': st(h, (P.flight == f).values) for f in sorted(P.flight.unique())}, kappa_median=float(np.median(k)))
    print(nm, json.dumps(R['single_image'][nm]['all']))
# kappa on the rendered test images, for reference (does the network know it is unsure?)
R['kappa_reference_rendered_test_median'] = None
try:
    A = pd.read_pickle('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading/pred_cnx_s_d8.pkl'); R['kappa_reference_rendered_test_median'] = float(A[A.part == 'test'].ah_kappa.median())
except Exception as e: print('kappa ref skipped', e)
P['raw_h'] = hr; P['raw_k'] = kr; P['bev_h'] = hb; P['bev_k'] = kb; P[['image', 'flight', 'heading', 'tilt', 'raw_h', 'raw_k', 'bev_h', 'bev_k']].to_pickle(f'{O}/real_single_image_pred.pkl')
# ---- the 600 evaluated pairs (archived engine output): fused heading of camera a and what it does to the ENU displacement
if not a.limit:
    D = pd.read_pickle(f'{REAL}/pred.pkl'); k = 60.0 / 220.18; Rp = {}
    for nm, m in (('all 600 pairs', np.ones(len(D), bool)), ('within training range (300)', (D.dist_equiv <= 38).values), ('beyond training range (300)', (D.dist_equiv > 38).values)):
        d = D[m]; r = {}
        for st_, pre in (('Stage I', 's1'), ('Stage II', 's2')):
            sg = wrap(d[f'{pre}_absh'] - d.heading_a); e = np.abs(sg); mu, Rl = circ(sg); lx, ly, z = d[f'{pre}_lx'].values * k, d[f'{pre}_ly'].values * k, d[f'{pre}_z'].values * k
            ecam = np.sqrt((lx - d.lx)**2 + (ly - d.ly)**2 + (z - d.z)**2); x, y = world(d[f'{pre}_absh'].values, lx, ly); eenu = np.sqrt((x - d.x)**2 + (y - d.y)**2 + (z - d.z)**2)
            r[st_] = dict(fused_heading=dict(mae=float(e.mean()), median=float(np.median(e)), lt10=float((e < 10).mean()), lt30=float((e < 30).mean()), signed_circular_mean=mu, resultant_length=Rl),
                          epe_cam_median=float(np.median(ecam)), epe_cam_mean=float(ecam.mean()), sr_epe_cam_1m=float((ecam < 1).mean()), epe_enu_median=float(np.median(eenu)), epe_enu_mean=float(eenu.mean()), sr_epe_enu_1m=float((eenu < 1).mean()),
                          x_mae_enu=float(np.abs(x - d.x).mean()), y_mae_enu=float(np.abs(y - d.y).mean()), lx_mae_cam=float(np.abs(lx - d.lx).mean()), ly_mae_cam=float(np.abs(ly - d.ly).mean()))
        r['true_distance_mean_m'] = float(d.dist.mean()); Rp[nm] = r; print(nm, json.dumps(r['Stage II']))
    R['pairs'] = Rp
json.dump(R, open(f'{O}/part_real.json', 'w'), indent=1)
