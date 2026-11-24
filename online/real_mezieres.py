#!/usr/bin/env python3
"""Zero-shot test on REAL drone photographs: HEIG-VD `p4rtk_mezieres` (DJI Phantom 4 RTK, 263 photos of a village,
RTK-fixed geotags + Metashape bundle-adjusted poses, CC BY 4.0; Hugging Face heig-vd-geo/scop-datasets).

  prep : parse the bundle-adjusted poses -> ENU position / heading / tilt / roll per photo (dataset conventions),
         resample every photo to the pipeline's canonical pinhole camera (undistorted, K0, 3686x2073), build pair lists
  run  : push the pairs through the online engine (no re-training, no fine-tuning, no test-time labels)
  eval : metrics
No model sees any label of this dataset. The only site-specific number used is the flight height above ground reported
by the aircraft itself (XMP RelativeAltitude), needed because the camera-height prior was calibrated at ~220 m.
"""
import sys, os, json, re, math, zipfile, xml.etree.ElementTree as ET
import numpy as np, pandas as pd, cv2
from pathlib import Path
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion'); sys.path.insert(0, '/root/rasr-pairuav/online')
from geom import R_wc
from pose_from_matches import K0
RAW = Path('/root/realdata/mezieres/raw/datasets/p4rtk_mezieres'); OUT = Path('/root/realdata/mezieres'); CAN = OUT / 'canonical'; W0, H0 = 3686, 2073
H_TRAIN = 220.18   # camera height above ground the pipeline's scale prior was calibrated at (median over well-matched train pairs)

def ecef_to_enu_R(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0], [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
def lla_to_ecef(lat, lon, h):
    a, e2 = 6378137.0, 6.69437999014e-3; la, lo = np.deg2rad(lat), np.deg2rad(lon); N = a / np.sqrt(1 - e2 * np.sin(la)**2)
    return np.array([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - e2) + h) * np.sin(la)])

def parse():
    z = zipfile.ZipFile(RAW / '_ms_p4/p4.files/0/chunk.zip'); r = ET.fromstring(z.read('doc.xml'))
    tr = r.find('transform'); Rc = np.array(tr.find('rotation').text.split(), float).reshape(3, 3); Tc = np.array(tr.find('translation').text.split(), float); sc = float(tr.find('scale').text)
    cal = [c for c in r.iter('calibration') if c.get('class') == 'adjusted'][0]; g = lambda k: float(cal.find(k).text) if cal.find(k) is not None else 0.0
    calib = dict(f=g('f'), cx=g('cx'), cy=g('cy'), k1=g('k1'), k2=g('k2'), k3=g('k3'), p1=g('p1'), p2=g('p2'), b1=g('b1'), b2=g('b2'), w=5472, h=3648)
    rows = []
    for c in r.iter('camera'):
        if c.find('transform') is None: continue
        T = np.array(c.find('transform').text.split(), float).reshape(4, 4); ref = c.find('reference')
        rows.append(dict(label=c.get('label'), C_ecef=sc * Rc @ T[:3, 3] + Tc, R_ecef=Rc @ T[:3, :3], lon=float(ref.get('x')), lat=float(ref.get('y')), alt=float(ref.get('z')), exif_yaw=float(ref.get('yaw')), exif_pitch=float(ref.get('pitch'))))
    lat0, lon0 = np.mean([x['lat'] for x in rows]), np.mean([x['lon'] for x in rows]); E = ecef_to_enu_R(lat0, lon0); O = np.mean([x['C_ecef'] for x in rows], 0); out = []
    for x in rows:
        p = E @ (x['C_ecef'] - O); R = E @ x['R_ecef']                          # camera (x right, y down, z forward) -> ENU
        right, down, fwd = R[:, 0], R[:, 1], R[:, 2]; tilt = np.rad2deg(np.arccos(np.clip(-fwd[2], -1, 1)))
        head = np.rad2deg(np.arctan2(fwd[0], fwd[1])) if tilt > 2 else np.rad2deg(np.arctan2(-right[1], right[0])); roll = np.rad2deg(np.arcsin(np.clip(right[2], -1, 1)))
        rtk = E @ (lla_to_ecef(x['lat'], x['lon'], x['alt']) - O)             # RTK geotag (antenna/camera position written by the aircraft)
        out.append(dict(image=x['label'], flight=x['label'][:8], e=p[0], n=p[1], u=p[2], heading=head % 360, tilt=tilt, roll=roll, rtk_e=rtk[0], rtk_n=rtk[1], rtk_u=rtk[2], exif_yaw=x['exif_yaw'] % 360, exif_tilt=x['exif_pitch']))
    return pd.DataFrame(out), calib

def canonical_maps(c):
    """For every canonical pixel (ideal pinhole K0, 3686x2073): where to sample in the distorted source photo (Metashape frame-camera model)."""
    u, v = np.meshgrid(np.arange(W0, dtype=np.float64), np.arange(H0, dtype=np.float64)); x = (u - K0[0, 2]) / K0[0, 0]; y = (v - K0[1, 2]) / K0[1, 1]; r2 = x * x + y * y
    rad = 1 + c['k1'] * r2 + c['k2'] * r2**2 + c['k3'] * r2**3; xd = x * rad + c['p1'] * (r2 + 2 * x * x) + 2 * c['p2'] * x * y; yd = y * rad + c['p2'] * (r2 + 2 * y * y) + 2 * c['p1'] * x * y
    return (c['w'] * 0.5 + c['cx'] + xd * c['f'] + xd * c['b1'] + yd * c['b2'] - 0.5).astype(np.float32), (c['h'] * 0.5 + c['cy'] + yd * c['f'] - 0.5).astype(np.float32)

def _conv(a):
    src, dst, mx, my = a
    if not os.path.exists(dst): cv2.setNumThreads(2); cv2.imwrite(dst, cv2.remap(cv2.imread(src), mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    return dst

def prep():
    P, calib = parse(); CAN.mkdir(exist_ok=True); print('cameras with adjusted pose:', len(P), '| calibration', {k: round(v, 5) for k, v in calib.items()})
    d = np.sqrt((P.e - P.rtk_e)**2 + (P.n - P.rtk_n)**2 + (P.u - P.rtk_u)**2); dy = (P.heading - P.exif_yaw + 180) % 360 - 180
    print(f'bundle-adjusted centre vs RTK geotag: median {d.median():.3f} m, p95 {d.quantile(.95):.3f} m | heading vs XMP gimbal yaw: median |diff| {dy.abs().median():.2f} deg | tilt {P.tilt.min():.1f}..{P.tilt.max():.1f} deg | |roll| median {P.roll.abs().median():.2f}, max {P.roll.abs().max():.2f} deg')
    print('height above take-off (XMP RelativeAltitude) = 60 m; camera altitude spread:', round(P.u.std(), 2), 'm')
    mx, my = canonical_maps(calib); print('source pixel range used: x', float(mx.min()), float(mx.max()), 'y', float(my.min()), float(my.max()))
    from multiprocessing import Pool
    jobs = [(str(RAW / 'images' / r.flight / f'{r.image}.JPG'), str(CAN / f'{r.image}.jpg'), mx, my) for r in P.itertuples()]
    with Pool(12) as p: p.map(_conv, jobs)
    P['path'] = [str(CAN / f'{i}.jpg') for i in P.image]; P.to_pickle(OUT / 'cameras.pkl')
    # ordered pairs: every (a, b) whose camera centres are within RANGE metres; labels follow the dataset convention (b minus a)
    X = P[['e', 'n', 'u']].values; D = np.linalg.norm(X[:, None] - X[None], axis=2); rows = []
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j and D[i, j] <= 30.0:
                a, b = P.iloc[i], P.iloc[j]; dx, dy_, dz = X[j] - X[i]; ha = np.deg2rad(a.heading)
                rows.append(dict(a=a.image, b=b.image, image_a=a.path, image_b=b.path, x=dx, y=dy_, z=dz, dist=D[i, j], heading=(b.heading - a.heading + 180) % 360 - 180, tilt=b.tilt - a.tilt, heading_a=a.heading, heading_b=b.heading,
                                 tilt_a=a.tilt, tilt_b=b.tilt, lx=np.cos(ha) * dx - np.sin(ha) * dy_, ly=np.sin(ha) * dx + np.cos(ha) * dy_, same_flight=a.flight == b.flight))
    Q = pd.DataFrame(rows); Q['dist_equiv'] = Q.dist * H_TRAIN / 60.0; Q.to_pickle(OUT / 'pairs_all.pkl')
    print('ordered pairs within 30 m:', len(Q), '| distance quantiles (m):', Q.dist.quantile([.05, .25, .5, .75, .95]).round(1).tolist(), '| |heading change| quantiles:', Q.heading.abs().quantile([.25, .5, .75, .95]).round(1).tolist())
    print('pairs by |heading change| bin:', pd.cut(Q.heading.abs(), [0, 15, 60, 120, 181]).value_counts().sort_index().to_dict())

def run(n=600, seed=0):
    from engine import Engine
    Q = pd.read_pickle(OUT / 'pairs_all.pkl')
    # stratified sample: half inside the baseline/height range seen in training (<= 38 m at 220 m), half beyond it
    rng = np.random.default_rng(seed); inr = Q[Q.dist_equiv <= 38]; outr = Q[Q.dist_equiv > 38]
    S = pd.concat([inr.sample(min(len(inr), n // 2), random_state=seed), outr.sample(min(len(outr), n - min(len(inr), n // 2)), random_state=seed)]).reset_index(drop=True)
    print('pairs:', len(S), '| in training range:', int((S.dist_equiv <= 38).sum())); eng = Engine(); R = []
    for i in range(0, len(S), 100): R.append(eng.predict(S.iloc[i:i + 100][['image_a', 'image_b']]))
    eng.close(); R = pd.concat(R, ignore_index=True).drop(columns=['manifest_index', 'image_a', 'image_b']); pd.concat([S, R], axis=1).to_pickle(OUT / 'pred.pkl'); print('saved', OUT / 'pred.pkl')

def evaluate():
    D = pd.read_pickle(OUT / 'pred.pkl'); wrap = lambda a: (np.asarray(a) + 180) % 360 - 180; k = 60.0 / H_TRAIN; res = {}; L = []
    def world(h, lx, ly): h = np.deg2rad(h); return np.cos(h) * lx + np.sin(h) * ly, -np.sin(h) * lx + np.cos(h) * ly
    def block(d, name):
        out = {}; gd = d.dist.values
        def row(lx, ly, z, tilt, head, absh=None, scale=k):
            lx, ly, z = np.asarray(lx) * scale, np.asarray(ly) * scale, np.asarray(z) * scale; pe = np.sqrt((lx - d.lx)**2 + (ly - d.ly)**2 + (z - d.z)**2); pdist = np.sqrt(lx**2 + ly**2 + z**2)
            cos = (lx * d.lx + ly * d.ly + z * d.z) / (pdist.clip(1e-9) * gd); he = np.abs(wrap(head - d.heading)); te = np.abs(tilt - d.tilt)
            r = dict(lx=float(np.abs(lx - d.lx).mean()), ly=float(np.abs(ly - d.ly).mean()), z=float(np.abs(z - d.z).mean()), distance=float(np.abs(pdist - gd).mean()), dist_rel=float((np.abs(pdist - gd) / gd).mean()), heading=float(he.mean()), heading_med=float(np.median(he)),
                     tilt=float(te.mean()), tilt_med=float(np.median(te)), pos3d=float(pe.mean()), pos3d_med=float(np.median(pe)), pos_rel=float((pe / gd).mean()), tdir_med=float(np.median(np.rad2deg(np.arccos(np.clip(cos, -1, 1))))),
                     succ_1m=float((pe < 1).mean()), succ_2m=float((pe < 2).mean()), succ_10pct=float((pe / gd < 0.10).mean()), succ_1m_1deg=float(((pe < 1) & (he < 1) & (te < 1)).mean()))
            if absh is not None:
                x, y = world(absh, lx, ly); r['x_world'] = float(np.abs(x - d.x).mean()); r['y_world'] = float(np.abs(y - d.y).mean()); r['abs_heading_err_med'] = float(np.median(np.abs(wrap(absh - d.heading_a))))
            return r
        out['DreamNav stage 2 (refined)'] = row(d.s2_lx, d.s2_ly, d.s2_z, d.s2_tilt, d.s2_heading, d.s2_absh)
        out['DreamNav stage 1 (coarse)'] = row(d.s1_lx, d.s1_ly, d.s1_z, d.s1_tilt, d.s1_heading, d.s1_absh)
        out['DreamNav stage 2, no altitude input'] = row(d.s2_lx, d.s2_ly, d.s2_z, d.s2_tilt, d.s2_heading, scale=1.0)
        out['MASt3R-stat MLP (RASR + sin/cos)'] = row(d.n_lx, d.n_ly, d.n_z, d.n_tilt, d.n_heading, scale=1.0)
        out['MASt3R-stat MLP, altitude-scaled'] = row(d.n_lx, d.n_ly, d.n_z, d.n_tilt, d.n_heading)
        z0 = np.zeros(len(d)); out['Constant (zero motion)'] = row(z0, z0, z0, z0, z0)
        res[name] = dict(n=int(len(d)), mean_gt_distance=float(gd.mean()), methods=out)
        L.extend([f'\n### {name} ({len(d)} pairs, mean true distance {gd.mean():.1f} m)\n', '| Method | pos. err mean (m) | median | rel. pos. err | <1 m | <2 m | <10% of dist. | heading MAE (deg) | median | tilt MAE (deg) | transl. dir. median (deg) | dist. rel. err |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for m, r in out.items(): L.append(f"| {m} | {r['pos3d']:.2f} | {r['pos3d_med']:.2f} | {100*r['pos_rel']:.1f}% | {100*r['succ_1m']:.1f}% | {100*r['succ_2m']:.1f}% | {100*r['succ_10pct']:.1f}% | {r['heading']:.3f} | {r['heading_med']:.3f} | {r['tilt']:.3f} | {r['tdir_med']:.2f} | {100*r['dist_rel']:.1f}% |")
    block(D, 'All sampled pairs'); block(D[D.dist_equiv <= 38], 'Baseline/height inside the training range'); block(D[D.dist_equiv > 38], 'Baseline/height beyond the training range')
    block(D[D.heading.abs() > 60], 'Large heading change (> 60 deg)'); block(D[D.heading.abs() <= 15], 'Small heading change (<= 15 deg)')
    a = res['All sampled pairs']['methods']['DreamNav stage 2 (refined)']
    L.append(f"\nWorld-frame x / y with the learned absolute-heading network (all pairs): MAE {a['x_world']:.2f} / {a['y_world']:.2f} m; median absolute-heading error {a['abs_heading_err_med']:.1f} deg.")
    L.append(f"Diagnostics: pass-1 inliers median {D.sg_inl.median():.0f}, zoom {D.zm_inl.median():.0f}, imagination re-matching {D.im_inl.median():.0f}; imagined-vs-real residual median {D.im_disp_med.median():.1f} px.")
    json.dump(res, open(OUT / 'real_metrics.json', 'w'), indent=1); open(OUT / 'real_metrics.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

if __name__ == '__main__':
    {'prep': prep, 'run': run, 'eval': evaluate}[sys.argv[1]]()
