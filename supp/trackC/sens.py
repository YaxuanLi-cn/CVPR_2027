#!/usr/bin/env python3
"""Track C, item 4: photometric sensitivity of the absolute-heading networks (inference only) + mechanism checks.

Images: a fixed random sample (seed 0) of the unique test images referenced by the test pairs. Perturbations are applied to
the cached network input (uint8 RGB, 448 x 256), i.e. at the scale the networks see. One network per process (--net raw|bev)
so that the two can run side by side; per-image predictions are saved and summarised by sens_combine.py (which also forms
the sum of natural parameters of the two networks, the single-image part of the Stage II ensemble).
Usage: sens.py --net raw|bev --tag main [--n 2000] [--subset all|heldout] [--conds all|focus] [--raw-weights model.pt] [--zoom] [--mech]
"""
import sys, json, argparse, numpy as np, pandas as pd, cv2
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, '/root/rasr-pairuav/supp/trackC')
from common import wrap, SG, S2, OUT
from nets import Heading, RAW_W, BEV_W
ap = argparse.ArgumentParser(); ap.add_argument('--net', required=True, choices=['raw', 'bev']); ap.add_argument('--tag', required=True); ap.add_argument('--n', type=int, default=2000); ap.add_argument('--dev', default='cuda'); ap.add_argument('--bs', type=int, default=48)
ap.add_argument('--raw-weights', default=None, help='raw-view network to test instead of the archived one'); ap.add_argument('--subset', default='all', choices=['all', 'heldout', 'other']); ap.add_argument('--conds', default='all', choices=['all', 'focus'])
ap.add_argument('--zoom', action='store_true', help='raw only: centre zooms of the original image'); ap.add_argument('--mech', action='store_true', help='raw only: single-view mirror / rotation checks'); a = ap.parse_args()
O = f'{OUT}/trackC'; D = f'{SG}/absheading'; m1 = pd.read_pickle(f'{D}/images.pkl'); I1 = np.load(f'{D}/images_448x256.npy', mmap_mode='r'); row = pd.Series(np.arange(len(m1)), index=m1.image.values)
it = pd.read_pickle(f'{S2}/pred_bev_s.pkl'); it = it[it.part == 'test'].drop_duplicates('image').reset_index(drop=True)          # unique test images of the test pairs, with the tilt estimate of their first pair
if a.subset != 'all':
    ST = pd.read_pickle(f'{O}/sites.pkl').set_index('tour'); hk = ST.heldout.reindex(it.tour).values.astype(bool); it = it[hk if a.subset == 'heldout' else ~hk].reset_index(drop=True)
S = it.sample(min(a.n, len(it)), random_state=0).sort_values('image').reset_index(drop=True); X = np.stack([I1[i] for i in row.reindex(S.image).values]); gt = S.heading_gt.values; n = len(S)
H = Heading(raw=(a.raw_weights or RAW_W) if a.net == 'raw' else None, bev=BEV_W if a.net == 'bev' else None, dev=a.dev, bs=a.bs); cv2.setNumThreads(2)

def per_image(f): return lambda x: np.stack([f(im) for im in x])
def _lab(im, fn):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB); lab[..., 0] = fn(lab[..., 0]); return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
def _lift(L):
    L = L.astype(np.float32); med = max(float(np.median(L)), 1.0); return np.clip(np.where(L < med, med * (L / med) ** 0.5, L), 0, 255).astype(np.uint8)
def _hue(im, deg):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); hsv[..., 0] = ((hsv[..., 0].astype(np.int32) + int(round(deg / 2))) % 180).astype(np.uint8); return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
def _jpeg(im, q): return cv2.cvtColor(cv2.imdecode(cv2.imencode('.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
def _contrast(x, f): m = x.reshape(len(x), -1).mean(1).reshape(-1, 1, 1, 1); return np.clip((x.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8)
def _shuffle(im, t, seed):
    h, w = im.shape[:2]; tiles = [im[y:y + t, x:x + t] for y in range(0, h, t) for x in range(0, w, t)]; perm = np.random.default_rng(seed).permutation(len(tiles)); out = np.empty_like(im); k = 0
    for y in range(0, h, t):
        for x in range(0, w, t): out[y:y + t, x:x + t] = tiles[perm[k]]; k += 1
    return out
def shuffle(t): return lambda x: np.stack([_shuffle(im, t, i) for i, im in enumerate(x)])
PERT = {
    'clean': lambda x: x,
    'grayscale': per_image(lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)),
    'brightness x0.6': lambda x: np.clip(x.astype(np.float32) * 0.6, 0, 255).astype(np.uint8), 'brightness x1.4': lambda x: np.clip(x.astype(np.float32) * 1.4, 0, 255).astype(np.uint8),
    'contrast x0.6': lambda x: _contrast(x, 0.6), 'contrast x1.4': lambda x: _contrast(x, 1.4),
    'gamma 0.6': lambda x: (255 * (x.astype(np.float32) / 255) ** 0.6).astype(np.uint8), 'gamma 1.6': lambda x: (255 * (x.astype(np.float32) / 255) ** 1.6).astype(np.uint8),
    'hue +90 deg': per_image(lambda im: _hue(im, 90)), 'hue +180 deg': per_image(lambda im: _hue(im, 180)),
    'JPEG quality 20': per_image(lambda im: _jpeg(im, 20)), 'Gaussian blur sigma 1 px': per_image(lambda im: cv2.GaussianBlur(im, (0, 0), 1.0)), 'Gaussian blur sigma 2 px': per_image(lambda im: cv2.GaussianBlur(im, (0, 0), 2.0)),
    'shadow lift (dark half, gamma 0.5 on L)': per_image(lambda im: _lab(im, _lift)), 'CLAHE on L (clip 3, 8x8)': per_image(lambda im: _lab(im, _clahe.apply)),
    'intensity inversion (negative)': lambda x: 255 - x, 'luminance inversion (L channel only, chroma kept)': per_image(lambda im: _lab(im, lambda L: 255 - L)),
    'tile shuffle, 64 px tiles (layout destroyed, local appearance kept)': shuffle(64), 'tile shuffle, 32 px tiles': shuffle(32),
}
FOCUS = ['clean', 'grayscale', 'hue +180 deg', 'JPEG quality 20', 'Gaussian blur sigma 2 px', 'shadow lift (dark half, gamma 0.5 on L)', 'intensity inversion (negative)', 'luminance inversion (L channel only, chroma kept)',
         'tile shuffle, 64 px tiles (layout destroyed, local appearance kept)', 'tile shuffle, 32 px tiles']
names = list(PERT) if a.conds == 'all' else FOCUS; out = {}; pred = (lambda Xp: H.raw_view(Xp)) if a.net == 'raw' else (lambda Xp: H.nadir(Xp, S.tilt_est.values))
for nm in names:
    h, k = pred(PERT[nm](X)); out[f'{nm}|h'] = h.astype(np.float32); out[f'{nm}|k'] = k.astype(np.float32)
    e = np.abs(wrap(h - gt)); print(f'{a.net} {nm:66s} MAE {e.mean():6.2f} med {np.median(e):5.2f} >45 {100 * (e > 45).mean():4.1f}%', flush=True)
extra = {}
if a.net == 'raw' and a.zoom:
    # scale shift: centre zoom of the ORIGINAL image (same optical axis, so the heading is unchanged). x3.6 gives a rendered view the ground sampling of the real 60 m flight (training views: about 215 m)
    from concurrent.futures import ThreadPoolExecutor
    ZS = (1.5, 2.0, 3.6); W0, H0 = 3686, 2073
    def zooms(rel):
        im = cv2.imread(f'/root/pairUAV-5D-temp/{rel}', cv2.IMREAD_COLOR); o = []
        for z in ZS:
            w, h = int(round(W0 / z)), int(round(H0 / z)); x0, y0 = (W0 - w) // 2, (H0 - h) // 2; o.append(cv2.cvtColor(cv2.resize(im[y0:y0 + h, x0:x0 + w], (448, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        return o
    with ThreadPoolExecutor(3) as ex: Z = list(ex.map(zooms, S.image.tolist()))
    for j, z in enumerate(ZS):
        h, k = H.raw_view(np.stack([q[j] for q in Z])); out[f'centre zoom x{z}|h'] = h.astype(np.float32); out[f'centre zoom x{z}|k'] = k.astype(np.float32); e = np.abs(wrap(h - gt)); print(f'raw centre zoom x{z}: MAE {e.mean():.2f} med {np.median(e):.2f}', flush=True)
    extra['zoom'] = [f'centre zoom x{z}' for z in ZS]; del Z
if a.net == 'raw' and a.mech:
    # mechanism checks, ONE view (centre crop), no test-time averaging
    def circ_mean(d): d = np.deg2rad(d); return float(np.rad2deg(np.arctan2(np.sin(d).mean(), np.cos(d).mean())))
    h0, k0 = H.raw_view(X, tta=False); nad = S.tilt_gt.values < 5; M = {'single_view_clean': dict(mae=float(np.abs(wrap(h0 - gt)).mean()), median=float(np.median(np.abs(wrap(h0 - gt))))), 'near_nadir_images': int(nad.sum())}
    hm, _ = H.raw_view(X, tta=False, flip=True); dv = np.abs(wrap(hm + h0)); M['mirror'] = dict(expected='h -> -h', deviation_mean=float(dv.mean()), deviation_median=float(np.median(dv)), within_10deg=float((dv < 10).mean()), error_vs_gt_mae=float(np.abs(wrap(-hm - gt)).mean()))
    for kk in (1, 2, 3):
        hk_, _ = H.raw_view(X, tta=False, k=kk); dv = np.abs(wrap(hk_ - h0 - 90 * kk))
        M[f'rot{90 * kk}'] = dict(expected=f'h -> h + {90 * kk}', near_nadir=dict(n=int(nad.sum()), deviation_mean=float(dv[nad].mean()), deviation_median=float(np.median(dv[nad])), within_10deg=float((dv[nad] < 10).mean()), signed_shift_mean=circ_mean(wrap(hk_ - h0)[nad])),
                                  all_images=dict(n=int(n), deviation_mean=float(dv.mean()), deviation_median=float(np.median(dv)), within_10deg=float((dv < 10).mean())))
    extra['mechanism_raw_view_single_view'] = M; print(json.dumps(M)[:600], flush=True)
np.savez_compressed(f'{O}/sens_{a.tag}_{a.net}.npz', image=S.image.values.astype(str), tour=S.tour.values.astype(str), gt=gt, tilt_est=S.tilt_est.values, tilt_gt=S.tilt_gt.values, archived_bev=S.ah_est.values, **out)
json.dump(dict(net=a.net, tag=a.tag, n_images=int(n), n_buildings=int(S.tour.nunique()), seed=0, subset=a.subset, conds=names, raw_weights=a.raw_weights or 'archived', **extra), open(f'{O}/sens_{a.tag}_{a.net}.json', 'w'), indent=1); print('saved', a.tag, a.net, flush=True)
