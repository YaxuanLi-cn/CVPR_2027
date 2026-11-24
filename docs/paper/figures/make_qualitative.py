"""fig_qualitative.pdf: views imagined by the plane-induced homography of the coarse (stage-I) pose, next to the real view (CPU only).

3 x 3 grid of square panels at single-column width (3.3 in). Columns: 2 PairUAV test pairs, then 1 pair of real
photographs (Mezieres, zero-shot). Rows: source S (square window of the frame; the outlined region is the part that is
re-rendered) | imagined target (S resampled in the target camera) | real target G. The last two rows are cropped to the
largest square seen by both cameras, so no panel has empty borders.

    python make_qualitative.py                 # imagined target next to the real target (default)
    python make_qualitative.py --view source   # direction of Eq. (homography) in method.tex: imagined source next to the real source

Selection: pairs were picked by hand for scene diversity among those where (a) at least 40% of the frame is seen by both
cameras, (b) the imagination-refined stage succeeded with a median residual displacement of at most 8 px (40 px for the real
photographs) and (c) the viewpoint changes visibly. The pose used for the warp is always the estimated one, never the GT."""
import sys, os, math, argparse, numpy as np, pandas as pd, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from matplotlib.patches import Polygon
sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
from geom import R_wc
from pose_from_matches import K0
ap = argparse.ArgumentParser(); ap.add_argument('--view', choices=['target', 'source'], default='target'); ap.add_argument('--out', default='fig_qualitative'); A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); SG = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; S2 = f'{SG}/stage2'; DATA = '/root/pairUAV-5D-temp'; W0, H0 = 3686, 2073
SYN = [432196, 74448]                                                               # manifest_index of PairUAV test pairs
REAL = [('100_0061_0045', '100_0061_0012')]                                          # (source, target) photographs of Mezieres
PW, PH = 640, 640                                                                    # pixels of one panel
YEL = '#FFD23F'

def homography(g):
    """source pixel -> target pixel, Eq. (homography) of method.tex: H = K (R + t n^T / h) K^-1"""
    Ra = R_wc(0.0, g['ta']); R = R_wc(g['dh'], g['tb']).T @ Ra
    t = -R @ (Ra.T @ np.array([g['lx'], g['ly'], g['z']], float)); n = np.array([0.0, math.sin(math.radians(g['ta'])), math.cos(math.radians(g['ta']))])
    return K0 @ (R + np.outer(t, n) / g['H']) @ np.linalg.inv(K0)

def inscribed(mask, ar=PW / PH):
    """largest axis-aligned rectangle of aspect ar inside a boolean mask (closest to the centre on ties) -> x0, y0, w, h"""
    Hh, Ww = mask.shape; ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))); lo, hi, best = 4, Ww, None
    while lo <= hi:
        w = (lo + hi) // 2; h = int(round(w / ar))
        if h > Hh: hi = w - 1; continue
        s = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]; ok = np.argwhere(s == w * h)
        if len(ok): c = ok[np.argmin(((ok + [h / 2, w / 2] - [Hh / 2, Ww / 2])**2).sum(1))]; best = (c[1], c[0], w, h); lo = w + 1
        else: hi = w - 1
    return best

def warp(im, M):   # 2x supersampling, then area averaging: the imagined and the real view go through the same resampling
    return cv2.resize(cv2.warpPerspective(im, np.diag([2.0, 2.0, 1.0]) @ M, (2 * PW, 2 * PH), flags=cv2.INTER_LINEAR), (PW, PH), interpolation=cv2.INTER_AREA)

def panels(i_in, i_out, Hio, ms=0.1):
    """i_in is re-rendered in the camera of i_out with Hio (pixels of i_in -> pixels of i_out)."""
    Zm = np.diag([ms, ms, 1.0]); m = cv2.warpPerspective(np.full((H0, W0), 255, np.uint8), Zm @ Hio, (int(W0 * ms), int(H0 * ms)), flags=cv2.INTER_NEAREST) > 250
    x0, y0, w, h = [v / ms for v in inscribed(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)]
    Zc = np.array([[PW / w, 0, -x0 * PW / w], [0, PH / h, -y0 * PH / h], [0, 0, 1.0]])
    q = np.linalg.inv(Hio) @ np.array([[x0, y0, 1], [x0 + w, y0, 1], [x0 + w, y0 + h, 1], [x0, y0 + h, 1]]).T; quad = (q[:2] / q[2]).T
    ww = H0 * PW / PH; xs = float(np.clip(quad[:, 0].mean() - ww / 2, 0, W0 - ww)); f = PH / H0; Zf = np.array([[f, 0, -xs * f], [0, f, 0], [0, 0, 1.0]])   # full-height window of the input frame around the region
    return dict(inp=warp(i_in, Zf), imag=warp(i_in, Zc @ Hio), real=warp(i_out, Zc), quad=(quad - [xs, 0]) * f, frac=w * h / (W0 * H0))

def make(ia, ib, g):
    H = homography(g); return panels(ia, ib, H) if A.view == 'target' else panels(ib, ia, np.linalg.inv(H))

cols = []
P = pd.read_pickle(f'{SG}/pairs.pkl').set_index('manifest_index'); G = pd.read_pickle(f'{S2}/guide.pkl').set_index('manifest_index')
for k in SYN:
    r, p = G.loc[k], P.loc[k]; assert r.split == 'test'
    R = make(cv2.imread(f'{DATA}/{r.image_a}'), cv2.imread(f'{DATA}/{r.image_b}'), dict(ta=r.g_ta, tb=r.g_tb, dh=r.g_dh, lx=r.p_lx, ly=r.p_ly, z=r.p_z, H=r.g_H))
    R['dh'], R['d'] = abs(p.heading), math.sqrt(p.x**2 + p.y**2 + p.z**2); cols.append(R)
    print(f'PairUAV {k}: tour {r.tour} dheading {p.heading:.0f} tilt {p.tilt_a:.0f}->{p.tilt_b:.0f} dz {p.z:.1f} dist {R["d"]:.1f} m | shown region {R["frac"]:.2f} of the frame')
M = pd.read_pickle('/root/realdata/mezieres/pred.pkl')
for a, b in REAL:
    r = M[(M.a == a) & (M.b == b)].iloc[0]
    R = make(cv2.imread(r.image_a), cv2.imread(r.image_b), dict(ta=r.g_ta, tb=r.g_tb, dh=r.g_dh, lx=r.s1_lx, ly=r.s1_ly, z=r.s1_z, H=r.g_H))
    R['dh'], R['d'] = abs(r.heading), r.dist; cols.append(R)
    print(f'real {a} -> {b}: dheading {r.heading:.0f} dist {r.dist:.1f} m | shown region {R["frac"]:.2f} of the frame')

# ---- layout: one pair per column; a slightly wider gap before the real photographs
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 7, 'pdf.fonttype': 42})
names = [('inp', 'source $S$'), ('imag', r'imagined $\widehat{G}$ (ours)'), ('real', 'real target $G$')] if A.view == 'target' else [('inp', 'target $G$'), ('imag', r'imagined $\widehat{S}$ (ours)'), ('real', 'real source $S$')]
n, ns = len(cols), len(SYN); fw, left, top, gap, ggap = 3.3, 0.14, 0.15, 0.02, 0.03
pw = (fw - left - (n - 1) * gap - ggap) / n; ph = pw * PH / PW; fh = top + 3 * ph + 2 * gap
x = lambda j: left + j * (pw + gap) + (ggap if j >= ns else 0); fig = plt.figure(figsize=(fw, fh))
for j, R in enumerate(cols):
    for i, (key, name) in enumerate(names):
        ax = fig.add_axes([x(j) / fw, 1 - (top + (i + 1) * ph + i * gap) / fh, pw / fw, ph / fh]); ax.imshow(R[key][:, :, ::-1], interpolation='lanczos', aspect='auto'); ax.axis('off')
        if key == 'inp': ax.add_patch(Polygon(R['quad'], closed=True, fill=False, ec=YEL, lw=0.9, joinstyle='round', path_effects=[pe.Stroke(linewidth=1.5, foreground='#1a1a1a'), pe.Normal()])); ax.set_xlim(0, PW); ax.set_ylim(PH, 0)
        if j == 0: fig.text((left - 0.03) / fw, 1 - (top + i * (ph + gap) + ph / 2) / fh, name, ha='right', va='center', rotation=90, fontsize=7)
for j0, j1, name in ((0, ns - 1, 'PairUAV'), (ns, n - 1, 'Real')):
    fig.text((x(j0) + x(j1) + pw) / 2 / fw, 1 - (top - 0.04) / fh, name, ha='center', va='bottom', fontsize=7.5)
suffix = '' if A.view == 'target' else '_source'
fig.savefig(f'{HERE}/{A.out}{suffix}.pdf', dpi=300); fig.savefig(f'{HERE}/{A.out}{suffix}.jpg', dpi=300, pil_kwargs=dict(quality=92)); print('wrote', A.out + suffix)
