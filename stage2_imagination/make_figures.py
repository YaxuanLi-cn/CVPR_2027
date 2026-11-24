import sys, math, types, numpy as np, pandas as pd, cv2, torch
sys.path.insert(0, '/root/rasr-pairuav/stage2_imagination'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
SG = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; S2 = f'{SG}/stage2'; OUT = '/root/rasr-pairuav/stage2_imagination/assets'
argv = sys.argv; sys.argv = ['x', '--out', '/tmp/x']; import imagine_match as IM; sys.argv = argv
P = pd.read_pickle(f'{SG}/pairs.pkl').set_index('manifest_index'); G = pd.read_pickle(f'{S2}/guide.pkl'); I = pd.read_pickle(f'{S2}/geom_imag.pkl'); Z = pd.read_pickle(f'{SG}/geom_zoom_half.pkl')[['manifest_index', 'zm_inl']].rename(columns={'zm_inl': 'zinl'})
D = G.merge(I, on='manifest_index').merge(Z, on='manifest_index'); D = D[D.split == 'test'].copy()
for c in ('heading', 'tilt', 'x', 'y', 'z', 'tilt_a', 'tilt_b'): D['gt_' + c] = P[c].reindex(D.manifest_index).values
D['gt_dist'] = np.sqrt(D.gt_x**2 + D.gt_y**2 + D.gt_z**2)
FONT = cv2.FONT_HERSHEY_SIMPLEX
def lab(im, t, sub=None):
    im = im.copy(); h = 34 if sub is None else 60; cv2.rectangle(im, (0, 0), (im.shape[1], h), (0, 0, 0), -1); cv2.putText(im, t, (8, 24), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if sub: cv2.putText(im, sub, (8, 50), FONT, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    return im
def chk(x, y, n=10):
    H, W = x.shape[:2]; yy, xx = np.mgrid[0:H, 0:W]; m = (((xx * n // W) + (yy * n // W)) % 2 == 0)[..., None]; return np.where(m, x, y)

def figure_A(r, name, title):
    sc = 0.25; W, H = int(IM.W0 * sc), int(IM.H0 * sc); Zs = np.diag([sc, sc, 1.0])
    ia = cv2.imread(str(IM.DATA / r.image_a)); ib = cv2.imread(str(IM.DATA / r.image_b))
    a = cv2.warpPerspective(ia, Zs, (W, H), flags=cv2.INTER_AREA); b = cv2.resize(ib, (W, H), interpolation=cv2.INTER_AREA)
    Hp = IM.homography_b2a(r, True); Hr = IM.homography_b2a(r, False); bp = cv2.warpPerspective(ib, Zs @ Hp, (W, H)); br = cv2.warpPerspective(ib, Zs @ Hr, (W, H))
    rows = [np.hstack([lab(b, 'real image b (source)'), lab(a, 'real image a (target)')]),
            np.hstack([lab(bp, 'IMAGINED a: b re-rendered with the stage-1 pose'), lab(chk(a, bp), 'checkerboard: real a / imagined a')]),
            np.hstack([lab(br, 'for comparison: rotation-only alignment (stage 1)'), lab(chk(a, br), 'checkerboard: real a / rotation-only')])]
    # native-resolution patch from the tile with the largest overlap, as fed to SuperGlue, + the residual displacement of the matches
    cfg = types.SimpleNamespace(pad=150, scale=1.0, max_grid=3, rot_only=False); idx, TA, TB, MK, ZI, ZHI, s, n = IM.Pairs(pd.DataFrame([r]), cfg)[0]
    if TA is not None:
        ti = int(np.argmax([m.float().mean().item() for m in MK])); ta = TA[ti].numpy(); tb = TB[ti].numpy(); mk = MK[ti].numpy(); PW, PH = 608, 400
        ys, xs = np.nonzero(cv2.erode(mk, np.ones((61, 61), np.uint8))); cx, cy = (int(np.median(xs)), int(np.median(ys))) if len(xs) else (640, 360)
        x0 = int(np.clip(cx - PW // 2, 0, IM.CW - PW)); y0 = int(np.clip(cy - PH // 2, 0, IM.CH - PH)); pa = cv2.cvtColor(ta[y0:y0 + PH, x0:x0 + PW], cv2.COLOR_GRAY2BGR); pb = cv2.cvtColor(tb[y0:y0 + PH, x0:x0 + PW], cv2.COLOR_GRAY2BGR)
        d = np.load(f'{S2}/matches_imag/{int(r.manifest_index)}.npz'); m = (d['tile'] == ti) & (d['conf'].astype(np.float32) > 0.3); p0 = d['p0'][m].astype(np.float64); ds = d['disp'][m].astype(np.float64) * s
        k0 = (np.c_[p0, np.ones(len(p0))] @ np.linalg.inv(ZI[ti]).T)[:, :2] - [x0, y0]; ov = (pa * 0.55).astype(np.uint8); inside = (k0[:, 0] > 0) & (k0[:, 0] < PW) & (k0[:, 1] > 0) & (k0[:, 1] < PH)
        for (u, v), (du, dv) in zip(k0[inside], ds[inside]):
            mag = math.hypot(du, dv); col = (80, 255, 80) if mag < 8 else (0, 200, 255) if mag < 30 else (60, 60, 255)
            cv2.line(ov, (int(u), int(v)), (int(u + du), int(v + dv)), col, 1, cv2.LINE_AA); cv2.circle(ov, (int(u), int(v)), 2, col, -1, cv2.LINE_AA)
        med = float(np.median(np.hypot(ds[:, 0], ds[:, 1]))) if len(ds) else float('nan')
        row = np.hstack([lab(pa, 'real a, native-res patch'), lab(pb, 'imagined a, same patch'), lab(ov, f'SuperGlue matches: {int(inside.sum())} in patch', f'line = imagined->real offset (median {med:.1f} px)')])
        row = cv2.resize(row, (2 * W, int(row.shape[0] * 2 * W / row.shape[1])), interpolation=cv2.INTER_AREA); rows.append(row)
    head = np.zeros((70, 2 * W, 3), np.uint8); cv2.putText(head, title, (8, 28), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(head, f'GT: dheading {r.gt_heading:.0f} deg, tilt a/b {r.gt_tilt_a:.0f}/{r.gt_tilt_b:.0f} deg, dz {r.gt_z:.1f} m, baseline {r.gt_dist:.1f} m | inliers: stage-1 zoom {r.zinl:.0f} -> imagination pass {r.im_inl:.0f}', (8, 58), FONT, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f'{OUT}/{name}.jpg', np.vstack([head] + rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(name, int(r.manifest_index))

def pick(q, seed): return D.query(q).sample(1, random_state=seed).iloc[0]
figure_A(pick('zinl > 400 and zinl < 800 and abs(gt_heading) < 60 and abs(gt_z) > 20 and gt_tilt_a < 25 and gt_tilt_b < 25', 1), 'A1_height_change', 'A1. Large height change, moderate heading change')
figure_A(pick('zinl > 250 and zinl < 600 and abs(gt_heading) > 140 and abs(gt_tilt) > 10', 2), 'A2_opposite_heading', 'A2. Nearly opposite headings')
figure_A(pick('zinl > 60 and zinl < 160 and im_inl > 1000 and gt_tilt_a > 25 and gt_tilt_b > 25', 3), 'A3_oblique_hard', 'A3. Both views oblique; stage 1 found few matches')
figure_A(pick('zinl < 60 and im_inl < 150', 4), 'A4_failure', 'A4. Failure case: almost no overlap, imagination cannot be verified')

# ---------------- module B ----------------
sys.argv = ['x', '--tag', 'bev_s']; import bev_heading as B; sys.argv = argv
B.net.load_state_dict(torch.load(f'{S2}/model_bev_s.pt')); B.net.eval(); it = pd.read_pickle(f'{S2}/pred_bev_s.pkl'); it = it[it.part == 'test']
row_of = pd.Series(np.arange(len(B.meta)), index=B.meta.image.values); rows = []; K = 12
def tile(t, size=None): x = t.permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)[:, :, ::-1]; return np.ascontiguousarray(x if size is None else cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA))
def cap(im, t, col=(255, 255, 255), sc=0.45):
    bar = np.zeros((22, im.shape[1], 3), np.uint8); cv2.putText(bar, t, (4, 16), FONT, sc, col, 1, cv2.LINE_AA); return np.vstack([bar, im])
for q, seed in (('tilt_gt < 6', 0), ('tilt_gt > 20 and tilt_gt < 28', 4), ('tilt_gt > 40', 2)):
    e = np.abs((it.ah_est - it.heading_gt + 180) % 360 - 180); r = it[e < 4].query(q).sample(1, random_state=seed).iloc[0]; ix = np.array([row_of[r.image]])
    with torch.inference_mode():
        x = torch.from_numpy(B.raw_np(ix)).to('cuda').permute(0, 3, 1, 2).float(); tl = torch.tensor([math.radians(r.tilt_est)], device='cuda'); cands = []
        for k in range(K):
            d = torch.tensor([2 * math.pi * k / K], device='cuda'); v = B.imagine_bev(x, tl, d)
            with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = B.net(B.norm(v))
            vote = math.degrees(math.atan2(mu[0, 0].float().item(), mu[0, 1].float().item()) - d.item()) % 360; cands.append((tile(v[0], 128), vote, kap.item()))
        nu_p = tile(B.imagine_bev(x, tl, torch.tensor([-math.radians(r.ah_est)], device='cuda'))[0]); nu_g = tile(B.imagine_bev(x, tl, torch.tensor([-math.radians(r.heading_gt)], device='cuda'))[0]); nad = tile(B.imagine_bev(x, tl, torch.zeros(1, device='cuda'))[0])
    src = np.zeros((256, 448, 3), np.uint8); src[:] = B.raw_np(ix)[0][:, :, ::-1]
    src = cap(src, f'real image: tilt {r.tilt_gt:.0f} (SuperGlue est. {r.tilt_est:.1f}), heading {r.heading_gt:.0f} deg'); nad = cap(nad, 'imagined nadir view (tilt removed)')
    grid = np.vstack([np.hstack([cap(c[0], f'+{30*k} -> {c[1]:.0f} deg', sc=0.42) for k, c in list(enumerate(cands))[i:i + 6]]) for i in (0, 6)])
    grid = np.vstack([grid, np.zeros((src.shape[0] - grid.shape[0], grid.shape[1], 3), np.uint8)]) if grid.shape[0] < src.shape[0] else cv2.resize(grid, (int(grid.shape[1] * src.shape[0] / grid.shape[0]), src.shape[0]), interpolation=cv2.INTER_AREA)
    nu_p = cap(nu_p, f'north-up by prediction ({r.ah_est % 360:.0f} deg)', (120, 255, 120)); nu_g = cap(nu_g, f'north-up by ground truth ({r.heading_gt:.0f} deg)', (120, 255, 120))
    rows.append(np.hstack([src, nad, grid, nu_p, nu_g]))
head = np.zeros((40, rows[0].shape[1], 3), np.uint8); cv2.putText(head, 'Module B: real image -> imagined nadir view -> 12 imagined heading candidates (rotation -> heading it votes for) -> north-up result', (8, 27), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(f'{OUT}/B_bev.jpg', np.vstack([head] + rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print('B done')
