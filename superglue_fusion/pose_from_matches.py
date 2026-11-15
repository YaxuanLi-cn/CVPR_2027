"""SuperGlue matches -> essential matrix -> (dheading, tilt_a, tilt_b, levelled translation direction) + quality stats."""
import numpy as np, cv2
from scipy.optimize import least_squares
from geom import decompose_R, level_dir_from_t

# Intrinsics in ORIGINAL pixel coords (3686x2073), calibrated on train-split GT poses (see SUPERGLUE_EXPERIMENTS.md §2)
K0 = np.array([[3324.9, 0, 1841.57], [0, 3324.9, 1078.04], [0, 0, 1.0]])   # refit on TRAIN-split pairs only (SUPERGLUE_EXPERIMENTS.md §15)

def _sampson(R, t, x0, x1):
    E = np.cross(np.eye(3), t) @ R  # [t]x R
    Ex0 = x0 @ E.T; Etx1 = x1 @ E
    num = (x1 * Ex0).sum(1)
    return num / np.sqrt(Ex0[:, 0]**2 + Ex0[:, 1]**2 + Etx1[:, 0]**2 + Etx1[:, 1]**2 + 1e-18)

def _refine(R, t, x0, x1, w=None):
    rv0, _ = cv2.Rodrigues(R); t0 = t / np.linalg.norm(t)
    # tangent-plane parametrisation of unit t
    b1 = np.cross(t0, [1, 0, 0] if abs(t0[0]) < 0.9 else [0, 1, 0]); b1 /= np.linalg.norm(b1); b2 = np.cross(t0, b1)
    def unpack(p):
        Rn, _ = cv2.Rodrigues(rv0.ravel() + p[:3]); tn = t0 + p[3] * b1 + p[4] * b2
        return Rn, tn / np.linalg.norm(tn)
    def fun(p):
        Rn, tn = unpack(p); r = _sampson(Rn, tn, x0, x1)
        return r if w is None else r * w
    res = least_squares(fun, np.zeros(5), loss='huber', f_scale=1.0 / K0[0, 0], max_nfev=30)
    Rn, tn = unpack(res.x)
    # covariance proxy: how well-conditioned is the translation direction
    try:
        J = res.jac; H = J.T @ J; ev = np.linalg.eigvalsh(H[3:, 3:] - H[3:, :3] @ np.linalg.solve(H[:3, :3] + 1e-12 * np.eye(3), H[:3, 3:]))
        t_info = float(np.log10(max(ev.min(), 1e-12)))
    except Exception:
        t_info = -12.0
    return Rn, tn, t_info

def _cheir(R, t, x0, x1):
    X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], x0[:, :2].T.copy(), x1[:, :2].T.copy())
    with np.errstate(all='ignore'):
        X = (X[:3] / X[3]).T
    z0 = X[:, 2]; z1 = (X @ R.T + t)[:, 2]
    return float(np.mean((z0 > 0) & (z1 > 0))), z0

def _tilt_penalty(R):
    _, ta, tb = decompose_R(R)
    return max(0.0, -8 - ta, ta - 53) + max(0.0, -8 - tb, tb - 53)

def _select_pose(E, x0, x1):
    """Pick among the 4 decompositions of E using cheirality plus the dataset prior (zero roll, tilt in [0,45])."""
    R1, R2, t = cv2.decomposeEssentialMat(E); t = t.ravel(); best = None
    for R in (R1, R2):
        for tt in (t, -t):
            sc = _cheir(R, tt, x0, x1)[0] - 0.05 * _tilt_penalty(R)
            if best is None or sc > best[0]: best = (sc, R, tt)
    return best[1], best[2]

NAN_ROW = dict(sg_ok=0.0, sg_dh=np.nan, sg_dh_sin=0.0, sg_dh_cos=0.0, sg_ta=np.nan, sg_tb=np.nan, sg_dtilt=np.nan,
               sg_vx=0.0, sg_vy=0.0, sg_vz=0.0, sg_inl=0.0, sg_inl_ratio=0.0, sg_resid=np.nan, sg_tinfo=-12.0, sg_par=np.nan, sg_cheir=0.0, sg_tsum=np.nan, sg_tpen=np.nan, sg_zspread=np.nan,
               **{f'sg_R{i}{j}': np.nan for i in range(3) for j in range(3)}, **{f'sg_t{i}': np.nan for i in range(3)},
               **{f'sg_hgt{c}_p{q}': np.nan for c in 'ab' for q in (25, 50, 75, 90)})

def pose_features(p0, p1, conf, thr_px=3.0, min_conf=0.2):
    """p0,p1: matched keypoints in ORIGINAL pixel coords (N,2). Returns dict of geometric features."""
    out = dict(NAN_ROW); n = len(p0)
    out.update(sg_n=float(n), sg_conf_mean=float(conf.mean()) if n else 0.0, sg_n_hi=float((conf > 0.6).sum()) if n else 0.0)
    if n:
        d = p1 - p0
        out.update(sg_flow_mx=float(np.median(d[:, 0])), sg_flow_my=float(np.median(d[:, 1])), sg_flow_sx=float(d[:, 0].std()), sg_flow_sy=float(d[:, 1].std()),
                   sg_cov0=float(p0.std(0).prod() / (W0H0)), sg_cov1=float(p1.std(0).prod() / (W0H0)))
    else:
        out.update(sg_flow_mx=0.0, sg_flow_my=0.0, sg_flow_sx=0.0, sg_flow_sy=0.0, sg_cov0=0.0, sg_cov1=0.0)
    keep = conf >= min_conf
    p0, p1, conf = p0[keep], p1[keep], conf[keep]
    if len(p0) < 8: return out
    Ki = np.linalg.inv(K0)
    x0 = np.c_[p0, np.ones(len(p0))] @ Ki.T; x1 = np.c_[p1, np.ones(len(p1))] @ Ki.T
    try:
        E, mask = cv2.findEssentialMat(x0[:, :2].copy(), x1[:, :2].copy(), np.eye(3), method=cv2.USAC_MAGSAC, prob=0.99999, threshold=thr_px / K0[0, 0])
    except cv2.error:
        return out
    if E is None or E.shape != (3, 3) or mask is None or mask.sum() < 8: return out
    inl = mask.ravel().astype(bool)
    R, t = _select_pose(E, x0[inl], x1[inl])
    try:
        R, t, t_info = _refine(R, t, x0[inl], x1[inl])
    except Exception:
        t_info = -12.0
    # re-score all matches with refined model, refine once more on the enlarged inlier set
    r = np.abs(_sampson(R, t, x0, x1)) * K0[0, 0]
    inl2 = r < thr_px
    if inl2.sum() >= 8:
        try: R, t, t_info = _refine(R, t, x0[inl2], x1[inl2]); r = np.abs(_sampson(R, t, x0, x1)) * K0[0, 0]; inl2 = r < thr_px
        except Exception: pass
    par = np.nan; ch = 0.0
    if inl2.sum():
        ch, z0 = _cheir(R, t, x0[inl2], x1[inl2])
        if ch < 0.5:
            ch2, z02 = _cheir(R, -t, x0[inl2], x1[inl2])
            if ch2 > ch: t, ch, z0 = -t, ch2, z02
        zz = z0[(z0 > 0) & np.isfinite(z0)]
        if len(zz): par = float(1.0 / np.median(zz)); zq = np.percentile(zz, [10, 90])
    dh, ta, tb = decompose_R(R)
    v = level_dir_from_t(R, t, ta, tb)
    # heights of triangulated inliers below each camera, in baseline units (gravity from the zero-roll decomposition)
    hg = {f'sg_hgt{c}_p{q}': np.nan for c in 'ab' for q in (25, 50, 75, 90)}
    if inl2.sum() >= 8:
        with np.errstate(all='ignore'):
            X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], x0[inl2, :2].T.copy(), x1[inl2, :2].T.copy()); X = (X[:3] / X[3]).T
        Xb = X @ R.T + t; good = (X[:, 2] > 0) & (Xb[:, 2] > 0) & np.isfinite(X).all(1)
        if good.sum() >= 8:
            for c, pts, tl in (('a', X[good], ta), ('b', Xb[good], tb)):
                up = np.array([0.0, -np.sin(np.deg2rad(tl)), -np.cos(np.deg2rad(tl))]); h = -(pts @ up)
                for q, val in zip((25, 50, 75, 90), np.percentile(h, [25, 50, 75, 90])): hg[f'sg_hgt{c}_p{q}'] = float(val)
    out.update(sg_ok=1.0, sg_dh=dh, sg_dh_sin=np.sin(np.deg2rad(dh)), sg_dh_cos=np.cos(np.deg2rad(dh)), sg_ta=ta, sg_tb=tb, sg_dtilt=tb - ta,
               sg_vx=v[0], sg_vy=v[1], sg_vz=v[2], sg_inl=float(inl2.sum()), sg_inl_ratio=float(inl2.mean()),
               sg_resid=float(np.median(r[inl2])) if inl2.any() else np.nan, sg_tinfo=t_info, sg_par=par, sg_cheir=ch,
               sg_tsum=ta + tb, sg_tpen=_tilt_penalty(R), sg_zspread=float(np.log(zq[1] / zq[0])) if np.isfinite(par) else np.nan,
               **{f'sg_R{i}{j}': float(R[i, j]) for i in range(3) for j in range(3)}, **{f'sg_t{i}': float(t[i]) for i in range(3)}, **hg)
    return out

W0H0 = 3686.0 * 2073.0

def load_existing_npz(path, min_conf=0.0):
    """Existing matches_superglue output (640x480 anisotropic resize) -> original pixel coords."""
    d = np.load(path); m = d['matches']; ok = m > -1
    s = np.array([3686 / 640, 2073 / 480])
    return d['keypoints0'][ok].astype(np.float64) * s, d['keypoints1'][m[ok]].astype(np.float64) * s, d['match_confidence'][ok].astype(np.float64)

ALT_KEYS = ['sg_ok', 'sg_dh_sin', 'sg_dh_cos', 'sg_ta', 'sg_tb', 'sg_vx', 'sg_vy', 'sg_vz', 'sg_inl', 'sg_inl_ratio', 'sg_par', 'sg_tinfo']

def pose_features_multi(p0, p1, conf, src, min_fine_inl=100):
    """Fine-only estimate (precise) and union estimate (robust). Primary = fine if it has enough inliers, else the better of the two.
    Also exports the non-primary estimate under the alt_ prefix so a downstream model can use their agreement."""
    fine = src >= 10
    F = pose_features(p0[fine], p1[fine], conf[fine])
    U = pose_features(p0, p1, conf)
    use_f = F['sg_ok'] == 1 and (F['sg_inl'] >= min_fine_inl or F['sg_inl'] >= 0.7 * U['sg_inl'])
    prim, alt = (F, U) if use_f else (U, F)
    out = dict(prim); out['sg_prim_fine'] = float(use_f)
    for k in ALT_KEYS: out['alt_' + k[3:]] = alt[k]
    out['sg_n_fine'] = float(fine.sum()); out['sg_n_coarse'] = float((~fine).sum())
    if prim['sg_ok'] == 1 and alt['sg_ok'] == 1:
        out['sg_alt_dh_diff'] = float(np.rad2deg(np.arctan2(prim['sg_dh_sin'] * alt['sg_dh_cos'] - prim['sg_dh_cos'] * alt['sg_dh_sin'], prim['sg_dh_cos'] * alt['sg_dh_cos'] + prim['sg_dh_sin'] * alt['sg_dh_sin'])))
        out['sg_alt_dir_cos'] = float(prim['sg_vx'] * alt['sg_vx'] + prim['sg_vy'] * alt['sg_vy'] + prim['sg_vz'] * alt['sg_vz'])
    else:
        out['sg_alt_dh_diff'] = np.nan; out['sg_alt_dir_cos'] = np.nan
    return out
