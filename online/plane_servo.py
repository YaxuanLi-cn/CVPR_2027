"""Short-range translation from the imagined-vs-real residual on the ground plane (used only by the closed-loop simulator).

The fusion models were fitted on pairs that are almost never closer than ~5 m, and the essential matrix loses the translation
direction as the baseline shrinks, so the last metres need a different cue. With the rotation fixed to the fused estimate,
ground points obey x_b ~ R x_a + w (n . x_a) with w = t / h (h = camera height above ground, n = ground normal in camera a).
That is linear in w and well conditioned at zero baseline. Matches come from the imagination re-matching step; no new model.
"""
import numpy as np
from geom import R_wc
from pose_from_matches import K0

def plane_translation(path, dh, ta, tb, min_conf=0.2, px_scale=2.0):
    """-> (wx, wy, wz, n_inliers): displacement of camera b from camera a in a's levelled frame (right, forward, up), in units of the camera height."""
    try:
        d = np.load(path); keep = d['conf'].astype(np.float64) >= min_conf; p0, p1 = d['p0'][keep].astype(np.float64), d['p1'][keep].astype(np.float64)
        if len(p0) < 30 or not np.isfinite([dh, ta, tb]).all(): return np.nan, np.nan, np.nan, 0
        Ki = np.linalg.inv(K0); xa = np.c_[p0, np.ones(len(p0))] @ Ki.T; xb = np.c_[p1, np.ones(len(p1))] @ Ki.T; Ra = R_wc(0.0, ta); R = R_wc(dh, tb).T @ Ra
        n = np.array([0.0, np.sin(np.deg2rad(ta)), np.cos(np.deg2rad(ta))]); y = xa @ R.T; s = xa @ n; u, v = xb[:, 0], xb[:, 1]
        A = np.concatenate([np.c_[s, np.zeros_like(s), -u * s], np.c_[np.zeros_like(s), s, -v * s]]); b = np.concatenate([u * y[:, 2] - y[:, 0], v * y[:, 2] - y[:, 1]]); wgt = np.ones(len(b)); w = np.zeros(3)
        for _ in range(12):                                   # IRLS with Cauchy weights on the reprojection residual (pixels); roofs and mismatches get down-weighted
            sw = np.sqrt(wgt); w = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
            den = (y[:, 2] + s * w[2]).clip(1e-6); r = np.hypot((y[:, 0] + s * w[0]) / den - u, (y[:, 1] + s * w[1]) / den - v) * K0[0, 0]; wgt = np.tile(1.0 / (1.0 + (r / px_scale)**2), 2)
        dl = -Ra @ R.T @ w                                   # t = -R Ra^T d  ->  d = -Ra R^T t
        return float(dl[0]), float(dl[1]), float(dl[2]), int((r < 2 * px_scale).sum())
    except Exception:
        return np.nan, np.nan, np.nan, 0

def work(a): return plane_translation(*a)
