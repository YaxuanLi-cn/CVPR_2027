"""Camera geometry for PairUAV-5D (Google Earth KML cameras) and SuperGlue-based relative pose."""
import numpy as np, cv2

W0, H0 = 3686, 2073  # original render size

def K_for(w, h, hfov_deg, w_src=W0, h_src=H0):
    """Intrinsics for an image resized (possibly anisotropically) from the original render."""
    f = (w_src / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
    sx, sy = w / w_src, h / h_src
    return np.array([[f * sx, 0, w / 2], [0, f * sy, h / 2], [0, 0, 1.0]])

def Rx(t):
    c, s = np.cos(t), np.sin(t); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Rz(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def R_wc(heading_deg, tilt_deg):
    """Camera(OpenCV: x right, y down, z fwd)->world(ENU). heading clockwise from north, tilt 0 = nadir."""
    h, t = np.deg2rad(heading_deg), np.deg2rad(tilt_deg)
    right = np.array([np.cos(h), -np.sin(h), 0.0])
    fwd = np.array([np.sin(t) * np.sin(h), np.sin(t) * np.cos(h), -np.cos(t)])
    up_img = np.array([np.cos(t) * np.sin(h), np.cos(t) * np.cos(h), np.sin(t)])
    return np.stack([right, -up_img, fwd], axis=1)

def gt_relative(ha, ta, hb, tb, d_enu):
    """R, t such that X_b = R X_a + t (t = position of cam a in cam b frame = -R_b^T d)."""
    Ra, Rb = R_wc(ha, ta), R_wc(hb, tb)
    return Rb.T @ Ra, -Rb.T @ np.asarray(d_enu, float)

def decompose_R(R):
    """R (a->b, X_b = R X_a) -> (dheading_deg, tilt_a_deg, tilt_b_deg) assuming zero roll.
    In level frames L (x right, y horizontal-forward, z up): cam = L rotated; derive via world vectors."""
    # R = Rb^T Ra. With Ra = Rz(-ha) A(ta), A(t) = R_wc(0,t):  R = A(tb)^T Rz(dh) A(ta)  (Rz(-ha)^T Rz(-hb)... sign below)
    # M = Rz(-hb)^T Rz(-ha) = Rz(hb-ha) = Rz(dh).  Entries of R give closed forms:
    # up-axis in cam frame: u(t) = A(t)^T e_z = (0, -sin t, -cos t).  R u(ta) = u(tb)  (gravity is shared).
    # Solve via: R[0,0] = cos(dh)  (right_b . right_a), right vectors are horizontal.
    # right_a = (cos ha, -sin ha, 0); right_b . right_a = cos(dh).  R[0,0] = e_x^T Rb^T Ra e_x = right_b.right_a
    # R[0,1] = right_b . down_a ; R[0,2] = right_b . fwd_a ; fwd_a = (st sinh, st cosh, -ct), right_b=(cos hb, -sin hb,0)
    # right_b.fwd_a = st_a * sin(ha-hb) = -st_a sin(dh);  right_b.down_a = -ct_a sin(ha-hb) = ct_a sin(dh)
    # R[1,0] = down_b.right_a = -ct_b * sin(hb-ha) = -ct_b sin(dh);  R[2,0] = fwd_b.right_a = st_b sin(dh)
    sa = np.hypot(R[0, 1], R[0, 2]); sb = np.hypot(R[1, 0], R[2, 0])  # both = |sin dh|
    # sign of sin(dh): tilt in [0,90) => cos t > 0, so sign(sin dh) = sign(R[0,1]) = -sign(R[1,0])
    sgn = np.sign(R[0, 1] - R[1, 0]) or 1.0
    dh = np.arctan2(sgn * 0.5 * (sa + sb), R[0, 0])
    ta = np.arctan2(-R[0, 2] * sgn, R[0, 1] * sgn)
    tb = np.arctan2(R[2, 0] * sgn, -R[1, 0] * sgn)
    return np.rad2deg(dh), np.rad2deg(ta), np.rad2deg(tb)

def level_dir_from_t(R, t, ta_deg, tb_deg):
    """Unit direction of displacement (C_b - C_a) in cam-a's levelled frame: (right, horizontal-forward, up)."""
    d_cam_a = -R.T @ t  # C_b - C_a expressed in cam a coords (since t = -R_b^T d, R^T t = -R_a^T d)
    A = R_wc(0.0, ta_deg)  # cam a -> world with heading 0: x=East=right, y=North=horizontal fwd, z=Up
    v = A @ d_cam_a
    return v / (np.linalg.norm(v) + 1e-12)
