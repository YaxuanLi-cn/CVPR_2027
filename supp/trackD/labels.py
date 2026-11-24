"""Camera files and pair labels of PairUAV, rebuilt from the KML files (Track D).

Label convention of the released pair files (checked against every selected pair in dataset_stats.py):
  x, y   = east, north components of (C_b - C_a) in the local tangent frame at the building ANCHOR (ECEF difference rotated to ENU at the anchor's latitude / longitude)
  z      = up component in the same frame
  tilt   = tilt_b - tilt_a,   heading = wrap(heading_b - heading_a) to (-180, 180]
"""
import os, re, json, glob, numpy as np, pandas as pd
DATA = '/root/pairUAV-5D-temp'
_pat = {k: re.compile(rf'<{k}>([-\d.eE+]+)</{k}>') for k in ('longitude', 'latitude', 'altitude', 'heading', 'tilt', 'roll')}; _fov = re.compile(r'<gx:horizFov>([-\d.eE+]+)</gx:horizFov>')
_A, _E2 = 6378137.0, 6.69437999014e-3
Z_MODE = 'up'      # 'up' (ENU up component at the anchor) or 'alt' (plain altitude difference); set from the check in dataset_stats.py


def load_cameras_raw(tour):
    rows = []
    for f in sorted(glob.glob(f'{DATA}/kml/{tour}/*.kml')):
        s = open(f).read(); r = {k: float(p.search(s).group(1)) for k, p in _pat.items()}; m = _fov.search(s); r['hfov'] = float(m.group(1)) if m else np.nan
        r['i'] = int(os.path.basename(f)[:-4]); r['image'] = f'{DATA}/images/{tour}/{os.path.basename(f)[:-4]}.jpg'; rows.append(r)
    return pd.DataFrame(rows)


def tour_anchor(tour):
    with open(f'{DATA}/pairs/{tour}.jsonl') as f: return json.loads(f.readline())['anchor']


def ecef(lat, lon, h):
    la, lo = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float)); h = np.asarray(h, float); N = _A / np.sqrt(1 - _E2 * np.sin(la)**2)
    return np.stack([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - _E2) + h) * np.sin(la)], -1)


def R_enu(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0], [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def enu_at_anchor(C, anc):
    """Adds e, n, u: camera position in the tangent frame at the anchor, origin on the terrain below the anchor."""
    C = C.copy(); X = ecef(C.latitude.values, C.longitude.values, C.altitude.values); X0 = ecef(anc['latitude'], anc['longitude'], anc['terrain_altitude_msl']); P = (X - X0) @ R_enu(anc['latitude'], anc['longitude']).T
    C['e'], C['n'], C['u'] = P[:, 0], P[:, 1], P[:, 2]; return C


def cameras(tour):
    """Camera table of a building with ENU positions (e, n, u) and height above the anchor terrain (hag)."""
    anc = tour_anchor(tour); C = enu_at_anchor(load_cameras_raw(tour), anc); C['hag'] = C.altitude - anc['terrain_altitude_msl']; C['tour'] = tour; C.attrs['anchor'] = anc; return C


def _z(C): return C.u.values if Z_MODE == 'up' else C.altitude.values


def pair_labels(C, a, b):
    """5-D labels (b minus a) for arrays of camera indices a, b of one building's camera table C."""
    Ci = C.set_index('i'); A = Ci.loc[np.asarray(a)]; B = Ci.loc[np.asarray(b)]; za = _z(A.reset_index()); zb = _z(B.reset_index())
    out = dict(x=B.e.values - A.e.values, y=B.n.values - A.n.values, z=zb - za, tilt=B.tilt.values - A.tilt.values, heading=(B.heading.values - A.heading.values + 180.0) % 360.0 - 180.0,
               heading_a=A.heading.values, heading_b=B.heading.values, tilt_a=A.tilt.values, tilt_b=B.tilt.values)
    out['dist'] = np.sqrt(out['x']**2 + out['y']**2 + out['z']**2); return out


def all_pairs_labels(C):
    """Labels of all ordered camera pairs (a != b) of one building; arrays of length n (n - 1)."""
    n = len(C); ia, ib = np.meshgrid(np.arange(n), np.arange(n), indexing='ij'); m = ia != ib; ia, ib = ia[m], ib[m]; z = _z(C)
    out = dict(a=C.i.values[ia], b=C.i.values[ib], x=C.e.values[ib] - C.e.values[ia], y=C.n.values[ib] - C.n.values[ia], z=z[ib] - z[ia], tilt=C.tilt.values[ib] - C.tilt.values[ia],
               heading=(C.heading.values[ib] - C.heading.values[ia] + 180.0) % 360.0 - 180.0)
    out['dist'] = np.sqrt(out['x']**2 + out['y']**2 + out['z']**2); return out
