"""Camera-frame labels for the Table 2 baselines (Track E).

The PairUAV label (x east, y north) is replaced by the displacement in the SOURCE camera's levelled frame
    lx = cos(h_a) x - sin(h_a) y   (to the right of the source camera)
    ly = sin(h_a) x + cos(h_a) y   (horizontal-forward of the source camera)
with h_a the absolute heading of the source camera (degrees, clockwise from north). z, tilt change and heading change are
unchanged. For a recipe that swaps the two images, the label of the swapped pair (source b, target a) is
    (lx', ly') = level(h_b, -x, -y),  z' = -z,  tilt' = -tilt,  heading' = -heading
i.e. it needs the heading of b; it is NOT the negated label as in the east-north frame.
"""
import numpy as np, pandas as pd
PAIRS = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/pairs.pkl'


def level(h_deg, x, y):
    h = np.deg2rad(np.asarray(h_deg, float)); return np.cos(h) * x - np.sin(h) * y, np.sin(h) * x + np.cos(h) * y


def table():
    """One row per pair of the 60k split, indexed by manifest_index: ENU label, camera-frame label, swapped camera-frame label."""
    P = pd.read_pickle(PAIRS)[['manifest_index', 'pair_id', 'split', 'tour', 'x', 'y', 'z', 'tilt', 'heading', 'heading_a', 'heading_b']].copy()
    P['lx'], P['ly'] = level(P.heading_a.values, P.x.values, P.y.values)
    P['lx_sw'], P['ly_sw'] = level(P.heading_b.values, -P.x.values, -P.y.values)
    return P.set_index('manifest_index')


def by_pair_id():
    T = table().reset_index(); return T.set_index('pair_id')
