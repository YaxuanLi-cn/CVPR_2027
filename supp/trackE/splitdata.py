"""Sample order of the json-based baselines (dinov3/train_pairuav5d.PairDataset): sorted tour dirs x sorted json files."""
import json
from pathlib import Path
import numpy as np
ROOT = Path('/root/baseline/_mast3r_split_data_v2'); IMAGE_ROOT = Path('/root/pairUAV-5D-temp')


def ordered(name):
    """-> list of dicts (manifest_index, image_a, image_b, x, y, z, tilt, heading) in PairDataset order."""
    cache = Path(__file__).parent / f'_order_{name}.npz'
    if cache.exists():
        d = np.load(cache, allow_pickle=False); return [dict(manifest_index=int(m), image_a=str(a), image_b=str(b), label=l.tolist()) for m, a, b, l in zip(d['manifest_index'], d['image_a'], d['image_b'], d['label'])]
    rows = []
    for tour_dir in sorted(p for p in (ROOT / name).iterdir() if p.is_dir()):
        for path in sorted(tour_dir.glob('*.json')):
            r = json.loads(path.read_text()); d = r['delta']
            rows.append(dict(manifest_index=int(r['manifest_index']), image_a=str(IMAGE_ROOT / r['image_a']), image_b=str(IMAGE_ROOT / r['image_b']),
                             label=[float(d['x_m']), float(d['y_m']), float(d['z_m']), float(r['tilt_num']), float(r['heading_num'])]))
    np.savez(cache, manifest_index=np.array([r['manifest_index'] for r in rows]), image_a=np.array([r['image_a'] for r in rows]), image_b=np.array([r['image_b'] for r in rows]), label=np.array([r['label'] for r in rows]))
    return rows


def labels(rows, frame):
    """[n, 5] float32 label array (x|lx, y|ly, z, tilt, heading) for the given frame, checked against the pair table."""
    from camlabels import table
    T = table().loc[[r['manifest_index'] for r in rows]]; L = np.array([r['label'] for r in rows], dtype=np.float64)
    assert np.abs(T[['x', 'y', 'z', 'tilt', 'heading']].values - L).max() < 1e-9
    if frame == 'cam': L[:, 0], L[:, 1] = T.lx.values, T.ly.values
    return L.astype(np.float32)
