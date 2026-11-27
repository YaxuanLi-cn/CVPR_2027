"""Independent absolute-image tilt constraints (not pair differences)."""
import numpy as np


def uniform_tilts(rng, n, low, high):
    if n < 1 or not np.isfinite([low, high]).all() or not low < high:
        raise ValueError('Invalid image tilt interval/count')
    counts = np.full(10, n//10)
    counts[:n % 10] += 1
    width = (high-low)/10
    values = np.concatenate([low+i*width+(np.arange(c)+rng.uniform(.05,.95,c))*width/c
                             for i,c in enumerate(counts) if c])
    rng.shuffle(values)
    return values


def validate_image_tilt(design, actual=None):
    spec = design.get('image_tilt_constraint')
    if spec is None:
        return None
    values = np.asarray([v['tilt'] for v in (actual if actual is not None else design['views'])])
    low, high = spec['range_deg']
    bins = spec['bins']
    if not len(values) or not np.isfinite(values).all() or np.any(values < low) or np.any(values > high):
        raise ValueError('Absolute image tilt outside required range')
    counts, edges = np.histogram(values, np.linspace(low, high, bins+1))
    # Counts differ by at most one for image counts not divisible by bin count.
    if counts.max()-counts.min() > 1:
        raise ValueError(f'Absolute image tilt is not balanced: {counts.tolist()}')
    return {'range_deg': [low, high], 'bins': bins, 'counts': counts.tolist(),
            'edges_deg': edges.tolist(), 'images': len(values),
            'min_deg': float(values.min()), 'max_deg': float(values.max()),
            'semantics': 'One vote per image; KML Camera tilt from nadir, not pair delta'}
