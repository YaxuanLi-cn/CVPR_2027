"""Necessary marginal capacity bounds, respecting unordered pair uniqueness.

These bounds identify shortages. Passing them does NOT establish simultaneous
five-axis feasibility, which remains the job of the integer solver.
"""
from itertools import combinations
from common import AXES


def capacity_diagnostics(edges, tours, per_tour, bins, tilt_limit):
    quota = len(tours) * per_tour // bins
    by_tour = {t: [] for t in tours}
    for edge in edges:
        by_tour[edge['tour']].append(edge)
    result = {'interpretation': 'Necessary optimistic bounds only; not a joint feasibility proof.',
              'per_tour_edges': {t: len(es) for t, es in by_tour.items()},
              'tour_shortfalls': {t: per_tour-len(es) for t, es in by_tour.items() if len(es)<per_tour},
              'axes': {}, 'shortages': []}
    for axis, (lo, hi) in AXES.items():
        if axis == 'tilt_deg': lo, hi = -tilt_limit, tilt_limit
        masks = {}
        for tour, es in by_tour.items():
            masks[tour] = []
            for edge in es:
                mask = 0
                for direction in (1, -1):
                    value = direction * edge['delta'][axis]
                    idx = max(0, min(bins-1, int((value-lo)/(hi-lo)*bins)))
                    mask |= 1 << idx
                masks[tour].append(mask)
        counts = {t: [sum(bool(mask & (1 << i)) for mask in ms) for i in range(bins)]
                  for t, ms in masks.items()}
        result['axes'][axis] = {'bin_edges': [lo+(hi-lo)*i/bins for i in range(bins+1)],
                               'per_tour_available_unordered_edges': counts,
                               'target_per_bin': quota}
        # Check single bins and unions, because the same unordered edge cannot
        # supply both orientations. All subsets are cheap for the default 5 bins.
        sizes = range(1, bins+1) if bins <= 10 else (1, 2, bins)
        for size in sizes:
            for subset in combinations(range(bins), size):
                subset_mask = sum(1 << i for i in subset)
                supply = {t: min(per_tour, sum(bool(m & subset_mask) for m in ms)) for t, ms in masks.items()}
                capacity = sum(supply.values())
                required = quota * size
                if capacity < required:
                    result['shortages'].append({'axis': axis, 'bins': list(subset),
                        'required': required, 'optimistic_capacity': capacity,
                        'deficit': required-capacity, 'per_tour_capacity': supply})
    result['necessary_bounds_pass'] = not result['tour_shortfalls'] and not result['shortages']
    return result
