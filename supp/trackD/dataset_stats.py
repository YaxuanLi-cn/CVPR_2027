#!/usr/bin/env python3
"""Track D, item 5: PairUAV statistics read from the released metadata only (KML camera files + pair jsonl files).

Nothing here is inferred about how pairs were annotated: only what the files contain is reported.
  cameras  : 1,412 buildings x 150 cameras (longitude, latitude, absolute altitude, heading, tilt, roll)
  pairs    : 1,412 x 1000 selected directed pairs with the 5-D label and the screening statistics stored with each pair
  anchors  : per building, the anchor point and the terrain altitude (mean sea level) probed below it
Outputs (outputs/supp/trackD/): dataset_stats.json, cache/cameras.pkl, cache/pairs_selected.pkl, fig_dataset_height_tilt.{pdf,png},
fig_dataset_pose_diff.{pdf,png}
Usage: dataset_stats.py [procs]
"""
import sys, os, re, json, glob, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/supp/trackD')
from labels import DATA, load_cameras_raw, tour_anchor, enu_at_anchor, pair_labels, all_pairs_labels
OUT = '/root/rasr-pairuav/outputs/supp/trackD'; os.makedirs(f'{OUT}/cache', exist_ok=True)
try:
    import orjson; loads = orjson.loads
except Exception: loads = json.loads
QS = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]


def q(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; return {('min' if p == 0 else 'max' if p == 100 else f'p{p:g}'): float(np.percentile(v, p)) for p in QS} | {'mean': float(v.mean()), 'std': float(v.std()), 'n': int(len(v))}


def read_pairs(tour):
    rows = []
    with open(f'{DATA}/pairs/{tour}.jsonl', 'rb') as f:
        for line in f:
            r = loads(line); d = r['delta']; a = r['anchor']
            rows.append((r['tour_id'], r['pair_id'], r['a'], r['b'], d['x_m'], d['y_m'], d['z_m'], d['tilt_deg'], d['heading_deg'], r['inliers'], r['matches'], r['inlier_ratio'], r['hull_fraction'], r['grid_cells'], r['model'], r.get('visual_screen', ''),
                         a['longitude'], a['latitude'], a['altitude'], a['terrain_altitude_msl'], a.get('terrain_calibration', {}).get('method', ''), a.get('feature_key', ''), a.get('semantic_note', '')))
    return rows


def per_building(tour):
    """Camera table of one building + pose differences of ALL 150 x 149 directed pairs (summary only)."""
    C = load_cameras_raw(tour); anc = tour_anchor(tour); C = enu_at_anchor(C, anc); C['tour'] = tour; C['hag'] = C.altitude - anc['terrain_altitude_msl']
    A = all_pairs_labels(C)                                  # dict of arrays over the 22,350 directed pairs
    hist = {k: np.histogram(A[k], bins=BINS[k])[0] for k in BINS}
    ext = dict(tour=tour, n_cam=len(C), e_range=float(C.e.max() - C.e.min()), n_range=float(C.n.max() - C.n.min()), u_range=float(C.u.max() - C.u.min()), max_pair_dist=float(A['dist'].max()), hag_mean=float(C.hag.mean()), hag_min=float(C.hag.min()), hag_max=float(C.hag.max()),
               lat=float(anc['latitude']), lon=float(anc['longitude']), terrain=float(anc['terrain_altitude_msl']), anchor_alt=float(anc['altitude']))
    mx = {k: (float(A[k].min()), float(A[k].max())) for k in ('x', 'y', 'z', 'tilt', 'heading', 'dist')}
    return C[['tour', 'i', 'longitude', 'latitude', 'altitude', 'heading', 'tilt', 'roll', 'hfov', 'e', 'n', 'u', 'hag']], hist, ext, mx, {k: A[k].astype(np.float32) for k in ('x', 'y', 'z', 'tilt', 'heading', 'dist')}


BINS = dict(x=np.linspace(-45, 45, 181), y=np.linspace(-45, 45, 181), z=np.linspace(-45, 45, 181), tilt=np.linspace(-50, 50, 201), heading=np.linspace(-180, 180, 181), dist=np.linspace(0, 70, 141))

if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tours = sorted(f[:-6] for f in os.listdir(f'{DATA}/pairs') if f.endswith('.jsonl')); print('buildings', len(tours), flush=True)
    # ---------------- selected pairs
    f = f'{OUT}/cache/pairs_selected.pkl'
    if os.path.exists(f): P = pd.read_pickle(f)
    else:
        with Pool(procs) as p: R = p.map(read_pairs, tours, chunksize=8)
        P = pd.DataFrame([r for rr in R for r in rr], columns=['tour', 'pair_id', 'a', 'b', 'x', 'y', 'z', 'tilt', 'heading', 'inliers', 'matches', 'inlier_ratio', 'hull_fraction', 'grid_cells', 'model', 'visual_screen',
                                                               'anchor_lon', 'anchor_lat', 'anchor_alt', 'terrain_msl', 'terrain_method', 'feature_key', 'semantic_note'])
        P.to_pickle(f)
    P['dist'] = np.sqrt(P.x**2 + P.y**2 + P.z**2); P['hdist'] = np.hypot(P.x, P.y); print('selected pairs', len(P), 'per building', P.groupby('tour').size().describe()[['min', 'max']].to_dict(), flush=True)
    # ---------------- cameras + all candidate pairs
    with Pool(procs) as p: R = p.map(per_building, tours, chunksize=4)
    C = pd.concat([r[0] for r in R], ignore_index=True); C.to_pickle(f'{OUT}/cache/cameras.pkl'); EXT = pd.DataFrame([r[2] for r in R]); EXT.to_pickle(f'{OUT}/cache/buildings.pkl')
    H_all = {k: sum(r[1][k] for r in R) for k in BINS}; print('cameras', len(C), flush=True)
    ALL = {k: np.concatenate([r[4][k] for r in R]) for k in ('x', 'y', 'z', 'tilt', 'heading', 'dist')}; n_all = len(ALL['x'])
    # ---------------- label check: jsonl labels vs labels rebuilt from the KML files (all selected pairs)
    Ci = C.set_index(['tour', 'i']); ia = pd.MultiIndex.from_arrays([P.tour, P.a]); ib = pd.MultiIndex.from_arrays([P.tour, P.b]); A = Ci.loc[ia].reset_index(drop=True); B = Ci.loc[ib].reset_index(drop=True)
    chk = dict(x=np.abs((B.e - A.e).values - P.x.values).max(), y=np.abs((B.n - A.n).values - P.y.values).max(), z_altitude_difference=np.abs((B.altitude - A.altitude).values - P.z.values).max(), z_enu_up=np.abs((B.u - A.u).values - P.z.values).max(),
               tilt=np.abs((B.tilt - A.tilt).values - P.tilt.values).max(), heading=np.abs(((B.heading - A.heading).values + 180) % 360 - 180 - P.heading.values).max())
    print('label reconstruction, max abs error over all selected pairs:', {k: float(v) for k, v in chk.items()}, flush=True)
    # ---------------- statistics
    S = {'buildings': len(tours), 'cameras': int(len(C)), 'cameras_per_building': C.groupby('tour').size().describe()[['min', 'max']].to_dict(), 'selected_pairs': int(len(P)), 'selected_pairs_per_building': P.groupby('tour').size().describe()[['min', 'max']].to_dict(),
         'candidate_directed_pairs_per_building': 150 * 149, 'selected_fraction': float(len(P) / n_all), 'label_reconstruction_max_abs_error': {k: float(v) for k, v in chk.items()}}
    S['camera'] = dict(height_above_terrain_m=q(C.hag), height_above_terrain_note='KML absolute altitude (altitudeMode=absolute, metres above mean sea level) minus anchor.terrain_altitude_msl, the terrain altitude at the building anchor; it is the height above the terrain at the anchor, not above roofs',
                       tilt_deg=q(C.tilt), heading_deg=q(C.heading), roll_deg=q(C.roll), kml_horizFov_values=C.hfov.value_counts().to_dict(),
                       heading_hist_12=np.histogram(C.heading % 360, bins=np.linspace(0, 360, 13))[0].tolist(), tilt_hist_9=np.histogram(C.tilt, bins=np.linspace(0, 45, 10))[0].tolist(),
                       building_mean_height_m=q(EXT.hag_mean), within_building_height_range_m=q(EXT.hag_max - EXT.hag_min), within_building_east_extent_m=q(EXT.e_range), within_building_north_extent_m=q(EXT.n_range), within_building_up_extent_m=q(EXT.u_range),
                       within_building_max_pair_distance_m=q(EXT.max_pair_dist), anchor_altitude_field=q(EXT.anchor_alt), terrain_altitude_msl=q(EXT.terrain),
                       camera_offset_from_anchor_east_m=q(C.e), camera_offset_from_anchor_north_m=q(C.n), camera_horizontal_offset_from_anchor_m=q(np.hypot(C.e, C.n)))
    S['selected_pairs_labels'] = {k: q(P[k]) for k in ('x', 'y', 'z', 'tilt', 'heading', 'dist', 'hdist')} | {f'abs_{k}': q(P[k].abs()) for k in ('x', 'y', 'z', 'tilt', 'heading')}
    S['selected_pairs_short_baselines'] = {f'lt_{t}m': float((P.dist < t).mean()) for t in (0.5, 1, 2, 3, 5, 10)}
    try:   # baselines of the pairs the fusion stages were fitted on (train rows of the archived result set)
        SGD = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; Tz = pd.read_pickle(f'{SGD}/half_zoom_pred.pkl'); Qz = pd.read_pickle(f'{SGD}/pairs.pkl'); Qz['dist'] = np.sqrt(Qz.x**2 + Qz.y**2 + Qz.z**2); trz = Qz[Qz.manifest_index.isin(Tz[Tz.split == 'train'].manifest_index)]
        S['fusion_training_rows_short_baselines'] = dict(rows=int(len(trz)), **{f'lt_{t}m': int((trz.dist < t).sum()) for t in (1, 2, 3, 5)})
    except Exception as e: print('fusion training rows not available:', e)
    S['all_candidate_pairs_labels'] = {k: q(ALL[k]) for k in ALL} | {f'abs_{k}': q(np.abs(ALL[k])) for k in ('x', 'y', 'z', 'tilt', 'heading')}
    rng_sel = {k: (float(P[k].min()), float(P[k].max())) for k in ('x', 'y', 'z', 'tilt', 'heading')}; inside = np.ones(n_all, bool); per = {}
    for k, (lo, hi) in rng_sel.items(): m = (ALL[k] >= lo) & (ALL[k] <= hi); per[k] = float(m.mean()); inside &= m
    S['candidates_vs_selected'] = dict(selected_label_range=rng_sel, fraction_of_all_candidates_inside_range_per_component=per, fraction_of_all_candidates_inside_all_five_ranges=float(inside.mean()),
                                       selected_over_candidates_inside_all_ranges=float(len(P) / inside.sum()), candidates=int(n_all))
    S['screening'] = {k: q(P[k]) for k in ('matches', 'inliers', 'inlier_ratio', 'hull_fraction', 'grid_cells')}
    S['screening']['grid_cells_counts'] = {int(k): int(v) for k, v in P.grid_cells.value_counts().sort_index().items()}
    S['screening']['model_counts'] = P.model.value_counts().to_dict(); S['screening']['visual_screen_strings'] = P.visual_screen.value_counts().to_dict()
    S['screening']['terrain_method'] = P.terrain_method.value_counts().to_dict(); S['screening']['feature_key'] = P.feature_key.value_counts().to_dict(); S['screening']['semantic_note'] = P.semantic_note.value_counts().to_dict()
    S['screening']['minima'] = {k: float(P[k].min()) for k in ('matches', 'inliers', 'inlier_ratio', 'hull_fraction', 'grid_cells')}
    S['screening']['share_below'] = dict(inliers_lt_30=float((P.inliers < 30).mean()), inliers_lt_50=float((P.inliers < 50).mean()), inliers_lt_100=float((P.inliers < 100).mean()), inlier_ratio_lt_0p7=float((P.inlier_ratio < 0.7).mean()), hull_lt_0p1=float((P.hull_fraction < 0.1).mean()))
    # balancing: groups of buildings named in the pair id (<tour>_group_<first>_<last>_<index>); five equal-width bins per component within each group
    g = P.pair_id.str.extract(r'_group_(\d+)_(\d+)_')[0] + '-' + P.pair_id.str.extract(r'_group_(\d+)_(\d+)_')[1]; P['group'] = g; gs = P.groupby('group'); sizes = gs.tour.nunique()
    bal = {}
    for k in ('x', 'y', 'z', 'tilt', 'heading'):
        dev = []
        for _, d in gs:
            v = d[k].values; c = np.histogram(v, bins=np.linspace(v.min(), v.max(), 6))[0]; dev.append((c.max() - c.min()) / len(v))
        bal[k] = dict(median_relative_spread_of_5_bin_counts=float(np.median(dev)), max=float(np.max(dev)))
    S['balancing'] = dict(groups=int(len(sizes)), buildings_per_group=sizes.value_counts().sort_index().to_dict(), five_equal_width_bins_within_group=bal,
                          global_5bin_counts={k: np.histogram(P[k], bins=np.linspace(P[k].min(), P[k].max(), 6))[0].tolist() for k in ('x', 'y', 'z', 'tilt', 'heading')})
    json.dump(S, open(f'{OUT}/dataset_stats.json', 'w'), indent=1)
    np.savez_compressed(f'{OUT}/cache/candidate_hist.npz', **{f'all_{k}': H_all[k] for k in BINS}, **{f'bins_{k}': BINS[k] for k in BINS}, **{f'sel_{k}': np.histogram(P[k], bins=BINS[k])[0] for k in BINS})
    # ---------------- figures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'font.family': 'serif', 'axes.linewidth': 0.6})
    C1, C2 = '#2a78d6', '#eb6834'
    fig, ax = plt.subplots(1, 2, figsize=(3.3, 1.45))
    ax[0].hist(C.hag, bins=np.arange(np.floor(C.hag.min() / 5) * 5, C.hag.max() + 5, 2.5), color=C1); ax[0].set_xlabel('camera height above terrain (m)'); ax[0].set_ylabel('cameras'); ax[0].set_xlim(np.percentile(C.hag, 0.05) - 5, np.percentile(C.hag, 99.95) + 5)
    ax[1].hist(C.tilt, bins=np.linspace(0, 45, 46), color=C1); ax[1].set_xlabel('camera tilt from nadir (deg)'); ax[1].set_xlim(0, 45)
    for a in ax: a.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); a.yaxis.get_offset_text().set_fontsize(6)
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(f'{OUT}/fig_dataset_height_tilt.pdf'); fig.savefig(f'{OUT}/fig_dataset_height_tilt.png', dpi=300); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(3.3, 2.3)); lab = dict(x=r'$\Delta x$ (m)', y=r'$\Delta y$ (m)', z=r'$\Delta z$ (m)', tilt=r'$\Delta\psi$ (deg)', heading=r'$\Delta\theta$ (deg)', dist='baseline (m)')
    for a, k in zip(ax.ravel(), ('x', 'y', 'z', 'tilt', 'heading', 'dist')):
        b = BINS[k]; cen = (b[:-1] + b[1:]) / 2; w = b[1] - b[0]; ha = H_all[k] / H_all[k].sum() / w; hs = np.histogram(P[k], bins=b)[0]; hs = hs / hs.sum() / w
        a.fill_between(cen, ha, step='mid', color='#c9c8c2', lw=0, label='all camera pairs'); a.step(cen, hs, where='mid', color=C2, lw=0.8, label='selected pairs'); a.set_xlabel(lab[k]); a.set_yticks([]); a.set_ylim(0)
        nz = np.nonzero(H_all[k])[0]; a.set_xlim(b[nz[0]], b[nz[-1] + 1])
    ax[0, 0].legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.05, 1.45), ncol=2, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.4, rect=(0, 0, 1, 0.93)); fig.savefig(f'{OUT}/fig_dataset_pose_diff.pdf'); fig.savefig(f'{OUT}/fig_dataset_pose_diff.png', dpi=300); plt.close(fig)
    print(json.dumps({k: S[k] for k in ('buildings', 'cameras', 'selected_pairs', 'selected_fraction', 'label_reconstruction_max_abs_error', 'candidates_vs_selected', 'balancing')}, indent=1))
    print('height above terrain', S['camera']['height_above_terrain_m']); print('tilt', S['camera']['tilt_deg']); print('screening minima', S['screening']['minima'], S['screening']['share_below'])
