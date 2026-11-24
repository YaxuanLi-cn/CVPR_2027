#!/usr/bin/env python3
"""Track B: assemble outputs/supp/trackB/REPORT.md from pairuav.json / real.json (tables, protocol, settings, timings).
The hand-written findings and caveats live in supp/trackB/findings.md and are inserted verbatim."""
import sys, os, json, glob
import numpy as np
ROOT = '/root/rasr-pairuav'; OUT = f'{ROOT}/outputs/supp/trackB'; sys.path.append(f'{ROOT}/supp/trackB')
P = json.load(open(f'{OUT}/pairuav.json')); R = json.load(open(f'{OUT}/real.json')) if os.path.exists(f'{OUT}/real.json') else None
f1 = lambda v: f'{100 * v:.1f}'; rows = P['rows']


def prow(s, full=True):
    c = s['ci']; a = f"| {s['method']} | {s['variant']} | {s['scale']} | {s['n']} | {s['fail']:.1f} | {s['d_mae']:.2f} [{c['d_mae'][0]:.2f}, {c['d_mae'][1]:.2f}] | {s['d_med']:.2f} | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {s['epe_mae']:.2f} | {s['epe_med']:.2f} | {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | {s['theta_mae']:.2f} / {s['theta_med']:.3f} | {s['psi_mae']:.2f} / {s['psi_med']:.3f} | {s['dir_med']:.1f} | {f1(s['auc_5'])} / {f1(s['auc_10'])} / {f1(s['auc_20'])} |"
    return a + (f" {f1(s['sr_5dof_1m_1deg'])} | {s.get('thr_px') or '-'} |" if full else '')


HEAD = '| Method | variant | scale | n | fail % | d MAE (m) [95% CI] | d med. | SR_d@1m % [CI] | EPE mean | EPE med. | SR_EPE@1m % [CI] | SR_EPE@2m | e_theta MAE / med. (deg) | e_psi MAE / med. (deg) | dir. med. (deg) | AUC@5 / 10 / 20 |'
SEP = '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
L = ['# Track B — third-party geometric two-view baselines\n',
     'Everything below is produced by the scripts in `supp/trackB/`; numbers are read from `pairuav.json` / `real.json` in this folder. '
     'All metrics come from `supp/common.py` (range error d, endpoint error EPE, SR_d@1m = the SR of Table 2, SR_EPE@1m, rotation-only pose AUC).\n']

PROTOCOL = """## 0. Protocol

**Pairs.** PairUAV: the 1000 random test pairs (seed 0) of the 6,364-pair test set behind Table 2 that the archived direct baselines used (`reviewer_baselines/rb_recon.subset(1000, 'test')`), and 300 random pairs from training tours (`subset(300, 'train')`) as calibration pairs. Real photographs: the 300 pairs of Table 4 (`/root/realdata/mezieres/pred.pkl`, `dist_equiv <= 38`), plus the 300 pairs beyond the training range for the methods that were run on them.

**Methods.** Frozen public weights; nothing is trained or fine-tuned on PairUAV. SIFT + ratio test (OpenCV), SuperPoint + SuperGlue (archived matches: off-the-shelf single pass, and the paper's rotation-aware pass-1 matching), SuperPoint + LightGlue, ALIKED + LightGlue, LoFTR (outdoor), DeDoDe with the SO(2) steerer (rotation-equivariant matching), RoMa (outdoor), MASt3R (reciprocal descriptor matches; pointmap + PnP), DUSt3R (pointmap + PnP). Settings, checkpoints and run times are in the table of Section 2.

**Rotation handling.** *plain*: the pair as it is. *rot. TTA*: the target image is rotated by k x 90 deg (k = 0..3, `np.rot90`), matched four times, keypoints mapped back exactly to the unrotated image, the pose is solved for every rotation and the rotation with the most essential-matrix inliers is kept. SIFT is rotation invariant by construction (plain only). DeDoDe is matched with the C8-discretised SO(2) steerer (max similarity over the 8 steered descriptions), so its plain variant is already rotation-equivariant; its rot. TTA variant is reported too. The paper's rotation-aware SuperGlue row uses the paper's own rotation search (coarse matching confidence).

**Pose.** One solver for every matcher: `superglue_fusion/pose_from_matches.pose_features` (the solver of the archived direct baselines and of the paper's coarse stage, without priors / MAP / fusion): calibrated essential matrix with `cv2.findEssentialMat(USAC_MAGSAC)` (MAGSAC++, confidence 0.99999) on K0-normalised points, choice among the four decompositions by cheirality plus the zero-roll / tilt-range prior of the dataset, Sampson refinement on the inliers (twice), zero-roll decomposition into heading change and the two camera tilts, translation direction in the levelled frame of the source camera. Every match a method returns under its own recommended filtering is given to the solver (the archived SuperGlue matches keep their confidence >= 0.2 filter). A pair without a pose predicts zero motion and counts in "fail %".

**Inlier threshold.** The archived baselines used one fixed threshold of 3 px (pixels of the 3686-px image) for all matchers, although the matchers work on 512 to 1843 px images. Here the threshold is chosen per matcher and variant among {0.2, 0.4, 0.75, 1.5, 3, 6, 12} px on the 300 calibration pairs (criterion: share of calibration pairs whose translation direction is within 5 deg; failures are misses). The paper's rotation-aware SuperGlue row keeps its archived 3 px.

**Metric scale.** `native`: MASt3R's metric depth of the source view at the inlier keypoints / its metric pointmap, untouched. `calib`: native x one global constant (median of true / predicted distance on the calibration pairs). `height_p75ab` (the flight-height rows of the tables): baseline = H / h with h = camera height above the triangulated scene in baseline units, taken as the mean over both cameras of the 75th percentile of the triangulated inlier heights (the ground level; this is the statistic of DreamNav's own flight-height cue `s_hgtab_p75`), and H = median of true distance x h on the calibration pairs. `height_p50a` (JSON only): the same with the median inlier height below the source camera, the statistic of the archived direct baselines, which is biased by roofs. `height_pointmap` (PnP rows): H / median height of the source pointmap. `learned`: unit translation direction of the method x the distance regressed by the frozen-MASt3R-statistics MLP (`n_dist`; trained on the 47,687 training pairs; the learned-prior branch of DreamNav). `blend`: log d = w log(H / h) + (1 - w) log(d_MLP) + b with (w, b) fitted on the calibration pairs.

**Evaluation.** `supp/common.py` only, in the levelled frame of the source camera (lx, ly, z against the labels lx, ly, z; no absolute heading involved). DreamNav Stage I / II are the archived predictions in the same frame (`common.dreamnav(T, pre, 'cam')`). 95% confidence intervals: bootstrap over buildings (PairUAV) or over source photographs (real), 1000 resamples.
"""
L.append(PROTOCOL)

# ---------------------------------------------------------------- compact PairUAV table
def best_variant(m):
    """rows of one matcher: the rot.-TTA variant if it exists, otherwise its only variant."""
    rs = [r for r in rows if r.get('matcher') == m]; v = 'rot. TTA' if any(r['variant'] == 'rot. TTA' for r in rs) else (rs[0]['variant'] if rs else None)
    return [r for r in rs if r['variant'] == v]
order = ['sift', 'sgplain', 'sgrot', 'splg', 'aliked', 'loftr', 'dedode', 'roma', 'mast3r', 'mast3r_pnp', 'dust3r_pnp']
L += ['## 1. PairUAV: summary table\n', f"{P['n_test']} random test pairs of the Table-2 test set (the same pairs as `outputs/reviewer_baselines`), evaluated in the levelled frame of the source camera. "
      'One line per method: its rotation-handling variant (rot. TTA where the method is not rotation invariant) with the flight-height scale prior and with the learned distance. Failed pairs predict zero motion. 95% confidence intervals: bootstrap over buildings.\n', HEAD, SEP]
for m in order:
    for r in best_variant(m):
        if r['scale'] in ('height_p75ab', 'height_pointmap', 'learned', 'blend'): L.append(prow(r, False))
for r in rows:
    if r.get('matcher') in ('mlp', 'dreamnav', 'zero'): L.append(prow(r, False))
sub = [r for r in rows if r.get('matcher') == 'dreamnav_subset']
if sub: L += ['', 'Reference rows on the subsets of the matchers that were run on fewer pairs:\n', HEAD, SEP] + [prow(r, False) for r in sub]

# ---------------------------------------------------------------- plain vs TTA failure / rotation table
L += ['\n### Rotation handling\n', 'Share of pairs without a pose ("fail %") and angle errors, target image as is versus rot. TTA (scale-independent columns; height-prior rows).\n',
      '| Method | variant | fail % | e_theta MAE / med. | e_psi MAE / med. | dir. med. | AUC@5 / 10 / 20 | rotations chosen (0 / 90 / 180 / 270 deg) | median inliers | inlier thr (px) |', '|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
for r in rows:
    if r['scale'] == 'height_p75ab' and r.get('matcher') in order[:9]:
        L.append(f"| {r['method']} | {r['variant']} | {r['fail']:.1f} | {r['theta_mae']:.2f} / {r['theta_med']:.3f} | {r['psi_mae']:.2f} / {r['psi_med']:.3f} | {r['dir_med']:.1f} | {f1(r['auc_5'])} / {f1(r['auc_10'])} / {f1(r['auc_20'])} | {r.get('rot_hist', '-')} | {r.get('inl_med', float('nan')):.0f} | {r.get('thr_px')} |")

# ---------------------------------------------------------------- full PairUAV table
L += ['\n## 2. PairUAV: all rows\n', 'Every matcher x variant x scale. `native` = MASt3R metric depth untouched; `calib` = native x one constant; `height` = flight-height prior; `learned` = direction of the method x distance of the frozen-MASt3R-statistics MLP.\n',
      HEAD + ' 5-DoF (1 m, 1 deg) | thr (px) |', SEP + '---:|---:|'] + [prow(r) for r in rows if r['scale'] != 'height_p50a']

# ---------------------------------------------------------------- thresholds / constants / timing
L += ['\n### Inlier threshold selected on the 300 calibration pairs\n', 'Candidates: 0.2, 0.4, 0.75, 1.5, 3, 6, 12 px (pixels of the 3686-px image). Criterion: share of calibration pairs whose translation direction is within 5 deg (failed pairs are misses).\n',
      '| matcher / variant | chosen (px) | calibration pairs | share with direction error < 5 deg, per candidate | median direction error (deg), per candidate |', '|---|---:|---:|---|---|']
for k, v in P['threshold_selection'].items():
    L.append(f"| {k} | {v['chosen']} | {v['calibration_pairs']} | " + ', '.join(f'{t}: {100 * x:.1f}%' for t, x in v['share_dir_err_below_5deg_calibration'].items()) + ' | ' + ', '.join(f'{t}: {x:.2f}' for t, x in v['median_dir_err_calibration'].items()) + ' |')
if P.get('threshold_sensitivity_test'):
    L += ['\n### Sensitivity to the inlier threshold on the test pairs (flight-height scale; for information, not used for selection)\n', '| matcher / variant | thr (px) | n | fail % | SR_d@1m | SR_EPE@1m | d MAE | EPE med. | dir. med. | e_theta med. | e_psi med. | fitted H (m) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k, d in P['threshold_sensitivity_test'].items():
        if len(d) < 2: continue
        for t, v in d.items(): L.append(f"| {k} | {t} | {v['n']} | {v['fail']:.1f} | {f1(v['sr_d_1m'])} | {f1(v['sr_epe_1m'])} | {v['d_mae']:.2f} | {v['epe_med']:.2f} | {v['dir_med']:.2f} | {v['theta_med']:.3f} | {v['psi_med']:.3f} | {v['H']:.1f} |")
L += ['\n### Scale constants fitted on the calibration pairs\n', '| method / variant / scale | constant | fitted on (pairs with a pose) |', '|---|---:|---:|']
for k, v in P['constants'].items(): L.append(f"| {k} | {v['const']:.2f}" + (f" (w), offset b = {v['offset']:+.3f}, H = {v['H']:.1f}" if 'offset' in v else '') + f" | {v['fitted_on']} |")
L += ['\n`calib` constants are dimensionless (true distance / distance in MASt3R units); `height` constants are the fitted flight height in metres.\n']
L += ['### Matcher settings and run time\n', 'Seconds per pair on the RTX 5090 **while it was shared with four other experiment tracks** (GPU at ~99% utilisation throughout), so these are upper bounds, not clean latencies. '
      '"plain" = one forward pass on (source, target); "4 rot." = all four rotations of the target. The essential-matrix solver runs on one CPU thread.\n',
      '| matcher | plain (s) | 4 rot. (s) | solver (s / pose) | peak GPU mem. of the process (GB) | settings |', '|---|---:|---:|---:|---:|---|']
for m, t in P['timing'].items():
    st = t.get('settings', {}); L.append(f"| {m} | {t.get('plain_s', float('nan')):.2f} | {(t.get('tta_s') or float('nan')):.2f} | {(t.get('solver_s_per_pose') or float('nan')):.3f} | {t.get('peak_gpu_gb', float('nan')):.1f} | " + '; '.join(f'{k}={v}' for k, v in st.items()) + ' |')

# ---------------------------------------------------------------- real photographs
if R:
    L += ['\n## 3. Real photographs (Mezieres, DJI Phantom 4 RTK), zero-shot\n',
          'Same pairs as Table 4. Nothing is fitted on these photographs; the inlier thresholds are those selected on PairUAV. `height60` = baseline = 60 m / (camera height above the triangulated scene in baseline units), the flight height the aircraft reports, '
          'which is the side information the DreamNav rows use too (their output is multiplied by 60 / 220.18). Pos. med. = median endpoint error in the frame of the first camera; SR@1m / SR@2m (range) = the columns of Table 4. 95% CI: bootstrap over source photographs.\n']
    RH = '| Method | variant | scale | n | fail % | Pos. med. (m) [95% CI] | EPE mean | SR@1m (range) % [CI] | SR@2m (range) | SR_EPE@1m % [CI] | SR_EPE@2m | d MAE | e_theta MAE / med. | e_psi MAE / med. | dir. med. | AUC@5 / 10 / 20 | thr (px) |'
    for which, title in (('real_in', 'Within the training range (the pairs of Table 4)'), ('real_out', 'Beyond the training range')):
        if which not in R['sets']: continue
        S = R['sets'][which]; L += [f"\n### {title}: {S['n']} pairs, mean true distance {S['mean_true_distance']:.1f} m\n", RH, '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for s in S['rows']:
            if s['scale'] == 'height60_p50a': continue
            c = s['ci']; L.append(f"| {s['method']} | {s['variant']} | {s['scale']} | {s['n']} | {s['fail']:.1f} | {s['epe_med']:.2f} [{c['epe_med'][0]:.2f}, {c['epe_med'][1]:.2f}] | {s['epe_mae']:.2f} | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {f1(s['sr_d_2m'])} | "
                                  f"{f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {f1(s['sr_epe_2m'])} | {s['d_mae']:.2f} | {s['theta_mae']:.2f} / {s['theta_med']:.3f} | {s['psi_mae']:.2f} / {s['psi_med']:.3f} | {s['dir_med']:.1f} | {f1(s['auc_5'])} / {f1(s['auc_10'])} / {f1(s['auc_20'])} | {s.get('thr_px') or '-'} |")
# ---------------------------------------------------------------- generated sub-reports: SIFT on the full test set, paired comparison, unscreened pairs
def include(fn, title, intro):
    f = f'{OUT}/{fn}'
    if os.path.exists(f):
        body = open(f).read().split('\n'); body = [('#' + b if b.startswith('#') else b) for b in body[1:] if True]      # drop the file's own title, demote its headings
        L.extend(['', f'## {title}\n', intro, ''] + body)
include('sift_full.md', '4. The classical baseline on the test set of Table 2 (`eval_sift_full.py`)', 'SIFT + ratio test + E (MAGSAC++) + flight-height prior on the test pairs behind Table 2 (all pairs processed so far, fixed random order), with the paired comparison against DreamNav, who succeeds where the other fails, and the comparison as a function of the SIFT inliers that the dataset screen recorded for each pair.')
include('paired_sift.md', '5. Paired comparison on the 1000-pair subset (`compare_best.py sift`)', 'Error tails, paired differences, scale diagnostics and per-bin success of the SIFT baseline against DreamNav on the 1000 test pairs of Sections 1-2.')
include('paired_roma_tta.md', '5b. Paired comparison of RoMa (rot. TTA) with DreamNav on the RoMa subset (`compare_best.py roma tta`)', 'Same analysis for the strongest learned matcher, on the random subset of test pairs it was run on.')
include('unscreened.md', '6. Unscreened pairs (`eval_unscreened.py`)', 'Every released PairUAV pair passed a SIFT + MAGSAC screen. These are pairs drawn uniformly from all camera pairs of the test buildings by Track D (`outputs/supp/trackD/cache/unfiltered_pairs.pkl`); DreamNav rows come from Track D\'s run of the online engine on the same pairs.')

# ---------------------------------------------------------------- run conditions and deviations (facts needed to read the tables)
nsub = {}
for r in rows:
    if r.get('matcher') in ('roma', 'mast3r') and r['variant'] == 'rot. TTA' and r['scale'] == 'height_p75ab': nsub[r['matcher']] = r['n']
ncal = {k.split('|')[0]: v['calibration_pairs'] for k, v in P['threshold_selection'].items() if k.endswith('|tta')}
L += ['\n## 7. Run conditions and deviations\n',
      '- **Shared GPU.** Four other experiment tracks used the GPU throughout (99-100 % utilisation, up to ten processes), so run times are upper bounds and the slowest methods were run on random subsets. Pairs are processed in a fixed random order, so every subset is a random sample; DreamNav reference rows on exactly those pairs are in the tables. '
      + 'Subsets on PairUAV: ' + '; '.join(f'{m}: {nsub[m]} test pairs, {ncal.get(m, "?")} calibration pairs' for m in nsub) + '. Subsets on the real pairs: see column n.',
      '- **RoMa** was run with `use_custom_corr=False` (the optional compiled `local_corr` extension is not installed; same model and weights, pure-PyTorch local correlation). The process needs about 7.4 GB of GPU memory.',
      '- **DeDoDe**: descriptor B with the SO(2) "spread" steerer discretised to C8 (the configuration recommended by the steerers authors), through kornia 0.9.2 with the official weights; the max-similarity matcher is re-implemented from the official source. DeDoDe-G (DINOv2 ViT-g/14, 4.5 GB of weights) was not run.',
      '- **MASt3R reciprocal matching**: `mast3r.fast_nn.fast_reciprocal_NNs` takes about 10 s per call on the shared GPU (hundreds of small GPU-CPU synchronisations). `supp/trackB/fastnn_gpu.py` is a GPU-native version with identical output (29,855 of 29,855 correspondences on 16 image pairs, `supp/trackB/check_fastnn.py`) and was used instead. On PairUAV only rotation 0 and the rotation that had the most inliers in the archived run were re-matched. '
      'The MASt3R / DUSt3R pointmap + PnP rows on PairUAV are the archived forward passes of `outputs/reviewer_baselines`, re-scored; their flight-height variant uses the median height of the source pointmap (`height_pointmap`), the only statistic that was archived.',
      '- **Inlier threshold** per matcher, chosen on the calibration pairs, instead of the fixed 3 px of the archived baselines. The paper\'s rotation-aware SuperGlue row keeps 3 px (its two-set solver has no threshold argument). For plain variants that fail on about half of the pairs the criterion is nearly flat, so the value chosen for them is arbitrary.',
      '- **Flight-height statistic**: `height_p75ab` everywhere in the tables; the `height_p50a` rows of the archived baselines are in the JSON files only.',
      '- **SIFT and the dataset screen.** The pair metadata of PairUAV records a SIFT + MAGSAC screen (`visual_screen: SIFT mutual ratio + USAC_MAGSAC`; `sift_inliers` >= 20 for every pair, median 392 on the 1000 test pairs), so SIFT matching succeeds on every PairUAV test pair by construction. The real photographs were not screened.']
for fn in ('files_and_repro.md',):
    if os.path.exists(f'{ROOT}/supp/trackB/{fn}'): L += ['', open(f'{ROOT}/supp/trackB/{fn}').read()]
open(f'{OUT}/REPORT.md', 'w').write('\n'.join(L) + '\n'); print('wrote', f'{OUT}/REPORT.md', len(L), 'lines')
