#!/usr/bin/env python3
"""Assemble outputs/supp/trackD/REPORT.md from the result files of the five items (whatever is available)."""
import os, json, glob, datetime
OUT = '/root/rasr-pairuav/outputs/supp/trackD'
def frag(name): f = f'{OUT}/{name}'; return open(f).read().strip() if os.path.exists(f) else f'*({name} not available yet)*'
def J(name): f = f'{OUT}/{name}'; return json.load(open(f)) if os.path.exists(f) else None
cl, it, og, un, ds = J('closed_loop.json'), J('iteration.json'), J('onestep_gap.json'), J('unfiltered.json'), J('dataset_stats.json')
ver = J('sim_verify_seed0/verify.json'); notes = frag('NOTES.md') if os.path.exists(f'{OUT}/NOTES.md') else ''
L = ['# Track D report: closed-loop uncertainty, internal iteration vs physical steps, unfiltered pairs, dataset statistics\n',
     f'Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC by `supp/trackD/make_report.py`. All pose metrics come from `supp/common.py` '
     '(d = range error, EPE = 3-D endpoint error, SR_d@1m = Table 2\'s success rate, SR_EPE@1m = endpoint success). Code: `supp/trackD/`; results: `outputs/supp/trackD/`.\n']
if notes: L += [notes, '']
# ------------------------------------------------------------------ 1
L += ['---\n# 1. Closed loop (Table 5) with repeated actuation-noise draws\n',
      '**Protocol.** Simulator, episodes, estimator, commands and noise model are those of the archived run (`online/sim_nav.py`, `outputs/sim/nav_log.pkl`): 8 held-out sites (`outputs/sim/tours.txt`), '
      'one 3-D Gaussian-splatting scene per site, 10 held-out goal photographs x 2 start cameras = 160 episodes, 5 steps, actuation noise per step 3 % of the commanded move per axis + 5 cm per axis, 0.2 deg heading, 0.1 deg tilt. '
      'Only the seed of the actuation noise changes (`supp/trackD/sim_nav_seeds.py`): each seed owns one random stream consumed in the order of the archived script (sites, steps, the five archived policies, episodes; '
      'the draws of the two policies that are not flown again are generated and discarded), so seed s gives what the archived script would have given with `np.random.default_rng(s)`. '
      'The archived run is seed 0. Stopping rule of Table 5: a flight stops at the first step whose own distance estimate is below tau (no move is executed at that step), otherwise after 5 steps.\n',
      '**Policies.** `s1` = coarse stage only (Table 5, "Ours (Stage I)"). `s2` = Stage I + imagination-refined stage, i.e. the estimator described in Sec. 4.3. '
      '`s2p` = `s2` **plus a short-range ground-plane solver** (`online/plane_servo.py`): when the fused distance estimate is below 5 m and the solver has at least 200 inliers, the translation is not taken from the fusion but solved linearly from the imagined-vs-real matches '
      'on the ground plane, with the rotation fixed to the fused estimate and the camera height dead-reckoned from the first step. **`s2p` is the row "Ours (Stage II)" of Table 5.** All three policies execute the estimated offset in the drone\'s own levelled frame (true heading and tilt read from the airframe; the learned absolute heading is not used).\n']
if ver:
    mx = max(v['max_abs_diff_m'] for p in ver.values() for v in p.values())
    L.append(f'**Check against the archived run.** `sim_nav_seeds.py --seeds 0 --tours 1407` (first site, 20 episodes, 3 policies, 5 steps) reproduces the archived per-episode position errors and distance estimates exactly: largest absolute difference {mx:.1e} m over 360 log rows (`sim_verify_seed0/verify.json`). The estimator is deterministic; the step-0 estimates are therefore computed once and shared by all seeds.\n')
L += [frag('closed_loop.md'), '']
if cl:
    s = cl['summary']; p = cl['paired']
    L.append(f"**Reading.** With {len(cl['seeds'])} noise draws, SR@1 m at tau = 1 m is {s['s2p']['sr_1.0']['mean']:.1f} ± {s['s2p']['sr_1.0']['std']:.1f} % for `s2p`, {s['s2']['sr_1.0']['mean']:.1f} ± {s['s2']['sr_1.0']['std']:.1f} % for `s2` and {s['s1']['sr_1.0']['mean']:.1f} ± {s['s1']['sr_1.0']['std']:.1f} % for `s1` (std over seeds); "
             f"episode-bootstrap 95 % CIs are about ±{(s['s2p']['sr_1.0']['ci_episode'][1] - s['s2p']['sr_1.0']['ci_episode'][0]) / 2:.0f} points. "
             f"`s2p - s1` at tau = 1 m: {p['s2p - s1']['sr_1.0']['delta']:+.1f} points [{p['s2p - s1']['sr_1.0']['ci_episode'][0]:+.1f}, {p['s2p - s1']['sr_1.0']['ci_episode'][1]:+.1f}]; `s2 - s1`: {p['s2 - s1']['sr_1.0']['delta']:+.1f} [{p['s2 - s1']['sr_1.0']['ci_episode'][0]:+.1f}, {p['s2 - s1']['sr_1.0']['ci_episode'][1]:+.1f}]. "
             'The closed-loop advantage of the Table 5 row "Ours (Stage II)" over "Ours (Stage I)" therefore comes from the short-range ground-plane solver; the imagination-refined fusion alone (`s2`) is not better than the coarse stage in closed loop.\n')
    sb = (ds or {}).get('selected_pairs_short_baselines')
    if sb: L.append(f"Context for the last metres: only {100 * sb['lt_1m']:.3f} % of the selected pairs have a baseline below 1 m and {100 * sb['lt_5m']:.2f} % below 5 m (the {ds['fusion_training_rows_short_baselines']['rows']:,} training rows of the fusion contain {ds['fusion_training_rows_short_baselines']['lt_1m']} pairs below 1 m, {ds['fusion_training_rows_short_baselines']['lt_2m']} below 2 m, {ds['fusion_training_rows_short_baselines']['lt_5m']} below 5 m), so the learned distance fusion is extrapolating once the drone is within a few metres of the goal; "
                       'without a stopping rule the success of `s1` and `s2` decreases after the second step (table "Fixed number of steps").\n')
# ------------------------------------------------------------------ 2
L += ['---\n# 2. Internal iteration without motion (fixed image pairs)\n',
      '**Protocol.** 1,000 test pairs (`reviewer_baselines/rb_recon.subset(1000, "test")`). Round 1 = archived Stage II. Round k+1: the round-k fused estimate becomes the guide pose '
      '(per-camera tilts from the round-k imagination MAP pose, shifted so that their difference equals the fused tilt change; heading change and metric displacement from the fusion; camera height = fused distance x triangulated ground depth, clipped to [0.75, 1.3] x 220.2 m, as in `build_guide2.py` / `engine.py`); '
      'the target is re-imagined and re-matched with the unmodified `stage2_imagination/imagine_match.py` (<= 3x3 tiles of 1280x720 at native resolution, 2048 keypoints per tile), poses are re-estimated with `imagine_geom.work`, and the FITTED Stage II fusion (`outputs/online_models/stage2.pkl`, predict mode, no re-training) is applied. '
      'Protocol A: the fusion keeps the Stage I estimate as its "stage1" candidate (and as the "s1" verification candidate); only imagination candidates/features are new. Protocol B: the round-k estimate replaces the Stage I estimate as "stage1" candidate and as verified candidate (true recursion; outside the distribution the fusion was fitted on). '
      'No motion, no new observation. Metrics in the camera frame with the paper\'s norm rescaling (exponent 0.5).\n', frag('iteration.md'), '']
if it and '_paired_vs_round1' in it:
    d = it['_paired_vs_round1']['deltas']; g = lambda k, m: (100 * d[k][m]['delta'], 100 * d[k][m]['ci'][0], 100 * d[k][m]['ci'][1]) if k in d else (float('nan'),) * 3
    r1 = it['Round 1 (archived Stage II)']; txt = []
    for k in ('Round 2, protocol A', 'Round 3, protocol A', 'Round 2, protocol B', 'Round 3, protocol B'):
        if k in d: a = g(k, 'sr_d_1m'); b = g(k, 'sr_epe_1m'); txt.append(f"{k}: SR_d {a[0]:+.1f} [{a[1]:+.1f}, {a[2]:+.1f}], SR_EPE {b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}]")
    res_px = [it[k]['imagination']['residual_px_median'] for k in ('Round 1 (archived Stage II)', 'Round 2, protocol B', 'Round 3, protocol B') if k in it and 'imagination' in it[k]]
    L.append(f"**Reading.** Starting from SR_d@1m {100 * r1['sr_d_1m']:.1f} % / SR_EPE@1m {100 * r1['sr_epe_1m']:.1f} % after the single round of the paper, further internal rounds change the success rates by (percentage points, 95 % CI): " + '; '.join(txt) + '. '
             f"The label-free residual between imagined and real view keeps shrinking ({' -> '.join(f'{v:.1f}' for v in res_px)} px), i.e. the guide pose does get better, but the pose accuracy is already limited by other factors. "
             'For comparison, the physical loop of Table 5 raises SR@1 m from 38.8 % (one step) to 81.9 % (2.2 steps on average), each step bringing a NEW observation from a closer pose. Repeating the refinement on a fixed image pair does not reproduce that gain; the two loops are different mechanisms.\n')
# ------------------------------------------------------------------ 3
L += ['---\n# 3. From Table 2 (83.3 %) to the one-step column of Table 5 (38.8 %)\n',
      '**Protocol.** For the 160 episodes the estimator is run once on (a) the real start photograph and (b) the simulator render at the start pose, both against the real held-out goal photograph (`supp/trackD/onestep_gap.py`; rows identical to the step-0 rows of the closed-loop runs). '
      'Rows 3-4 score these estimates like Table 2 (range error <= 1 m, norm rescaled with the paper\'s exponent) and by the 3-D endpoint error in the camera frame. The last block executes the command of each policy from the rendered observation: without actuation noise, and with the noise model of the simulator (2,000 Monte-Carlo draws per episode; the archived table is one draw).\n',
      frag('onestep_gap.md'), '']
if og:
    e = og['episodes_first_step']; x = og['executed_first_step']; r = og['test_pairs']['Stage II']
    L.append(f"**Reading.** (i) Definition: on the test pairs the range-error SR is {100 * r['all_test']['sr_d_1m']:.1f} %, the endpoint SR in the camera frame {100 * r['all_test']['sr_epe_1m']:.1f} %. "
             f"(ii) Pair distribution: the episodes pair a held-out goal with an arbitrary camera of the site, i.e. they are unfiltered pairs; {100 * (1 - og['distances']['episodes_inside_selected_label_ranges']):.0f} % of them lie outside the label ranges of the selected pairs (|dx| or |dy| > 15 m). On real start photographs Stage II drops to {100 * e['Stage II | real']['paper_protocol']['sr_d_1m']:.1f} % / {100 * e['Stage II | real']['paper_protocol']['sr_epe_1m']:.1f} % (range / endpoint). "
             f"(iii) Rendering: with the simulator render as start view, {100 * e['Stage II | rendered']['paper_protocol']['sr_d_1m']:.1f} % / {100 * e['Stage II | rendered']['paper_protocol']['sr_epe_1m']:.1f} %. "
             f"(iv) Actuation: executing the (unrescaled) command gives {100 * x['s2p']['no_noise']['sr_1m']:.1f} % without and {100 * x['s2p']['with_noise']['sr_1m_expected']:.1f} % with actuation noise in expectation (95 % of single draws in [{100 * x['s2p']['with_noise']['sr_1m_range_95'][0]:.1f}, {100 * x['s2p']['with_noise']['sr_1m_range_95'][1]:.1f}]; archived draw {100 * x['s2p']['archived_single_draw']['sr_1m']:.1f} %). "
             f"The sites themselves are not harder than average ({100 * r['sim_sites']['sr_d_1m']:.1f} % on their {r['n_sim_site_pairs']} selected test pairs). Start distances are comparable (median {og['distances']['episodes']['p50']:.1f} m vs {og['distances']['test_pairs']['p50']:.1f} m).\n")
# ------------------------------------------------------------------ 4
L += ['---\n# 4. Unfiltered pairs\n',
      '**Protocol.** For each of the 290 test buildings, 4 directed pairs drawn uniformly at random from all 150 x 149 camera pairs (seed `[2026, building id]`, nothing excluded; 1,160 pairs). Labels rebuilt from the KML files (`supp/trackD/labels.py`; identical to the released labels to < 1e-6 m on the pairs that happen to be selected). '
      'Full online engine (Stage I + II + learned absolute heading, fitted stores of `outputs/online_models`, no re-training), scored like the archived predictions on the 6,364 selected test pairs of the same buildings; CIs = bootstrap over buildings. '
      '"Inside the label ranges" = all five pose differences within the min-max range of the selected pairs (dataset-wide). Overlap = fraction of a 48 x 27 grid of source pixels whose intersection with the horizontal plane at the terrain altitude of the building anchor projects inside the target image (true poses). '
      '"Screen proxy" = re-implementation of the screen named in the metadata (`supp/trackD/sift_screen.py`: SIFT at 864 px width, mutual nearest neighbours with ratio 0.75, fundamental matrix by USAC_MAGSAC at 1 px, accept if >= 20 inliers covering >= 4 cells of a 4 x 4 grid); the original screening code is not in the release and the stored counts are only reproduced approximately, so pass rates are estimates.\n',
      frag('unfiltered.md'), '']
if un:
    m = lambda blk, key='Stage II | camera frame': un[blk]['methods'][key]; A = m('selected test pairs (archived predictions)'); B = m('unfiltered pairs (all)'); Ci = m('unfiltered pairs inside the label ranges of the selected pairs'); Co = m('unfiltered pairs outside those ranges')
    Be = m('unfiltered pairs (all)', 'Stage II | ENU, learned heading'); Ae = m('selected test pairs (archived predictions)', 'Stage II | ENU, learned heading'); B1 = m('unfiltered pairs (all)', 'Stage I | camera frame'); ov = {r['bin']: r for r in un['by_overlap']['unfiltered']}; os_ = {r['bin']: r for r in un['by_overlap']['selected']}
    txt = (f"**Reading.** On a uniform sample of camera pairs Stage II reaches SR_d@1m {100 * B['sr_d_1m']:.1f} % [{100 * B['ci']['sr_d_1m'][0]:.1f}, {100 * B['ci']['sr_d_1m'][1]:.1f}] and SR_EPE@1m {100 * B['sr_epe_1m']:.1f} % (camera frame; {100 * Be['sr_epe_1m']:.1f} % in ENU with the learned heading), against {100 * A['sr_d_1m']:.1f} % / {100 * A['sr_epe_1m']:.1f} % ({100 * Ae['sr_epe_1m']:.1f} %) on the selected pairs; "
           f"the median endpoint error stays below one metre ({B['epe_med']:.2f} m vs {A['epe_med']:.2f} m) while the means are dominated by failures (d MAE {B['d_mae']:.2f} m vs {A['d_mae']:.2f} m). Stage I alone: {100 * B1['sr_d_1m']:.1f} % / {100 * B1['sr_epe_1m']:.1f} %, so the refinement still helps. "
           f"Two properties of the selection explain the gap. (a) Co-visibility: {100 * ov['< 0.02']['share']:.1f} % of the unfiltered pairs have a ground-footprint overlap below 0.02 (selected: {100 * os_['< 0.02']['share']:.1f} %), where SR_EPE is {100 * ov['< 0.02']['sr_epe_1m_cam']:.1f} %; for overlap >= 0.2 the unfiltered pairs are within a few points of the selected ones. "
           f"(b) Displacement range: {100 * un['range_check']['fraction_unfiltered_outside']:.1f} % of the unfiltered pairs have |dx| or |dy| beyond the 15 m covered by the selected (and training) pairs; SR_d is {100 * Ci['sr_d_1m']:.1f} % inside and {100 * Co['sr_d_1m']:.1f} % outside. ")
    if 'screen_proxy' in un:
        P_ = m('unfiltered pairs passing the screen proxy'); Fl = m('unfiltered pairs failing the screen proxy'); IP = m('unfiltered pairs inside the label ranges and passing the screen proxy'); sp = un['screen_proxy']
        txt += (f"With the re-implemented screen, {100 * sp['unfiltered']['pass_rate']:.1f} % of the unfiltered pairs are accepted; on them Stage II gives {100 * P_['sr_d_1m']:.1f} % / {100 * P_['sr_epe_1m']:.1f} %, on the rejected ones {100 * Fl['sr_d_1m']:.1f} % / {100 * Fl['sr_epe_1m']:.1f} %; "
                f"accepted pairs inside the label ranges: {100 * IP['sr_d_1m']:.1f} % / {100 * IP['sr_epe_1m']:.1f} %, i.e. the level of the selected test pairs. ")
    L.append(txt + 'The 83.3 % of Table 2 therefore describes pairs that share enough ground and lie in the sampled displacement range; it is not an open-set number.\n')
# ------------------------------------------------------------------ 5
L += ['---\n# 5. Dataset statistics (metadata only)\n', frag('dataset_stats.md'), '\nFigures: `fig_dataset_height_tilt.{pdf,png}` (camera height above terrain, tilt), `fig_dataset_pose_diff.{pdf,png}` (pose differences and baseline length: all camera pairs vs selected pairs).\n']
if un and 'screen_proxy' in un:
    sp = un['screen_proxy']; L.append(f"Estimated effect of the visual screen on a uniform sample of camera pairs (item 4): the proxy accepts {100 * sp['unfiltered']['pass_rate']:.1f} % of 1,160 uniformly drawn pairs ({100 * sp['unfiltered']['pass_rate_corrected_for_proxy_misses']:.1f} % after dividing by its {100 * sp['control_selected_pairs']['pass_rate']:.1f} % acceptance of pairs known to have passed the original screen); "
                                      f"{100 * sp['unfiltered']['inside_ranges_and_pass']:.1f} % are both inside the label ranges and accepted, against {100 * ds['selected_fraction']:.2f} % finally selected after balancing.\n")
L += ['---\n# Files\n', '| file | content |', '|---|---|', '| `closed_loop.json`, `closed_loop.md` | item 1: every Table-5 cell per seed, mean/std, bootstrap CIs, paired differences |', '| `sim_seed*/nav_log.pkl`, `sim_verify_seed0/` | per-step logs of the new runs; verification against the archived run |',
      '| `iteration.json`, `iteration.md`, `iter/` | item 2 |', '| `onestep_gap.json`, `onestep_gap.md`, `cache/onestep_pred.pkl` | item 3 |', '| `unfiltered.json`, `unfiltered.md`, `cache/unfiltered_{pairs,pred,scored,sift}.pkl` | item 4 |',
      '| `dataset_stats.json`, `dataset_stats.md`, `fig_dataset_*.{pdf,png}`, `cache/{cameras,buildings,pairs_selected}.pkl` | item 5 |', '| `logs/` | all run logs |']
open(f'{OUT}/REPORT.md', 'w').write('\n'.join(L) + '\n'); print('REPORT.md written,', sum(len(x) for x in L), 'chars')
