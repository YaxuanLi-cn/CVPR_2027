# Internal iteration, closed loop with repeated noise draws, one-step chain, unfiltered pairs by overlap (Track D; executed by make_tables.py)
def pm(s, d=1): return f'{s["mean"]:.{d}f}' + (f'\\,$\\pm$\\,{s["std"]:.{d}f}' if s.get('std', 0) > 0 else '')
def sgn(v, d=1): return f'{v:+.{d}f}'
# ---------------- internal iteration on fixed pairs
it = J('trackD/iteration.json')
if it is None: placeholder('iteration', 'tab:iteration', 'internal iteration without motion')
else:
    B = []; P = it['_paired_vs_round1']['deltas']
    for k, lab in (('Stage I', 'Coarse stage'), ('Round 1 (re-applied fitted fusion)', 'Refinement, 1 round (paper)'), ('Round 2, protocol A', '2 rounds, A'), ('Round 3, protocol A', '3 rounds, A'), ('Round 2, protocol B', '2 rounds, B'), ('Round 3, protocol B', '3 rounds, B')):
        if k not in it: continue
        r = it[k]; im = r.get('imagination') or {}; p = P.get(k)
        dl = (f'${100 * p["sr_epe_1m"]["delta"]:+.1f}$ \\ci{{{100 * p["sr_epe_1m"]["ci"][0]:+.1f}}}{{{100 * p["sr_epe_1m"]["ci"][1]:+.1f}}}') if p else '--'
        B.append(f'{lab} & {r["d_mae"]:.2f} & {pct(r["sr_d_1m"])} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} & {dl} & {num(im.get("residual_px_median"), 1) if im else "--"} \\\\' + ('\\midrule' if k == 'Round 1 (re-applied fitted fusion)' else ''))
    w('iteration', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Repeating the refinement on a fixed image pair, without motion} (' + f'{it["_n"]:,}'.replace(',', '{,}') + ' test pairs, camera frame). The estimate of a round guides the imagination of the next. A: the fusion keeps the coarse estimate as a candidate; B: the estimate of the previous round replaces it. $\\Delta$: paired change of \\SRe{} with respect to one round. Res.: median displacement between imagined and real view.}\n\\label{tab:iteration}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n'
      '& $d$ & \\SRd & EPE med. & \\SRe & $\\Delta$\\SRe & Res. \\\\\n& (m) & (\\%) & (m) & (\\%) & (points) & (px) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    macro('itN', f'{it["_n"]:,}'.replace(',', '{,}'))
    for k, tag in (('Round 2, protocol A', 'itTwoA'), ('Round 3, protocol A', 'itThreeA'), ('Round 2, protocol B', 'itTwoB'), ('Round 3, protocol B', 'itThreeB')):
        if k not in P: continue
        for m, t in (('sr_d_1m', 'SRd'), ('sr_epe_1m', 'SRe')): macro(tag + t, sgn(100 * P[k][m]['delta'])); macro(tag + t + 'Lo', sgn(100 * P[k][m]['ci'][0])); macro(tag + t + 'Hi', sgn(100 * P[k][m]['ci'][1]))
        macro(tag + 'Res', num(it[k]['imagination']['residual_px_median'], 1)); macro(tag + 'AbsSRe', pct(it[k]['sr_epe_1m'])); macro(tag + 'AbsSRd', pct(it[k]['sr_d_1m']))
    r1 = it['Round 1 (re-applied fitted fusion)']; macro('itOneRes', num(r1['imagination']['residual_px_median'], 1)); macro('itOneSRe', pct(r1['sr_epe_1m'])); macro('itOneSRd', pct(r1['sr_d_1m'])); macro('itGainSRe', sgn(100 * P['Round 3, protocol A']['sr_epe_1m']['delta']))
# ---------------- closed loop with repeated actuation-noise draws
cl = J('trackD/closed_loop.json')
if cl is None: placeholder('closedloop', 'tab:closedloop', 'closed loop with repeated noise draws', True)
else:
    S = cl['summary']; ns_ = len(cl['seeds']); B = []; NAMES = (('s1', 'Coarse (Table~5, ``Stage~I\'\')'), ('s2', 'Coarse + refined'), ('s2p', 'Coarse + refined + short-range (Table~5, ``Stage~II\'\')'))
    for k, lab in NAMES:
        s = S[k]; cells = [pm(s['one_sr'])]
        for tau in ('0.5', '1.0', '1.5', '2.0'):
            cells += [f'{s["steps_" + tau]["mean"]:.1f}'] + ([f'{s["err_" + tau]["mean"]:.2f}'] if tau == '1.0' else []) + [pm(s['sr_' + tau]) + (f' \\ci{{{s["sr_" + tau]["ci_episode"][0]:.1f}}}{{{s["sr_" + tau]["ci_episode"][1]:.1f}}}' if tau == '1.0' else '')]
        B.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    seedtxt = (f'SR: mean\\,$\\pm$\\,standard deviation over {ns_} draws of the actuation noise, the first being the draw of Table~5; other columns: mean' if ns_ > 1 else 'one draw of the actuation noise (the draw of Table~5)')
    w('closedloop', '\\begin{table*}[t]\n\\centering\n\\caption{\\textbf{Closed loop of Table~5 with the three estimators separated} (' + str(cl['episodes']) + ' episodes on ' + str(len(cl['sites'])) + ' held-out sites, at most ' + str(cl['steps']) + ' steps; ' + seedtxt + '). The run stops when the estimated remaining distance is below $\\tau$. SR: final position within 1\\,m; Err.: median final position error; brackets: 95\\,\\% interval of a bootstrap over episodes.}\n\\label{tab:closedloop}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n'
      '& One step & \\multicolumn{2}{c}{$\\tau{=}0.5$\\,m} & \\multicolumn{3}{c}{$\\tau{=}1$\\,m} & \\multicolumn{2}{c}{$\\tau{=}1.5$\\,m} & \\multicolumn{2}{c}{$\\tau{=}2$\\,m} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\nEstimator & SR (\\%) & Steps & SR (\\%) & Steps & Err. (m) & SR (\\%) & Steps & SR (\\%) & Steps & SR (\\%) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
    B = []
    for k, lab in (('s1', 'Coarse'), ('s2', 'Coarse + refined'), ('s2p', 'Coarse + refined + short-range')):
        s = S[k]; B.append(lab + ' & ' + ' & '.join(f'{s[f"fixed_sr_{i}"]["mean"]:.1f}' for i in range(1, 6)) + f' & {s["fixed_err_5"]["mean"]:.2f} \\\\')
    w('fixedsteps', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Without a stopping rule}: share of the episodes of \\cref{tab:closedloop} within 1\\,m of the goal after each step (\\%, mean over the draws), and median position error after the fifth step.}\n\\label{tab:fixedsteps}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n'
      '& \\multicolumn{5}{c}{SR after step} & Err. \\\\\n\\cmidrule(lr){2-6}\nEstimator & 1 & 2 & 3 & 4 & 5 & (m) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    macro('clSeeds', str(ns_)); macro('clEpisodes', str(cl['episodes'])); macro('clSites', str(len(cl['sites'])))
    W_ = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}.get(ns_, str(ns_))
    macro('clDrawsText', 'only the draw of Table~5 is available' if ns_ == 1 else f'{W_} independent draws, the first being that of Table~5')
    macro('clSpreadText', '' if ns_ == 1 else f' Over the {W_} draws the success of the three estimators at $\\tau{{=}}1$\\,m ranges over {S["s1"]["sr_1.0"]["min"]:.1f}--{S["s1"]["sr_1.0"]["max"]:.1f}\\%, {S["s2"]["sr_1.0"]["min"]:.1f}--{S["s2"]["sr_1.0"]["max"]:.1f}\\% and {S["s2p"]["sr_1.0"]["min"]:.1f}--{S["s2p"]["sr_1.0"]["max"]:.1f}\\%.')
    for k, tag in (('s1', 'clSi'), ('s2', 'clSiiPlain'), ('s2p', 'clSii')):
        s = S[k]; macro(tag + 'SR', f'{s["sr_1.0"]["mean"]:.1f}'); macro(tag + 'SRstd', f'{s["sr_1.0"]["std"]:.1f}'); macro(tag + 'SRmin', f'{s["sr_1.0"]["min"]:.1f}'); macro(tag + 'SRmax', f'{s["sr_1.0"]["max"]:.1f}'); macro(tag + 'SRLo', f'{s["sr_1.0"]["ci_episode"][0]:.1f}'); macro(tag + 'SRHi', f'{s["sr_1.0"]["ci_episode"][1]:.1f}')
        macro(tag + 'One', f'{s["one_sr"]["mean"]:.1f}'); macro(tag + 'Steps', f'{s["steps_1.0"]["mean"]:.1f}'); macro(tag + 'FixTwo', f'{s["fixed_sr_2"]["mean"]:.1f}'); macro(tag + 'FixFive', f'{s["fixed_sr_5"]["mean"]:.1f}'); macro(tag + 'SRhalf', f'{s["sr_0.5"]["mean"]:.1f}'); macro(tag + 'SRtwo', f'{s["sr_2.0"]["mean"]:.1f}')
    for k, tag in (('s2p - s1', 'clDfull'), ('s2 - s1', 'clDrefine'), ('s2p - s2', 'clDplane')):
        p = cl['paired'][k]['sr_1.0']; macro(tag, sgn(p['delta'])); macro(tag + 'Lo', sgn(p['ci_episode'][0])); macro(tag + 'Hi', sgn(p['ci_episode'][1])); macro(tag + 'SiteLo', sgn(p['ci_site'][0])); macro(tag + 'SiteHi', sgn(p['ci_site'][1]))
        p = cl['paired'][k]['one_sr']; macro(tag + 'One', sgn(p['delta'])); macro(tag + 'OneLo', sgn(p['ci_episode'][0])); macro(tag + 'OneHi', sgn(p['ci_episode'][1]))
# ---------------- from Table 2 to the one-step column of Table 5
og = J('trackD/onestep_gap.json')
if og is None: placeholder('onestep', 'tab:onestep', 'from Table 2 to the one-step column of Table 5')
else:
    tp = og['test_pairs']; ep = og['episodes_first_step']; ex = og['executed_first_step']; B = []
    def row(lab, n, a, b, ci=None):
        c = (' ' + cipct(ci['sr_epe_1m'])) if ci else ''
        return f'{lab} & {n} & {pct(a["sr_d_1m"])} & {pct(a["sr_epe_1m"])} & {pct(b["sr_d_1m"])} & {pct(b["sr_epe_1m"])}{c} \\\\'
    nt = f'{int(tp["Stage II"]["all_test"].get("n", 6364)):,}'.replace(',', '{,}')
    B.append(row('Test pairs of Table~2', nt, tp['Stage I']['all_test'], tp['Stage II']['all_test'], tp['Stage II'].get('ci_all')))
    B.append(row('\\quad of the simulator sites', tp['Stage II']['n_sim_site_pairs'], tp['Stage I']['sim_sites'], tp['Stage II']['sim_sites']))
    B.append(row('Episode pairs, real start photograph', ep['Stage II | real']['paper_protocol']['n'], ep['Stage I | real']['paper_protocol'], ep['Stage II | real']['paper_protocol'], ep['Stage II | real']['paper_protocol'].get('ci')))
    B.append(row('Episode pairs, rendered start view', ep['Stage II | rendered']['paper_protocol']['n'], ep['Stage I | rendered']['paper_protocol'], ep['Stage II | rendered']['paper_protocol'], ep['Stage II | rendered']['paper_protocol'].get('ci')))
    ne = ep['Stage II | rendered']['paper_protocol']['n']; B.append('\\midrule')
    B.append(f'After the move, no actuation noise & {ne} & -- & {pct(ex["s1"]["no_noise"]["sr_1m"])} & -- & {pct(ex["s2"]["no_noise"]["sr_1m"])} {cipct(ex["s2"]["no_noise"]["ci_sr"])} \\\\')
    B.append(f'After the move, with actuation noise & {ne} & -- & {pct(ex["s1"]["with_noise"]["sr_1m_expected"])} & -- & {pct(ex["s2"]["with_noise"]["sr_1m_expected"])} {cipct(ex["s2"]["with_noise"]["sr_1m_range_95"])} \\\\')
    B.append(f'\\quad the single draw of Table~5 & {ne} & -- & {pct(ex["s1"]["archived_single_draw"]["sr_1m"])} & -- & {pct(ex["s2"]["archived_single_draw"]["sr_1m"])} \\\\')
    w('onestep', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{From Table~2 to the one-step column of Table~5.} Success of one estimate on the test pairs, on the start/goal pairs of the simulated episodes, and after the commanded move is executed (position within 1\\,m of the goal; with noise: expectation over 2{,}000 draws and the range that contains 95\\,\\% of single draws). Camera frame.}\n\\label{tab:onestep}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lrcccc@{}}\n\\toprule\n'
      '& & \\multicolumn{2}{c}{Coarse} & \\multicolumn{2}{c}{Refined} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n& Pairs & \\SRd & \\SRe & \\SRd & \\SRe \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    e = ep['Stage II | real']['paper_protocol']; macro('ogRealSRd', pct(e['sr_d_1m'])); macro('ogRealSRe', pct(e['sr_epe_1m'])); e = ep['Stage II | rendered']['paper_protocol']; macro('ogRendSRd', pct(e['sr_d_1m'])); macro('ogRendSRe', pct(e['sr_epe_1m']))
    macro('ogSiteSRd', pct(tp['Stage II']['sim_sites']['sr_d_1m'])); macro('ogSiteSRe', pct(tp['Stage II']['sim_sites']['sr_epe_1m'])); macro('ogSiteN', str(tp['Stage II']['n_sim_site_pairs']))
    macro('ogNoNoise', pct(ex['s2']['no_noise']['sr_1m'])); macro('ogNoise', pct(ex['s2']['with_noise']['sr_1m_expected'])); macro('ogNoiseLo', pct(ex['s2']['with_noise']['sr_1m_range_95'][0])); macro('ogNoiseHi', pct(ex['s2']['with_noise']['sr_1m_range_95'][1])); macro('ogNoiseStd', pct(ex['s2']['with_noise']['sr_1m_std_over_draws']))
    macro('ogTableFive', pct(ex['s2p']['archived_single_draw']['sr_1m'])); macro('ogNoiseDisp', num(ex['s2']['noise_only_displacement_m']['median'], 2)); macro('ogInside', pct(og['distances']['episodes_inside_selected_label_ranges'], 0)); macro('ogOutside', pct(og['distances']['episodes_abs_dx_or_dy_gt_15m'], 0))
    io = og['inside_vs_outside_label_range']['Stage II | real']; macro('ogRealInSRe', pct(io['inside']['sr_epe_1m'])); macro('ogRealOutSRe', pct(io['outside']['sr_epe_1m'])); macro('ogRealInSRd', pct(io['inside']['sr_d_1m'])); macro('ogRealOutSRd', pct(io['outside']['sr_d_1m']))
# ---------------- unfiltered pairs by overlap
uf = J('trackD/unfiltered.json')
if uf is not None:
    B = []
    for b in uf['by_overlap']['unfiltered']: B.append(f'{b["bin"].replace("<", "$<$").replace(">=", "$\\geq$")} & {pct(b["share"])} & {pct(b["sr_d_1m"])} & {pct(b["sr_epe_1m_stage1_cam"])} & {pct(b["sr_epe_1m_cam"])} & {b["epe_med_cam"]:.2f} \\\\')
    w('overlap', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Unscreened pairs by overlap of the two ground footprints} (pairs of \\cref{tab:unscreened}; refined stage unless stated, camera frame).}\n\\label{tab:overlap}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n'
      '& Share & \\SRd & \\SRe & \\SRe & EPE med. \\\\\nOverlap & (\\%) & (\\%) & coarse & refined & (m) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    sp = uf['screen_proxy']; macro('ufProxyPass', pct(sp['control_selected_pairs']['pass_rate'])); macro('ufFewInl', pct(sp['unfiltered']['fewer_than_20_inliers']))
    rc = uf['range_check']['fraction_unfiltered_inside_per_component']; macro('ufInX', pct(rc['x'])); macro('ufInY', pct(rc['y']))
# ---------------- how often the short-range solver acts; how rare short baselines are in the released pairs
if cl is not None and 'plane_solver_use' in cl:
    pu = cl['plane_solver_use']; macro('clPlaneSteps', pct(pu['share_of_steps'])); macro('clPlaneFlights', pct(pu['episodes_using_it_at_least_once'])); macro('clPlaneFirst', pct(pu['by_step']['0'])); macro('clPlaneSecond', pct(pu['by_step']['1'])); macro('clPlaneThird', pct(pu['by_step']['2']))
ds_ = J('trackD/dataset_stats.json')
if ds_ is not None and 'selected_pairs_short_baselines' in ds_:
    sb = ds_['selected_pairs_short_baselines']; macro('dsShortFive', pct(sb['lt_5m'], 2)); macro('dsShortTwo', pct(sb['lt_2m'], 2)); fr = ds_.get('fusion_training_rows_short_baselines')
    if fr: macro('dsFusionShortFive', str(fr['lt_5m'])); macro('dsFusionShortTwo', str(fr['lt_2m'])); macro('dsFusionRows', f'{fr["rows"]:,}'.replace(',', '{,}'))
