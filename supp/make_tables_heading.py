# Absolute-heading prior (Track C; executed by make_tables.py)
hd = J('trackC/heading.json')
if hd is None:
    for n_, l_, w_ in (('heading_err', 'tab:heading_err', 'error of the heading prior'), ('heading_cost', 'tab:heading_cost', 'cost of the heading error'), ('heading_site', 'tab:heading_site', 'site-disjoint re-training'), ('heading_sens', 'tab:heading_sens', 'photometric sensitivity')): placeholder(n_, l_, w_)
else:
    ts = hd['test_set']; es = ts['error_statistics']; B = []
    for k, lab, tag in (('raw-view network, single image (12,728 pair sides)', 'Raw-view network, one image', 'hdRawOne'), ('nadir-rectified network, single image (12,728 pair sides)', 'Nadir-rectified network, one image', 'hdBevOne'),
                        ('raw-view, fused a+b (Stage I)', 'Coarse stage: raw-view, both images', 'hdSi'), ('two-network ensemble, fused a+b (Stage II)', 'Refined stage: both networks, both images', 'hdSii')):
        r = es[k]; ci = r.get('mean_ci'); B.append(f'{lab} & {r["mean"]:.2f}' + (f' \\ci{{{ci[0]:.2f}}}{{{ci[1]:.2f}}}' if ci else '') + f' & {r["median"]:.2f} & {r["p90"]:.1f} & {pct(r["gt10"])} & {pct(r["gt45"])} \\\\')
        macro(tag + 'Mean', f'{r["mean"]:.1f}'); macro(tag + 'Med', f'{r["median"]:.1f}'); macro(tag + 'GtTen', pct(r['gt10'])); macro(tag + 'GtFortyFive', pct(r['gt45'])); macro(tag + 'PNinety', f'{r["p90"]:.1f}')
    w('heading_err', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Error of the absolute-heading prior} on the ' + f'{ts["n_pairs"]:,}'.replace(',', '{,}') + ' test pairs (degrees; heading of the source view, wrapped absolute error).}\n\\label{tab:heading_err}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n'
      'Estimator & Mean & Median & P90 & $>10^\\circ$ (\\%) & $>45^\\circ$ (\\%) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    # cost in the pose output
    c1, c2 = ts['consequences']['Stage I'], ts['consequences']['Stage II']; B = []
    for k, lab, tag in (('camera frame (no heading needed)', 'Camera frame (no heading)', 'hdCam'), ('ENU, learned heading', 'ENU, learned heading (paper)', 'hdEnu'), ('ENU, compass (true heading + N(0, 1 deg))', 'ENU, compass, $\\sigma{=}1^\\circ$', 'hdCompOne'),
                        ('ENU, compass (true heading + N(0, 2 deg))', 'ENU, compass, $\\sigma{=}2^\\circ$', 'hdCompTwo'), ('ENU, compass (true heading + N(0, 5 deg))', 'ENU, compass, $\\sigma{=}5^\\circ$', 'hdCompFive'), ('ENU, compass (true heading + N(0, 10 deg))', 'ENU, compass, $\\sigma{=}10^\\circ$', 'hdCompTen')):
        a, b = c1[k], c2[k]; ci = b.get('sr_epe_1m_ci')
        B.append(f'{lab} & {pct(a["sr_epe_1m"])} & {b["x_mae"]:.2f} / {b["y_mae"]:.2f} & {b["epe_med"]:.2f} & {pct(b["sr_epe_1m"])}' + (f' {cipct(ci)}' if ci else '') + ' \\\\' + ('\\midrule' if tag == 'hdEnu' else ''))
        macro(tag + 'SRe', pct(b['sr_epe_1m'])); macro(tag + 'SiSRe', pct(a['sr_epe_1m'])); macro(tag + 'EPEmed', f'{b["epe_med"]:.2f}')
    w('heading_cost', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{What the heading costs in the pose output.} Same estimates, expressed in the camera frame or rotated into ENU with different sources of the absolute heading; the compass rows add Gaussian noise to the true heading. Horizontal MAE and EPE in metres.}\n\\label{tab:heading_cost}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n'
      '& Coarse & \\multicolumn{3}{c}{Refined} \\\\\n\\cmidrule(lr){2-2}\\cmidrule(lr){3-5}\nFrame and heading source & \\SRe & $\\Delta x$ / $\\Delta y$ & EPE med. & \\SRe \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    fa = c2['failure_attribution']; macro('hdFailFixed', pct(fa['fixed_by_true_heading'])); macro('hdFailRate', pct(fa['fail_rate']))
    for b in ts['by_heading_error']:
        t = {'0-1': 'hdBinA', '1-2': 'hdBinB', '2-5': 'hdBinC', '5-10': 'hdBinD', '10-20': 'hdBinE'}.get(b['bin'])
        if t: macro(t + 'Share', pct(b['frac'])); macro(t + 'SRe', pct(b['sr_epe_enu']))
    D = ts['by_distance_to_nearest_training_building']
    for b, t in zip(D, ('hdDistA', 'hdDistB', 'hdDistC', 'hdDistD', 'hdDistE', 'hdDistF')):
        macro(t + 'Med', f'{b["heading_s2"]["median"]:.1f}'); macro(t + 'Mean', f'{b["heading_s2"]["mean"]:.1f}'); macro(t + 'Enu', pct(b['sr_epe_enu'])); macro(t + 'Cam', pct(b['sr_epe_cam'])); macro(t + 'N', str(b['buildings']))
    bc = ts['building_level_correlation']; macro('hdDistMedian', f'{bc["m_to_nearest_train"]["median"]:.0f}'); macro('hdDistPNinety', f'{bc["m_to_nearest_train"]["p90"]:.0f}'); macro('hdDistSpearman', f'{bc["spearman_meanerr_vs_distance"]:.2f}')
    for b in ts['by_hemisphere']:
        if b['bin'] == 'south': macro('hdSouthN', str(b['buildings'])); macro('hdSouthPairs', str(b['pairs'])); macro('hdSouthErr', f'{b["heading_s2"]["median"]:.0f}')
        else: macro('hdNorthMean', f'{b["heading_s2"]["mean"]:.1f}'); macro('hdNorthMed', f'{b["heading_s2"]["median"]:.1f}')
    macro('hdSouthSigned', f'{ts["southern_signed_error_circular_mean"]:.0f}')
    rc = {round(x['coverage'], 2): x for x in ts['risk_coverage']['by_kappa']}; macro('hdCovMae', f'{rc[0.95]["heading_mae"]:.1f}'); macro('hdCovGtFortyFive', pct(rc[0.95]['heading_gt45'], 2)); macro('hdCovSRe', pct(rc[0.95]['sr_epe_enu']))
    bn = ts['bias_vs_noise']; macro('hdBiasOnly', f'{bn["heading_mae"]["building_bias_only"]:.1f}'); macro('hdNoiseOnly', f'{bn["heading_mae"]["image_noise_only"]:.1f}'); macro('hdBadBuildings', str(bn['buildings_with_mean_error_gt20'])); macro('hdTopFiveShare', pct(bn['share_of_total_abs_error_from_worst_buildings']['top5'], 0))
    # site-disjoint re-training
    sd = hd['site_disjoint_retraining']; S_ = sd['subsets']; B = []
    for sk, sl in (('held-out sites', 'Held-out sites'), ('other sites', 'Other sites')):
        s = S_[sk]; B.append(('\\midrule' if B else '') + f'\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{sl}}} ({s["buildings"]} test buildings, ' + f'{s["pairs"]:,}'.replace(',', '{,}') + f' pairs; camera frame: {pct(s["camera_frame"]["sr_epe_1m"])}\\% \\SRe)}}\\\\')
        for k, lab in (('raw-view | A archived', '\\quad paper network'), ('raw-view | B site-disjoint', '\\quad re-trained without the six sites')):
            r = s[k]; f_ = r['fused_heading']; ci = r['fused_heading_mean_ci']
            B.append(f'{lab} & {f_["mean"]:.1f} \\ci{{{ci[0]:.1f}}}{{{ci[1]:.1f}}} & {f_["median"]:.1f} & {pct(f_["gt10"])} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} {cipct(r["sr_epe_1m_ci"])} \\\\')
    w('heading_site', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Heading prior on sites that were never seen.} Raw-view heading network re-trained after removing every building of six sites from its training set; estimates of the refined stage rotated into ENU with the fused heading of this network. Heading error in degrees.}\n\\label{tab:heading_site}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n'
      '& \\multicolumn{3}{c}{Heading error} & EPE med. & \\SRe, ENU \\\\\n\\cmidrule(lr){2-4}\nHeading network & mean & med. & $>10^\\circ$ (\\%) & (m) & (\\%) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    ho = S_['held-out sites']; a, b = ho['raw-view | A archived'], ho['raw-view | B site-disjoint']; pp = ho['paired']['raw-view | B site-disjoint minus raw-view | A archived']
    macro('hdSdBuildings', str(ho['buildings'])); macro('hdSdPairs', f'{ho["pairs"]:,}'.replace(',', '{,}')); macro('hdSdRemoved', str(hd['sites']['train_buildings_removed'])); macro('hdSdSites', str(len(hd['sites']['heldout_sites'])))
    macro('hdSdSeenMed', f'{a["fused_heading"]["median"]:.1f}'); macro('hdSdSeenMean', f'{a["fused_heading"]["mean"]:.1f}'); macro('hdSdUnseenMed', f'{b["fused_heading"]["median"]:.1f}'); macro('hdSdUnseenMean', f'{b["fused_heading"]["mean"]:.1f}'); macro('hdSdSeenSRe', pct(a['sr_epe_1m'])); macro('hdSdUnseenSRe', pct(b['sr_epe_1m'])); macro('hdSdCamSRe', pct(ho['camera_frame']['sr_epe_1m']))
    macro('hdSdDelta', f'{100 * pp["sr_epe_1m"]["delta"]:+.1f}'); macro('hdSdDeltaLo', f'{100 * pp["sr_epe_1m"]["ci"][0]:+.1f}'); macro('hdSdDeltaHi', f'{100 * pp["sr_epe_1m"]["ci"][1]:+.1f}')
    po = S_['other sites']['paired']['raw-view | B site-disjoint minus raw-view | A archived']; macro('hdSdOtherDelta', f'{100 * po["sr_epe_1m"]["delta"]:+.1f}'); macro('hdSdOtherDeltaLo', f'{100 * po["sr_epe_1m"]["ci"][0]:+.1f}'); macro('hdSdOtherDeltaHi', f'{100 * po["sr_epe_1m"]["ci"][1]:+.1f}')
    bb = sd['heldout_bias_vs_noise_B']; macro('hdSdBias', f'{bb["heading_mae"]["building_bias_only"]:.1f}'); macro('hdSdNoise', f'{bb["heading_mae"]["image_noise_only"]:.1f}')
    offs = [x['raw-view | B site-disjoint']['signed_circular_mean'] for x in sd['heldout_per_site']]; rl = [x['raw-view | B site-disjoint']['resultant_length'] for x in sd['heldout_per_site']]
    macro('hdSdOffMin', f'{min(offs):+.0f}'); macro('hdSdOffMax', f'{max(offs):+.0f}'); macro('hdSdResLo', f'{min(rl):.2f}'); macro('hdSdResHi', f'{max(rl):.2f}')
    tr = sd['training']['raw_sitedisjoint']; macro('hdSdTrainBuildings', str(tr['train_buildings'])); macro('hdSdEpochs', str(tr['epochs']))
    # photometric sensitivity
    ps = hd['photometric_sensitivity']; C_ = ps['conditions']; B = []
    def cond(lab, keys, mid=False):
        vals = [C_[k]['ensemble'] for k in keys if k in C_]
        if not vals: return
        B.append(('\\midrule ' if mid else '') + f'{lab} & ' + ', '.join(f'{v["mae"]:.1f}' for v in vals) + ' & ' + ', '.join(f'{v["median"]:.1f}' for v in vals) + ' & ' + ', '.join(pct(v['gt45'], 0) for v in vals) + ' \\\\')
    cond('None', ['clean']); cond('Brightness $\\times0.6$, $\\times1.4$', ['brightness x0.6', 'brightness x1.4'], True); cond('Contrast $\\times0.6$, $\\times1.4$', ['contrast x0.6', 'contrast x1.4']); cond('Gamma 0.6, 1.6', ['gamma 0.6', 'gamma 1.6'])
    cond('Shadows lifted', ['shadow lift (dark half, gamma 0.5 on L)']); cond('Tiles of 64, 32 px shuffled', ['tile shuffle, 64 px tiles (layout destroyed, local appearance kept)', 'tile shuffle, 32 px tiles'])
    cond('Blur, $\\sigma{=}1$, 2 px', ['Gaussian blur sigma 1 px', 'Gaussian blur sigma 2 px'], True); cond('JPEG, quality 20', ['JPEG quality 20']); cond('Grey levels only', ['grayscale']); cond('Hue rotated by $90^\\circ$, $180^\\circ$', ['hue +90 deg', 'hue +180 deg'])
    cond('Luminance inverted, colour kept', ['luminance inversion (L channel only, chroma kept)']); cond('Negative image', ['intensity inversion (negative)'])
    z = ps.get('zoom_raw_view', {})
    if z: B.append('\\midrule Centre zoom $\\times1.5$, $\\times2$, $\\times3.6^\\dagger$ & ' + ', '.join(f'{z[k]["mae"]:.1f}' for k in z) + ' & ' + ', '.join(f'{z[k]["median"]:.1f}' for k in z) + ' & ' + ', '.join(pct(z[k]['gt45'], 0) for k in z) + ' \\\\')
    w('heading_sens', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{What the heading prior reads.} Error of the heading predicted from one image (both networks) after an image operation, on ' + f'{ps["n_images"]:,}'.replace(',', '{,}') + ' test images (degrees). $^\\dagger$raw-view network only; a zoom of 3.6 corresponds to the flight height of the real photographs.}\n\\label{tab:heading_sens}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n'
      'Operation on the image & Mean & Median & $>45^\\circ$ (\\%) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    e = lambda k: C_[k]['ensemble']
    macro('hdSensClean', f'{e("clean")["median"]:.1f}'); macro('hdSensGray', f'{e("grayscale")["median"]:.0f}'); macro('hdSensHue', f'{e("hue +180 deg")["median"]:.0f}'); macro('hdSensNegFlip', pct(e('intensity inversion (negative)')['flips_180_within_30deg'], 0)); macro('hdSensNeg', f'{e("intensity inversion (negative)")["median"]:.0f}')
    macro('hdSensJpeg', f'{e("JPEG quality 20")["median"]:.0f}'); macro('hdSensShuffle', f'{e("tile shuffle, 64 px tiles (layout destroyed, local appearance kept)")["median"]:.1f}'); macro('hdSensZoom', f'{z["centre zoom x3.6"]["median"]:.0f}' if z else 'n/a'); macro('hdSensN', f'{ps["n_images"]:,}'.replace(',', '{,}'))
    # real photographs
    rp = hd['real_photographs']; si = rp['single_image']; pw = rp['pairs']['within training range (300)']['Stage II']
    macro('hdRealPhotos', str(rp['photos'])); macro('hdRealOneMed', f'{si["ensemble"]["all"]["median"]:.0f}'); macro('hdRealOneMae', f'{si["ensemble"]["all"]["mae"]:.0f}'); macro('hdRealFusedMed', f'{pw["fused_heading"]["median"]:.0f}'); macro('hdRealEnuSRe', pct(pw['sr_epe_enu_1m'])); macro('hdRealCamSRe', pct(pw['sr_epe_cam_1m'])); macro('hdRealEnuEPE', f'{pw["epe_enu_median"]:.1f}')
    macro('hdRenderKappa', f'{rp["kappa_reference_rendered_test_median"]:.0f}'); macro('hdRealKappa', f'{si["raw-view"]["kappa_median"]:.0f}')
    km = None
    for k_ in ('kappa_median', 'kappa_median_real'):
        if k_ in si['ensemble']['all']: km = si['ensemble']['all'][k_]
    if km is not None: macro('hdRealKappa', f'{km:.0f}')
