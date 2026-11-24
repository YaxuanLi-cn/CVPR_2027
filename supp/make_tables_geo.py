# Geometric baselines (Track B): PairUAV table, scale table, SIFT on the full test set, unscreened pairs, real photographs (executed by make_tables.py)
def ang(v): return f'{v:.3f}' if v < 1 else (f'{v:.2f}' if v < 10 else f'{v:.1f}')
def nfmt(n): return f'{n:,}'.replace(',', '{,}')
H = 'height_p75ab'
GEO_SPEC = [('SIFT + ratio test', 'SIFT + ratio test + E', 'plain', None, 'inv.', 'Sift'), ('SuperPoint + SuperGlue, one pass', 'SuperPoint + SuperGlue + E', 'plain', None, 'none', 'SgPlain'),
            ('SuperPoint + SuperGlue, our first pass', 'SuperPoint + SuperGlue, rotation-aware matching of the paper + E', 'rotation search', 'SuperPoint + SuperGlue + E', 'ours', 'SgRot'),
            ('SuperPoint + LightGlue', 'SuperPoint + LightGlue + E', 'rot. TTA', 'SuperPoint + LightGlue + E', 'TTA', 'Lg'), ('ALIKED + LightGlue', 'ALIKED + LightGlue + E', 'rot. TTA', 'ALIKED + LightGlue + E', 'TTA', 'Aliked'), ('LoFTR', 'LoFTR + E', 'rot. TTA', 'LoFTR + E', 'TTA', 'Loftr'),
            ('DeDoDe + rotation steerer', 'DeDoDe-B + SO(2) steerer (C8) + E', 'rot. TTA', 'DeDoDe-B + SO(2) steerer (C8) + E', 'eq.+TTA', 'Dedode'), ('RoMa', 'RoMa + E', 'rot. TTA', 'RoMa + E', 'TTA', 'Roma'), ('MASt3R descriptors', 'MASt3R matches + E', 'rot. TTA', 'MASt3R matches + E', 'TTA', 'MastMatch')]
def geo_get(rows, method, variant=None, scale=None):
    for r in rows:
        if r['method'] == method and (variant is None or r.get('variant', '').startswith(variant)) and (scale is None or r['scale'] == scale): return r
    return None
d = J('trackB/pairuav.json')
if d is None: placeholder('geometric', 'tab:geometric', 'geometric baselines on PairUAV', True)
else:
    rows = d['rows']; get = lambda *a: geo_get(rows, *a)
    def line(lab, r, plain=None, rot=''):
        if r is None: return None
        c = r['ci']
        if rot == 'none': f1, f2 = f'{r["fail"]:.1f}', '--'
        else: f1, f2 = (f'{plain["fail"]:.1f}' if plain is not None else '--'), f'{r["fail"]:.1f}'
        return (f'{lab} & {rot} & {nfmt(r["n"])} & {f1} & {f2} & {r["d_mae"]:.2f} & {pct(r["sr_d_1m"])} {cipct(c["sr_d_1m"])} & {r["epe_mae"]:.2f} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} {cipct(c["sr_epe_1m"])} & '
                f'{ang(r["theta_mae"])} / {ang(r["theta_med"])} & {ang(r["psi_mae"])} / {ang(r["psi_med"])} & {r["dir_med"]:.1f} \\\\')
    B = ['\\multicolumn{13}{@{}l}{\\emph{Matcher + calibrated essential matrix (MAGSAC++); length from the flight-height prior}}\\\\']
    for lab, m, v, pm, rot, tag in GEO_SPEC:
        l = line(lab, get(m, v, H), get(pm, 'plain', H) if pm else None, rot)
        if l: B.append(l)
    B.append('\\midrule\\multicolumn{13}{@{}l}{\\emph{Feed-forward reconstruction used as a pose estimator (pointmap + PnP)}}\\\\')
    for lab, m, sc in (('MASt3R, its own metric scale', 'MASt3R pointmap + PnP', 'native'), ('MASt3R, flight-height prior', 'MASt3R pointmap + PnP', 'height_pointmap'), ('DUSt3R, flight-height prior', 'DUSt3R pointmap + PnP', 'height_pointmap')):
        l = line(lab, get(m, 'rot. TTA', sc), get(m, 'plain', sc), 'TTA')
        if l: B.append(l)
    B.append('\\midrule\\multicolumn{13}{@{}l}{\\emph{Learned}}\\\\')
    for lab, m in (('Regression on pooled MASt3R statistics', 'Regression: MLP on frozen MASt3R statistics (camera-frame targets)'), ('Ours (Stage I)', 'DreamNav Stage I'), ('Ours (Stage II)', 'DreamNav Stage II')):
        l = line(lab, get(m), None, '')
        if l: B.append(l)
    cap = ('\\textbf{Geometric two-view baselines on PairUAV} (' + nfmt(d['n_test']) + ' random test pairs, a random subset of them where fewer are listed; inlier thresholds and the constant of the flight-height prior from ' + str(d['n_calibration']) + ' training pairs). Camera frame. '
           'Rot.: handling of the in-plane rotation (inv.: invariant descriptor; eq.: rotation-equivariant matching; TTA: the target is matched under four rotations and the one with most inliers is kept; ours: rotation search of Sec.~4.2). Fail: pairs without a pose, without and with this handling; they predict zero motion. Angular errors are mean / median.')
    w('geometric', '\\begin{table*}[t]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:geometric}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrcccccccccc@{}}\n\\toprule\n'
      '& & & \\multicolumn{2}{c}{Fail (\\%)} & $d$ MAE & \\SRd@1\\,m & \\multicolumn{2}{c}{EPE (m)} & \\SRe@1\\,m & $\\dth$ error ($^\\circ$) & $\\dpsi$ error ($^\\circ$) & Dir. \\\\\nMethod & Rot. & Pairs & as is & handled & (m) & (\\%) & mean & med. & (\\%) & mean / med. & mean / med. & med. ($^\\circ$) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
    for lab, m, v, pm, rot, tag in GEO_SPEC:
        r = get(m, v, H); tag = 'gb' + tag
        if r is None: continue
        macro(tag + 'SRd', pct(r['sr_d_1m'])); macro(tag + 'SRe', pct(r['sr_epe_1m'])); macro(tag + 'Fail', f'{r["fail"]:.1f}'); macro(tag + 'D', f'{r["d_mae"]:.2f}'); macro(tag + 'EPEmed', f'{r["epe_med"]:.2f}'); macro(tag + 'Theta', ang(r['theta_mae'])); macro(tag + 'Dir', f'{r["dir_med"]:.1f}'); macro(tag + 'Inl', f'{r["inl_med"]:.0f}'); macro(tag + 'N', nfmt(r['n']))
        macro(tag + 'SRdLo', pct(r['ci']['sr_d_1m'][0])); macro(tag + 'SRdHi', pct(r['ci']['sr_d_1m'][1])); macro(tag + 'SReLo', pct(r['ci']['sr_epe_1m'][0])); macro(tag + 'SReHi', pct(r['ci']['sr_epe_1m'][1]))
    for m, tag in (('DreamNav Stage I', 'gbSi'), ('DreamNav Stage II', 'gbSii')):
        r = get(m); macro(tag + 'SRd', pct(r['sr_d_1m'])); macro(tag + 'SRe', pct(r['sr_epe_1m'])); macro(tag + 'SRdLo', pct(r['ci']['sr_d_1m'][0])); macro(tag + 'SRdHi', pct(r['ci']['sr_d_1m'][1])); macro(tag + 'SReLo', pct(r['ci']['sr_epe_1m'][0])); macro(tag + 'SReHi', pct(r['ci']['sr_epe_1m'][1])); macro(tag + 'Dir', f'{r["dir_med"]:.1f}')
    r = get('MASt3R pointmap + PnP', 'rot. TTA', 'native'); macro('gbMastNativeSRd', pct(r['sr_d_1m'])); macro('gbMastNativeEPE', f'{r["epe_med"]:.1f}'); macro('gbMastTheta', ang(r['theta_mae'])); r = get('MASt3R pointmap + PnP', 'rot. TTA', 'height_pointmap'); macro('gbMastHeightSRd', pct(r['sr_d_1m'])); macro('gbMastDir', f'{r["dir_med"]:.0f}')
    r = get('DUSt3R pointmap + PnP', 'rot. TTA', 'height_pointmap'); macro('gbDustHeightSRd', pct(r['sr_d_1m'])); macro('gbDustDir', f'{r["dir_med"]:.0f}')
    lo = min(get(m, 'plain', H)['fail'] for m in ('SuperPoint + SuperGlue + E', 'SuperPoint + LightGlue + E', 'ALIKED + LightGlue + E', 'LoFTR + E')); hi = max(get(m, 'plain', H)['fail'] for m in ('SuperPoint + SuperGlue + E', 'SuperPoint + LightGlue + E', 'ALIKED + LightGlue + E', 'LoFTR + E'))
    macro('gbPlainFailLo', f'{lo:.0f}'); macro('gbPlainFailHi', f'{hi:.0f}'); macro('gbN', nfmt(d['n_test'])); macro('gbNcal', str(d['n_calibration']))
    # ---- metric scale for the strongest geometric front-end (SIFT + E)
    ps = J('trackB/paired_sift.json'); B = []
    for lab, sc in (('MASt3R metric depth, as is', 'native'), ('MASt3R metric depth $\\times$ one constant', 'calib'), ('Flight-height prior', H), ('Learned distance regression', 'learned'), ('Flight height and regression, blended', 'blend')):
        r = get('SIFT + ratio test + E', 'plain', sc)
        if r: B.append(f'{lab} & {r["d_mae"]:.2f} & {pct(r["sr_d_1m"])} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} \\\\'); t = 'gbScale' + {'native': 'Native', 'calib': 'Calib', H: 'Height', 'learned': 'Learned', 'blend': 'Blend'}[sc]; macro(t + 'SRd', pct(r['sr_d_1m'])); macro(t + 'D', f'{r["d_mae"]:.1f}')
    if ps is not None:
        hy = ps['hybrids']; B.append('\\midrule')
        for lab, k, tag in (('Fused distance of our Stage I', 'DreamNav Stage I fused distance', 'gbHybOne'), ('Fused distance of our Stage II', 'DreamNav Stage II fused distance', 'gbHybTwo'), ('True distance (oracle)', 'true distance (oracle)', 'gbHybOracle')):
            r = hy[k]; B.append(f'{lab} & {r["d_mae"]:.2f} & {r["sr_d"]:.1f} & {r["epe_med"]:.2f} & {r["sr_epe1"]:.1f} \\\\'); macro(tag + 'SRd', f'{r["sr_d"]:.1f}'); macro(tag + 'SRe', f'{r["sr_epe1"]:.1f}'); macro(tag + 'EPEmed', f'{r["epe_med"]:.2f}'); macro(tag + 'D', f'{r["d_mae"]:.2f}')
        j = ps['joint_epe_1m_vs_stage2']; macro('gbJointBoth', pct(j['both'])); macro('gbJointSift', pct(j['baseline_only'])); macro('gbJointOurs', pct(j['dreamnav_only'])); macro('gbJointNone', pct(j['neither'])); macro('gbJointEither', pct(j['both'] + j['baseline_only'] + j['dreamnav_only']))
        pd2 = ps['paired']['DreamNav Stage II']; macro('gbSiftVsIISRd', f'{pd2["sr_d"]["delta"]:+.1f}'); macro('gbSiftVsIISRdLo', f'{pd2["sr_d"]["ci"][0]:+.1f}'); macro('gbSiftVsIISRdHi', f'{pd2["sr_d"]["ci"][1]:+.1f}'); macro('gbSiftVsIISRe', f'{pd2["sr_epe1"]["delta"]:+.1f}'); macro('gbSiftVsIISReLo', f'{pd2["sr_epe1"]["ci"][0]:+.1f}'); macro('gbSiftVsIISReHi', f'{pd2["sr_epe1"]["ci"][1]:+.1f}')
        pd1 = ps['paired']['DreamNav Stage I']; macro('gbSiftVsISRd', f'{pd1["sr_d"]["delta"]:+.1f}'); macro('gbSiftVsISRe', f'{pd1["sr_epe1"]["delta"]:+.1f}')
    od = J('trackA/oracle_distance.json')
    if od is not None:
        B.append('\\midrule\\multicolumn{5}{@{}l}{\\emph{Direction and rotation of our stages, true distance (oracle)}}\\\\')
        for lab, k, tag in (('Stage I', 'sub1000|I', 'gbOracleSi'), ('Stage II', 'sub1000|II', 'gbOracleSii')):
            r = od[k]; B.append(f'{lab} & 0.00 & 100.0 & {r["epe_med_true_dist"]:.2f} & {r["sr_epe_true_dist"]:.1f} \\\\'); macro(tag + 'SRe', f'{r["sr_epe_true_dist"]:.1f}'); macro(tag + 'EPEmed', f'{r["epe_med_true_dist"]:.2f}')
    w('geo_scale', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Where the metric scale comes from.} Upper part: translation direction and rotation from SIFT + essential matrix on the pairs of \\cref{tab:geometric}; only the source of the length changes. Lower part: our own direction with the true length, for reference.}\n\\label{tab:geo_scale}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n'
      '& $d$ MAE & \\SRd & EPE med. & \\SRe \\\\\nSource of the length & (m) & (\\%) & (m) & (\\%) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
f = J('trackB/sift_full.json')
if f is not None:
    macro('sfN', nfmt(f['n'])); macro('sfNfull', nfmt(f['n_full'])); R = f['rows']
    for k, tag in (('SIFT + ratio test + E (MAGSAC++) + flight-height prior (camera frame)', 'sfSift'), ('DreamNav Stage I (camera frame)', 'sfSi'), ('DreamNav Stage II (camera frame)', 'sfSii'), ('SIFT + ratio test + E (MAGSAC++) + flight-height prior (ENU, heading network of the paper)', 'sfSiftEnu')):
        if k in R: macro(tag + 'SRd', pct(R[k]['sr_d_1m'])); macro(tag + 'SRe', pct(R[k]['sr_epe_1m'])); macro(tag + 'D', f'{R[k]["d_mae"]:.2f}'); macro(tag + 'EPEmed', f'{R[k]["epe_med"]:.2f}')
    p = f['paired_baseline_minus_dreamnav_camera_frame']['DreamNav Stage II']; macro('sfVsIISRd', f'{p["sr_d_points"]["delta"]:+.1f}'); macro('sfVsIISRdLo', f'{p["sr_d_points"]["ci"][0]:+.1f}'); macro('sfVsIISRdHi', f'{p["sr_d_points"]["ci"][1]:+.1f}')
    macro('sfVsIISRe', f'{p["sr_epe1_points"]["delta"]:+.1f}'); macro('sfVsIISReLo', f'{p["sr_epe1_points"]["ci"][0]:+.1f}'); macro('sfVsIISReHi', f'{p["sr_epe1_points"]["ci"][1]:+.1f}')
    p = f['paired_baseline_minus_dreamnav_camera_frame']['DreamNav Stage I']; macro('sfVsISRd', f'{p["sr_d_points"]["delta"]:+.1f}'); macro('sfVsISRdLo', f'{p["sr_d_points"]["ci"][0]:+.1f}'); macro('sfVsISRdHi', f'{p["sr_d_points"]["ci"][1]:+.1f}'); macro('sfVsISRe', f'{p["sr_epe1_points"]["delta"]:+.1f}'); macro('sfVsISReLo', f'{p["sr_epe1_points"]["ci"][0]:+.1f}'); macro('sfVsISReHi', f'{p["sr_epe1_points"]["ci"][1]:+.1f}')
u = J('trackB/unscreened.json')
if u is None: placeholder('unscreened', 'tab:unscreened', 'unscreened pairs')
else:
    S = u['sets_common']; B = []
    for key, lab in (('all unscreened pairs', 'All sampled pairs'), ('inside the ranges and passing the screen proxy', 'Within the released ranges, passing the screen'), ('inside the pose-difference ranges of the release', 'Within the released ranges'), ('outside those ranges', 'Outside the released ranges'), ('failing the screen proxy', 'Failing the screen')):
        if key not in S: continue
        s = S[key]; n = s['DreamNav Stage II']['n']; B.append(('\\midrule' if B else '') + f'\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{lab}}} ({nfmt(n)} pairs)}}\\\\')
        for m, ml in (('SIFT + ratio test + E + flight-height prior', 'SIFT + E, flight height'), ('DreamNav Stage I', 'Ours (Stage I)'), ('DreamNav Stage II', 'Ours (Stage II)')):
            r = s[m]; B.append(f'\\quad {ml} & {r["fail"]:.1f} & {pct(r["sr_d_1m"])} {cipct(r["ci"]["sr_d_1m"])} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} {cipct(r["ci"]["sr_epe_1m"])} & {ang(r["theta_med"])} \\\\')
    w('unscreened', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Pairs that were not screened.} ' + nfmt(u['n_pairs']) + ' ordered pairs drawn uniformly from all camera pairs of the ' + str(u['buildings']) + ' test buildings. Camera frame. Fail: pairs without a pose (zero motion is predicted).}\n\\label{tab:unscreened}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n'
      '& Fail & \\SRd@1\\,m & EPE med. & \\SRe@1\\,m & $\\dth$ med. \\\\\n& (\\%) & (\\%) & (m) & (\\%) & ($^\\circ$) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    a = S['all unscreened pairs']
    for m, tag in (('SIFT + ratio test + E + flight-height prior', 'unSift'), ('DreamNav Stage I', 'unSi'), ('DreamNav Stage II', 'unSii')):
        r = a[m]; macro(tag + 'SRd', pct(r['sr_d_1m'])); macro(tag + 'SRe', pct(r['sr_epe_1m'])); macro(tag + 'Fail', f'{r["fail"]:.1f}'); macro(tag + 'EPEmed', f'{r["epe_med"]:.2f}'); macro(tag + 'D', f'{r["d_mae"]:.2f}'); macro(tag + 'ThetaMae', ang(r['theta_mae']))
        macro(tag + 'SRdLo', pct(r['ci']['sr_d_1m'][0])); macro(tag + 'SRdHi', pct(r['ci']['sr_d_1m'][1])); macro(tag + 'SReLo', pct(r['ci']['sr_epe_1m'][0])); macro(tag + 'SReHi', pct(r['ci']['sr_epe_1m'][1]))
    macro('unN', nfmt(u['n_pairs'])); macro('unInside', pct(u['share_inside'])); macro('unScreenPass', pct(u['share_screen_pass'])); macro('unBoth', pct(u['share_inside_and_pass'])); macro('unNoOverlap', pct(u['by_overlap']['< 0.02']['share']))
    for k2, pre in (('inside the ranges and passing the screen proxy', 'unIn'), ('failing the screen proxy', 'unFail'), ('passing the screen proxy', 'unPass')):
        if k2 not in S: continue
        for m, tag in (('SIFT + ratio test + E + flight-height prior', 'Sift'), ('DreamNav Stage II', 'Sii'), ('DreamNav Stage I', 'Si')): macro(pre + tag + 'SRd', pct(S[k2][m]['sr_d_1m'])); macro(pre + tag + 'SRe', pct(S[k2][m]['sr_epe_1m'])); macro(pre + tag + 'Fail', f'{S[k2][m]["fail"]:.1f}')
        macro(pre + 'N', nfmt(S[k2]['DreamNav Stage II']['n']))
    for k2, pre in (('all unscreened pairs', 'unSiiVsSift'), ('inside the ranges and passing the screen proxy', 'unInSiiVsSift')):
        p = u['paired'].get(k2, {}).get('DreamNav Stage II')
        if p:
            macro(pre + 'SRd', f'{p["sr_d_points"]["delta"]:+.1f}'); macro(pre + 'SRdLo', f'{p["sr_d_points"]["ci"][0]:+.1f}'); macro(pre + 'SRdHi', f'{p["sr_d_points"]["ci"][1]:+.1f}'); macro(pre + 'SRe', f'{p["sr_epe1_points"]["delta"]:+.1f}'); macro(pre + 'SReLo', f'{p["sr_epe1_points"]["ci"][0]:+.1f}'); macro(pre + 'SReHi', f'{p["sr_epe1_points"]["ci"][1]:+.1f}')
    j = u['joint']['all unscreened pairs']; macro('unJointBoth', pct(j['both'])); macro('unJointSift', pct(j['sift_only'])); macro('unJointOurs', pct(j['dreamnav_only'])); macro('unJointNone', pct(j['neither']))
# ---- real photographs
rr = J('trackB/real.json')
if rr is None: placeholder('real_geo', 'tab:real_geo', 'geometric baselines on the real photographs', True)
else:
    HR = 'height60_p75ab'; B = []
    def rline(lab, r, rot='', ref=False):
        if r is None: return None
        c = r['ci']; pm = (r.get('paired_minus_stage2') or {}).get('sr_epe_1m_points')
        dl = '--' if (ref or not pm) else f'${pm["delta"]:+.1f}$ \\ci{{{pm["ci"][0]:+.1f}}}{{{pm["ci"][1]:+.1f}}}'
        return f'{lab} & {rot} & {r["n"]} & {r["fail"]:.1f} & {pct(r["sr_d_1m"])} {cipct(c["sr_d_1m"])} & {r["epe_med"]:.2f} \\ci{{{c["epe_med"][0]:.2f}}}{{{c["epe_med"][1]:.2f}}} & {pct(r["sr_epe_1m"])} {cipct(c["sr_epe_1m"])} & {ang(r["theta_med"])} & {r["dir_med"]:.1f} & {dl} \\\\'
    for sk, title in (('real_in', 'Within the training range'), ('real_out', 'Beyond the training range')):
        if sk not in rr['sets']: continue
        S = rr['sets'][sk]; rows = S['rows']; B.append(('\\midrule' if B else '') + f'\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{title}}} (mean true distance {S["mean_true_distance"]:.1f}\\,m)}}\\\\')
        for lab, m, v, pm, rot, tag in GEO_SPEC:
            if rot == 'none': continue
            r = geo_get(rows, m, v, HR); l = rline(lab, r, rot)
            if l:
                B.append(l); t = ('gr' if sk == 'real_in' else 'gro') + tag
                macro(t + 'SRd', pct(r['sr_d_1m'])); macro(t + 'SRe', pct(r['sr_epe_1m'])); macro(t + 'EPEmed', f'{r["epe_med"]:.2f}'); macro(t + 'EPELo', f'{r["ci"]["epe_med"][0]:.2f}'); macro(t + 'EPEHi', f'{r["ci"]["epe_med"][1]:.2f}'); macro(t + 'Fail', f'{r["fail"]:.1f}'); macro(t + 'SReLo', pct(r['ci']['sr_epe_1m'][0])); macro(t + 'SReHi', pct(r['ci']['sr_epe_1m'][1]))
                macro(t + 'N', str(r['n']))
                p = (r.get('paired_minus_stage2') or {}).get('sr_epe_1m_points')
                if p: macro(t + 'VsII', f'{p["delta"]:+.1f}'); macro(t + 'VsIILo', f'{p["ci"][0]:+.1f}'); macro(t + 'VsIIHi', f'{p["ci"][1]:+.1f}')
        for lab, m, sc, tag in (('MASt3R pointmap + PnP, flight height', 'MASt3R pointmap + PnP', 'height60_pointmap', 'MastPnp'), ('Ours (Stage I)', 'DreamNav Stage I', 'rescaled by 60/220.18', 'Si'), ('Ours (Stage II)', 'DreamNav Stage II', 'rescaled by 60/220.18', 'Sii')):
            r = geo_get(rows, m, 'rot. TTA' if 'PnP' in m else None, sc); l = rline(lab, r, 'TTA' if 'PnP' in m else '', ref=(tag == 'Sii'))
            if l: B.append(l); t = ('gr' if sk == 'real_in' else 'gro') + tag; macro(t + 'SRd', pct(r['sr_d_1m'])); macro(t + 'SRe', pct(r['sr_epe_1m'])); macro(t + 'EPEmed', f'{r["epe_med"]:.2f}'); macro(t + 'Dir', f'{r["dir_med"]:.0f}')
    w('real_geo', '\\begin{table*}[t]\n\\centering\n\\caption{\\textbf{Geometric baselines on the real photographs}, zero-shot, with the flight height of 60\\,m given to every method (our rows: displacement rescaled by $60/220$; baselines: flight-height prior with $H{=}60$\\,m). Camera frame; 95\\,\\% intervals by bootstrap over source photographs. Last column: difference of \\SRe{} to our refined stage on the same pairs.}\n\\label{tab:real_geo}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrccccccc@{}}\n\\toprule\n'
      '& & & Fail & \\SRd@1\\,m & EPE med. & \\SRe@1\\,m & $\\dth$ med. & Dir. med. & $\\Delta$\\SRe{} \\vs \\\\\nMethod & Rot. & Pairs & (\\%) & (\\%) & (m) & (\\%) & ($^\\circ$) & ($^\\circ$) & Stage~II (points) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
# ---- SIFT pose and scale cues as additional candidates of our fusion (control subset)
sfu = J('trackA/sift_fusion.json')
if sfu is None: placeholder('sift_fusion', 'tab:sift_fusion', 'SIFT candidates in the fusion')
else:
    Rr = sfu['rows']; B = []
    for k, lab, tag in (('SIFT + E + flight-height prior', 'SIFT + E, flight-height prior', 'sfuSift'), ('Coarse stage', 'Ours, coarse stage', 'sfuCoarse'), ('Coarse stage + SIFT candidates', '\\quad + SIFT candidates', 'sfuCoarseSift'), ('Refined stage', 'Ours, refined stage', 'sfuRef'), ('Refined stage + SIFT candidates', '\\quad + SIFT candidates', 'sfuRefSift')):
        if k not in Rr: continue
        r = Rr[k]; B.append(('\\midrule ' if k in ('Coarse stage', 'Refined stage') else '') + f'{lab} & {r["d_mae"]:.2f} & {r["sr_d"]:.1f} \\ci{{{r["ci_sr_d"][0]:.1f}}}{{{r["ci_sr_d"][1]:.1f}}} & {r["epe_med"]:.2f} & {r["sr_epe"]:.1f} \\ci{{{r["ci_sr_epe"][0]:.1f}}}{{{r["ci_sr_epe"][1]:.1f}}} & {r["dir_med"]:.2f} \\\\')
        macro(tag + 'SRd', f'{r["sr_d"]:.1f}'); macro(tag + 'SRe', f'{r["sr_epe"]:.1f}'); macro(tag + 'D', f'{r["d_mae"]:.2f}')
    w('sift_fusion', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{The SIFT pose and scale cues as additional candidates of our fusion} (' + nfmt(sfu['n']) + ' test pairs; fusions fitted on ' + nfmt(sfu['n_train']) + ' training pairs, as in \\cref{tab:controls}). Camera frame.}\n\\label{tab:sift_fusion}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n'
      '& $d$ & \\SRd@1\\,m & EPE med. & \\SRe@1\\,m & Dir. med. \\\\\n& (m) & (\\%) & (m) & (\\%) & ($^\\circ$) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    for k, tag in (('Refined stage + SIFT candidates - Refined stage', 'sfuAddRef'), ('Refined stage + SIFT candidates - SIFT + E + flight-height prior', 'sfuAllVsSift'), ('Coarse stage + SIFT candidates - Coarse stage', 'sfuAddCoarse'), ('Coarse stage + SIFT candidates - SIFT + E + flight-height prior', 'sfuCoarseSiftVsSift'),
                   ('Refined stage - Coarse stage + SIFT candidates', 'sfuRefVsCoarseSift'), ('Refined stage - SIFT + E + flight-height prior', 'sfuRefVsSift')):
        p = sfu['paired'].get(k)
        if p is None: continue
        for m, t in (('sr_d', 'SRd'), ('sr_epe', 'SRe')): macro(tag + t, f'{p[m]["delta"]:+.1f}'); macro(tag + t + 'Lo', f'{p[m]["ci"][0]:+.1f}'); macro(tag + t + 'Hi', f'{p[m]["ci"][1]:+.1f}')
    macro('sfuN', nfmt(sfu['n'])); macro('sfuNtrain', nfmt(sfu['n_train']))
# ---- final additions: RoMa paired with our stages, sensitivity of the SIFT baseline to its two settings, SIFT vs ours by baseline length
pr_ = J('trackB/paired_roma_tta.json')
if pr_ is not None:
    for k, tag in (('DreamNav Stage II', 'gbRomaVsII'), ('DreamNav Stage I', 'gbRomaVsI')):
        p = pr_['paired'][k]
        for m, t in (('sr_d', 'SRd'), ('sr_epe1', 'SRe')): macro(tag + t, f'{p[m]["delta"]:+.1f}'); macro(tag + t + 'Lo', f'{p[m]["ci"][0]:+.1f}'); macro(tag + t + 'Hi', f'{p[m]["ci"][1]:+.1f}')
    macro('gbRomaOracleSRe', f'{pr_["hybrids"]["true distance (oracle)"]["sr_epe1"]:.1f}')
if d is not None:
    for m, tag in (('DreamNav Stage II [same 200 pairs as RoMa]', 'gbSiiOnRoma'), ('DreamNav Stage I [same 200 pairs as RoMa]', 'gbSiOnRoma')):
        r = geo_get(d['rows'], m)
        if r: macro(tag + 'SRd', pct(r['sr_d_1m'])); macro(tag + 'SRe', pct(r['sr_epe_1m']))
    ts = d.get('threshold_sensitivity_test', {}).get('sift|plain', {})
    for kk in ('3', '3.0'):
        if kk in ts: macro('gbSiftThrThreeSRd', pct(ts[kk]['sr_d_1m'])); macro('gbSiftThrThreeSRe', pct(ts[kk]['sr_epe_1m']))
    r = geo_get(d['rows'], 'SIFT + ratio test + E', 'plain', 'height_p50a')
    if r: macro('gbSiftMedHeightSRd', pct(r['sr_d_1m'])); macro('gbSiftMedHeightSRe', pct(r['sr_epe_1m']))
    r = geo_get(d['rows'], 'RoMa + E', 'plain', H)
    if r: macro('gbRomaPlainFail', f'{r["fail"]:.1f}'); macro('gbRomaPlainSRd', pct(r['sr_d_1m']))
if f is not None:
    for b in f['bins']['true distance (m)']:
        tag = {'< 10': 'sfShort', '>= 30': 'sfLong'}.get(b['bin'])
        if tag:
            p = b['stage2_minus_baseline_sr_epe_points']; macro(tag + 'Delta', f'{p["delta"]:+.1f}'); macro(tag + 'DeltaLo', f'{p["ci"][0]:+.1f}'); macro(tag + 'DeltaHi', f'{p["ci"][1]:+.1f}'); macro(tag + 'N', nfmt(b['n'])); macro(tag + 'SiftSRe', pct(b['baseline']['sr_epe_1m'])); macro(tag + 'SiiSRe', pct(b['stage2']['sr_epe_1m']))
    j = f['joint_success_vs_stage2']['EPE < 1 m']; macro('sfJointBoth', pct(j['both'])); macro('sfJointSift', pct(j['baseline_only'])); macro('sfJointOurs', pct(j['dreamnav2_only'])); macro('sfJointNone', pct(j['neither'])); macro('sfJointEither', pct(j['oracle_best_of_two']))
    macro('sfFail', f'{f["fail_pct"]:.2f}')
    for k, tag in (('SIFT + ratio test + E (MAGSAC++) + flight-height prior (camera frame)', 'sfSift'),):
        r = f['rows'][k]; c = r.get('ci', {})
        if 'sr_d_1m' in c: macro(tag + 'SRdLo', pct(c['sr_d_1m'][0])); macro(tag + 'SRdHi', pct(c['sr_d_1m'][1])); macro(tag + 'SReLo', pct(c['sr_epe_1m'][0])); macro(tag + 'SReHi', pct(c['sr_epe_1m'][1]))
    e = f['rows'].get('DreamNav Stage II (ENU, learned heading)')
    if e: macro('sfSiiEnuSRe', pct(e['sr_epe_1m']))
