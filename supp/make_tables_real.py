# Real-photograph tables (executed by make_tables.py). Rows of DreamNav / policies from trackA/real.json; geometric rows from trackB/real.json when present.
d = J('trackA/real.json')
if d is None: placeholder('real', 'tab:real', 'real photographs', True)
else:
    gb = J('trackB/real.json')
    def line(lab, s, extra=''):
        c = s['ci']
        return (f'{lab} & {s["epe_med"]:.2f} {cinum(c["epe_med"])} & {s["epe_mae"]:.2f} & {pct(s["sr_epe_1m"])} {cipct(c["sr_epe_1m"])} & {pct(s["sr_epe_2m"])} & {pct(s["sr_d_1m"])} & {pct(s["sr_d_2m"])} & {s["theta_mae"]:.2f} / {s["theta_med"]:.2f} & {s["psi_mae"]:.2f} & {s["dir_med"]:.1f} \\\\')
    def block(key, with_pol):
        r = d[key]; B = []
        if with_pol:
            for k, lab in (('ViNT', 'ViNT'), ('NoMaD', 'NoMaD'), ('pi0-FAST', '$\\pi_0$-FAST')):
                if k in r: B.append(line(lab, r[k]))
        B.append(line('Constant (zero motion)', r['Constant (zero motion)'])); B.append(line('MASt3R-statistics MLP, flight height', r['MASt3R-statistics MLP, flight height']))
        B.append('\\cmidrule(l){1-10}'); B.append(line('Ours (Stage I), no flight height', r['DreamNav Stage I, no flight height'])); B.append(line('Ours (Stage II), no flight height', r['DreamNav Stage II, no flight height']))
        B.append(line('Ours (Stage I)', r['DreamNav Stage I'])); B.append(line('Ours (Stage II)', r['DreamNav Stage II']))
        B.append(line('Ours (Stage II), ENU with learned heading', r['DreamNav Stage II, ENU (learned heading)']))
        return B
    i1, i2 = d['in_range']['_info'], d['beyond_range']['_info']
    B = [f'\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{Within the training range: {i1["n"]} pairs, baselines {i1["dist_min"]:.1f}--{i1["dist_max"]:.1f}\\,m}}}}\\\\'] + block('in_range', True) + ['\\midrule', f'\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{Beyond the training range: {i2["n"]} pairs, baselines {i2["dist_min"]:.1f}--{i2["dist_max"]:.1f}\\,m}}}}\\\\'] + block('beyond_range', False)
    cap = ('\\textbf{Real photographs: Table~4 re-scored with uncertainty}, in the camera frame of the first image unless noted, for the pairs within the baseline-to-height range of the training data (the pairs of Table~4) and beyond it. EPE: endpoint error; Table~4 reports its median as ``Pos.\'\' and \\SRd{} as SR. '
           'Brackets: 95\\% bootstrap intervals over source images. Angular errors of $\\dth$ are mean / median. ``No flight height\'\': the output is not rescaled by the height reported by the aircraft.')
    w('real', '\\begin{table*}[t]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:real}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n'
      '& \\multicolumn{2}{c}{EPE (m)} & \\SRe@1\\,m & \\SRe@2\\,m & \\SRd@1\\,m & \\SRd@2\\,m & $\\dth$ error ($^\\circ$) & $\\dpsi$ MAE & Dir. med. \\\\\nMethod & median & mean & (\\%) & (\\%) & (\\%) & (\\%) & mean / med. & ($^\\circ$) & ($^\\circ$) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
    a, b = d['in_range'], d['beyond_range']
    for key, tag in (('DreamNav Stage II', 'rlSii'), ('DreamNav Stage I', 'rlSi')):
        s = a[key]; macro(f'{tag}Pos', f'{s["epe_med"]:.2f}'); macro(f'{tag}SRe', pct(s['sr_epe_1m'])); macro(f'{tag}SReLo', pct(s['ci']['sr_epe_1m'][0])); macro(f'{tag}SReHi', pct(s['ci']['sr_epe_1m'][1])); macro(f'{tag}SRd', pct(s['sr_d_1m'])); macro(f'{tag}EPEmean', f'{s["epe_mae"]:.2f}')
        s = b[key]; macro(f'{tag}OutPos', f'{s["epe_med"]:.1f}'); macro(f'{tag}OutSRe', pct(s['sr_epe_1m'])); macro(f'{tag}OutSRd', pct(s['sr_d_1m'])); macro(f'{tag}OutTheta', f'{s["theta_mae"]:.1f}'); macro(f'{tag}OutThetaMed', f'{s["theta_med"]:.1f}'); macro(f'{tag}OutDir', f'{s["dir_med"]:.0f}')
    macro('rlNoHeightPos', f'{a["DreamNav Stage II, no flight height"]["epe_med"]:.1f}'); macro('rlEnuSRe', pct(a['DreamNav Stage II, ENU (learned heading)']['sr_epe_1m'])); macro('rlEnuHead', f'{a["DreamNav Stage II, ENU (learned heading)"]["abs_heading_err_med"]:.0f}')
    p = a['_paired_II_minus_I']; macro('rlGainSRe', f'{100 * p["sr_epe_1m"]["delta"]:.1f}'); macro('rlGainSReLo', f'{100 * p["sr_epe_1m"]["ci"][0]:.1f}'); macro('rlGainSReHi', f'{100 * p["sr_epe_1m"]["ci"][1]:.1f}')
    macro('rlNin', str(i1['n'])); macro('rlNout', str(i2['n'])); macro('rlImages', str(d['all']['_info']['images'])); macro('rlInMax', f'{i1["dist_max"]:.1f}'); macro('rlOutMax', f'{i2["dist_max"]:.1f}'); macro('rlTilt', f'{d["all"]["_info"]["tilt_a_mean"]:.0f}'); macro('rlZeroOutPos', f'{b["Constant (zero motion)"]["epe_med"]:.1f}')
    macro('rlRangeLimit', f'{d["range_limit_m"]:.1f}'); macro('rlMlpPos', f'{a["MASt3R-statistics MLP, flight height"]["epe_med"]:.1f}')
