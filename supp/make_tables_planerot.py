# Where the plane-induced part of the warp helps (executed by make_tables.py)
d = J('trackA/plane_vs_rot.json')
if d is None: placeholder('plane_vs_rot', 'tab:plane_vs_rot', 'plane-induced warp vs rotation-only alignment by regime')
else:
    def blk(key, unit):
        return [f'{r["bin"].replace(">=", "$\\geq$").replace("<", "$<$").replace("%", "\\%")}{unit} & {r["n"]} & {pct(r["coarse"])} & {pct(r["rot"])} & {pct(r["plane"])} & {100 * r["diff"]:+.1f} \\ci{{{100 * r["ci"][0]:+.1f}}}{{{100 * r["ci"][1]:+.1f}}} \\\\' for r in d[key]]
    B = ['\\multicolumn{6}{@{}l}{\\emph{Larger tilt of the pair}}\\\\'] + blk('by_tilt_max', '$^\\circ$') + ['\\midrule\\multicolumn{6}{@{}l}{\\emph{Altitude change $|\\Delta z|$}}\\\\'] + blk('by_abs_dz_m', '\\,m') + ['\\midrule\\multicolumn{6}{@{}l}{\\emph{Relief}}\\\\'] + blk('by_relief', '')
    w('plane_vs_rot', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Where the plane-induced part of the warp helps.} Endpoint success (\\%) of the coarse stage, of the compute-matched refinement with rotation-only alignment and of the refinement with the plane-induced warp, and their paired difference in points with a 95\\% bootstrap interval. Same pairs and protocol as \\cref{tab:controls}.}\n\\label{tab:plane_vs_rot}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}lrcccc@{}}\n\\toprule\n'
      '& & & Rot. & Plane & Plane $-$ rot. \\\\\nBin & Pairs & Coarse & only & warp & (points) \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    bt = {r['bin']: r for r in d['by_tilt_max']}; bz = {r['bin']: r for r in d['by_abs_dz_m']}
    for r, tag in ((bt['35-45'], 'prTiltHi'), (bt['0-15'], 'prTiltLo'), (bz['>=25'], 'prDzHi'), (bz['0-5'], 'prDzLo')):
        macro(tag, f'{100 * r["diff"]:+.1f}'); macro(tag + 'Lo', f'{100 * r["ci"][0]:+.1f}'); macro(tag + 'Hi', f'{100 * r["ci"][1]:+.1f}')
