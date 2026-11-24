# Table 2 baselines re-trained with camera-frame targets (Track E; executed by make_tables.py)
cf = J('trackE/camframe.json')
if cf is None: placeholder('camframe', 'tab:camframe', 'baselines re-trained in the camera frame')
else:
    R = cf['rows']; B = []
    def M(r):                       # mean over the seeds of the recipe where it was repeated, otherwise the single run
        m = dict(d_mae=r['d_mae'], sr_d_1m=r['sr_d_1m'], epe_med=r['epe_med'], sr_epe_1m=r['sr_epe_1m'], x_mae=r['x_mae'], y_mae=r['y_mae'], corr_h=(r['corr']['x'] + r['corr']['y']) / 2, seeds=1)
        if 'seed_mean' in r: m.update({k: r['seed_mean'][k] for k in ('d_mae', 'sr_d_1m', 'epe_med', 'sr_epe_1m', 'x_mae', 'y_mae', 'corr_h')}); m['seeds'] = r.get('seeds', 1)
        return m
    def hor(r): return (r['x_mae'] + r['y_mae']) / 2
    def cor(r): return r['corr_h']
    SPEC = [('ViNT', 'ViNT | ENU-trained (Table 2)', 'ViNT | camera-frame-trained', 'Vint'), ('NoMaD', 'NoMaD | ENU-trained (Table 2)', 'NoMaD | camera-frame-trained', 'Nomad'),
            ('DINO-WM', 'DINO-WM | ENU-trained, re-run (control)', 'DINO-WM | camera-frame-trained', 'Dinowm'), ('Visual Foresight', 'Visual Foresight | ENU-trained, re-run (control)', 'Visual Foresight | camera-frame-trained', 'Vf'),
            ('DINOv3 ViT-7B', 'DINOv3 ViT-7B | ENU-trained, re-run (control)', 'DINOv3 ViT-7B | camera-frame-trained', 'Dinov'), ('Sample4Geo', 'Sample4Geo | ENU-trained, re-run (control), batch 32', 'Sample4Geo | camera-frame-trained, batch 32', 'Sfg'),
            ('Regression on MASt3R statistics', 'MASt3R-statistics MLP | ENU-trained x/y', 'MASt3R-statistics MLP | camera-frame-trained x/y', 'Mlp')]
    lo = {'d': [], 'srd': [], 'cor': []}
    for lab, ke, kc, tag in SPEC:
        if ke not in R or kc not in R: continue
        a, b = M(R[ke]), M(R[kc]); lab = lab + (f' ({b["seeds"]})' if b['seeds'] > 1 else '')
        if tag == 'Mlp': B.append('\\midrule')
        B.append(f'{lab} & {a["d_mae"]:.2f} & {b["d_mae"]:.2f} & {pct(a["sr_d_1m"])} & {pct(b["sr_d_1m"])} & {hor(a):.2f} & {hor(b):.2f} & {cor(a):+.2f} & {cor(b):+.2f} & {a["epe_med"]:.1f} & {b["epe_med"]:.1f} \\\\'.replace('+0.00', '0.00').replace('-0.00', '0.00'))
        macro('cf' + tag + 'EnuD', f'{a["d_mae"]:.2f}'); macro('cf' + tag + 'CamD', f'{b["d_mae"]:.2f}'); macro('cf' + tag + 'EnuSRd', pct(a['sr_d_1m'])); macro('cf' + tag + 'CamSRd', pct(b['sr_d_1m'])); macro('cf' + tag + 'EnuHor', f'{hor(a):.2f}'); macro('cf' + tag + 'CamHor', f'{hor(b):.2f}'); macro('cf' + tag + 'EnuCorr', f'{cor(a):.2f}'); macro('cf' + tag + 'CamCorr', f'{cor(b):.2f}')
        macro('cf' + tag + 'CamSRe', pct(b['sr_epe_1m'])); macro('cf' + tag + 'EnuEPE', f'{a["epe_med"]:.1f}'); macro('cf' + tag + 'CamEPE', f'{b["epe_med"]:.1f}')
        if tag != 'Mlp': lo['d'].append(b['d_mae']); lo['srd'].append(100 * b['sr_d_1m']); lo['cor'] += [abs(R[kc]['corr']['x']), abs(R[kc]['corr']['y'])]
    z = R.get('Constant (zero motion) | scored in the camera frame'); ze = R.get('Constant (zero motion) | scored in east-north')
    if z and ze: z, ze = M(z) if 'corr' in z and isinstance(z['corr'], dict) and z['corr'].get('x') is not None else dict(z), M(ze) if 'corr' in ze and isinstance(ze['corr'], dict) and ze['corr'].get('x') is not None else dict(ze); B.append('\\midrule'); B.append(f'No motion & {ze["d_mae"]:.2f} & {z["d_mae"]:.2f} & {pct(ze["sr_d_1m"])} & {pct(z["sr_d_1m"])} & {hor(ze):.2f} & {hor(z):.2f} & -- & -- & {ze["epe_med"]:.1f} & {z["epe_med"]:.1f} \\\\'); macro('cfZeroCamHor', f'{hor(z):.2f}'); macro('cfZeroEnuHor', f'{hor(ze):.2f}')
    cd = R.get('Constant distance (20.11 m, train median), direction unknown | only d and SR_d are meaningful')
    if cd: macro('cfConstD', f'{cd["d_mae"]:.2f}'); macro('cfConstSRd', pct(cd['sr_d_1m']))
    macro('cfMaxCamSRd', f'{max(lo["srd"]):.1f}'); macro('cfMinCamD', f'{min(lo["d"]):.2f}'); macro('cfMaxCamCorr', f'{max(lo["cor"]):.2f}'); macro('cfMaxCamSRe', pct(max(R[kc]['sr_epe_1m'] for _, _, kc, t in SPEC if t != 'Mlp' and kc in R), 2))
    w('camframe', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Baselines of Table~2 re-trained with the horizontal target in the camera frame} (same recipes; ' + f'{cf["n"]:,}'.replace(',', '{,}') + ' test pairs). Hor.: mean absolute error of the two horizontal components; Corr.: correlation between predicted and true horizontal components, averaged over the two. A number in parentheses: recipes repeated with that many seeds, mean reported. $\\pi_{0.5}$ and $\\pi_0$-FAST were not re-trained.}\n\\label{tab:camframe}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n'
      '& \\multicolumn{2}{c}{$d$ (m)} & \\multicolumn{2}{c}{\\SRd{} (\\%)} & \\multicolumn{2}{c}{Hor. (m)} & \\multicolumn{2}{c}{Corr.} & \\multicolumn{2}{c}{EPE med. (m)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\nTraining target & ENU & cam. & ENU & cam. & ENU & cam. & ENU & cam. & ENU & cam. \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
