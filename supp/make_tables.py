#!/usr/bin/env python3
"""Generate the LaTeX tables and number macros of the supplementary material from the result files under outputs/supp/.
Nothing is typed by hand: every number of docs/paper/supp/tables/*.tex and numbers.tex comes from a JSON written by an experiment script.
A table whose result file does not exist yet is written as a visible placeholder so that the document still compiles.
Usage: make_tables.py
"""
import json, os, sys, math
import numpy as np
R = '/root/rasr-pairuav/outputs/supp'; T = '/root/rasr-pairuav/docs/paper/supp/tables'; os.makedirs(T, exist_ok=True)
NUM = {}                                     # macro name -> string


def J(path):
    f = f'{R}/{path}'
    return json.load(open(f)) if os.path.exists(f) else None


FIT = {'geo_scale', 'servo', 'dataset', 'fixedsteps'}        # single-column tables that are slightly too wide: scaled to the column width
def w(name, tex):
    if name in FIT and '\\resizebox' not in tex:
        tex = tex.replace('\\begin{tabular}', '\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}', 1); i = tex.rindex('\\end{tabular}') + len('\\end{tabular}'); tex = tex[:i] + '}' + tex[i:]
    import re
    tex = re.sub(r'(?<=& )([+-]\d+(?:\.\d+)?)(?= (?:\\ci|&|\\\\))', r'$\1$', tex)      # signed numbers in table cells: true minus sign
    open(f'{T}/{name}.tex', 'w').write(tex.rstrip() + '\n'); print('wrote', name)


def placeholder(name, label, what, star=False):
    env = 'table*' if star else 'table'
    w(name, f'\\begin{{{env}}}[t]\\centering\\fbox{{\\parbox{{0.9\\linewidth}}{{\\centering\\textbf{{Pending:}} {what}}}}}\\caption{{Pending result.}}\\label{{{label}}}\\end{{{env}}}')


def pct(v, d=1): return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{100 * v:.{d}f}'
def num(v, d=2): return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{d}f}'
def cipct(c): return f'\\ci{{{100 * c[0]:.1f}}}{{{100 * c[1]:.1f}}}'
def cinum(c, d=2): return f'\\ci{{{c[0]:.{d}f}}}{{{c[1]:.{d}f}}}'
def macro(k, v):
    assert k.isalpha(), f'macro name must be alphabetic: {k}'
    v = str(v)
    if len(v) > 1 and v[0] in '+-' and v[1].isdigit(): v = '\\ensuremath{' + v + '}'      # signed numbers: a true minus sign also in running text
    NUM[k] = v


# ----------------------------------------------------------------------------------------------- runtime
def t_runtime():
    d = J('runtime/runtime.json')
    if d is None: return placeholder('runtime', 'tab:runtime', 'runtime benchmark')
    L = d['latency_ms']; rows = [('MASt3R forward pass, pooled statistics, prior heads', 'mast3r_priors'), ('First matching pass (4 coarse + 1--2 fine)', 'match_pass1'), ('Pose: essential matrix and \\cref{eq:map}, first pass', 'pose_pass1'),
                                 ('Zoom re-matching', 'match_zoom'), ('Pose, zoom pass', 'pose_zoom'), ('Fusion, coarse stage', 'fusion1'), ('\\emph{Coarse stage, total}', None),
                                 ('Imagination: warp and tile-wise matching', 'match_imagine'), ('Pose and verification on $\\mathcal{M}_{\\mathrm{im}}$', 'pose_imagine'), ('Fusion, refinement stage', 'fusion2'), ('\\emph{Refinement stage, total}', None),
                                 ('Absolute heading (2 networks, 96 views)', 'abs_heading')]
    c1 = sum(L[k]['mean'] for k in ('mast3r_priors', 'match_pass1', 'pose_pass1', 'match_zoom', 'pose_zoom', 'fusion1')); c2 = sum(L[k]['mean'] for k in ('match_imagine', 'pose_imagine', 'fusion2')); tot = L['total_wall']['mean']
    body = []
    for lab, k in rows:
        if k is None: v = c1 if 'Coarse' in lab else c2; body.append(f'\\rowcolor{{rowgray}} {lab} & {v:.0f} & {100 * v / tot:.0f} \\\\')
        else: body.append(f'{lab} & {L[k]["mean"]:.0f} & {100 * L[k]["mean"] / tot:.0f} \\\\')
    p = d['passes']
    w('runtime', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Running time} of DreamNav for one image pair on an RTX~5090 (mean over ' + str(d['latency_pairs']) + ' test pairs, single process, including image decoding).}\n\\label{tab:runtime}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrr@{}}\n\\toprule\nStep & ms & \\% \\\\\n\\midrule\n'
      + '\n'.join(body) + f'\n\\midrule\nTotal & {tot:.0f} & 100 \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}')
    macro('rtTotal', f'{tot / 1000:.1f}'); macro('rtHz', f'{1000 / tot:.2f}'); macro('rtStageOne', f'{d["latency_stage1_no_heading_ms"]["mean"] / 1000:.2f}'); macro('rtBoth', f'{d["latency_stage2_no_heading_ms"]["mean"] / 1000:.1f}')
    macro('rtBatch', f'{d["throughput"]["s_per_pair"]:.2f}'); macro('rtMem', f'{d["peak_gpu_mem_gb"]:.1f}'); macro('rtSG', f'{p["superglue_total_mean"]:.1f}'); macro('rtSGmax', str(p['superglue_total_max'])); macro('rtSP', f'{p["superpoint_total_mean"]:.1f}')
    macro('rtFine', f'{p["superglue_pass1_fine_mean"]:.2f}'); macro('rtTiles', f'{p["imag_tiles_mean"]:.2f}'); macro('rtMast', f'{L["mast3r_priors"]["mean"] / 1000:.2f}')


# ----------------------------------------------------------------------------------------------- all methods, explicit metrics
ORDER = [('pi0.5', '$\\pi_{0.5}$'), ('pi0-FAST', '$\\pi_0$-FAST'), ('ViNT', 'ViNT'), ('NoMaD', 'NoMaD'), ('DINO-WM', 'DINO-WM'), ('Visual Foresight', 'Visual Foresight'), ('DINOv3 ViT-7B', 'DINOv3 ViT-7B'), ('Sample4Geo', 'Sample4Geo')]
def t_main():
    d = J('metrics/main.json')
    if d is None: return placeholder('main_metrics', 'tab:main_metrics', 'all methods, explicit metrics', True)
    M = d['methods']
    def row(lab, s, frame=None, dh=False):
        c = s['ci']; dm = f'{s["d_mae"]:.2f}'; sr = f'{pct(s["sr_d_1m"])} {cipct(c["sr_d_1m"])}'
        return (f'{lab} & {frame or ("cam." if s["frame"] == "cam" else "ENU")} & {dm} & {sr} & {s["epe_mae"]:.2f} & {s["epe_med"]:.2f} & {pct(s["sr_epe_1m"])} {cipct(c["sr_epe_1m"])} & {pct(s["sr_epe_2m"])} & {pct(s["sr_epe_yaw_oracle_1m"])} & '
                f'{s["z_mae"]:.2f} & {s["dir_med"]:.1f} & {s["theta_mae"]:.2f} & {s["psi_mae"]:.2f} & {pct(s["sr_5dof_1m_1deg"])} & {pct(s["sr_5dof_1m_5deg"])} \\\\')
    B = [row(lab, M[k]) for k, lab in ORDER]
    B.append('\\midrule'); B.append(row('MASt3R-statistics regression~\\cite{liang2026rasr}', M['MASt3R-statistics regression (RASR heads)'])); B.append(row('\\quad MLP heads, sin/cos heading', M['MASt3R-statistics MLP, world-frame x/y'])); B.append(row('\\quad MLP heads, camera-frame target', M['MASt3R-statistics MLP, camera-frame x/y']))
    B.append(row('Constant (zero motion)', M['Constant (zero motion)'], frame='--')); B.append('\\midrule')
    for st in ('I', 'II'):
        B.append(row(f'Ours (Stage {st})', M[f'DreamNav Stage {st} (ENU, learned heading)'])); B.append(row(f'\\quad camera frame / true heading', M[f'DreamNav Stage {st} (camera frame)'], frame='cam.'))
    cap = ('\\textbf{All methods of Table~2 under explicit definitions} (' + f'{d["n"]:,}'.replace(',', '{,}') + ' test pairs; brackets: 95\\% bootstrap intervals over buildings). $d$: range error (the $d$ column of Table~2). \\SRd: success under the range error (the SR column of Table~2). EPE: 3-D endpoint error; \\SRe: endpoint success. '
           '``Yaw-free\'\': endpoint success after the best rotation of the prediction about the vertical axis, an upper bound for any heading source. \\SRf: endpoint within 1\\,m and both angles within $1^\\circ$ or $5^\\circ$. '
           'For ViNT and NoMaD, Table~2 reports $d$ and SR from their distance heads (' + f'{M["ViNT"]["dist_head"]["d_mae"]:.2f}\\,m, {pct(M["ViNT"]["dist_head"]["sr_d_1m"])}\\% and {M["NoMaD"]["dist_head"]["d_mae"]:.2f}\\,m, {pct(M["NoMaD"]["dist_head"]["sr_d_1m"])}\\%' + '); here every row uses the norm of the predicted displacement.')
    w('main_metrics', '\\begin{table*}[t]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:main_metrics}\n\\footnotesize\n\\setlength{\\tabcolsep}{3.2pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llcccccccccccccc@{}}\n\\toprule\n'
      '& & $d$ $\\downarrow$ & \\SRd@1\\,m $\\uparrow$ & \\multicolumn{2}{c}{EPE (m) $\\downarrow$} & \\SRe@1\\,m $\\uparrow$ & \\SRe@2\\,m & Yaw-free & $\\Delta z$ & Dir. & $\\dth$ & $\\dpsi$ & \\multicolumn{2}{c}{\\SRf (1\\,m) $\\uparrow$} \\\\\n'
      'Method & Frame & MAE (m) & (\\%) & mean & med. & (\\%) & (\\%) & \\SRe@1\\,m & MAE & med.\\,($^\\circ$) & MAE\\,($^\\circ$) & MAE\\,($^\\circ$) & $1^\\circ$ & $5^\\circ$ \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
    s2, s2c, s1, s1c = (M[k] for k in ('DreamNav Stage II (ENU, learned heading)', 'DreamNav Stage II (camera frame)', 'DreamNav Stage I (ENU, learned heading)', 'DreamNav Stage I (camera frame)'))
    for tag, s in (('SiiEnu', s2), ('SiiCam', s2c), ('SiEnu', s1), ('SiCam', s1c)):
        macro(f'{tag}D', f'{s["d_mae"]:.2f}'); macro(f'{tag}SRd', pct(s['sr_d_1m'])); macro(f'{tag}EPEmean', f'{s["epe_mae"]:.2f}'); macro(f'{tag}EPEmed', f'{s["epe_med"]:.2f}'); macro(f'{tag}SRe', pct(s['sr_epe_1m'])); macro(f'{tag}SRf', pct(s['sr_5dof_1m_1deg']))
        macro(f'{tag}SReLo', pct(s['ci']['sr_epe_1m'][0])); macro(f'{tag}SReHi', pct(s['ci']['sr_epe_1m'][1])); macro(f'{tag}SRdLo', pct(s['ci']['sr_d_1m'][0])); macro(f'{tag}SRdHi', pct(s['ci']['sr_d_1m'][1]))
        macro(f'{tag}X', f'{s["x_mae"]:.2f}'); macro(f'{tag}Y', f'{s["y_mae"]:.2f}'); macro(f'{tag}Z', f'{s["z_mae"]:.2f}')
    macro('nTest', f'{d["n"]:,}'.replace(',', '{,}')); macro('nBuildings', str(d['tours']))
    macro('baseMaxSRe', pct(max(M[k]['sr_epe_1m'] for k, _ in ORDER))); macro('baseMaxYawFree', pct(max(M[k]['sr_epe_yaw_oracle_1m'] for k, _ in ORDER))); macro('baseMinZ', f'{min(M[k]["z_mae"] for k, _ in ORDER):.1f}'); macro('baseMaxZ', f'{max(M[k]["z_mae"] for k, _ in ORDER):.1f}'); macro('zeroZ', f'{M["Constant (zero motion)"]["z_mae"]:.1f}')
    mw, mc = M['MASt3R-statistics MLP, world-frame x/y'], M['MASt3R-statistics MLP, camera-frame x/y']; macro('mlpWorldD', f'{mw["d_mae"]:.2f}'); macro('mlpCamD', f'{mc["d_mae"]:.2f}'); macro('mlpWorldSRd', pct(mw['sr_d_1m'])); macro('mlpCamSRd', pct(mc['sr_d_1m'])); macro('mlpCamSRe', pct(mc['sr_epe_1m']))
    P = M['_paired_stage2_minus_stage1']
    for k, tag in (('sr_d_1m', 'gainSRd'), ('sr_epe_1m_cam', 'gainSReCam'), ('sr_epe_1m_enu', 'gainSReEnu')): macro(tag, f'{100 * P[k]["delta"]:.1f}'); macro(tag + 'Lo', f'{100 * P[k]["ci"][0]:.1f}'); macro(tag + 'Hi', f'{100 * P[k]["ci"][1]:.1f}')


# ----------------------------------------------------------------------------------------------- component inventory (Table 3)
def t_inventory():
    d = J('trackA/ablation_full.json')
    if d is None: return placeholder('inventory', 'tab:inventory', 'component inventory', True)
    rows = d['rows']; by = {(r['id'], r['label']): r for r in rows}
    def get(i, key): return next(r for r in rows if r['id'] == i and key in r['label'])
    def line(lab, r, enu=True, mark=''):
        en = r.get('enu') if enu else None
        return (f'{lab} & {pct(r["fail"])} & {r["d_mae"]:.2f} & {pct(r["sr_d_1m"])} & {r["epe_mae"]:.2f} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} & {r["theta_mae"]:.2f} / {r["theta_med"]:.3f} & {r["psi_mae"]:.2f} / {r["psi_med"]:.3f} & {r["dir_med"]:.2f} & '
                + (f'{pct(en["sr_epe_1m"])}' if en else '--') + ' \\\\')
    B = ['\\multicolumn{11}{@{}l}{\\emph{Learned priors only (no correspondences)}}\\\\', line('P\\quad MLP on pooled MASt3R statistics', get('P', 'Learned priors'))]
    B.append('\\midrule\\multicolumn{11}{@{}l}{\\emph{Two-view geometry only: no learned prior, no fusion; scale from the flight-height prior}}\\\\')
    if any(r['id'] == 'G0' for r in rows): B.append(line('G0\\quad Frozen SuperGlue, one pass, E', get('G0', 'flight-height')))
    for i, lab in (('G1', 'Rotation-aware matching, E'), ('G3', '\\quad + zoom re-matching, E'), ('G5', '\\quad + imagination re-matching, E')): B.append(line(f'{i}\\quad {lab}', get(i, 'flight-height')))
    B.append('\\midrule\\multicolumn{11}{@{}l}{\\emph{Geometry with learned priors in the pose refinement (\\cref{eq:map}); scale from the flight-height prior; no fusion}}\\\\')
    for i, lab in (('G2', 'Rotation-aware matching, refined'), ('G4', '\\quad + zoom re-matching, refined'), ('G6', '\\quad + imagination re-matching, refined')): B.append(line(f'{i}\\quad {lab}', get(i, 'flight-height')))
    B.append('\\midrule\\multicolumn{11}{@{}l}{\\emph{Same geometry, scale from the learned distance regression instead of the flight-height prior}}\\\\')
    for i, lab in (('G2', 'Rotation-aware matching, refined'), ('G6', 'Imagination re-matching, refined')): B.append(line(f"{i}$'$\\quad {lab}", get(i, 'learned distance')))
    hm = J('trackA/homography.json')
    if hm is not None:
        B.append('\\midrule\\multicolumn{11}{@{}l}{\\emph{Planar pose model: homography decomposition instead of the essential matrix; scale from the flight height; no learning}}\\\\')
        for k, i, lab in (('P1', 'H1', 'Rotation-aware matching, H'), ('Z', 'H3', '\\quad + zoom re-matching, H'), ('IM', 'H5', '\\quad + imagination re-matching, H')):
            r = dict(hm[k]); B.append(line(f'{i}\\quad {lab}', r, enu=False)); nm = {'1': 'One', '3': 'Three', '5': 'Five'}[i[1:]]; macro(f'hm{nm}SRd', pct(r['sr_d_1m'])); macro(f'hm{nm}SRe', pct(r['sr_epe_1m'])); macro(f'hm{nm}EPEmed', f'{r["epe_med"]:.1f}')
        macro('hmHzero', f'{hm["IM"]["H0"]:.0f}')
    B.append('\\midrule\\multicolumn{11}{@{}l}{\\emph{Reliability-weighted fusion of all candidates and scale cues (the rows of Table~3)}}\\\\')
    B.append(line('F1\\quad Rotation-aware pass only', get('F1', 'Fused'))); B.append(line('F2\\quad + zoom re-matching (Stage I)', get('F2', 'Coarse'))); B.append(line('F3\\quad + geometric imagination (Stage II)', get('F3', 'Refined')))
    cap = ('\\textbf{Component inventory} on the ' + f'{d["n"]:,}'.replace(',', '{,}') + ' test pairs. All columns but the last are in the camera frame and involve no absolute heading. Pairs whose pose cannot be solved predict zero motion (``fail\'\'). '
           'Angular errors are mean / median. The last column is the endpoint success in ENU with the learned absolute heading.')
    w('inventory', '\\begin{table*}[t]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:inventory}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lcccccccccc@{}}\n\\toprule\n'
      '& fail & $d$ MAE & \\SRd@1\\,m & \\multicolumn{2}{c}{EPE (m)} & \\SRe@1\\,m & $\\dth$ error ($^\\circ$) & $\\dpsi$ error ($^\\circ$) & Dir. med. & ENU \\\\\n'
      'Configuration & (\\%) & (m) & (\\%) & mean & med. & (\\%) & mean / med. & mean / med. & ($^\\circ$) & \\SRe@1\\,m \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
    for i, key, tag in (('G1', 'flight-height', 'gOne'), ('G2', 'flight-height', 'gTwo'), ('G4', 'flight-height', 'gFour'), ('G5', 'flight-height', 'gFive'), ('G6', 'flight-height', 'gSix'), ('F1', 'Fused', 'fOne'), ('P', 'Learned', 'pri'), ('G6', 'learned distance', 'gSixLearned'), ('G2', 'learned distance', 'gTwoLearned')) + ((('G0', 'flight-height', 'gZero'),) if any(r['id'] == 'G0' for r in rows) else ()):
        r = get(i, key); macro(f'{tag}SRd', pct(r['sr_d_1m'])); macro(f'{tag}SRe', pct(r['sr_epe_1m'])); macro(f'{tag}D', f'{r["d_mae"]:.2f}'); macro(f'{tag}EPEmed', f'{r["epe_med"]:.2f}'); macro(f'{tag}EPEmean', f'{r["epe_mae"]:.2f}'); macro(f'{tag}Fail', pct(r['fail'])); macro(f'{tag}Theta', f'{r["theta_mae"]:.2f}'); macro(f'{tag}Psi', f'{r["psi_mae"]:.2f}')
        if r.get('enu'): macro(f'{tag}EnuSRe', pct(r['enu']['sr_epe_1m']))
    e = d['extra']['table3_row2_as_printed (z, angles fused; x/y from the world-frame regression)']; macro('tThreeRowTwoD', f'{e["d_mae"]:.2f}'); macro('tThreeRowTwoSR', pct(e['sr_d_1m']))


# ----------------------------------------------------------------------------------------------- controls for the refinement stage
def t_controls():
    d = J('trackA/controls.json')
    if d is None: return placeholder('controls', 'tab:controls', 'refinement controls', True)
    rows = {r['label']: r for r in d['rows']}
    def line(lab, key, note=''):
        if key not in rows: return f'{lab} & \\multicolumn{{8}}{{c}}{{pending}} \\\\'
        r = rows[key]; c = r['ci']; vc = r['vs_coarse']
        g = lambda k: f'{100 * vc[k]["delta"]:+.1f} \\ci{{{100 * vc[k]["ci"][0]:+.1f}}}{{{100 * vc[k]["ci"][1]:+.1f}}}'
        return f'{lab} & {r["d_mae"]:.2f} & {pct(r["sr_d_1m"])} & {r["epe_mae"]:.2f} & {r["epe_med"]:.2f} & {pct(r["sr_epe_1m"])} & {r["theta_mae"]:.3f} & {r["psi_mae"]:.3f} & {g("sr_d_1m") if key != "Coarse stage (Stage I)" else "--"} & {g("sr_epe_1m") if key != "Coarse stage (Stage I)" else "--"} \\\\'
    B = [line('Coarse stage (input of all rows below)', 'Coarse stage (Stage I)'), '\\midrule\\multicolumn{10}{@{}l}{\\emph{(a) Learned capacity without new correspondences}}\\\\', line('Second fusion stage on the coarse candidates', 'Second fusion stage only'),
         '\\midrule\\multicolumn{10}{@{}l}{\\emph{(b) New correspondences, full fusion; what the second matching pass sees}}\\\\', line('Rotation-only alignment, $\\le 3{\\times}3$ tiles (compute-matched)', 'Tile-wise re-matching, rotation-only alignment'),
         line('Plane-induced warp, 1 tile (no extra budget)', 'Plane-induced warp, single tile'), line('Plane-induced warp at $512{\\times}288$, 1 tile', 'Plane-induced warp at 512x288'), line('Plane-induced warp, $\\le 3{\\times}3$ tiles = Stage II', 'Full refinement (Stage II)'),
         '\\midrule\\multicolumn{10}{@{}l}{\\emph{(c) Plane-induced warp with $\\le 3{\\times}3$ tiles; how the re-estimated pose is used}}\\\\', line('Re-estimated pose replaces the coarse pose (no fusion)', 'Warp + tiles, re-estimated pose used directly'),
         line('Uniform candidate weights, no residual', 'Warp + tiles, uniform candidate weights'), line('Predicted-error weights, no residual', 'Warp + tiles, predicted-error weights'), line('Full fusion without verification scores', 'Warp + tiles, fusion without verification scores'),
         line('Full fusion, scale kept from the coarse stage', 'Warp + tiles, fusion with the coarse scale'), line('Full fusion = Stage II', 'Full refinement (Stage II)')]
    cap = ('\\textbf{Controls for the refinement stage} on ' + f'{d["n"]:,}'.replace(',', '{,}') + ' test pairs; every fusion is fitted on the same ' + f'{d["train_rows"]:,}'.replace(',', '{,}') + ' training pairs. Camera frame. The last two columns are paired differences to the coarse stage in percentage points with 95\\% bootstrap intervals over buildings.')
    w('controls', '\\begin{table*}[t]\n\\centering\n\\caption{' + cap + '}\n\\label{tab:controls}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n'
      '& $d$ MAE & \\SRd@1\\,m & \\multicolumn{2}{c}{EPE (m)} & \\SRe@1\\,m & $\\dth$ & $\\dpsi$ & $\\Delta$\\SRd & $\\Delta$\\SRe \\\\\nVariant & (m) & (\\%) & mean & med. & (\\%) & MAE ($^\\circ$) & MAE ($^\\circ$) & \\vs coarse & \\vs coarse \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table*}')
    RG = d['raw_geometry']; B = []
    def rawline(k, lab):
        if k not in RG: return f'{lab} & \\multicolumn{{8}}{{c}}{{pending}} \\\\'
        r = RG[k]; return (f'{lab} & {num(r["tiles_mean"], 1) if "tiles_mean" in r else "1"} & {r.get("inliers_med", float("nan")):.0f} & {pct(r.get("inl_lt100", float("nan")))} & {num(r["resid_px_med"], 1) if "resid_px_med" in r else "--"} & '
                           f'{r["MAP"]["dir_med"]:.2f} / {r["MAP"]["dir_mean"]:.1f} & {pct(r["MAP"]["dir_lt2"])} & {r["MAP"]["theta_mean"]:.2f} & {r["MAP"]["psi_mean"]:.2f} \\\\')
    B.append(f'\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{All {d["n"]:,} test pairs of \\cref{{tab:controls}}}}}}\\\\'.replace(',', '{,}'))
    for k, lab in (('Coarse stage, zoom pass', 'Zoom pass of the coarse stage'), ('Rotation-only alignment, <= 3x3 tiles', 'Rotation-only alignment, tiles'), ('Plane-induced warp, 1 tile', 'Plane-induced warp, 1 tile'), ('Plane-induced warp, 512x288', 'Plane-induced warp at $512{\\times}288$'), ('Plane-induced warp, <= 3x3 tiles (ours)', 'Plane-induced warp, tiles (ours)')): B.append(rawline(k, lab))
    nsub = RG.get('Plane-induced warp under the true pose, <= 3x3 tiles', {}).get('n'); n90 = RG.get('No pose-guided alignment, <= 3x3 tiles, oracle subset', {}).get('n')
    B.append('\\midrule' + f'\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Subset of {nsub:,} pairs}}}}\\\\'.replace(',', '{,}') if nsub else '\\midrule')
    for k, lab in (('Coarse stage, zoom pass, oracle subset', 'Zoom pass of the coarse stage'), ('Plane-induced warp, <= 3x3 tiles (ours), oracle subset', 'Plane-induced warp, tiles (ours)'), ('Plane-induced warp under the true pose, <= 3x3 tiles', 'Plane-induced warp from the true pose, tiles')): B.append(rawline(k, lab))
    if n90:
        B.append('\\midrule' + f'\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{Subset of {n90:,} pairs}}}}\\\\'.replace(',', '{,}'))
        for k, lab in (('No pose-guided alignment, <= 3x3 tiles, oracle subset', 'No pose-guided alignment, tiles'), ('Rotation-only alignment, <= 3x3 tiles, oracle subset', 'Rotation-only alignment, tiles')): B.append(rawline(k, lab))
    w('raw_geometry', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{What each second matching pass delivers before any fusion}: tiles matched (at most $3{\\times}3$), median number of inliers, share of pairs with fewer than 100 inliers, median displacement between the aligned and the real view, and accuracy of the pose refined with \\cref{eq:map}. Unsolved pairs count as $90^\\circ$ / $20^\\circ$.}\n\\label{tab:raw_geometry}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n'
      '& & Inl. & $<$100 & Res. & Dir. ($^\\circ$) & Dir. & $\\dth$ & $\\dpsi$ \\\\\nSecond matching pass & Tiles & med. & (\\%) & (px) & med. / mean & $<2^\\circ$ & mean & mean \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}}\n\\end{table}')
    for key, tag in (('Coarse stage (Stage I)', 'ctCoarse'), ('Second fusion stage only', 'ctFusionOnly'), ('Tile-wise re-matching, rotation-only alignment', 'ctRot'), ('Tile-wise re-matching, no pose-guided alignment', 'ctNoAlign'), ('Plane-induced warp, single tile', 'ctOneTile'), ('Plane-induced warp at 512x288', 'ctLowRes'),
                     ('Warp + tiles, re-estimated pose used directly', 'ctDirect'), ('Warp + tiles, uniform candidate weights', 'ctUniform'), ('Warp + tiles, predicted-error weights', 'ctLearnedNoRes'), ('Warp + tiles, fusion without verification scores', 'ctNoVer'),
                     ('Warp + tiles, fusion with the coarse scale', 'ctNoScale'), ('Full refinement (Stage II)', 'ctFull')):
        if key not in rows: continue
        r = rows[key]; macro(f'{tag}SRd', pct(r['sr_d_1m'])); macro(f'{tag}SRe', pct(r['sr_epe_1m'])); macro(f'{tag}D', f'{r["d_mae"]:.2f}'); macro(f'{tag}EPEmed', f'{r["epe_med"]:.2f}'); macro(f'{tag}dSRd', f'{100 * r["vs_coarse"]["sr_d_1m"]["delta"]:+.1f}'); macro(f'{tag}dSRe', f'{100 * r["vs_coarse"]["sr_epe_1m"]["delta"]:+.1f}')
        macro(f'{tag}ShareD', pct(r['share_of_full_gain']['sr_d_1m'], 0)); macro(f'{tag}ShareE', pct(r['share_of_full_gain']['sr_epe_1m'], 0)); macro(f'{tag}Theta', f'{r["theta_mae"]:.3f}'); macro(f'{tag}Psi', f'{r["psi_mae"]:.3f}')
    macro('ctN', f'{d["n"]:,}'.replace(',', '{,}'))
    ref = rows.get('Full refinement (Stage II)'); alt = [rows[k] for k in ('Stage II, fusion fitted on 6,000 pairs', 'Stage II, fusion fitted on 12,000 pairs (main paper)') if k in rows]
    if ref and alt:
        macro('ctTrainSizeGap', f'{100 * max(max(abs(a["sr_d_1m"] - ref["sr_d_1m"]), abs(a["sr_epe_1m"] - ref["sr_epe_1m"])) for a in alt):.1f}')
        for a, tag in zip(alt, ('ctSixK', 'ctTwelveK')): macro(f'{tag}SRd', pct(a['sr_d_1m'])); macro(f'{tag}SRe', pct(a['sr_epe_1m']))
    macro('ctTrainRows', f'{d["train_rows"]:,}'.replace(',', '{,}'))
    for key, tag in (('Tile-wise re-matching, rotation-only alignment', 'ctRot'), ('Tile-wise re-matching, no pose-guided alignment', 'ctNoAlign'), ('Plane-induced warp, single tile', 'ctOneTile'), ('Plane-induced warp at 512x288', 'ctLowRes')):
        if key not in rows: continue
        vf = rows[key]['vs_full']
        for k, kt in (('sr_d_1m', 'VsFullSRd'), ('sr_epe_1m', 'VsFullSRe')): macro(tag + kt, f'{100 * vf[k]["delta"]:+.1f}'); macro(tag + kt + 'Lo', f'{100 * vf[k]["ci"][0]:+.1f}'); macro(tag + kt + 'Hi', f'{100 * vf[k]["ci"][1]:+.1f}')
    if 'Tile-wise re-matching, rotation-only alignment' in rows:
        vf = rows['Tile-wise re-matching, rotation-only alignment']['vs_full']['sr_epe_1m']; macro('ctPlaneGainSRe', f'{-100 * vf["delta"]:.1f}'); macro('ctPlaneGainSReLo', f'{-100 * vf["ci"][1]:.1f}'); macro('ctPlaneGainSReHi', f'{-100 * vf["ci"][0]:.1f}')
        vd = rows['Tile-wise re-matching, rotation-only alignment']['vs_full']['sr_d_1m']; macro('ctPlaneGainSRd', f'{-100 * vd["delta"]:.1f}'); macro('ctPlaneGainSRdLo', f'{-100 * vd["ci"][1]:.1f}'); macro('ctPlaneGainSRdHi', f'{-100 * vd["ci"][0]:.1f}')
    RGm = d['raw_geometry']
    for key, tag in (('Rotation-only alignment, <= 3x3 tiles', 'rgRot'), ('Plane-induced warp, <= 3x3 tiles (ours)', 'rgOurs'), ('Plane-induced warp, 1 tile', 'rgOneTile'), ('Plane-induced warp, 512x288', 'rgLowRes'), ('Coarse stage, zoom pass', 'rgZoom'),
                     ('Plane-induced warp under the true pose, <= 3x3 tiles', 'rgOracle'), ('Plane-induced warp, <= 3x3 tiles (ours), oracle subset', 'rgOursSub'), ('No pose-guided alignment, <= 3x3 tiles, oracle subset', 'rgNoAlign'), ('Rotation-only alignment, <= 3x3 tiles, oracle subset', 'rgRotSub')):
        if key not in RGm: continue
        r = RGm[key]; macro(tag + 'Inl', f'{r.get("inliers_med", float("nan")):,.0f}'.replace(',', '{,}')); macro(tag + 'Dir', f'{r["MAP"]["dir_med"]:.2f}'); macro(tag + 'DirMean', f'{r["MAP"]["dir_mean"]:.1f}')
        if 'resid_px_med' in r: macro(tag + 'Res', f'{r["resid_px_med"]:.0f}')
        macro(tag + 'N', f'{r.get("n", 0):,}'.replace(',', '{,}'))
        macro(tag + 'Solved', pct(r['MAP']['solved'])); macro(tag + 'DirLtTwo', pct(r['MAP']['dir_lt2'])); macro(tag + 'LtHundred', pct(r.get('inl_lt100', float('nan'))))


# ----------------------------------------------------------------------------------------------- planarity
def t_planarity():
    d = J('trackA/planarity.json')
    if d is None: return placeholder('planarity', 'tab:planarity', 'planarity analysis')
    B = []
    for r in d['by_relief']: B.append(f'{r["bin"].replace("%", "\\%").replace(">=", "$\\geq$").replace("<", "$<$")} & {r["n"]} & {r["struct_med"]:.0f} & {r["disp_med"]:.1f} & {pct(r["s1_sr_d"])} & {pct(r["s2_sr_d"])} & {pct(r["s1_sr_epe"])} & {pct(r["s2_sr_epe"])} & {r["s1_psi_mae"]:.2f} & {r["s2_psi_mae"]:.2f} \\\\')
    w('planarity', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Accuracy against scene relief} (' + f'{d["n_valid"]:,}'.replace(',', '{,}') + ' test pairs). Relief: height of the tallest matched structures over the flight height. $S$: that height in metres (median). Res.: median displacement between the imagined and the real view. I / II: coarse / refined stage, camera frame.}\n\\label{tab:planarity}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{@{}lrrrcccccc@{}}\n\\toprule\n'
      '& & $S$ & Res. & \\multicolumn{2}{c}{\\SRd@1\\,m} & \\multicolumn{2}{c}{\\SRe@1\\,m} & \\multicolumn{2}{c}{$\\dpsi$ MAE} \\\\\nRelief & Pairs & (m) & (px) & I & II & I & II & I & II \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    B = []
    for r in d['by_tilt_max']: B.append(f'{r["bin"]}$^\\circ$ & {r["n"]} & {r["disp_med"]:.1f} & {pct(r["s1_sr_d"])} & {pct(r["s2_sr_d"])} & {pct(r["s1_sr_epe"])} & {pct(r["s2_sr_epe"])} \\\\')
    B.append('\\midrule')
    for r in d['by_ground_m']: B.append(f'{r["bin"].replace(">=", "$\\geq$").replace("<", "$<$")}\\,m & {r["n"]} & {r["disp_med"]:.1f} & {pct(r["s1_sr_d"])} & {pct(r["s2_sr_d"])} & {pct(r["s1_sr_epe"])} & {pct(r["s2_sr_epe"])} \\\\')
    B.append('\\midrule')
    for r in d['by_baseline_over_height']: B.append(f'{r["bin"].replace("%", "\\%").replace(">=", "$\\geq$").replace("<", "$<$")} & {r["n"]} & {r["disp_med"]:.1f} & {pct(r["s1_sr_d"])} & {pct(r["s2_sr_d"])} & {pct(r["s1_sr_epe"])} & {pct(r["s2_sr_epe"])} \\\\')
    w('planarity_tilt', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Accuracy against the larger tilt of the pair} (top), \\textbf{the height of the source camera above the ground} (middle) \\textbf{and the baseline-to-height ratio} (bottom). Same pairs and columns as \\cref{tab:planarity}.}\n\\label{tab:planarity_tilt}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}lrrcccc@{}}\n\\toprule\n'
      '& & Res. & \\multicolumn{2}{c}{\\SRd@1\\,m} & \\multicolumn{2}{c}{\\SRe@1\\,m} \\\\\nBin & Pairs & (px) & I & II & I & II \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    D = d['dist']; macro('plGroundMed', f'{D["ground"]["p50"]:.0f}'); macro('plGroundLo', f'{D["ground"]["p5"]:.0f}'); macro('plGroundHi', f'{D["ground"]["p95"]:.0f}'); macro('plStructMed', f'{D["struct"]["p50"]:.0f}'); macro('plStructHi', f'{D["struct"]["p95"]:.0f}')
    macro('plReliefMed', pct(D['relief']['p50'])); macro('plReliefLo', pct(D['relief']['p5'])); macro('plReliefHi', pct(D['relief']['p95'])); macro('plOffTenMed', pct(D['off10']['p50'], 0)); macro('plOffThreeMed', pct(D['off3']['p50'], 0)); macro('plNearMed', pct(D['near2']['p50'], 0))
    macro('plParallaxMed', f'{D["parallax_px"]["p50"]:.0f}'); macro('plParallaxHi', f'{D["parallax_px"]["p95"]:.0f}'); macro('plSlopeMed', f'{D["slope"]["p50"]:.1f}'); macro('plSlopeHi', f'{D["slope"]["p95"]:.1f}')
    macro('plTerrCorr', f'{d["ground_vs_terrain"]["corr"]:.2f}'); macro('plTerrDiff', f'{d["ground_vs_terrain"]["median_abs_diff"]:.1f}'); macro('plSpearEPEi', f'{d["spearman"]["epe"]["s1"]:+.2f}'); macro('plSpearEPEii', f'{d["spearman"]["epe"]["s2"]:+.2f}'); macro('plSpearRes', f'{d["spearman"]["relief_vs_residual_px"]:+.2f}')
    for r in d['gain_by_relief']: tag = {'<5%': 'A', '5-10%': 'B', '10-20%': 'C', '>=20%': 'D'}[r['bin']]; macro(f'plGain{tag}', f'{100 * r["gain_sr_epe"]:.1f}'); macro(f'plGain{tag}Lo', f'{100 * r["ci_epe"][0]:.1f}'); macro(f'plGain{tag}Hi', f'{100 * r["ci_epe"][1]:.1f}')
    macro('plN', f'{d["n_valid"]:,}'.replace(',', '{,}'))
    br = {r['bin']: r for r in d['by_relief']}; mid = [br[k] for k in ('5-10%', '10-15%', '15-20%', '20-30%') if k in br]
    macro('plMidSRdLo', pct(min(r['s2_sr_d'] for r in mid))); macro('plMidSRdHi', pct(max(r['s2_sr_d'] for r in mid))); macro('plMidSReLo', pct(min(r['s2_sr_epe'] for r in mid))); macro('plMidSReHi', pct(max(r['s2_sr_epe'] for r in mid)))
    lo = [br[k] for k in ('<2%', '2-5%') if k in br]; nlo = sum(r['n'] for r in lo); macro('plFlatN', str(nlo)); macro('plFlatSiSRd', pct(sum(r['s1_sr_d'] * r['n'] for r in lo) / nlo)); macro('plFlatSiiSRd', pct(sum(r['s2_sr_d'] * r['n'] for r in lo) / nlo))
    macro('plFlatSiSRe', pct(sum(r['s1_sr_epe'] * r['n'] for r in lo) / nlo)); macro('plFlatSiiSRe', pct(sum(r['s2_sr_epe'] * r['n'] for r in lo) / nlo)); macro('plFlatRes', f'{sum(r["disp_med"] * r["n"] for r in lo) / nlo:.0f}')
    if '>=30%' in br: macro('plTallN', str(br['>=30%']['n'])); macro('plTallSiiSRd', pct(br['>=30%']['s2_sr_d'])); macro('plTallSiiSRe', pct(br['>=30%']['s2_sr_epe'])); macro('plTallSiSRe', pct(br['>=30%']['s1_sr_epe']))
    bt = {r['bin']: r for r in d['by_tilt_max']}; macro('plTiltLoSiiSRd', pct(bt['0-15']['s2_sr_d'])); macro('plTiltHiSiiSRd', pct(bt['35-45']['s2_sr_d'])); macro('plTiltHiSiSRd', pct(bt['35-45']['s1_sr_d'])); macro('plTiltLoSiSRd', pct(bt['0-15']['s1_sr_d']))
    macro('plTiltHiSiiSRe', pct(bt['35-45']['s2_sr_epe'])); macro('plTiltHiSiSRe', pct(bt['35-45']['s1_sr_epe'])); macro('plTiltLoSiiSRe', pct(bt['0-15']['s2_sr_epe']))
    bg = {r['bin']: r for r in d['by_ground_m']}
    if '>=250' in bg: macro('plHighN', str(bg['>=250']['n'])); macro('plHighSiiSRd', pct(bg['>=250']['s2_sr_d'])); macro('plHighSiSRd', pct(bg['>=250']['s1_sr_d']))
    macro('plResMid', f'{np.median([r["disp_med"] for r in mid]):.0f}')
    bb = {r['bin']: r for r in d['by_baseline_over_height']}
    if '<3%' in bb: macro('plShortN', str(bb['<3%']['n'])); macro('plShortSiSRe', pct(bb['<3%']['s1_sr_epe'])); macro('plShortSiiSRe', pct(bb['<3%']['s2_sr_epe'])); macro('plLongSiiSRe', pct(bb['>=12%']['s2_sr_epe']))


if __name__ == '__main__':
    for fn in (t_runtime, t_main, t_inventory, t_controls, t_planarity):
        try: fn()
        except Exception as e: print('FAILED', fn.__name__, repr(e))
    for extra in sorted(os.listdir(os.path.dirname(os.path.abspath(__file__)))):
        if extra.startswith('make_tables_') and extra.endswith('.py'):
            ns = {'J': J, 'w': w, 'placeholder': placeholder, 'pct': pct, 'num': num, 'cipct': cipct, 'cinum': cinum, 'macro': macro, 'R': R, 'T': T, 'np': np, 'json': json, 'os': os, 'math': math}
            try: exec(compile(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), extra)).read(), extra, 'exec'), ns)
            except Exception as e: print('FAILED', extra, repr(e))
    import re, glob
    used = set()
    for f in glob.glob('/root/rasr-pairuav/docs/paper/supp/sec/*.tex') + glob.glob('/root/rasr-pairuav/docs/paper/supp/tables/*.tex'): used |= set(re.findall(r'\\((?:ct|pl|rt|g|f|pri|mlp|base|gain|Si|Sii|n|t|zero|hd|cl|it|un|ds|rl|gb|cf|bl|sv|sg|im|fg|hm|rg|pr|gb|sf|gr|gro|og|uf|sfu|cf|hd)[A-Za-z]+)', open(f).read()))
    BUILTIN = {'ci', 'cite', 'cref', 'Cref', 'caption', 'centering', 'gamma', 'geq', 'footnotesize', 'frac', 'fbox', 'begin', 'best', 'bm', 'big', 'Big', 'cdot', 'cos', 'cap', 'cup', 'bar', 'beta', 'bottomrule', 'ie', 'eg', 'input', 'item', 'in', 'infty', 'int', 'itemize', 'gg', 'ge', 'times', 'top', 'textbf', 'text', 'tilde', 'theta', 'tau', 'toprule', 'to', 'textit', 'texttt', 'tfrac', 'topsep', 'pm', 'psi', 'phi', 'pi', 'par', 'parbox', 'prime', 'phantom', 'paragraph', 'Pr', 'prod', 'propto', 'noindent', 'nabla', 'neq', 'ne', 'ni', 'not', 'nu', 'newcommand', 'main', 'mathbf', 'mathrm', 'mathcal', 'max', 'min', 'midrule', 'multicolumn', 'multirow', 'makecell', 'mu', 'rowcolor', 'right', 'rho', 'rule', 'renewcommand', 'cline', 'cmidrule', 'cdots', 'circ', 'colorbox', 'color', 'checkmark', 'chi', 'cong', 'footnote', 'fboxsep', 'forall', 'flat', 'gets', 'gtrsim', 'tabcolsep', 'tabular', 'table', 'textcolor', 'textsc', 'textsuperscript', 'tiny', 'triangle', 'pagebreak', 'partial', 'perp', 'ptm', 'nolinebreak', 'nonumber', 'normalsize', 'num', 'zeta', 'baselineskip', 'bigskip', 'bf', 'bullet', 'backslash', 'clearpage', 'columnwidth', 'cr', 'fi', 'fill', 'flushleft', 'gdef', 'global', 'rm', 'raggedright', 'raisebox', 'ref', 'relax', 'resizebox', 'rightarrow', 'Rightarrow', 'nobreak', 'nodes', 'newline', 'newpage', 'noalign', 'plus', 'pt', 'pdf', 'prec', 'preceq', 'tt', 'thanks', 'the', 'then', 'title', 'today', 'textwidth', 'textheight', 'tableofcontents', 'tag', 'tan', 'tanh', 'tabularnewline', 'basename', 'bibliography', 'bibliographystyle', 'box', 'break', 'flushright', 'frame', 'framebox', 'ctrl', 'gbox', 'iff', 'it', 'itshape', 'itemsep', 'iint', 'imath', 'indent', 'index', 'inf', 'intertext', 'unskip', 'underline', 'underbrace', 'unitlength', 'unless', 'usepackage', 'usebox', 'dth', 'dpsi', 'dpv', 'dpvh', 'lv', 'SRd', 'SRe', 'SRf', 'wrapd', 'todoauthors', 'bfseries', 'cline', 'dots', 'ds', 'displaystyle', 'rlap', 'clap', 'cfrac', 'blacksquare', 'bot', 'boldsymbol', 'bmod', 'binom', 'bigl', 'bigr', 'Bigl', 'Bigr', 'bigg', 'biggl', 'biggr', 'gcd', 'geqslant', 'ggg', 'grave', 'cl', 'un', 'hd', 'hdots', 'iota', 'itemindent', 'sum', 'sin', 'sqrt', 'small', 'scriptsize', 'section', 'subsection', 'setlength', 'sigma', 'sim', 'simeq', 'star', 'subset', 'top', 'tfrac', 'unit', 'cfbox', 'pagestyle', 'plGain', 'nolimits', 'notin', 'nmid', 'ng', 'rtimes', 'rfloor', 'rceil', 'rangle', 'rbrace', 'rbrack', 'rgroup', 'blank', 'gray', 'bg', 'fg', 'tex', 'footnotemark', 'footnotetext', 'fontsize', 'frenchspacing', 'pitchfork', 'plain', 'pounds', 'printindex', 'protect', 'put', 'prefix', 'zref', 'thead'}
    for k in sorted(used - set(NUM) - BUILTIN):
        if re.fullmatch(r'(ct|pl|rt|g|f|pri|mlp|base|gain|Si|Sii|n|t|zero|hd|cl|it|un|ds|rl|gb|cf|bl|sv|sg|im|fg|hm|rg|pr|gb|sf|gr|gro|og|uf|sfu|cf|hd)[A-Z][A-Za-z]*', k): NUM[k] = '\\textbf{\\color{red}??}'; print('  missing number macro ->', k)
    open('/root/rasr-pairuav/docs/paper/supp/numbers.tex', 'w').write('% generated by supp/make_tables.py\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(NUM.items())))
    print(len(NUM), 'number macros')
