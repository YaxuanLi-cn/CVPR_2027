# Classical servoing table and site-familiarity table (executed by make_tables.py)
d = J('trackA/servo.json')
if d is None: placeholder('servo', 'tab:servo', 'classical visual servoing')
else:
    R = d['rows']; B = []
    for k, lab in (('ibvs', 'IBVS, gain 0.5'), ('ibvs1', 'IBVS, gain 1'), ('geo', 'Direct geometric servo'), ('s1', 'Ours, coarse stage'), ('s2', 'Ours, refined stage'), ('s2p', 'Ours, refined + short-range')):
        r = R[k]; g = lambda key: pct(r[key]) if key in r else '--'
        B.append(f'{lab} & {r["budget"]} & {g("sr_after_1")} & {g("sr_after_2")} & {g("sr_after_5")} & {pct(r["final_sr"])} & {r["final_med"]:.2f} & {pct(r["excursion_gt50"])} & {r["stop_tau1"]["steps"]:.1f} & {pct(r["stop_tau1"]["sr"])} \\\\')
        if k == 'geo': B.append('\\midrule')
    w('servo', '\\begin{table}[t]\n\\centering\n\\caption{\\textbf{Classical visual servoing in the simulator of Table~5} (same 160 episodes and actuation noise, one noise draw). SR: true position within 1\\,m of the goal, after 1, 2, 5 steps and at the end of the step budget $T$. ``$>$50\\,m\'\': episodes that move more than 50\\,m away from the goal during the flight. Last two columns: steps flown and SR when the policy stops once its own estimate of the remaining distance is below 1\\,m.}\n\\label{tab:servo}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{@{}lccccccccc@{}}\n\\toprule\n'
      '& & \\multicolumn{4}{c}{SR (\\%) after} & Err. & $>$50\\,m & \\multicolumn{2}{c}{Stop at 1\\,m} \\\\\nPolicy & $T$ & 1 & 2 & 5 & $T$ & (m) & (\\%) & steps & SR \\\\\n\\midrule\n' + '\n'.join(B) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}')
    for k, tag in (('ibvs', 'svIbvs'), ('ibvs1', 'svIbvsOne'), ('geo', 'svGeo'), ('s1', 'svSi'), ('s2', 'svSii'), ('s2p', 'svSiip')):
        r = R[k]; macro(f'{tag}Final', pct(r['final_sr'])); macro(f'{tag}Exc', pct(r['excursion_gt50'])); macro(f'{tag}StopSR', pct(r['stop_tau1']['sr'])); macro(f'{tag}StopSteps', f'{r["stop_tau1"]["steps"]:.1f}'); macro(f'{tag}Med', f'{r["final_med"]:.2f}')
        for j in (1, 2, 5):
            if f'sr_after_{j}' in r: macro(f'{tag}After{["", "One", "Two", "", "", "Five"][j]}', pct(r[f'sr_after_{j}']))
    macro('svHprior', f'{d["flight_height_prior"]:.0f}')
d = J('trackA/site_geometry.json')
if d is not None:
    for r in d['rows']:
        tag = {'0': 'sgZero', '1-5': 'sgFew', '6-20': 'sgSome', '>20': 'sgMany'}[r['bin']]
        macro(f'{tag}N', str(r['buildings'])); macro(f'{tag}SRe', pct(r['s2']['sr_epe'])); macro(f'{tag}SReLo', pct(r['s2']['sr_epe_ci'][0])); macro(f'{tag}SReHi', pct(r['s2']['sr_epe_ci'][1])); macro(f'{tag}SRd', pct(r['s2']['sr_d']))
    macro('sgRho', f'{d["spearman_building_success_vs_site_n_train"]["rho"]:+.2f}')

# mean number of tiles of the imagination pass over the test pairs (archived geometry)
try:
    import pandas as pd
    _G = pd.read_pickle('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/stage2/geom_imag.pkl'); _T = pd.read_pickle('/root/rasr-pairuav/outputs/supp/common/T.pkl')
    _g = _G.set_index('manifest_index').reindex(_T.manifest_index); macro('imTilesMean', f'{_g.im_ntiles.mean():.1f}'); macro('imInlMed', f'{_g.im_inl.median():.0f}'); macro('imResMed', f'{_g.im_disp_med.median():.1f}')
except Exception as e: print('tiles macro failed', e)

# example pairs of the qualitative imagination figure
d = J('trackA/fig_imagination.json')
if d is not None:
    for r, tag in zip(d, ('fgLow', 'fgHigh')):
        macro(f'{tag}Relief', f'{100 * r["relief"]:.0f}'); macro(f'{tag}Struct', f'{r["struct"]:.0f}'); macro(f'{tag}Res', f'{r["resid_px"]:.0f}'); macro(f'{tag}Inl', f'{r["inliers"]:,.0f}'.replace(',', '{,}')); macro(f'{tag}Dist', f'{r["dist"]:.0f}'); macro(f'{tag}Dz', f'{abs(r["dz"]):.0f}')
