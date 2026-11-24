#!/usr/bin/env python3
"""Track B: one-screen digest of the headline numbers (read from the JSON files) for the written summary."""
import json, os
O = '/root/rasr-pairuav/outputs/supp/trackB'; f1 = lambda v: f'{100 * v:.1f}'
P = json.load(open(f'{O}/pairuav.json')); R = json.load(open(f'{O}/real.json'))
def line(r): c = r['ci']; return f"{r['method'][:46]:46s} | {r['variant'][:9]:9s} | {r['scale'][:13]:13s} | n {r['n']:4d} | fail {r['fail']:4.1f} | d {r['d_mae']:5.2f} | SRd {f1(r['sr_d_1m'])} [{f1(c['sr_d_1m'][0])},{f1(c['sr_d_1m'][1])}] | EPEmed {r['epe_med']:5.2f} | SRepe {f1(r['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])},{f1(c['sr_epe_1m'][1])}] | th {r['theta_mae']:.2f}/{r['theta_med']:.3f} | psi {r['psi_mae']:.2f}/{r['psi_med']:.3f} | dir {r['dir_med']:.1f}"
print('=== PairUAV'); keep = ('height_p75ab', 'height_pointmap')
for r in P['rows']:
    if r['scale'] in keep or r.get('matcher') in ('mlp', 'dreamnav', 'zero', 'dreamnav_subset') or (r.get('matcher') in ('sift', 'roma') and r['scale'] in ('learned', 'blend')) or (r.get('matcher') == 'mast3r_pnp' and r['scale'] == 'native'): print(line(r))
print('thr:', {k: v['chosen'] for k, v in P['threshold_selection'].items()}); print('timing:', {m: (round(t.get('plain_s', 0) or 0, 2), round(t.get('tta_s', 0) or 0, 2)) for m, t in P['timing'].items()})
print('sens:', {k: {t: (f1(v['sr_d_1m']), f1(v['sr_epe_1m'])) for t, v in d.items()} for k, d in P['threshold_sensitivity_test'].items() if k in ('sift|plain', 'roma|tta', 'dedode|tta')})
for which, S in R['sets'].items():
    print(f'=== real {which} n={S["n"]}')
    for s in S['rows']:
        if s['scale'] == 'height60_p50a': continue
        c = s['ci']; p = s.get('paired_minus_stage2', {})
        print(f"{s['method'][:46]:46s} | {s['variant'][:9]:9s} | {s['scale'][:15]:15s} | n {s['n']:3d} | fail {s['fail']:4.1f} | pos {s['epe_med']:5.2f} [{c['epe_med'][0]:.2f},{c['epe_med'][1]:.2f}] | SR1 {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])},{f1(c['sr_d_1m'][1])}] | SR2 {f1(s['sr_d_2m'])} | SRepe1 {f1(s['sr_epe_1m'])} | th {s['theta_mae']:.2f}/{s['theta_med']:.2f} | psi {s['psi_mae']:.2f} | dir {s['dir_med']:.1f} | dSR1 {p.get('sr_d_1m_points', {}).get('delta', 0):+.1f} [{p.get('sr_d_1m_points', {}).get('ci', [0, 0])[0]:+.1f},{p.get('sr_d_1m_points', {}).get('ci', [0, 0])[1]:+.1f}]")
for fn in ('sift_full.json', 'unscreened.json', 'paired_sift.json', 'paired_roma_tta.json', 'paired_dedode_tta.json'):
    if not os.path.exists(f'{O}/{fn}'): continue
    J = json.load(open(f'{O}/{fn}')); print('===', fn)
    if fn == 'sift_full.json':
        print('n', J['n'], 'of', J['n_full'], 'fail', round(J['fail_pct'], 2), 'H', round(J['H'], 1))
        for k, s in J['rows'].items(): c = s['ci']; print(f"  {k[:74]:74s} | x {s['x_mae']:.2f} y {s['y_mae']:.2f} z {s['z_mae']:.2f} d {s['d_mae']:.2f} th {s['theta_mae']:.3f} psi {s['psi_mae']:.3f} | SRd {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])},{f1(c['sr_d_1m'][1])}] | EPE {s['epe_mae']:.2f}/{s['epe_med']:.2f} | SRepe {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])},{f1(c['sr_epe_1m'][1])}] | 5dof {f1(s['sr_5dof_1m_1deg'])} | auc {f1(s['auc_5'])}/{f1(s['auc_10'])}/{f1(s['auc_20'])}")
        for nm, r in J['paired_baseline_minus_dreamnav_camera_frame'].items(): print('  paired vs', nm, {k: (round(v['delta'], 2), [round(x, 2) for x in v['ci']]) for k, v in r.items()})
        print('  joint', {k: {a: (round(100 * b, 1) if not isinstance(b, list) else [round(100 * x, 1) for x in b]) for a, b in v.items()} for k, v in J['joint_success_vs_stage2'].items()})
        for r in J['bins']['SIFT inliers recorded by the dataset screen']: print('  inl', r['bin'], r['n'], 'base', [round(100 * r['baseline'][k], 1) for k in ('sr_d_1m', 'sr_epe_1m')], 's1', [round(100 * r['stage1'][k], 1) for k in ('sr_d_1m', 'sr_epe_1m')], 's2', [round(100 * r['stage2'][k], 1) for k in ('sr_d_1m', 'sr_epe_1m')], 'd', round(r['stage2_minus_baseline_sr_epe_points']['delta'], 1), [round(x, 1) for x in r['stage2_minus_baseline_sr_epe_points']['ci']])
    elif fn == 'unscreened.json':
        print('n', J['n_pairs'], 'dreamnav', J['n_dreamnav'], 'inside', f1(J['share_inside']), 'pass', f1(J['share_screen_pass']), 'both', f1(J['share_inside_and_pass']))
        for sname, d in J['sets_common'].items():
            for nm, s in d.items(): c = s['ci']; print(f"  {sname[:40]:40s} | {nm[:28]:28s} | n {s['n']:4d} | fail {s['fail']:4.1f} | SRd {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])},{f1(c['sr_d_1m'][1])}] | EPEmed {s['epe_med']:.2f} | SRepe {f1(s['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])},{f1(c['sr_epe_1m'][1])}] | @2m {f1(s['sr_epe_2m'])}")
        for sname, d in J['paired'].items():
            for nm, r in d.items(): print(f"  paired {sname[:40]:40s} {nm}: dSRd {r['sr_d_points']['delta']:+.1f} [{r['sr_d_points']['ci'][0]:+.1f},{r['sr_d_points']['ci'][1]:+.1f}] dSRepe {r['sr_epe1_points']['delta']:+.1f} [{r['sr_epe1_points']['ci'][0]:+.1f},{r['sr_epe1_points']['ci'][1]:+.1f}]")
        print('  joint', {k: {a: round(100 * b, 1) for a, b in v.items() if a != 'n'} for k, v in J['joint'].items()})
    else:
        print(' ', J['baseline'], 'n', J['n']); print('  paired', {nm: {k: (round(v['delta'], 2), [round(x, 2) for x in v['ci']]) for k, v in r.items()} for nm, r in J['paired'].items()}); print('  joint', {k: round(100 * v, 1) for k, v in J['joint_epe_1m_vs_stage2'].items()}); print('  hybrids', {k: (round(v['sr_d'], 1), round(v['sr_epe1'], 1)) for k, v in J['hybrids'].items()})
