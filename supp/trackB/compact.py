#!/usr/bin/env python3
"""Track B: the two compact tables of the final summary (one line per method: its rotation-handling variant with its best non-oracle scale)."""
import json, os
OUT = '/root/rasr-pairuav/outputs/supp/trackB'; P = json.load(open(f'{OUT}/pairuav.json')); f1 = lambda v: f'{100 * v:.1f}'; rows = P['rows']; tm = P['timing']
order = ['sift', 'sgplain', 'sgrot', 'splg', 'aliked', 'loftr', 'dedode', 'roma', 'mast3r', 'mast3r_pnp', 'dust3r_pnp']
L = ['| Method (PairUAV, camera frame) | variant | scale | n | fail % | d MAE | SR_d@1m [95% CI] | EPE med. | SR_EPE@1m [95% CI] | e_theta MAE / med. | e_psi MAE / med. | dir. med. |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
def line(r): c = r['ci']; return f"| {r['method']} | {r['variant']} | {r['scale']} | {r['n']} | {r['fail']:.1f} | {r['d_mae']:.2f} | {f1(r['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {r['epe_med']:.2f} | {f1(r['sr_epe_1m'])} [{f1(c['sr_epe_1m'][0])}, {f1(c['sr_epe_1m'][1])}] | {r['theta_mae']:.2f} / {r['theta_med']:.3f} | {r['psi_mae']:.2f} / {r['psi_med']:.3f} | {r['dir_med']:.1f} |"
for m in order:
    rs = [r for r in rows if r.get('matcher') == m]
    if not rs: continue
    for v in dict.fromkeys(r['variant'] for r in rs):
        if v == 'plain' and any(r['variant'] == 'rot. TTA' for r in rs):   # plain variant of a rotation-sensitive matcher: one line, flight-height scale
            L += [line(r) for r in rs if r['variant'] == v and r['scale'] in ('height_p75ab', 'height_pointmap')]; continue
        L += [line(r) for r in rs if r['variant'] == v and r['scale'] in ('height_p75ab', 'height_pointmap', 'learned')]
        if m == 'mast3r_pnp': L += [line(r) for r in rs if r['variant'] == v and r['scale'] == 'native']
L += [line(r) for r in rows if r.get('matcher') in ('mlp', 'dreamnav', 'zero', 'dreamnav_subset')]
print('\n'.join(L))
if os.path.exists(f'{OUT}/real.json'):
    R = json.load(open(f'{OUT}/real.json'))
    for which, S in R['sets'].items():
        print(f"\n| Method (real photographs, {which}, n={S['n']}) | variant | scale | n | fail % | Pos. med. [95% CI] | SR@1m range [95% CI] | SR@2m range | SR_EPE@1m | e_theta MAE / med. | e_psi MAE | dir. med. |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in S['rows']:
            if s['scale'] in ('height60_p50a',) or (s['variant'] == 'plain' and s.get('matcher') in ('splg', 'aliked', 'loftr', 'roma', 'mast3r', 'sgplain') and s['scale'] != 'height60_p75ab'): continue
            c = s['ci']; print(f"| {s['method']} | {s['variant']} | {s['scale']} | {s['n']} | {s['fail']:.1f} | {s['epe_med']:.2f} [{c['epe_med'][0]:.2f}, {c['epe_med'][1]:.2f}] | {f1(s['sr_d_1m'])} [{f1(c['sr_d_1m'][0])}, {f1(c['sr_d_1m'][1])}] | {f1(s['sr_d_2m'])} | {f1(s['sr_epe_1m'])} | {s['theta_mae']:.2f} / {s['theta_med']:.3f} | {s['psi_mae']:.2f} | {s['dir_med']:.1f} |")
