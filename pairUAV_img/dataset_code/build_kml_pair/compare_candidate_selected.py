#!/usr/bin/env python3
"""Compare visual-audit candidate supply with final selected group pairs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AXES = {'x_m': (-15, 15), 'y_m': (-15, 15), 'z_m': (-33, 33),
        'heading_deg': (-180, 180), 'tilt_deg': (-45, 45)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--json-root', type=Path, default=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'))
    a = ap.parse_args()
    lo, hi = int(a.group[:4]), int(a.group[5:])
    candidate = {k: None for k in AXES}
    selected = {k: [] for k in AXES}
    accepted = 0
    for i in range(lo, hi + 1):
        t = f'{i:04d}'
        audit = json.loads((a.json_root / t / 'visual_audit_redesigned/summary.json').read_text())
        accepted += int(audit['accepted_pairs'])
        for k in AXES:
            values = np.asarray(audit['available_oriented_histograms'][k], dtype=int)
            candidate[k] = values.copy() if candidate[k] is None else candidate[k] + values
        pair = json.loads((a.json_root / f'{t}_group_{a.group}.json').read_text())
        for row in pair['records']:
            for k in AXES:
                selected[k].append(row['delta'][k])
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for col, (k, (low, high)) in enumerate(AXES.items()):
        c_bins = len(candidate[k])
        c_edges = np.linspace(low, high, c_bins + 1)
        c_centers = (c_edges[:-1] + c_edges[1:]) / 2
        c_width = np.diff(c_edges) * .72
        axes[0, col].bar(c_centers, candidate[k], width=c_width,
                         label='candidate orientations (both directions)', color='#8c564b')
        axes[0, col].set_title(f'{k}: candidate supply')
        axes[0, col].set_xlabel(f'{low} to {high}')
        axes[0, col].set_ylabel('count')
        axes[0, col].grid(axis='y', alpha=.25)
        s_edges = np.linspace(low, high, 11)
        s_centers = (s_edges[:-1] + s_edges[1:]) / 2
        s_width = np.diff(s_edges) * .72
        hist, _ = np.histogram(selected[k], bins=s_edges)
        axes[1, col].bar(s_centers, hist, width=s_width,
                         label='selected unordered pairs', color='#1f77b4')
        target = len(selected[k]) / 10
        axes[1, col].axhline(target, color='black', linestyle='--',
                             label=f'10-bin target={target:g}')
        axes[1, col].set_title(f'{k}: final selection')
        axes[1, col].set_xlabel(f'{low} to {high}')
        axes[1, col].set_ylabel('count')
        axes[1, col].grid(axis='y', alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f'Candidate supply vs final selected pairs: group {a.group}\n'
                 f'candidate accepted edges={accepted}; selected pairs={len(selected["x_m"])}', fontsize=16)
    fig.tight_layout()
    out = a.json_root / f'group_{a.group}_candidate_vs_selected.png'
    fig.savefig(out, dpi=180)
    print(out)

if __name__ == '__main__':
    main()
