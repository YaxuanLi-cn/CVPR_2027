#!/usr/bin/env python3
"""Plot fine-grained distributions for one five-tour selected-pair group."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AXES = {
    'x_m': (-15, 15), 'y_m': (-15, 15), 'z_m': (-33, 33),
    'heading_deg': (-180, 180), 'tilt_deg': (-45, 45),
}

def load_records(root: Path, group: str):
    lo, hi = map(int, (group[:4], group[5:]))
    records = []
    per_tour = {}
    for i in range(lo, hi + 1):
        tour = f'{i:04d}'
        data = json.loads((root / f'{tour}_group_{group}.json').read_text())
        rows = data['records']
        per_tour[tour] = rows
        records.extend(rows)
    return records, per_tour

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--json-root', type=Path, default=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'))
    ap.add_argument('--bins', type=int, default=10)
    a = ap.parse_args()
    records, per_tour = load_records(a.json_root, a.group)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = plt.get_cmap('tab10').colors
    for ax, (key, (lo, hi)) in zip(axes.flat, AXES.items()):
        edges = np.linspace(lo, hi, a.bins + 1)
        bottom = np.zeros(a.bins, dtype=int)
        for idx, (tour, rows) in enumerate(per_tour.items()):
            values = [r['delta'][key] for r in rows]
            hist, _ = np.histogram(values, bins=edges)
            ax.bar((edges[:-1] + edges[1:]) / 2, hist, width=np.diff(edges) * .9,
                   bottom=bottom, label=tour, color=colors[idx])
            bottom += hist
        ax.axhline(len(records) / a.bins, color='black', linestyle='--', linewidth=1,
                   label='equal target' if key == 'x_m' else None)
        ax.set_title(f'{key}: {a.bins} equal-width bins')
        ax.set_xlabel(f'{lo} to {hi}')
        ax.set_ylabel('pair count')
        ax.grid(axis='y', alpha=.25)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='upper left', title='tour contribution')
    axes.flat[-1].text(0, .45,
        f'{a.group}: {len(records)} selected unordered pairs\n'
        f'{len(per_tour)} tours, {a.bins} bins per axis\n'
        'Dashed line = equal-count target\n'
        'This plot exposes within-5-bin concentration.',
        transform=axes.flat[-1].transAxes, fontsize=12, va='center')
    fig.suptitle(f'Fine-grained selected-pair distributions: group {a.group}', fontsize=18)
    fig.tight_layout()
    out = a.json_root / f'group_{a.group}_fine_{a.bins}bins.png'
    fig.savefig(out, dpi=180)
    plt.close(fig)

    tilt = np.array([r['delta']['tilt_deg'] for r in records])
    summary = {
        'group': a.group, 'pairs': len(records), 'bins': a.bins,
        'tilt_histogram': np.histogram(tilt, bins=np.linspace(-45, 45, a.bins + 1))[0].tolist(),
        'tilt_mean': float(tilt.mean()), 'tilt_median': float(np.median(tilt)),
        'tilt_min': float(tilt.min()), 'tilt_max': float(tilt.max()),
    }
    (a.json_root / f'group_{a.group}_fine_{a.bins}bins.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(out)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
