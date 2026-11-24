#!/usr/bin/env python3
"""Table metrics of docs/paper/experiment.tex for a predictions file, on the common 6,364-pair test set.

Same definitions as rasr-pairuav/analysis/extended_metrics.py (errs / auc): the pose error is
the rotation error max(heading error, tilt error) and the AUC is the area under the
recall-vs-threshold curve up to the threshold. Rows are matched to docs/paper/figures/T.pkl by label.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

T = pd.read_pickle("/root/rasr-pairuav/docs/paper/figures/T.pkl")


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def auc(e, thr):
    e = np.sort(e); r = (np.arange(len(e)) + 1) / len(e); i = np.searchsorted(e, thr)
    x = np.r_[0.0, e[:i], thr]; y = np.r_[0.0, r[:i], r[i - 1] if i else 0.0]
    return float(np.trapezoid(y, x) / thr)


def table_row(path, pred_key="pred_xyz_tilt_heading", true_key="true_xyz_tilt_heading"):
    d = np.load(path); p, t = d[pred_key], d[true_key]
    dist, ix = cKDTree(t[:, :4]).query(T[["x", "y", "z", "tilt"]].values)
    assert dist.max() < 1e-2 and len(set(ix)) == len(ix)
    p = p[ix]; g = T[["x", "y", "z", "tilt", "heading"]].values
    e = np.abs(p[:, :3] - g[:, :3]); pn = np.linalg.norm(p[:, :3], axis=1); gn = np.linalg.norm(g[:, :3], axis=1)
    distance = np.abs(pn - gn); tilt = np.abs(p[:, 3] - g[:, 3]); heading = np.abs(wrap(p[:, 4] - g[:, 4]))
    cos = (p[:, :3] * g[:, :3]).sum(1) / (pn.clip(1e-9) * gn.clip(1e-9)); tdir = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
    pose = np.maximum(heading, tilt)  # rotation only
    return {"rows": len(T), "x": e[:, 0].mean(), "y": e[:, 1].mean(), "z": e[:, 2].mean(), "distance": distance.mean(),
            "heading": heading.mean(), "tilt": tilt.mean(), "SR@1m": 100 * (distance <= 1.0).mean(),
            "pose_auc@5": 100 * auc(pose, 5), "pose_auc@10": 100 * auc(pose, 10), "pose_auc@20": 100 * auc(pose, 20),
            "transl_dir_median_deg": float(np.median(tdir)), "pos3d_mean_m": float(np.linalg.norm(e, axis=1).mean())}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        row = {k: (round(float(v), 4) if not isinstance(v, int) else v) for k, v in table_row(path).items()}
        print(path); print(json.dumps(row))
