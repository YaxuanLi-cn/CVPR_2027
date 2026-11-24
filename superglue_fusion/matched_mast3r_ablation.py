#!/usr/bin/env python3
"""Test whether MASt3R features pooled at SuperGlue geometric inliers help pose prediction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path('/root/rasr-pairuav/outputs/pairuav5d_60k')
SG = ROOT / 'superglue'
DATA = Path('/root/pairUAV-5D-temp')
MAST3R_RUNTIME = Path('/root/mast3r_runtime')
MODEL_PATH = MAST3R_RUNTIME / 'models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
IMAGE_W, IMAGE_H = 3686.0, 2073.0


def load_runtime():
    for path in (MAST3R_RUNTIME, MAST3R_RUNTIME / 'mast3r', MAST3R_RUNTIME / 'mast3r/dust3r'):
        sys.path.insert(0, str(path))
    try:
        import torch
        from torch.library import Library
        try:
            torch._C._dispatch_find_schema_or_throw('torchvision::nms', '')
        except Exception:
            Library('torchvision', 'DEF').define(
                'nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor'
            )
    except Exception:
        import torch
    from run_mast3r_probe import prepare_image
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.device import collate_with_cat, to_cpu
    from mast3r.model import AsymmetricMASt3R
    return torch, prepare_image, make_pairs, collate_with_cat, to_cpu, AsymmetricMASt3R


def geometric_inliers(p0: np.ndarray, p1: np.ndarray, conf: np.ndarray) -> np.ndarray:
    from pose_from_matches import K0
    keep = np.isfinite(conf) & (conf >= 0.2)
    indices = np.flatnonzero(keep)
    if len(indices) < 8:
        return np.zeros(len(p0), dtype=bool)
    q0 = np.c_[p0[keep], np.ones(len(indices))] @ np.linalg.inv(K0).T
    q1 = np.c_[p1[keep], np.ones(len(indices))] @ np.linalg.inv(K0).T
    try:
        _essential, mask = cv2.findEssentialMat(
            q0[:, :2].copy(), q1[:, :2].copy(), np.eye(3), method=cv2.USAC_MAGSAC,
            prob=0.99999, threshold=3.0 / K0[0, 0],
        )
    except cv2.error:
        return np.zeros(len(p0), dtype=bool)
    out = np.zeros(len(p0), dtype=bool)
    if mask is not None and mask.size == len(indices):
        out[indices] = mask.ravel().astype(bool)
    return out


def sample_map(array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h, w = array.shape[:2]
    x = np.clip(np.rint(xy[:, 0] * w / IMAGE_W).astype(int), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1] * h / IMAGE_H).astype(int), 0, h - 1)
    return array[y, x]


def summarize(values: np.ndarray, prefix: str, row: dict[str, float]) -> None:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if not len(values):
        for i in range(values.shape[1]):
            row[f'{prefix}_c{i}_mean'] = 0.0
            row[f'{prefix}_c{i}_std'] = 0.0
        return
    for i in range(values.shape[1]):
        finite = values[:, i][np.isfinite(values[:, i])]
        row[f'{prefix}_c{i}_mean'] = float(finite.mean()) if len(finite) else 0.0
        row[f'{prefix}_c{i}_std'] = float(finite.std()) if len(finite) else 0.0


def extract_pair_features(row, model, runtime, device) -> dict[str, float]:
    torch, prepare_image, make_pairs, collate_with_cat, to_cpu, _ = runtime
    idx = int(row.manifest_index)
    match_path = SG / 'matches_rot' / f'{idx}.npz'
    if not match_path.is_file():
        raise FileNotFoundError(match_path)
    matches = np.load(match_path)
    p0, p1 = matches['p0'].astype(np.float32), matches['p1'].astype(np.float32)
    conf = matches['conf'].astype(np.float32)
    inliers = geometric_inliers(p0, p1, conf)
    p0, p1 = p0[inliers], p1[inliers]

    image_a, image_b = DATA / row.image_a, DATA / row.image_b
    views = [
        prepare_image(str(image_a), idx=0, size=512, patch_size=16),
        prepare_image(str(image_b), idx=1, size=512, patch_size=16),
    ]
    pairs = list(make_pairs(views, scene_graph='complete', symmetrize=True))
    view1, view2 = collate_with_cat(pairs)
    ignore = {'depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'}
    for view in (view1, view2):
        for name, value in list(view.items()):
            if name not in ignore and isinstance(value, torch.Tensor):
                view[name] = value.to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                               enabled=device.type == 'cuda'):
        pred1, pred2 = model(view1, view2)
    output = to_cpu({'view1': view1, 'view2': view2, 'pred1': pred1, 'pred2': pred2})
    lookup = {
        (int(output['view1']['idx'][i]), int(output['view2']['idx'][i])): i
        for i in range(len(output['view1']['idx']))
    }
    ab = lookup[(0, 1)]
    desc_a = np.asarray(output['pred1']['desc'][ab], dtype=np.float32)
    desc_b = np.asarray(output['pred2']['desc'][ab], dtype=np.float32)
    conf_a = np.asarray(output['pred1']['desc_conf'][ab], dtype=np.float32)
    conf_b = np.asarray(output['pred2']['desc_conf'][ab], dtype=np.float32)
    pts_a = np.asarray(output['pred1']['pts3d'][ab], dtype=np.float32)
    pts_b = np.asarray(output['pred2']['pts3d_in_other_view'][ab], dtype=np.float32)

    da, db = sample_map(desc_a, p0), sample_map(desc_b, p1)
    ca, cb = sample_map(conf_a, p0), sample_map(conf_b, p1)
    pa, pb = sample_map(pts_a, p0), sample_map(pts_b, p1)
    feats: dict[str, float] = {'mf_num_inliers': float(len(p0))}
    for name, values in (
        ('desc_a', da), ('desc_b', db), ('desc_absdiff', np.abs(da - db)),
        ('point_a', pa), ('point_b', pb), ('point_absdiff', np.abs(pa - pb)),
    ):
        summarize(values, f'mf_{name}', feats)
    cosine = np.sum(da * db, axis=1) / (np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1) + 1e-8)
    feats['mf_desc_cos_mean'] = float(np.mean(cosine)) if len(cosine) else 0.0
    feats['mf_desc_cos_std'] = float(np.std(cosine)) if len(cosine) else 0.0
    for name, values in (
        ('conf_a', ca), ('conf_b', cb), ('point_conf_a', sample_map(np.asarray(output['pred1']['conf'][ab]), p0)),
        ('point_conf_b', sample_map(np.asarray(output['pred2']['conf'][ab]), p1)),
        ('point_norm_a', np.linalg.norm(pa, axis=1)), ('point_norm_b', np.linalg.norm(pb, axis=1)),
    ):
        feats[f'mf_{name}_mean'] = float(np.mean(values)) if len(values) else 0.0
        feats[f'mf_{name}_std'] = float(np.std(values)) if len(values) else 0.0
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-limit', type=int, default=1000)
    parser.add_argument('--test-limit', type=int, default=500)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=SG / 'matched_mast3r_ablation.pkl')
    args = parser.parse_args()

    pairs = pd.read_pickle(SG / 'pairs.pkl')
    rng = np.random.default_rng(args.seed)
    selected = []
    for split, limit in (('train', args.train_limit), ('test', args.test_limit)):
        available = pairs[pairs.split == split]
        chosen = rng.choice(available.index.to_numpy(), min(limit, len(available)), replace=False)
        selected.append(available.loc[chosen])
    sample = pd.concat(selected).sort_values('manifest_index').reset_index(drop=True)

    runtime = load_runtime()
    torch = runtime[0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = runtime[-1].from_pretrained(str(MODEL_PATH)).to(device).eval()
    model.requires_grad_(False)
    rows = []
    for number, row in enumerate(sample.itertuples(index=False), start=1):
        result = extract_pair_features(row, model, runtime, device)
        result['manifest_index'] = int(row.manifest_index)
        rows.append(result)
        if number % 50 == 0 or number == len(sample):
            print(f'{number}/{len(sample)} rows extracted', flush=True)
    features = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(args.output)

    from fusion_lib import add_targets, load_mast3r, oof_fit, report
    mast3r = load_mast3r()
    geom = pd.read_pickle(SG / 'geom_rot_full.pkl')
    stage0 = pd.read_pickle(SG / 'stage0.pkl')
    mlp = pd.read_pickle(SG / 'stage0_mlp.pkl')
    data = add_targets(pairs).merge(mast3r.drop(columns='split'), on='manifest_index').merge(
        geom, on='manifest_index', how='left'
    )
    data = data.merge(stage0, on='manifest_index', suffixes=('', '_stage0'))
    data = data.merge(mlp, on='manifest_index', suffixes=('', '_mlp')).merge(features, on='manifest_index')
    train, test = data.split.eq('train').to_numpy(), data.split.eq('test').to_numpy()
    metadata = {'manifest_index', 'split', 'tour', 'pair_id', 'image_a', 'image_b'}
    base_cols = [c for c in mast3r.columns if c not in metadata]
    base_cols += [c for c in geom.columns if c.startswith(('sg_', 'alt_')) and c in data.columns]
    base_cols += [c for c in stage0.columns if c.startswith('m_') and c in data.columns]
    base_cols += [c for c in mlp.columns if c.startswith('n_') and c in data.columns]
    base_cols = list(dict.fromkeys(base_cols))
    match_cols = [c for c in features.columns if c.startswith('mf_')]
    X_base = data[base_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    X_plus = data[base_cols + match_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    groups = data.loc[train, 'tour'].astype(str).to_numpy()
    targets = {
        'x': ('lx', False), 'y': ('ly', False), 'z': ('z', False),
        'tilt': ('tilt', False), 'distance': ('logd', False),
        'heading_sin': ('hsin', False), 'heading_cos': ('hcos', False),
    }
    results = {'sample': {'train': int(train.sum()), 'test': int(test.sum()), 'seed': args.seed}, 'features': len(match_cols)}
    for name, (target, _) in targets.items():
        y = data[target].to_numpy(np.float64)
        pred_base = oof_fit(X_base[train], y[train], groups, X_base[test])[1]
        pred_plus = oof_fit(X_plus[train], y[train], groups, X_plus[test])[1]
        if name == 'heading_sin':
            results['heading_sin_base'] = pred_base
            results['heading_sin_plus'] = pred_plus
            continue
        if name == 'heading_cos':
            sin_base, cos_base = results.pop('heading_sin_base'), results.pop('heading_sin_plus')
            base = np.rad2deg(np.arctan2(sin_base, pred_base))
            plus = np.rad2deg(np.arctan2(cos_base, pred_plus))
            truth = data.loc[test, 'heading'].to_numpy()
            wrap = lambda d: (d + 180.0) % 360.0 - 180.0
            results['heading_mae_base'] = float(np.mean(np.abs(wrap(base - truth))))
            results['heading_mae_plus'] = float(np.mean(np.abs(wrap(plus - truth))))
            continue
        if name == 'distance':
            pred_base, pred_plus = np.exp(pred_base), np.exp(pred_plus)
            y = data.loc[test, 'dist'].to_numpy()
        else:
            y = data.loc[test, target].to_numpy()
        results[f'{name}_mae_base'] = float(np.mean(np.abs(pred_base - y)))
        results[f'{name}_mae_plus'] = float(np.mean(np.abs(pred_plus - y)))
    print(pd.Series(results).to_string(), flush=True)
    result_path = args.output.with_suffix('.json')
    import json
    result_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'saved {args.output} and {result_path}', flush=True)


if __name__ == '__main__':
    main()
