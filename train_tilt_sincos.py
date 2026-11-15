#!/usr/bin/env python3
"""Fit a sin/cos tilt head and compare it on the fixed 60k PairUAV split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models_src.train_head import load_config, load_training_arrays, standardize_features


class TiltSinCosHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_test(path: Path, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_valid", "true").lower() not in {"true", "1", "yes"}:
                continue
            try:
                x = [float(row[name]) for name in feature_names]
                y = float(row["tilt_num"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(x).all() and np.isfinite(y):
                xs.append(x)
                ys.append(y)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int((config.get("heads") or {}).get("distance_head_a", {}).get("seed", 51))
    np.random.seed(seed)
    torch.manual_seed(seed)
    target = "tilt_num"
    x_raw, y_deg, feature_names = load_training_arrays(
        args.train_features,
        target_column=target,
        metadata_columns=[str(value) for value in config["metadata_columns"]],
        limit=None,
    )
    x, mean, std = standardize_features(x_raw)
    radians = np.deg2rad(y_deg)
    y = np.stack([np.sin(radians), np.cos(radians)], axis=1).astype(np.float32)

    device = torch.device(args.device)
    batch_size = min(int(config.get("batch_size", 256)), len(x))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = TiltSinCosHead(
        x.shape[1], int(config.get("hidden_dim", 256)), float(config.get("dropout", 0.1))
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion = nn.MSELoss()
    max_steps = int(config.get("max_epochs", 30)) * len(loader)
    losses: list[float] = []
    model.train()
    while len(losses) < max_steps:
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(bx.to(device))
            loss = criterion(pred, by.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if len(losses) >= max_steps:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "tilt_sincos.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_dim": int(x.shape[1]),
        "hidden_dim": int(config.get("hidden_dim", 256)),
        "dropout": float(config.get("dropout", 0.1)),
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "target_column": target,
        "target_encoding": "sin(tilt_deg), cos(tilt_deg)",
        "seed": seed,
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    torch.save(checkpoint, checkpoint_path)

    tx_raw, ty_deg = load_test(args.test_features, feature_names)
    tx = (tx_raw - mean) / std
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tx), 2048):
            out = model(torch.as_tensor(tx[start : start + 2048], device=device)).cpu().numpy()
            predictions.append(out)
    pred = np.concatenate(predictions)
    pred_deg = np.rad2deg(np.arctan2(pred[:, 0], pred[:, 1]))
    mae = float(np.mean(np.abs(pred_deg - ty_deg)))
    circular_mae = float(np.mean(np.abs((pred_deg - ty_deg + 180.0) % 360.0 - 180.0)))

    baseline_path = args.output_dir.parent / "metrics.json"
    baseline = json.loads(baseline_path.read_text())
    report = {
        "split": "same tour-disjoint 60k sampled split",
        "train_rows": len(x),
        "test_rows": len(ty_deg),
        "model": "same MLP size and optimizer as direct tilt head; two outputs trained against sin/cos targets",
        "MAE_tilt_deg": mae,
        "MAE_tilt_deg_circular": circular_mae,
        "direct_regression_MAE_tilt_deg": baseline["MAE_tilt_deg"],
        "difference_sincos_minus_direct_deg": mae - baseline["MAE_tilt_deg"],
        "steps": len(losses),
        "checkpoint": str(checkpoint_path),
    }
    report_path = args.output_dir.parent / "tilt_ablation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
