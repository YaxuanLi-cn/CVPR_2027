#!/usr/bin/env python3
"""Evaluate xyz, tilt, and heading predictions on the fixed PairUAV split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from models_src.distance_head import DistanceHead


def read_checkpoint(path: Path, device: torch.device) -> tuple[DistanceHead, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DistanceHead(
        checkpoint["feature_dim"], checkpoint["hidden_dim"], checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--heading-mode", choices=["sincos", "raw"], default="sincos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    coordinate_models = {
        axis: read_checkpoint(args.model_dir / f"coordinate_head_{axis}.pt", device)
        for axis in "xyz"
    }
    tilt_model, tilt_checkpoint = read_checkpoint(args.model_dir / "tilt_head.pt", device)
    heading_name = "heading_baseline.json" if args.heading_mode == "sincos" else "heading_raw.json"
    heading = json.loads((args.model_dir / heading_name).read_text(encoding="utf-8"))
    feature_names = coordinate_models["x"][1]["feature_names"]
    heading_mean = np.asarray(heading["mean"], dtype=np.float32)
    heading_std = np.asarray(heading["std"], dtype=np.float32)
    heading_feature_names = heading["feature_names"]
    if args.heading_mode == "sincos":
        sin_weights = np.asarray(heading["sin_weights"], dtype=np.float64)
        cos_weights = np.asarray(heading["cos_weights"], dtype=np.float64)
    else:
        heading_weights = np.asarray(heading["weights"], dtype=np.float64)

    errors = {key: [] for key in ("x", "y", "z", "distance", "tilt", "heading")}
    valid = 0
    invalid = 0
    batch: list[dict[str, str]] = []

    def flush(rows: list[dict[str, str]]) -> None:
        nonlocal valid, invalid
        if not rows:
            return
        kept: list[dict[str, str]] = []
        for row in rows:
            try:
                if row.get("is_valid", "true").lower() not in ("true", "1", "yes"):
                    raise ValueError
                values = [float(row[key]) for key in (*feature_names, "x_num", "y_num", "z_num", "tilt_num", "heading_num")]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError
                kept.append(row)
            except (KeyError, TypeError, ValueError):
                invalid += 1
        if not kept:
            return

        x = np.asarray([[float(row[name]) for name in feature_names] for row in kept], dtype=np.float32)
        targets = np.asarray(
            [[float(row[f"{axis}_num"]) for axis in "xyz"] for row in kept], dtype=np.float64
        )
        tx = np.asarray(
            [[float(row[name]) for name in tilt_checkpoint["feature_names"]] for row in kept],
            dtype=np.float32,
        )
        predictions: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for axis, (model, checkpoint) in coordinate_models.items():
                mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
                std = np.maximum(np.asarray(checkpoint["feature_std"], dtype=np.float32), 1e-6)
                normalized = (x - mean) / std
                prediction = model(torch.as_tensor(normalized, device=device)).cpu().numpy()
                predictions[axis] = prediction * float(checkpoint["target_scale"])

            tilt_mean = np.asarray(tilt_checkpoint["feature_mean"], dtype=np.float32)
            tilt_std = np.maximum(np.asarray(tilt_checkpoint["feature_std"], dtype=np.float32), 1e-6)
            tilt_normalized = (tx - tilt_mean) / tilt_std
            tilt_prediction = tilt_model(torch.as_tensor(tilt_normalized, device=device)).cpu().numpy()
            tilt_prediction *= float(tilt_checkpoint["target_scale"])

        predicted_xyz = np.column_stack([predictions[axis] for axis in "xyz"])
        predicted_distance = np.linalg.norm(predicted_xyz, axis=1)
        true_distance = np.linalg.norm(targets, axis=1)
        for index, axis in enumerate("xyz"):
            errors[axis].extend(np.abs(predicted_xyz[:, index] - targets[:, index]).tolist())
        errors["distance"].extend(np.abs(predicted_distance - true_distance).tolist())
        true_tilt = np.asarray([float(row["tilt_num"]) for row in kept], dtype=np.float64)
        errors["tilt"].extend(np.abs(tilt_prediction - true_tilt).tolist())

        hidx = [feature_names.index(name) for name in heading_feature_names]
        hx = (x[:, hidx].astype(np.float64) - heading_mean) / np.where(heading_std < 1e-8, 1.0, heading_std)
        hx = np.concatenate([hx, np.ones((len(hx), 1))], axis=1)
        if args.heading_mode == "sincos":
            heading_prediction = (np.rad2deg(np.arctan2(hx @ sin_weights, hx @ cos_weights)) + 360.0) % 360.0
        else:
            heading_prediction = hx @ heading_weights
        true_heading = np.asarray([float(row["heading_num"]) for row in kept], dtype=np.float64)
        heading_error = np.abs((heading_prediction - true_heading + 180.0) % 360.0 - 180.0)
        errors["heading"].extend(heading_error.tolist())
        valid += len(kept)

    with args.test_features.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            batch.append(row)
            if len(batch) >= args.batch_size:
                flush(batch)
                batch = []
    flush(batch)

    report = {
        "split": "tour-disjoint 60k sampled test split",
        "test_rows_valid": valid,
        "test_rows_invalid": invalid,
        "heading_mode": args.heading_mode,
        "distance_metric": "MAE of Euclidean distance computed from predicted xyz vs true xyz",
        "MAE_x_m": float(np.mean(errors["x"])) if errors["x"] else None,
        "MAE_y_m": float(np.mean(errors["y"])) if errors["y"] else None,
        "MAE_z_m": float(np.mean(errors["z"])) if errors["z"] else None,
        "MAE_tilt_deg": float(np.mean(errors["tilt"])) if errors["tilt"] else None,
        "MAE_heading_deg_circular": float(np.mean(errors["heading"])) if errors["heading"] else None,
        "MAE_distance_from_xyz_m": float(np.mean(errors["distance"])) if errors["distance"] else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
