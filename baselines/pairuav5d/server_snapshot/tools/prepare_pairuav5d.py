#!/usr/bin/env python3
"""Materialize a small legacy-style PairUAV dataset from completed 5D pairs."""

import argparse
import json
import math
from pathlib import Path


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/root/pairUAV-5D-temp"))
    parser.add_argument("--output", type=Path, default=Path("/root/baseline/_pairuav5d_data"))
    parser.add_argument("--min-matches", type=int, default=900,
                        help="Only use tours with at least this many saved SuperGlue matches")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    completed = []
    for pair_file in sorted((dataset / "pairs").glob("*.jsonl")):
        tour_id = pair_file.stem
        match_dir = dataset / "matches_superglue" / tour_id
        count = sum(1 for _ in match_dir.glob("*.npz")) if match_dir.is_dir() else 0
        if count >= args.min_matches:
            completed.append((tour_id, pair_file, count))
    if len(completed) < 2:
        raise SystemExit(f"Need at least two completed tours; found {len(completed)}")

    train_tours = completed[:-1]
    test_tours = completed[-1:]
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")

    counts = {"train": 0, "test": 0, "skipped_missing_match": 0, "skipped_missing_image": 0}
    for split, tours in (("train", train_tours), ("test", test_tours)):
        for tour_id, pair_file, _ in tours:
            for row in read_rows(pair_file):
                pair_id = row["pair_id"]
                match_path = dataset / "matches_superglue" / tour_id / f"{pair_id}.npz"
                if not match_path.is_file():
                    counts["skipped_missing_match"] += 1
                    continue
                image_a = dataset / row["image_a"]
                image_b = dataset / row["image_b"]
                if not image_a.is_file() or not image_b.is_file():
                    counts["skipped_missing_image"] += 1
                    continue

                delta = row["delta"]
                sample = dict(row)
                sample["heading_num"] = float(delta["heading_deg"])
                sample["distance_num"] = math.sqrt(
                    float(delta["x_m"]) ** 2
                    + float(delta["y_m"]) ** 2
                    + float(delta["z_m"]) ** 2
                )
                sample["tilt_num"] = float(delta["tilt_deg"])
                sample["match_path"] = str(match_path)
                sample["source_dataset"] = str(dataset)
                target = output / split / tour_id / f"{pair_id}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(sample, separators=(",", ":")) + "\n")
                counts[split] += 1

    summary = {
        "dataset": str(dataset),
        "label_mapping": {
            "heading_num": "delta.heading_deg",
            "distance_num": "sqrt(delta.x_m^2 + delta.y_m^2 + delta.z_m^2), meters",
            "tilt_num": "delta.tilt_deg, degrees",
        },
        "train_tours": [tour[0] for tour in train_tours],
        "test_tours": [tour[0] for tour in test_tours],
        "counts": counts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
