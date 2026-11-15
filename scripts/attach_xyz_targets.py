#!/usr/bin/env python3
"""Join x/y/z labels from sampled pair JSONL files onto an existing feature CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_targets(manifest: Path, data_root: Path) -> dict[int, tuple[float, float, float]]:
    targets: dict[int, tuple[float, float, float]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            source_index, reference = line.rstrip("\n").split("\t", 1)
            pair_ref, offset = reference.rsplit("#", 1)
            with (data_root / pair_ref).open("rb") as pair_file:
                pair_file.seek(int(offset))
                payload = json.loads(pair_file.readline())
            delta = payload["delta"]
            targets[int(source_index)] = tuple(
                float(delta[key]) for key in ("x_m", "y_m", "z_m")
            )
    return targets


def main() -> None:
    args = parse_args()
    targets = load_targets(args.manifest, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.features.open("r", encoding="utf-8", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        for name in ("x_num", "y_num", "z_num"):
            if name not in fields:
                fields.append(name)
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            idx = int(row["manifest_index"])
            if idx not in targets:
                raise RuntimeError(f"manifest has no target for feature row {idx}")
            row.update(zip(("x_num", "y_num", "z_num"), targets[idx]))
            writer.writerow(row)
            count += 1
    if count != len(targets):
        raise RuntimeError(f"feature rows ({count}) do not match manifest rows ({len(targets)})")
    print(json.dumps({"output": str(args.output), "rows": count}, indent=2))


if __name__ == "__main__":
    main()
