#!/usr/bin/env python3
"""Run the bundled SuperGlue matcher on PairUAV-5D JSONL pair files."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import cv2

from models.matching import Matching


def read_image(path, device, resize):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    height, width = image.shape
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(height, width)
        new_width, new_height = int(round(width * scale)), int(round(height * scale))
    elif len(resize) == 1 and resize[0] == -1:
        new_width, new_height = width, height
    else:
        new_width, new_height = resize
    image = cv2.resize(image, (new_width, new_height)).astype(np.float32)
    tensor = torch.from_numpy(image / 255.0).float()[None, None].to(device)
    return image, tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SuperGlue matches for PairUAV-5D JSONL pairs")
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("/root/pairUAV-5D-temp"),
        help="Dataset root containing images/ and pairs/.")
    parser.add_argument(
        "--pairs_dir", type=Path, default=None,
        help="Directory containing per-tour JSONL files (default: DATASET/pairs).")
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output root (default: DATASET/matches_superglue).")
    parser.add_argument(
        "--tour", action="append", dest="tours", default=None,
        help="Tour id to process; may be repeated. Default: all JSONL files.")
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Maximum pairs per tour, useful for smoke tests.")
    parser.add_argument(
        "--resize", type=int, nargs="+", default=[640, 480],
        help="Resize like match_pairs.py: WIDTH HEIGHT, one max dimension, or -1.")
    parser.add_argument("--superglue", choices=("indoor", "outdoor"), default="outdoor")
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.2)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Recompute results even when the output NPZ already exists.")
    return parser.parse_args()


def load_pairs(pairs_file, dataset_dir, max_pairs):
    pairs = []
    with pairs_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {pairs_file}:{line_number}: {exc}") from exc
            for key in ("pair_id", "image_a", "image_b"):
                if key not in pair:
                    raise ValueError(f"Missing {key!r} in {pairs_file}:{line_number}")
            image0 = (dataset_dir / pair["image_a"]).resolve()
            image1 = (dataset_dir / pair["image_b"]).resolve()
            if not image0.is_file() or not image1.is_file():
                raise FileNotFoundError(
                    f"Image missing for pair {pair['pair_id']}: {image0} or {image1}")
            pairs.append((pair["pair_id"], pair["image_a"], pair["image_b"]))
            if max_pairs is not None and len(pairs) >= max_pairs:
                break
    return pairs


def main():
    opt = parse_args()
    dataset_dir = opt.dataset_dir.resolve()
    pairs_dir = (opt.pairs_dir or dataset_dir / "pairs").resolve()
    output_dir = (opt.output_dir or dataset_dir / "matches_superglue").resolve()
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[:1]
    if len(opt.resize) not in (1, 2):
        raise ValueError("--resize accepts one or two integers")

    pair_files = ([pairs_dir / f"{tour}.jsonl" for tour in opt.tours]
                  if opt.tours else sorted(pairs_dir.glob("*.jsonl")))
    if not pair_files:
        raise FileNotFoundError(f"No .jsonl pair files found in {pairs_dir}")
    missing = [path for path in pair_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Pair file(s) not found: " + ", ".join(map(str, missing)))

    device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
    print(f"Running SuperGlue on {device}; writing results to {output_dir}")
    config = {
        "superpoint": {
            "nms_radius": opt.nms_radius,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
        },
        "superglue": {
            "weights": opt.superglue,
            "sinkhorn_iterations": opt.sinkhorn_iterations,
            "match_threshold": opt.match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)

    total = 0
    for pairs_file in pair_files:
        tour_id = pairs_file.stem
        pairs = load_pairs(pairs_file, dataset_dir, opt.max_pairs)
        tour_output = output_dir / tour_id
        tour_output.mkdir(parents=True, exist_ok=True)
        print(f"{tour_id}: {len(pairs)} pairs")
        for index, (pair_id, name0, name1) in enumerate(pairs, start=1):
            matches_path = tour_output / f"{pair_id}.npz"
            if matches_path.exists() and not opt.no_cache:
                total += 1
                continue

            image0, inp0 = read_image(dataset_dir / name0, device, opt.resize)
            image1, inp1 = read_image(dataset_dir / name1, device, opt.resize)
            if image0 is None or image1 is None:
                raise RuntimeError(f"Could not read pair images: {name0}, {name1}")

            with torch.inference_mode():
                prediction = matching({"image0": inp0, "image1": inp1})
            prediction = {key: value[0].cpu().numpy()
                          for key, value in prediction.items()}
            np.savez_compressed(
                matches_path,
                keypoints0=prediction["keypoints0"],
                keypoints1=prediction["keypoints1"],
                matches=prediction["matches0"],
                match_confidence=prediction["matching_scores0"],
            )
            total += 1
            if index % 100 == 0 or index == len(pairs):
                print(f"  {tour_id}: {index}/{len(pairs)}")
    print(f"Done. Results present for {total} processed pairs.")


if __name__ == "__main__":
    main()
