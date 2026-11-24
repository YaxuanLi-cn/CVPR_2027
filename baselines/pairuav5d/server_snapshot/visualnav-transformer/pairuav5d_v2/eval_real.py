#!/usr/bin/env python3
"""Zero-shot evaluation of the PairUAV-5D ViNT / NoMaD checkpoints on real drone photographs.

Pairs: the Mezieres pairs already scored for DreamNav (/root/realdata/mezieres/pred.pkl), restricted to the
baseline/height range seen in training (dist_equiv <= 38 m). No weight is updated and no label is read by the
models. Images go through exactly the PairUAV-5D pipeline of build_cache.py / train_v2.py: centre crop to 4:3,
256x192 uint8 cache, antialiased resize to the official input size, ImageNet normalisation. Inference is the
test protocol of train_v2.py (final-epoch EMA weights; NoMaD: first of 8 DDPM samples, 10 steps, seed 0).
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_cache import decode  # noqa: E402  (absolute paths override its IMAGE_ROOT)
from train_v2 import DDPM, NATIVE, NoMaDAdapter, Normalizer, ViNTAdapter, predict  # noqa: E402

REAL = Path("/root/realdata/mezieres")
RANGE = 38.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["vint", "nomad"])
    parser.add_argument("--run", type=Path, default=None, help="run directory holding final.pt")
    parser.add_argument("--out", type=Path, default=REAL / "baselines")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run = args.run or HERE / "runs" / f"{args.model}_native"
    device = torch.device(args.device)

    pairs = pd.read_pickle(REAL / "pred.pkl")
    pairs = pairs[pairs.dist_equiv <= RANGE]
    paths = sorted(set(pairs.image_a) | set(pairs.image_b))
    slot = {p: i for i, p in enumerate(paths)}
    with Pool(8) as pool:
        cache = np.stack(pool.map(decode, paths))  # [n, 192, 256, 3] uint8, same as images_256x192.npy

    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    width, height = checkpoint["size"]
    assert (width, height) == NATIVE[args.model]
    block = torch.from_numpy(cache).to(device).permute(0, 3, 1, 2).float()  # same steps as train_v2.load_images
    block = F.interpolate(block, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
    images = block.round().clamp(0, 255).to(torch.uint8)

    official = HERE.parent / "pretrained" / f"{args.model}.pth"
    model = ViNTAdapter(official) if args.model == "vint" else NoMaDAdapter(official)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    index = {"a": torch.tensor([slot[p] for p in pairs.image_a], device=device),
             "b": torch.tensor([slot[p] for p in pairs.image_b], device=device)}
    rows = torch.arange(len(pairs), device=device)
    first, mean, distance = predict(model, args.model, images, index, rows, Normalizer().to(device),
                                    DDPM(10, device) if args.model == "nomad" else None,
                                    samples=8 if args.model == "nomad" else 1)

    args.out.mkdir(parents=True, exist_ok=True)
    extra = {"pred_mean_of_8_samples": mean.cpu().numpy()} if args.model == "nomad" else {}
    np.savez_compressed(args.out / f"{args.model}_predictions.npz", pair_index=pairs.index.values,
                        a=np.array(pairs.a.tolist(), dtype=str), b=np.array(pairs.b.tolist(), dtype=str),
                        pred_xyz_tilt_heading=first.cpu().numpy(), pred_distance_head_m=distance.cpu().numpy(),
                        true_xyz_tilt_heading=pairs[["x", "y", "z", "tilt", "heading"]].values, **extra)
    print(json.dumps({"model": args.model, "checkpoint": str(run / "final.pt"), "pairs": len(pairs),
                      "images": len(paths), "input_size_wh": [width, height], "device": str(device),
                      "pred_std": first.std(dim=0).tolist(), "distance_head_mean_m": distance.mean().item()}))


if __name__ == "__main__":
    main()
