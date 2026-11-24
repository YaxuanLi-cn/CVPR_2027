#!/usr/bin/env python3
"""Decode every image used by the fixed PairUAV-5D split once and cache it as uint8.

Preprocessing follows vint_train.data.data_utils.resize_and_aspect_crop: centre crop to
4:3, then resize. The cache is stored at 256x192 so that the native ViNT (85x64) and NoMaD
(96x96) inputs, and higher-resolution variants, can all be derived from it on the GPU.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/baseline")
IMAGE_ROOT = Path("/root/pairUAV-5D-temp")
OUT = Path(__file__).parent / "cache"
W, H = 256, 192


def load_split(name):
    rows = []
    for tour_dir in sorted(p for p in (ROOT / "_mast3r_split_data_v2" / name).iterdir() if p.is_dir()):
        for path in sorted(tour_dir.glob("*.json")):
            r = json.loads(path.read_text())
            d = r["delta"]
            rows.append((r["image_a"], r["image_b"], r["tour_id"], r["pair_id"],
                         float(d["x_m"]), float(d["y_m"]), float(d["z_m"]),
                         float(r["tilt_num"]), float(r["heading_num"])))
    return rows


def decode(path):
    with Image.open(IMAGE_ROOT / path) as im:
        im.draft("RGB", (W * 4 // 3, H * 4 // 3))
        im = im.convert("RGB")
        w, h = im.size
        cw = int(h * 4 / 3)
        left = (w - cw) // 2
        im = im.crop((left, 0, left + cw, h)).resize((W, H), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def main():
    OUT.mkdir(exist_ok=True)
    splits = {name: load_split(name) for name in ("train", "test")}
    paths = sorted({p for rows in splits.values() for r in rows for p in r[:2]})
    index = {p: i for i, p in enumerate(paths)}
    for name, rows in splits.items():
        np.savez(OUT / f"{name}_index.npz",
                 a=np.array([index[r[0]] for r in rows], dtype=np.int64),
                 b=np.array([index[r[1]] for r in rows], dtype=np.int64),
                 tour=np.array([r[2] for r in rows]),
                 pair_id=np.array([r[3] for r in rows]),
                 target=np.array([r[4:] for r in rows], dtype=np.float64))
        print(name, len(rows), flush=True)
    (OUT / "paths.json").write_text(json.dumps(paths))
    print("images", len(paths), flush=True)
    arr = np.lib.format.open_memmap(OUT / f"images_{W}x{H}.tmp.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(paths), H, W, 3))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for i, img in enumerate(pool.imap(decode, paths, chunksize=64)):
            arr[i] = img
            if i % 5000 == 0:
                print(i, flush=True)
    arr.flush()
    del arr
    os.rename(OUT / f"images_{W}x{H}.tmp.npy", OUT / f"images_{W}x{H}.npy")
    print("done", flush=True)


if __name__ == "__main__":
    main()
