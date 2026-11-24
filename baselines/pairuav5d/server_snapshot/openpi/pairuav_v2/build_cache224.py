#!/usr/bin/env python3
"""Cache every image of the fixed PairUAV-5D split as openpi sees it: the full frame,
resize_with_pad to 224x224 (same call the openpi ResizeImages transform makes), uint8.
Image order and the train/test index files are shared with visualnav-transformer/pairuav5d_v2/cache."""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from openpi_client import image_tools

IMAGE_ROOT = Path("/root/pairUAV-5D-temp")
SRC = Path("/root/baseline/visualnav-transformer/pairuav5d_v2/cache")
OUT = Path(__file__).parent / "cache"


def decode(path):
    with Image.open(IMAGE_ROOT / path) as im:
        im.draft("RGB", (896, 504))
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return image_tools.resize_with_pad(image[None], 224, 224)[0]


def main():
    OUT.mkdir(exist_ok=True)
    paths = json.loads((SRC / "paths.json").read_text())
    arr = np.lib.format.open_memmap(OUT / "images_224_pad.tmp.npy", mode="w+", dtype=np.uint8, shape=(len(paths), 224, 224, 3))
    with Pool(int(sys.argv[1])) as pool:
        for i, img in enumerate(pool.imap(decode, paths, chunksize=64)):
            arr[i] = img
            if i % 5000 == 0:
                print(i, flush=True)
    arr.flush()
    del arr
    os.rename(OUT / "images_224_pad.tmp.npy", OUT / "images_224_pad.npy")
    print("done", flush=True)


if __name__ == "__main__":
    main()
