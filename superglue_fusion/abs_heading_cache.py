#!/usr/bin/env python3
"""Cache down-sized RGB images (448x256) of every image used by the 60k split, for the absolute-heading CNN."""
import numpy as np, pandas as pd, cv2, re
from multiprocessing import Pool
from pathlib import Path
DATA = Path('/root/pairUAV-5D-temp'); OUT = Path('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading'); OUT.mkdir(parents=True, exist_ok=True)
W, H = 448, 256
def load(rel):
    cv2.setNumThreads(1)
    im = cv2.imread(str(DATA / rel), cv2.IMREAD_REDUCED_COLOR_4)
    return cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
if __name__ == '__main__':
    df = pd.read_pickle('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/pairs.pkl')
    a = df[['split', 'tour', 'image_a', 'heading_a', 'tilt_a', 'latitude_a']].set_axis(['split', 'tour', 'image', 'heading', 'tilt', 'lat'], axis=1)
    b = df[['split', 'tour', 'image_b', 'heading_b', 'tilt_b', 'latitude_b']].set_axis(['split', 'tour', 'image', 'heading', 'tilt', 'lat'], axis=1)
    im = pd.concat([a, b]).drop_duplicates('image').reset_index(drop=True)
    print(im.groupby('split').size())
    arr = np.lib.format.open_memmap(OUT / 'images_448x256.npy', mode='w+', dtype=np.uint8, shape=(len(im), H, W, 3))
    with Pool(40) as p:
        for i, x in enumerate(p.imap(load, im.image.tolist(), chunksize=64)):
            arr[i] = x
            if i % 10000 == 0: print(i, flush=True)
    arr.flush(); im.to_pickle(OUT / 'images.pkl'); print('done', len(im))
