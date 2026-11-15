#!/usr/bin/env python3
"""Cache the remaining images of the TRAIN tours (not referenced by the 60k pair sample) for the absolute-heading CNN."""
import numpy as np, pandas as pd, cv2, re, os
from multiprocessing import Pool
from pathlib import Path
DATA = Path('/root/pairUAV-5D-temp'); OUT = Path('/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading'); W, H = 448, 256
pat = {k: re.compile(rf'<{k}>([-\d.eE+]+)</{k}>') for k in ('heading', 'tilt', 'latitude')}
def load(rel):
    cv2.setNumThreads(1); im = cv2.imread(str(DATA / rel), cv2.IMREAD_REDUCED_COLOR_4)
    return cv2.cvtColor(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
if __name__ == '__main__':
    old = pd.read_pickle(OUT / 'images.pkl'); have = set(old.image); rows = []
    for t in sorted(old[old.split == 'train'].tour.unique()):
        for f in sorted(os.listdir(DATA / 'images' / t)):
            rel = f'images/{t}/{f}'
            if rel in have: continue
            s = (DATA / 'kml' / t / f.replace('.jpg', '.kml')).read_text()
            rows.append(dict(split='train', tour=t, image=rel, heading=float(pat['heading'].search(s).group(1)), tilt=float(pat['tilt'].search(s).group(1)), lat=float(pat['latitude'].search(s).group(1))))
    im = pd.DataFrame(rows); print(len(im), flush=True)
    arr = np.lib.format.open_memmap(OUT / 'images2_448x256.npy', mode='w+', dtype=np.uint8, shape=(len(im), H, W, 3))
    with Pool(8) as p:
        for i, x in enumerate(p.imap(load, im.image.tolist(), chunksize=64)):
            arr[i] = x
            if i % 20000 == 0: print(i, flush=True)
    arr.flush(); im.to_pickle(OUT / 'images2.pkl'); print('done', len(im))
