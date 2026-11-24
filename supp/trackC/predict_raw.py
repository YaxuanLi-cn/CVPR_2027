#!/usr/bin/env python3
"""Track C: 24-view predictions of a re-trained raw-view network on all cached val + test images (one pass), written in the
format of the archived pred_cnx_s_d8.pkl. Used instead of the end-of-training evaluation of train_raw.py, which repeats the
24-view pass several times.   Usage: predict_raw.py <tag> [train_seconds]"""
import sys, json, re, os, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/supp/trackC')
from nets import Heading
tag = sys.argv[1]; O = '/root/rasr-pairuav/outputs/supp/trackC'; D = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading'
m1 = pd.read_pickle(f'{D}/images.pkl'); I1 = np.load(f'{D}/images_448x256.npy', mmap_mode='r')
rng = np.random.default_rng(0); tr_tours = np.array(sorted(m1[m1.split == 'train'].tour.unique())); val_tours = set(rng.choice(tr_tours, len(tr_tours) // 10, replace=False))
m1['part'] = np.where(m1.split == 'test', 'test', np.where(m1.tour.isin(val_tours), 'val', 'train')); ST = pd.read_pickle(f'{O}/sites.pkl').set_index('tour'); m1['heldout'] = ST.heldout.reindex(m1.tour).values
ix = np.where(m1.part.isin(['val', 'test']))[0]; H = Heading(raw=f'{O}/model_{tag}.pt', bev=None, bs=64); hs, ks = [], []
for i in range(0, len(ix), 2048):
    X = np.stack([I1[j] for j in ix[i:i + 2048]]); h, k = H.raw_view(X); hs.append(h); ks.append(k); print(i, len(ix), flush=True)
o = m1.iloc[ix][['image', 'part', 'tour', 'heading', 'tilt', 'heldout']].copy(); o['ah_pred'] = np.concatenate(hs); o['ah_kappa'] = np.concatenate(ks); o.to_pickle(f'{O}/pred_{tag}.pkl')
e = np.abs((o.ah_pred - o.heading + 180) % 360 - 180)
for part in ('val', 'test'):
    for held in (True, False): m = (o.part == part) & (o.heldout == held); print(f'{part} heldout={held}: n {m.sum()} MAE {e[m].mean():.2f} med {e[m].median():.2f}')
# training record reconstructed from the log (the training process is stopped after the last checkpoint)
log = open(f'{O}/logs/train_{tag}.log').read(); hd = re.search(r"excluded training buildings (\d+) \| training images used (\d+) from (\d+) buildings", log); st = re.findall(r'step (\d+)/(\d+) loss [-\d.]+ (\d+)s peak mem ([\d.]+) GB', log)
rec = dict(tag=tag, excluded_buildings=int(hd.group(1)), train_images=int(hd.group(2)), train_buildings=int(hd.group(3)), epochs=10, bs=128, chunk=12, lr=2e-4, steps=int(st[-1][1]), train_seconds=float(sys.argv[2]) if len(sys.argv) > 2 else float(st[-1][2]) * int(st[-1][1]) / int(st[-1][0]),
           peak_mem_gb=float(st[-1][3]), seed=0, note='wall-clock on a GPU shared with other jobs, including the per-epoch monitoring passes', epoch_log=re.findall(r'== epoch \d+ .*', log))
json.dump(rec, open(f'{O}/train_{tag}.json', 'w'), indent=1); print('saved', tag)
