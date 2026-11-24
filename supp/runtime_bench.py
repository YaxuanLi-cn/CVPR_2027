#!/usr/bin/env python3
"""Runtime of the full DreamNav pipeline (online engine) on one RTX 5090.

latency    : one pair at a time, everything in the main process (no loader workers), per-phase wall clock
throughput : a batch of pairs with loader workers and the CPU geometry pool, as in the archived batch runs
Also counts the SuperPoint / SuperGlue forward passes per pair from the saved match files.
Usage: runtime_bench.py [n_latency] [n_throughput]
"""
import sys, os, json, time, glob, shutil, numpy as np, pandas as pd
sys.path.insert(0, '/root/rasr-pairuav/online'); sys.path.insert(0, '/root/rasr-pairuav/superglue_fusion')
OUT = '/root/rasr-pairuav/outputs/supp/runtime'; os.makedirs(OUT, exist_ok=True)
nl = int(sys.argv[1]) if len(sys.argv) > 1 else 40; nt = int(sys.argv[2]) if len(sys.argv) > 2 else 96
from engine import Engine, SGD
import torch
from torch.utils.data._utils.collate import default_convert
P = pd.read_pickle(f'{SGD}/stage2/s2_pred.pkl'); P = P[P.split == 'test'][['manifest_index']].merge(pd.read_pickle(f'{SGD}/pairs.pkl'), on='manifest_index')
S = P.sample(nl + nt + 4, random_state=123).reset_index(drop=True); pairs = [(f'/root/pairUAV-5D-temp/{a}', f'/root/pairUAV-5D-temp/{b}') for a, b in zip(S.image_a, S.image_b)]
eng = Engine(verbose=False); res = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}

# ---- latency: single pair, in-process loading
orig_loader = eng._loader
def seq_loader(ds):
    for i in range(len(ds)): yield default_convert(ds[i])
eng._loader = seq_loader
for p in pairs[:4]: eng.predict([p])                        # warm-up (cuDNN autotune, lazy inits)
torch.cuda.reset_peak_memory_stats(); rows = []
for p in pairs[4:4 + nl]:
    t0 = time.time(); R = eng.predict([p]); torch.cuda.synchronize(); T = dict(R.attrs['timing']); T['total_wall'] = time.time() - t0; rows.append(T)
L = pd.DataFrame(rows); res['latency_ms'] = {k: dict(mean=float(1000 * L[k].mean()), median=float(1000 * L[k].median()), p90=float(1000 * L[k].quantile(.9))) for k in L.columns}
res['latency_pairs'] = int(len(L)); res['peak_gpu_mem_gb'] = float(torch.cuda.max_memory_allocated() / 2**30)
# stage-1-only latency (no imagination, no absolute heading) and heading-free full pipeline
rows1, rows2 = [], []
for p in pairs[4:4 + nl]:
    t0 = time.time(); eng.predict([p], stage2=False, abs_heading=False); torch.cuda.synchronize(); rows1.append(time.time() - t0)
    t0 = time.time(); eng.predict([p], stage2=True, abs_heading=False); torch.cuda.synchronize(); rows2.append(time.time() - t0)
res['latency_stage1_no_heading_ms'] = dict(mean=float(1000 * np.mean(rows1)), median=float(1000 * np.median(rows1)))
res['latency_stage2_no_heading_ms'] = dict(mean=float(1000 * np.mean(rows2)), median=float(1000 * np.median(rows2)))

# ---- throughput: batch with workers
eng._loader = orig_loader; kd = '/dev/shm/dreamnav_bench'; shutil.rmtree(kd, ignore_errors=True)
t0 = time.time(); R = eng.predict(pairs[4 + nl:4 + nl + nt], keep_dir=kd); torch.cuda.synchronize(); wall = time.time() - t0
res['throughput'] = dict(pairs=nt, wall_s=wall, s_per_pair=wall / nt, phases_s_per_pair={k: float(v / nt) for k, v in R.attrs['timing'].items()})
# ---- forward-pass counts
fine = [len(np.load(f)['fine_counts']) for f in glob.glob(f'{kd}/rot/*.npz')]; tiles = [int(np.load(f)['ntiles']) for f in glob.glob(f'{kd}/imag/*.npz')]
kp = [int((np.load(f)['src'] >= 10).sum()) for f in glob.glob(f'{kd}/rot/*.npz')]
res['passes'] = dict(superglue_pass1_coarse=4, superglue_pass1_fine_mean=float(np.mean(fine)), superglue_zoom=1, imag_tiles_mean=float(np.mean(tiles)), imag_tiles_max=int(np.max(tiles)), imag_tiles_hist=np.bincount(tiles, minlength=10).tolist(),
                     superglue_total_mean=float(4 + np.mean(fine) + 1 + np.mean(tiles)), superglue_total_max=int(4 + np.max(fine) + 1 + np.max(tiles)),
                     superpoint_total_mean=float(5 + 1 + np.mean(fine) + 2 + 2 * np.mean(tiles)))
shutil.rmtree(kd, ignore_errors=True); eng.close()
json.dump(res, open(f'{OUT}/runtime.json', 'w'), indent=1); print(json.dumps(res, indent=1))
