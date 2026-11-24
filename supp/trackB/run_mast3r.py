#!/usr/bin/env python3
"""Track B: MASt3R used directly as a two-view pose estimator (frozen metric checkpoint, 512 x 288 input, nothing fitted).

  pointmap + PnP : PnP-RANSAC on the target pointmap expressed in the source frame (reviewer_baselines/rb_recon.cpu_work)
  matches + E    : reciprocal nearest neighbours of the dense descriptors (fast_reciprocal_NNs, subsample 4), saved in the
                   common match format so that the essential-matrix solver and its inlier threshold are those of the other matchers
The source-view metric depth map of every forward pass is saved for the native-scale variants.

real_in / real_out : all four rotations of the target, PnP rows + matches + depth.
test / train       : PnP rows, depth and the per-rotation inlier counts already exist in outputs/reviewer_baselines; here only
                     the matches of rotation 0 and of the archived best rotation are regenerated.
Usage: run_mast3r.py <real_in|real_out|test|train>
"""
import sys, os, time
from multiprocessing import get_context
import numpy as np, pandas as pd, cv2
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
import rb_recon as rb
from run_matchers import pair_set, OUT
W5, H5 = rb.W5, rb.H5; which = sys.argv[1]; real = which.startswith('real')


class Pairs:
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        from dust3r.utils.image import load_images
        r = self.df.iloc[i]; a, b = (load_images([p], size=512, verbose=False)[0]['img'] for p in (r.a, r.b)); return int(r.id), a, b


def main():
    df = pair_set(which).sample(frac=1.0, random_state=0)      # fixed random order: a partial run is a random subset
    if len(sys.argv) > 2: df = df.iloc[:int(sys.argv[2])]
    os.makedirs(f'{OUT}/matches/mast3r/{which}', exist_ok=True); os.makedirs(f'{OUT}/mast3r/depth_{which}', exist_ok=True); os.makedirs(f'{OUT}/mast3r/pnp_{which}', exist_ok=True)
    if real: rots = {int(i): [0, 1, 2, 3] for i in df.id}
    else:
        R = pd.read_pickle(f'{rb.OUT}/recon_mast3r_{which}.pkl'); best = R.loc[R.fillna({'mm_inl': -1}).groupby('manifest_index').mm_inl.idxmax()].set_index('manifest_index').k
        rots = {int(i): sorted({0, int(best.loc[i])}) for i in df.id}
    todo = df[[not os.path.exists(f'{OUT}/matches/mast3r/{which}/{i}.npz') for i in df.id]]; print(f'{which}: {len(todo)} / {len(df)} pairs to do', flush=True)
    pool = get_context('fork').Pool(3)
    import matched_mast3r_ablation as mm; mm.load_runtime()
    import torch
    from mast3r.model import AsymmetricMASt3R
    from fastnn_gpu import fast_reciprocal_gpu
    dev = 'cuda'; model = AsymmetricMASt3R.from_pretrained(str(mm.MODEL_PATH)).to(dev).eval().requires_grad_(False)
    dl = torch.utils.data.DataLoader(Pairs(todo), batch_size=None, num_workers=2, prefetch_factor=2); ys, xs = torch.meshgrid(torch.arange(H5), torch.arange(W5), indexing='ij'); t0 = time.time(); tt = []
    q = lambda v: np.clip(np.rint(4 * v), 0, 65535).astype(np.uint16)
    for n_, (idx, ia, ib) in enumerate(dl):
        assert ia.shape[-2:] == (H5, W5), ia.shape
        ia, ib = ia.to(dev), ib.to(dev); deps = {}; out = {}; tk = []; jobs = []
        for k in rots[idx]:
            torch.cuda.synchronize(); t1 = time.time()
            for attempt in range(40):
                try:
                    ibr = torch.rot90(ib, k, (2, 3)); v1 = dict(img=ia, true_shape=torch.tensor([[H5, W5]]), idx=[0], instance=['0']); v2 = dict(img=ibr, true_shape=torch.tensor([list(ibr.shape[-2:])]), idx=[1], instance=['1'])
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16): p1, p2 = model(v1, v2)
                    X1 = p1['pts3d'][0].float(); X2 = p2['pts3d_in_other_view'][0].float(); c2 = p2['conf'][0].float()
                    a1, a2 = fast_reciprocal_gpu(p1['desc'][0].float(), p2['desc'][0].float(), 4); break      # identical output to mast3r.fast_nn.fast_reciprocal_NNs(subsample 4, dot), one GPU synchronisation (supp/trackB/check_fastnn.py)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); print(f'pair {idx} k={k}: out of memory, retry {attempt + 1}', flush=True); time.sleep(30); t1 = time.time()
            else: print('out of memory after 40 retries, giving up', flush=True); os._exit(3)
            torch.cuda.synchronize(); tk.append(time.time() - t1)
            mx, my = torch.rot90(xs, k, (0, 1)), torch.rot90(ys, k, (0, 1)); Hr, Wr = ibr.shape[-2:]
            keep = (a1[:, 0] >= 3) & (a1[:, 0] < W5 - 3) & (a1[:, 1] >= 3) & (a1[:, 1] < H5 - 3) & (a2[:, 0] >= 3) & (a2[:, 0] < Wr - 3) & (a2[:, 1] >= 3) & (a2[:, 1] < Hr - 3); a1, a2 = a1[keep], a2[keep]
            xy1 = a1.astype(np.float64); xy2 = np.stack([mx.numpy()[a2[:, 1], a2[:, 0]], my.numpy()[a2[:, 1], a2[:, 0]]], 1).astype(np.float64)
            out[f'p0_{k}'] = q(rb.to_full(xy1)); out[f'p1_{k}'] = q(rb.to_full(xy2)); out[f'c_{k}'] = np.full(len(xy1), 255, np.uint8)
            X1n = X1.cpu().numpy(); deps[k] = X1n[..., 2].astype(np.float16)
            if real:
                sl = (slice(2, None, 4), slice(2, None, 4)); uv2 = torch.stack([mx[sl], my[sl]], -1).reshape(-1, 2).numpy().astype(np.float32)
                jobs.append(pool.apply_async(rb.cpu_work, ((idx, k, X2[sl].reshape(-1, 3).cpu().numpy(), uv2, c2[sl].reshape(-1).cpu().numpy(), X1n, None, None),)))
        if real:
            pd.DataFrame([j.get() for j in jobs]).rename(columns={'manifest_index': 'id'}).to_pickle(f'{OUT}/mast3r/pnp_{which}/{idx}.pkl'); np.save(f'{OUT}/mast3r/depth_{which}/{idx}.npy', np.stack([deps[k] for k in range(4)]))
        np.savez_compressed(f'{OUT}/matches/mast3r/{which}/{idx}.npz', **out); tt.append(tk); torch.cuda.empty_cache()
        if (n_ + 1) % 50 == 0: print(f'{n_+1}/{len(todo)} {(time.time()-t0)/(n_+1):.2f}s/pair', flush=True)
    if real:
        import glob
        R = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f'{OUT}/mast3r/pnp_{which}/*.pkl'))], ignore_index=True); R.to_pickle(f'{OUT}/mast3r/recon_{which}.pkl'); print('saved', R.shape, flush=True)
    if tt:
        import json; json.dump(dict(set=which, pairs=len(tt), plain_s=float(np.mean([t[0] for t in tt])), tta_s=float(np.mean([sum(t) for t in tt])) if real else None, peak_gpu_gb=float(torch.cuda.max_memory_allocated() / 2**30),
                                    settings=dict(model='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', input='512 x 288', matching='fast reciprocal nearest neighbours on the dense descriptors, subsample 4, dot product (GPU-native re-implementation with identical output)', pnp='solvePnPRansac (SQPNP), 2 px at 512 px, 500 iterations + LM refinement')),
                               open(f'{OUT}/matches/mast3r/timing_{which}.json', 'w'), indent=1)
    print('done', which, f'{time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
