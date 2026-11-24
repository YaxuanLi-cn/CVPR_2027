"""Equivalence and timing of fastnn_gpu.fast_reciprocal_gpu against mast3r.fast_nn.fast_reciprocal_NNs on a few pairs (all four rotations)."""
import sys, time, numpy as np, torch
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']; sys.path.append(f'{ROOT}/supp/trackB')
import rb_recon as rb
from run_matchers import pair_set
from fastnn_gpu import fast_reciprocal_gpu
import matched_mast3r_ablation as mm; mm.load_runtime()
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.utils.image import load_images
dev = 'cuda'; model = AsymmetricMASt3R.from_pretrained(str(mm.MODEL_PATH)).to(dev).eval().requires_grad_(False); H5, W5 = rb.H5, rb.W5; tot = same = 0
for which, sl in (('real_in', slice(5, 7)), ('test', slice(0, 2))):
    for r in pair_set(which).sample(frac=1.0, random_state=0).iloc[sl].itertuples():
        ia, ib = (load_images([p], size=512, verbose=False)[0]['img'].to(dev) for p in (r.a, r.b))
        for k in range(4):
            ibr = torch.rot90(ib, k, (2, 3)); v1 = dict(img=ia, true_shape=torch.tensor([[H5, W5]]), idx=[0], instance=['0']); v2 = dict(img=ibr, true_shape=torch.tensor([list(ibr.shape[-2:])]), idx=[1], instance=['1'])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16): p1, p2 = model(v1, v2)
            d1, d2 = p1['desc'][0].float(), p2['desc'][0].float(); torch.cuda.synchronize(); t = time.time(); a1, a2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=4, device=dev, dist='dot', block_size=2**13); t_off = time.time() - t
            torch.cuda.synchronize(); t = time.time(); b1, b2 = fast_reciprocal_gpu(d1, d2, 4); t_new = time.time() - t
            A = set(map(tuple, np.c_[a1, a2].tolist())); B = set(map(tuple, np.c_[b1, b2].tolist())); tot += len(A | B); same += len(A & B)
            print(f'{which} {r.id} k={k}: official {len(A)} matches in {t_off:.2f}s | gpu-native {len(B)} in {t_new:.2f}s | identical pairs {len(A & B)} | only official {len(A - B)} | only new {len(B - A)}', flush=True)
print(f'overall: {same} of {tot} correspondences identical ({100 * same / max(tot, 1):.3f}%)')
