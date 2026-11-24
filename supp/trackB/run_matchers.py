#!/usr/bin/env python3
"""Track B, part 1: off-the-shelf matchers on the sampled pairs (frozen public weights, nothing trained on PairUAV).

For every pair and every in-plane rotation k*90 deg (counter-clockwise, k = 0..3) of the TARGET image, the matcher is run on
(source, rotated target); matched keypoints are mapped back to the ORIGINAL pixel grid (3686 x 2073, pixel centres at
integers) of the two unrotated images and saved, quantised to 1/4 px, as  <out>/<matcher>/<set>/<id>.npz  with arrays
p0_k, p1_k (uint16, = round(4 * xy)), c_k (uint8, = round(255 * confidence)) for k = 0..3.

Matchers (settings = the public defaults / recommendations of each method, see SETTINGS):
  splg     SuperPoint + LightGlue           aliked   ALIKED-n16 + LightGlue
  loftr    LoFTR, outdoor weights (kornia)  roma     RoMa outdoor (romatch)
  dedode   DeDoDe-L detector + DeDoDe-B descriptor, SO(2) "spread" weights, matched with the C8 discretised steerer
           (max-similarity + dual softmax), i.e. the rotation-equivariant configuration recommended by the steerers authors
Usage: run_matchers.py <test|train|real_in|real_out> <matcher[,matcher...]> [limit]
"""
import sys, os, time, json, math
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
ROOT = '/root/rasr-pairuav'; sys.path[:0] = [f'{ROOT}/superglue_fusion', f'{ROOT}/reviewer_baselines']
OUT = f'{ROOT}/outputs/supp/trackB'; DATA = '/root/pairUAV-5D-temp'; W0, H0 = 3686, 2073; MAXM = 6000
SETTINGS = {
    'splg': dict(extractor='SuperPoint (LightGlue repo)', max_num_keypoints=2048, resize_long_side=1024, matcher='LightGlue(superpoint)', depth_confidence=-1, width_confidence=-1, filter_threshold=0.1, weights='github.com/cvg/LightGlue releases v0.1_arxiv'),
    'aliked': dict(extractor='ALIKED-n16 (LightGlue repo)', max_num_keypoints=2048, resize_long_side=1024, matcher='LightGlue(aliked)', depth_confidence=-1, width_confidence=-1, filter_threshold=0.1, weights='github.com/cvg/LightGlue releases v0.1_arxiv'),
    'loftr': dict(model='kornia.feature.LoFTR(pretrained="outdoor")', input='grayscale 1152 x 648 (648 x 1152 when rotated)', coarse_match_threshold=0.2, weights='loftr_outdoor.ckpt (kornia mirror of the official checkpoint)'),
    'roma': dict(model='romatch.roma_outdoor (romatch 0.1.2, use_custom_corr=False)', coarse_res=560, upsample_res=864, symmetric=True, sampled_matches=5000, weights='roma_outdoor.pth + DINOv2 ViT-L/14 (official release)'),
    'dedode': dict(model='kornia.feature.DeDoDe.from_pretrained("L-SO2", "B-SO2")', input='RGB 1120 x 630 (630 x 1120 when rotated)', keypoints=10000, steerer='SO2 generator discretised to C8 (kornia DiscreteSteerer.create_dedode_default("SO2", 8))',
                   matching='max similarity over the 8 steered descriptions, dual softmax, inv_temp=20, threshold=0.01, mutual nearest neighbours', weights='rotation-steerers release-2 (setting C)'),
}


def pair_set(name):
    if name in ('test', 'train'):
        from rb_recon import subset
        T = subset(1000 if name == 'test' else 300, split=name); return pd.DataFrame(dict(id=T.manifest_index.values, a=[f'{DATA}/{p}' for p in T.image_a], b=[f'{DATA}/{p}' for p in T.image_b]))
    if name == 'unscreened':   # Track D: 4 directed pairs per test building, uniform over all camera pairs (not passed through the SIFT screen of the release)
        U = pd.read_pickle(f'{ROOT}/outputs/supp/trackD/cache/unfiltered_pairs.pkl').reset_index(drop=True); return pd.DataFrame(dict(id=U.index.values, a=U.image_a.values, b=U.image_b.values))
    if name == 'test_full':   # all 6,364 test pairs behind Table 2 (used for the CPU-only SIFT baseline)
        T = pd.read_pickle(f'{ROOT}/outputs/supp/common/T.pkl'); return pd.DataFrame(dict(id=T.manifest_index.values, a=[f'{DATA}/{p}' for p in T.image_a], b=[f'{DATA}/{p}' for p in T.image_b]))
    D = pd.read_pickle('/root/realdata/mezieres/pred.pkl'); m = (D.dist_equiv <= 38) if name == 'real_in' else (D.dist_equiv > 38); d = D[m]
    return pd.DataFrame(dict(id=d.index.values, a=d.image_a.values, b=d.image_b.values))


class Pairs(torch.utils.data.Dataset):
    def __init__(self, df): self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        cv2.setNumThreads(1); r = self.df.iloc[i]; f = lambda p: torch.from_numpy(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_2), cv2.COLOR_BGR2RGB))
        return int(r.id), f(r.a), f(r.b)


def unrotate(xy, k, H, W):
    """Index coordinates in np.rot90(img, k) (counter-clockwise) -> index coordinates in img (H x W). Exact for sub-pixel positions."""
    x, y = xy[:, 0], xy[:, 1]
    if k % 4 == 0: return xy
    if k % 4 == 1: return np.stack([W - 1 - y, x], 1)
    if k % 4 == 2: return np.stack([W - 1 - x, H - 1 - y], 1)
    return np.stack([y, H - 1 - x], 1)


def to_res(xy, w_from, h_from, w_to, h_to):
    """Index coordinates in a (w_from x h_from) image -> index coordinates in the same image resampled to (w_to x h_to)."""
    return np.stack([(xy[:, 0] + 0.5) * w_to / w_from - 0.5, (xy[:, 1] + 0.5) * h_to / h_from - 0.5], 1)


# ------------------------------------------------------------------ matcher wrappers: (a_rgb uint8 HxWx3, b_rgb uint8 (rotated), cache) -> xy_a, xy_b (index coords of the arrays given), conf
class LG:
    def __init__(self, kind, dev):
        from lightglue import LightGlue, SuperPoint, ALIKED
        self.ext = (SuperPoint if kind == 'superpoint' else ALIKED)(max_num_keypoints=2048).eval().to(dev); self.m = LightGlue(features=kind, depth_confidence=-1, width_confidence=-1).eval().to(dev); self.dev = dev
    def feats(self, im): return self.ext.extract(torch.from_numpy(np.ascontiguousarray(im)).to(self.dev).permute(2, 0, 1).float() / 255.0, resize=1024)
    @torch.inference_mode()
    def __call__(self, a, b, cache):
        if 'fa' not in cache: cache['fa'] = self.feats(a)
        fa, fb = cache['fa'], self.feats(b); o = self.m({'image0': fa, 'image1': fb}); m = o['matches'][0]; s = o['scores'][0]
        return fa['keypoints'][0][m[:, 0]].float().cpu().numpy(), fb['keypoints'][0][m[:, 1]].float().cpu().numpy(), s.float().cpu().numpy()


class Loftr:
    def __init__(self, dev):
        import kornia.feature as KF
        self.m = KF.LoFTR(pretrained='outdoor').eval().to(dev); self.dev = dev
    def prep(self, im):
        h, w = im.shape[:2]; s = (1152, 648) if w >= h else (648, 1152); g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), s, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(g).to(self.dev).float()[None, None] / 255.0, s
    @torch.inference_mode()
    def __call__(self, a, b, cache):
        if 'ta' not in cache: cache['ta'] = self.prep(a)
        (ta, sa), (tb, sb) = cache['ta'], self.prep(b); o = self.m({'image0': ta, 'image1': tb})
        ka, kb, c = o['keypoints0'].float().cpu().numpy(), o['keypoints1'].float().cpu().numpy(), o['confidence'].float().cpu().numpy()
        return to_res(ka, sa[0], sa[1], a.shape[1], a.shape[0]), to_res(kb, sb[0], sb[1], b.shape[1], b.shape[0]), c


class Dedode:
    def __init__(self, dev, n=10000):
        import kornia.feature as KF
        self.m = KF.DeDoDe.from_pretrained(detector_weights='L-SO2', descriptor_weights='B-SO2', amp_dtype=torch.float16).eval().to(dev)
        self.st = KF.steerers.DiscreteSteerer.create_dedode_default(generator_type='SO2', steerer_order=8).to(dev); self.dev = dev; self.n = n
    def feats(self, im):
        h, w = im.shape[:2]; s = (1120, 630) if w >= h else (630, 1120); t = torch.from_numpy(cv2.resize(im, s, interpolation=cv2.INTER_AREA)).to(self.dev).permute(2, 0, 1).float()[None] / 255.0
        kp, sc, de = self.m(t, n=self.n); return kp[0].float() - 0.5, F.normalize(de[0].float(), dim=-1), s      # kornia returns pixel i at i + 0.5
    @torch.inference_mode()
    def __call__(self, a, b, cache):
        if 'fa' not in cache: cache['fa'] = self.feats(a)
        (ka, da, sa), (kb, db, sb) = cache['fa'], self.feats(b); corr = None; d = da
        for p in range(8):                      # official MaxSimilarityMatcher: steer the descriptions of A, keep the best similarity
            if p: d = F.normalize(self.st(d), dim=-1)
            c = 20.0 * (d @ db.T); corr = c if corr is None else torch.maximum(corr, c)
        P = corr.softmax(dim=-2) * corr.softmax(dim=-1); del corr
        ia, ib = torch.nonzero((P == P.max(dim=-1, keepdim=True).values) & (P == P.max(dim=-2, keepdim=True).values) & (P > 0.01), as_tuple=True); conf = P[ia, ib]; del P
        return to_res(ka[ia].cpu().numpy(), sa[0], sa[1], a.shape[1], a.shape[0]), to_res(kb[ib].cpu().numpy(), sb[0], sb[1], b.shape[1], b.shape[0]), conf.float().cpu().numpy()


class Roma:
    def __init__(self, dev):
        from romatch import roma_outdoor
        self.m = roma_outdoor(device=dev, use_custom_corr=False); self.dev = dev      # pure-PyTorch local correlation (the optional compiled `local_corr` extension is not installed)
    @torch.inference_mode()
    def __call__(self, a, b, cache):
        from PIL import Image
        if 'pa' not in cache: cache['pa'] = Image.fromarray(a)
        pa, pb = cache['pa'], Image.fromarray(np.ascontiguousarray(b)); warp, cert = self.m.match(pa, pb, device=self.dev); m, c = self.m.sample(warp, cert, num=5000)
        ka, kb = self.m.to_pixel_coordinates(m, a.shape[0], a.shape[1], b.shape[0], b.shape[1])
        return ka.float().cpu().numpy() - 0.5, kb.float().cpu().numpy() - 0.5, c.float().cpu().numpy()      # continuous coords (pixel i spans [i, i+1)) -> index coords


def build(name, dev): return {'splg': lambda: LG('superpoint', dev), 'aliked': lambda: LG('aliked', dev), 'loftr': lambda: Loftr(dev), 'dedode': lambda: Dedode(dev), 'roma': lambda: Roma(dev)}[name]()


def main():
    which = sys.argv[1]; names = sys.argv[2].split(','); df = pair_set(which).sample(frac=1.0, random_state=0)      # fixed random order: any prefix is a random subset
    if len(sys.argv) > 3: df = df.iloc[:int(sys.argv[3])]
    sh = [int(x) for x in os.environ.get('SHARD', '0/1').split('/')]; df = df.iloc[sh[0]::sh[1]]
    ROTS = [int(x) for x in os.environ.get('ROTS', '0,1,2,3').split(',')]      # rotations of the target image to run
    dev = 'cuda'; M = {n: build(n, dev) for n in names}; q = lambda v: np.clip(np.rint(4 * v), 0, 65535).astype(np.uint16)
    for n in names: os.makedirs(f'{OUT}/matches/{n}/{which}', exist_ok=True)
    todo = df[[not all(os.path.exists(f'{OUT}/matches/{n}/{which}/{i}.npz') for n in names) for i in df.id]]; print(f'{which}: {len(todo)} / {len(df)} pairs to do for {names}', flush=True)
    dl = torch.utils.data.DataLoader(Pairs(todo), batch_size=None, num_workers=int(os.environ.get('LOADERS', '2')), prefetch_factor=2); T = {n: [] for n in names}; t00 = time.time()
    for it, (idx, a, b) in enumerate(dl):
        a = a.numpy(); b = b.numpy(); Hh, Wh = b.shape[:2]
        for n in names:
            f = f'{OUT}/matches/{n}/{which}/{idx}.npz'
            if os.path.exists(f): continue
            cache = {}; out = {}; tk = []
            for k in ROTS:
                torch.cuda.synchronize(); t0 = time.time()
                try:
                    for attempt in range(40):       # the GPU is shared: on out-of-memory free the cache, wait and retry
                        try: xa, xb, c = M[n](a, np.ascontiguousarray(np.rot90(b, k)), cache); break      # each method runs in its own default precision
                        except torch.cuda.OutOfMemoryError:
                            cache.clear(); torch.cuda.empty_cache(); print(f'[{n}] pair {idx} k={k}: out of memory, retry {attempt + 1}', flush=True); time.sleep(30); t0 = time.time()
                    else: print('out of memory after 40 retries: giving up on this run (nothing written for this pair)', flush=True); os._exit(3)
                except Exception as e:
                    print(f'[{n}] pair {idx} k={k}: {type(e).__name__}: {str(e)[:120]}', flush=True); xa = np.zeros((0, 2)); xb = np.zeros((0, 2)); c = np.zeros(0)
                torch.cuda.synchronize(); tk.append(time.time() - t0)
                if len(c) > MAXM: o = np.argsort(-c)[:MAXM]; xa, xb, c = xa[o], xb[o], c[o]
                xb = unrotate(np.asarray(xb, np.float64), k, Hh, Wh)
                out[f'p0_{k}'] = q(to_res(np.asarray(xa, np.float64), a.shape[1], a.shape[0], W0, H0)); out[f'p1_{k}'] = q(to_res(xb, Wh, Hh, W0, H0)); out[f'c_{k}'] = np.clip(np.rint(255 * np.asarray(c, np.float64)), 0, 255).astype(np.uint8)
            np.savez_compressed(f, **out); T[n].append(tk)
        torch.cuda.empty_cache()      # keep the reserved pool small: the GPU is shared
        if (it + 1) % 50 == 0: print(f'{it+1}/{len(todo)} {(time.time()-t00)/(it+1):.2f}s/pair | ' + ' '.join(f'{n}: k0 {np.mean([t[0] for t in T[n]]):.3f}s, 4 rot {np.mean([sum(t) for t in T[n]]):.3f}s' for n in names if T[n]), flush=True)
    for n in names:
        if T[n]:
            json.dump(dict(set=which, pairs=len(T[n]), plain_s=float(np.mean([t[0] for t in T[n][2:] or T[n]])), tta_s=float(np.mean([sum(t) for t in T[n][2:] or T[n]])), peak_gpu_gb=float(torch.cuda.max_memory_allocated() / 2**30), settings=SETTINGS[n]),
                      open(f'{OUT}/matches/{n}/timing_{which}.json', 'w'), indent=1)
    print('done', which, names, f'{time.time()-t00:.0f}s', flush=True)


if __name__ == '__main__':
    main()
