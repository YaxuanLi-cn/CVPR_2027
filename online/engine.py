"""Online DreamNav inference: run the full two-stage pipeline on arbitrary new image pairs.

The archived experiment scripts are batch jobs over the dataset manifest. This engine loads every model once and
pushes a list of (image_a, image_b) pairs through the same code (the original Dataset classes and geometry workers
are imported and reused, not re-implemented), using the fitted stores from online/train_stores.py.

Input images must be in the canonical camera of the pipeline: 3686x2073 px, intrinsics K0 (pose_from_matches.K0),
zero roll. Use online/canonical.py to bring other cameras into that form.
"""
import sys, os, math, time, shutil, tempfile, pickle, importlib.util, argparse
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd, cv2
ROOT = '/root/rasr-pairuav'
for p in (f'{ROOT}/superglue_fusion', f'{ROOT}/stage2_imagination', f'{ROOT}/online'):
    if p not in sys.path: sys.path.insert(0, p)
import plane_servo
import extract_geom, map_pose, extract_geom_zoom, imagine_geom          # CPU geometry workers (imported before the pool forks)
import fusion_core as fc
from fusion_lib import wrap
SGD = f'{ROOT}/outputs/pairuav5d_60k/superglue'; MODELS = f'{ROOT}/outputs/online_models'; W0, H0 = 3686, 2073
RESCALE = {'s1': 0.75, 's2': 0.5}     # xyz-norm rescaling exponents chosen on the validation tours in the archived runs


def _prior(r): return dict(h=float(r.n_heading), ta=float(np.clip(r.n_tilt_a, 0, 47)), tb=float(np.clip(r.n_tilt_b, 0, 47)), u=np.array([r.n_ux, r.n_uy, r.n_uz], float))


class Engine:
    def __init__(self, workers=16, loader_workers=6, device='cuda', verbose=True):
        self.dev = device; self.verbose = verbose; self.lw = loader_workers
        self.pool = get_context('fork').Pool(workers)                    # forked before CUDA is initialised
        import torch, torch.nn as nn, torch.nn.functional as F, timm, lightgbm as lgb
        self.torch = torch; self.F = F
        # ---- stage 0 priors
        s0 = pickle.load(open(f'{MODELS}/stage0.pkl', 'rb')); self.mfeats = s0['feats']; self.s0_lgb = {t: lgb.Booster(model_str=s) for t, s in s0['lgb'].items()}
        from stage0_mlp import Heads
        m = s0['mlp']; self.mlp_t = m['targets']; self.mlp = Heads(len(m['targets']), len(self.mfeats)).to(device); self.mlp.load_state_dict(m['state']); self.mlp.eval()
        self.mlp_mu, self.mlp_sd, self.mlp_ys = m['mu'].to(device), m['sd'].to(device), m['ys'].to(device)
        # ---- SuperPoint / SuperGlue modules first: their top-level package is called `models`, the same name CroCo (inside MASt3R) uses,
        #      so import them, then drop the name from sys.modules / sys.path before MASt3R is imported
        import run_superglue_rot as rot, run_superglue_zoom as zoom, imagine_match as imag
        self.rot, self.zoom, self.imag = rot, zoom, imag
        for k in [k for k in sys.modules if k == 'models' or k.startswith('models.')]: sys.modules['sg_' + k] = sys.modules.pop(k)
        sys.path[:] = [q for q in sys.path if q != '/root/PairUAV/baseline/SuperGlue']
        # ---- MASt3R statistics extractor (the baseline's 422-d descriptor)
        spec = importlib.util.spec_from_file_location('extract_features', f'{ROOT}/data_processing/03_extract_features.py'); self.fx = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.fx)
        self.rt = self.fx.load_runtime(Path('/root/mast3r_runtime'))
        self.mast3r = self.rt['AsymmetricMASt3R'].from_pretrained('/root/mast3r_runtime/models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth').to(device).eval().requires_grad_(False)
        # ---- SuperPoint / SuperGlue (frozen, outdoor weights)
        mk = lambda n: rot.SuperPoint({'nms_radius': 4 if n <= 1024 else 3, 'keypoint_threshold': 0.005, 'max_keypoints': n}).eval().to(device)
        self.sp_c, self.sp_f = mk(768), mk(2048); self.sg = rot.SuperGlue({'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}).eval().to(device)
        # ---- fusion stores
        self.st1 = fc.Store(f'{MODELS}/stage1.pkl'); self.st2 = fc.Store(f'{MODELS}/stage2.pkl'); self.Hcam = self.st2.d['s2/H0']
        # ---- absolute-heading networks (single image): CNN v2 on the raw view, and the nadir-rectified (BEV) one
        class Net(nn.Module):
            def __init__(s):
                super().__init__(); s.b = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=False, num_classes=0, drop_path_rate=0.2); s.h = nn.Linear(s.b.num_features, 3)
            def forward(s, x): o = s.h(s.b(x)); return F.normalize(o[:, :2], dim=1), F.softplus(o[:, 2]) + 1e-3
        self.h_v2 = Net().to(device).to(memory_format=torch.channels_last); self.h_v2.load_state_dict(torch.load(f'{SGD}/absheading/model_cnx_s_d8.pt', map_location=device)); self.h_v2.eval()
        self.h_bev = Net().to(device).to(memory_format=torch.channels_last); self.h_bev.load_state_dict(torch.load(f'{SGD}/stage2/model_bev_s.pt', map_location=device)); self.h_bev.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255; self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

    def log(self, *a):
        if self.verbose: print(*a, flush=True)

    # ------------------------------------------------------------------ pieces
    def _mast3r_feats(self, df):
        self.rt['probe_ref']._prepared.cache_clear(); rows = []; dt = self.torch.float16
        recs = [dict(manifest_index=int(r.manifest_index), scene_id='x', pair_id=str(r.manifest_index), json_rel_path='', image_a=r.image_a, image_b=r.image_b, x_num=0.0, y_num=0.0, z_num=0.0,
                     heading_num=0.0, range_num=0.0, tilt_num=0.0, image_a_path=Path(r.image_a), image_b_path=Path(r.image_b)) for r in df.itertuples()]
        for i in range(0, len(recs), 4): rows += self.fx.process_batch(recs[i:i + 4], self.rt, self.mast3r, self.torch.device(self.dev), dt, 512, 16)
        M = pd.DataFrame(rows); bad = ~M.is_valid.astype(bool)
        if bad.any(): raise RuntimeError(f'MASt3R failed on {int(bad.sum())} pairs: {M.failure_reason[bad].iloc[0]}')
        return M[['manifest_index'] + self.mfeats].astype({c: np.float32 for c in self.mfeats})

    def _stage0(self, M):
        X = M[self.mfeats].values.astype(np.float32); o = pd.DataFrame({'manifest_index': M.manifest_index.values})
        for t, b in self.s0_lgb.items(): o['m_' + t] = b.predict(X)
        o['m_heading'] = np.rad2deg(np.arctan2(o.m_hsin, o.m_hcos)); o['m_hconf'] = np.hypot(o.m_hsin, o.m_hcos); o['m_dist'] = np.exp(o.m_logd)
        with self.torch.inference_mode(): p = (self.mlp((self.torch.tensor(X, device=self.dev) - self.mlp_mu) / self.mlp_sd) * self.mlp_ys).cpu().numpy()
        for j, t in enumerate(self.mlp_t): o['n_' + t] = p[:, j]
        o['n_heading'] = np.rad2deg(np.arctan2(o.n_hsin, o.n_hcos)); o['n_logd'] = np.log(o.n_dist.clip(lower=0.3)); return o

    def _loader(self, ds): return self.torch.utils.data.DataLoader(ds, batch_size=None, num_workers=min(self.lw, max(1, len(ds))), prefetch_factor=2)

    def _pass1(self, df, out):
        torch = self.torch; dev = self.dev; coarse, fine = (640, 360), (1280, 720); sc_f = np.array([W0 / fine[0], H0 / fine[1]], np.float32); sc_c = np.array([W0 / coarse[0], H0 / coarse[1]], np.float32)
        d = df.assign(tour='', pair_id='')
        for idx, tour, pid, ims_a, ims_b in self._loader(self.rot.Pairs(d, [coarse, fine])):
            f = lambda t: (t.to(dev).float() / 255.0)[None, None]
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                cs, _ = self.rot.match_rot(self.sp_c, self.sg, f(ims_a[0]), f(ims_b[0]), [0, 1, 2, 3])
                cnt = [int(len(c[0])) for c in cs]; csum = [float(c[2].sum()) for c in cs]; order = np.argsort(csum)[::-1]; kbest = int(order[0])
                ks = [kbest] + ([int(order[1])] if csum[order[1]] >= 0.5 * csum[kbest] else [])
                fn, _ = self.rot.match_rot(self.sp_f, self.sg, f(ims_a[1]), f(ims_b[1]), ks)
            P0, P1, C, S = [], [], [], []
            for k, (k0, k1, conf) in zip(ks, fn): P0.append(k0.float().cpu().numpy() * sc_f); P1.append(k1.float().cpu().numpy() * sc_f); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), 10 + k))
            for k, (k0, k1, conf) in enumerate(cs): P0.append(k0.float().cpu().numpy() * sc_c); P1.append(k1.float().cpu().numpy() * sc_c); C.append(conf.float().cpu().numpy()); S.append(np.full(len(k0), k))
            np.savez(f'{out}/{idx}.npz', p0=np.concatenate(P0).astype(np.float32), p1=np.concatenate(P1).astype(np.float32), conf=np.concatenate(C).astype(np.float16), src=np.concatenate(S).astype(np.int8),
                     rot=kbest, rot_counts=np.array(cnt, dtype=np.int32), fine_counts=np.array([len(x[0]) for x in fn], dtype=np.int32))

    def _match_tiles(self, ia, ib, mask):
        """SuperPoint + SuperGlue on one aligned tile pair; keypoints of image b are restricted to the valid (warped-in) area."""
        torch = self.torch
        with torch.autocast('cuda', dtype=torch.float16):
            fa = self.sp_f({'image': ia}); fb = self.sp_f({'image': ib}); kb = fb['keypoints'][0]; keep = mask[kb[:, 1].long(), kb[:, 0].long()] > 0
            if keep.sum() < 2 or len(fa['keypoints'][0]) < 2: return None
            d = {'image0': ia, 'image1': ib, 'keypoints0': fa['keypoints'][0][None], 'scores0': fa['scores'][0][None], 'descriptors0': fa['descriptors'][0][None],
                 'keypoints1': kb[keep][None], 'scores1': fb['scores'][0][keep][None], 'descriptors1': fb['descriptors'][0][:, keep][None]}
            o = self.sg(d); m = o['matches0'][0]; ok = m > -1
        return d['keypoints0'][0][ok].float().cpu().numpy().astype(np.float64), d['keypoints1'][0][m[ok]].float().cpu().numpy().astype(np.float64), o['matching_scores0'][0][ok].float().cpu().numpy()

    @staticmethod
    def _back(k, Hm):
        if not len(k): return k
        p = np.c_[k, np.ones(len(k))] @ np.asarray(Hm).T; return p[:, :2] / p[:, 2:3]

    def _zoom(self, df, out):
        torch = self.torch; dev = self.dev
        with torch.inference_mode():
            for idx, ia, ib, mask, Zi, ZHi, src, s in self._loader(self.zoom.Pairs(df, 300)):
                f = lambda t: (t.to(dev).float() / 255.0)[None, None]; r = self._match_tiles(f(ia), f(ib), mask.to(dev))
                k0, k1, conf = r if r is not None else (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, np.float32))
                np.savez(f'{out}/{idx}.npz', p0=self._back(k0, Zi).astype(np.float32), p1=self._back(k1, ZHi).astype(np.float32), conf=conf.astype(np.float16), guide_src=int(src), zoom=float(s))

    def _imagine(self, df, out, rot_only=False):
        torch = self.torch; dev = self.dev; a = argparse.Namespace(rot_only=rot_only, pad=150, scale=1.0, max_grid=3)
        with torch.inference_mode():
            for idx, TA, TB, MK, ZI, ZHI, s, n in self._loader(self.imag.Pairs(df, a)):
                P0, P1, CF, TI, DS = [np.zeros((0, 2))], [np.zeros((0, 2))], [np.zeros(0)], [np.zeros(0)], [np.zeros((0, 2))]
                for ti in range(0 if TA is None else len(TA)):
                    f = lambda t: (t.to(dev).float() / 255.0)[None, None]; r = self._match_tiles(f(TA[ti]), f(TB[ti]), MK[ti].to(dev))
                    if r is None: continue
                    k0, k1, conf = r; P0.append(self._back(k0, ZI[ti])); P1.append(self._back(k1, ZHI[ti])); CF.append(conf); TI.append(np.full(len(k0), ti)); DS.append((k1 - k0) / s)
                np.savez(f'{out}/{idx}.npz', p0=np.concatenate(P0).astype(np.float32), p1=np.concatenate(P1).astype(np.float32), conf=np.concatenate(CF).astype(np.float16),
                         tile=np.concatenate(TI).astype(np.int8), disp=np.concatenate(DS).astype(np.float32), zoom=float(s), grid=int(n), ntiles=0 if TA is None else len(TA))

    # ---- absolute heading
    def _small(self, paths):
        def load(p): im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4); return cv2.cvtColor(cv2.resize(im, (448, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return self.torch.from_numpy(np.stack([load(p) for p in paths])).to(self.dev).permute(0, 3, 1, 2).float()

    def _heading_v2(self, x):
        torch = self.torch; n = len(x); s = torch.zeros(n, device=self.dev); c = torch.zeros(n, device=self.dev)
        with torch.inference_mode():
            for x0 in (0, 96, 192):
                xc = (x[:, :, :, x0:x0 + 256] - self.mean) / self.std
                for flip in (False, True):
                    xf = torch.flip(xc, dims=[3]) if flip else xc
                    for k in range(4):
                        with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = self.h_v2(torch.rot90(xf, k, (2, 3)).contiguous(memory_format=torch.channels_last))
                        ang = torch.atan2(mu[:, 0].float(), mu[:, 1].float()) - k * math.pi / 2
                        if flip: ang = -ang
                        s += kap.float() * torch.sin(ang); c += kap.float() * torch.cos(ang)
        s, c = (s / 24).cpu().numpy(), (c / 24).cpu().numpy(); return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)

    def _bev(self, x, tilt, delta, flip):
        """Imagined nadir view rotated by `delta` (port of bev_heading.imagine_bev, rectified variant)."""
        torch = self.torch; F = self.F; dev = self.dev; B = len(x); z = torch.zeros(B, device=dev); WC, HC, S = 448, 256, 256
        FX, FY, CX, CY = 3324.9 * WC / 3686, 3324.9 * HC / 2073, 1841.57 * WC / 3686, 1078.04 * HC / 2073; FV = 212.0 / 0.6
        def R_wc(h, t):
            zz = torch.zeros_like(h); right = torch.stack([torch.cos(h), -torch.sin(h), zz], 1)
            fwd = torch.stack([torch.sin(t) * torch.sin(h), torch.sin(t) * torch.cos(h), -torch.cos(t)], 1); up = torch.stack([torch.cos(t) * torch.sin(h), torch.cos(t) * torch.cos(h), torch.sin(t)], 1)
            return torch.stack([right, -up, fwd], 2)
        x = torch.where(flip.view(B, 1, 1, 1), torch.flip(x, dims=[3]), x); cx = torch.where(flip, WC - CX, CX)
        Rt = R_wc(z, tilt); Rn = R_wc(delta, z); M = Rt.transpose(1, 2) @ Rn; g = Rn.transpose(1, 2) @ Rt[:, :, 2:3]; c = g[:, :2, 0] / g[:, 2:3, 0]; fv = FV * torch.cos(tilt)
        u = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5 - S / 2); yy, xx = torch.meshgrid(u, u, indexing='ij')
        rn = torch.stack([xx[None] / fv.view(B, 1, 1) + c[:, 0].view(B, 1, 1), yy[None] / fv.view(B, 1, 1) + c[:, 1].view(B, 1, 1), torch.ones(B, S, S, device=dev)], -1)
        rc = rn @ M.transpose(1, 2).unsqueeze(1); zc = rc[..., 2].clamp(min=1e-6); px = FX * rc[..., 0] / zc + cx.view(B, 1, 1); py = FY * rc[..., 1] / zc + CY
        grid = torch.stack([2 * px / WC - 1, 2 * py / HC - 1], -1); grid = torch.where((rc[..., 2:3] > 1e-6), grid, torch.full_like(grid, -5.0))
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def _heading_bev(self, x, tilt_deg, krot=12):
        torch = self.torch; n = len(x); tl = torch.tensor(np.deg2rad(tilt_deg), dtype=torch.float32, device=self.dev); s = torch.zeros(n, device=self.dev); c = torch.zeros(n, device=self.dev)
        with torch.inference_mode():
            for fl in (False, True):
                for k in range(krot):
                    d = torch.full((n,), 2 * math.pi * k / krot, device=self.dev)
                    xt = ((self._bev(x, tl, d, torch.full((n,), fl, device=self.dev, dtype=torch.bool)) - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
                    with torch.autocast('cuda', dtype=torch.bfloat16): mu, kap = self.h_bev(xt)
                    ang = torch.atan2(mu[:, 0].float(), mu[:, 1].float()) - d
                    if fl: ang = -ang
                    s += kap.float() * torch.sin(ang); c += kap.float() * torch.cos(ang)
        s, c = (s / (2 * krot)).cpu().numpy(), (c / (2 * krot)).cpu().numpy(); return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)

    # ------------------------------------------------------------------ full pipeline
    def predict(self, pairs, stage2=True, abs_heading=True, keep_dir=None, imag_rot_only=False, priors_only=False, plane=False):
        """pairs: list of (path_a, path_b) or a DataFrame with image_a / image_b (absolute paths, canonical camera).
        Returns one row per pair, in input order: s1_* (coarse), s2_* (refined), control predictions, diagnostics."""
        t0 = time.time(); T = {}
        df = pairs.copy() if isinstance(pairs, pd.DataFrame) else pd.DataFrame(pairs, columns=['image_a', 'image_b'])
        df = df.reset_index(drop=True); df['manifest_index'] = np.arange(len(df)); df['image_a'] = df.image_a.astype(str); df['image_b'] = df.image_b.astype(str)
        wd = keep_dir or tempfile.mkdtemp(prefix='dreamnav_', dir='/dev/shm'); os.makedirs(wd, exist_ok=True)
        for s in ('rot', 'zoom', 'imag'): os.makedirs(f'{wd}/{s}', exist_ok=True)
        try:
            def lap(k):
                nonlocal t0; self.torch.cuda.synchronize(); T[k] = time.time() - t0; t0 = time.time()
            M = self._mast3r_feats(df); S0 = self._stage0(M); lap('mast3r_priors')
            if priors_only: return df[['manifest_index', 'image_a', 'image_b']].merge(S0, on='manifest_index').sort_values('manifest_index').reset_index(drop=True)
            self._pass1(df, f'{wd}/rot'); lap('match_pass1')
            ids = df.manifest_index.tolist(); f = lambda sub, i: f'{wd}/{sub}/{i}.npz'
            G = pd.DataFrame(self.pool.map(extract_geom.work, [f('rot', i) for i in ids], chunksize=2)); N = S0.set_index('manifest_index'); Gd = G.set_index('manifest_index')
            cols = ['sg_ok', 'sg_dh', 'sg_ta', 'sg_tb', 'sg_t0', 'sg_t1', 'sg_t2', 'sg_vx', 'sg_vy', 'sg_vz', 'sg_prim_fine']
            MP = pd.DataFrame(self.pool.map(map_pose.work, [(f('rot', i), Gd.loc[i, cols].to_dict(), _prior(N.loc[i])) for i in ids], chunksize=2)); lap('pose_pass1')
            dz = df.merge(G, on='manifest_index').merge(MP, on='manifest_index').merge(S0, on='manifest_index')
            self._zoom(dz, f'{wd}/zoom'); lap('match_zoom')
            Z = pd.DataFrame(self.pool.map(extract_geom_zoom.work, [(f('zoom', i), _prior(N.loc[i])) for i in ids], chunksize=2)); lap('pose_zoom')
            D = df[['manifest_index']].merge(S0, on='manifest_index').merge(M, on='manifest_index').merge(G, on='manifest_index').merge(MP, on='manifest_index', how='left').merge(Z, on='manifest_index', how='left')
            s1 = fc.stage1(D, self.st1); lap('fusion1')
            out = df[['manifest_index', 'image_a', 'image_b']].merge(s1.rename(columns=lambda c: c.replace('p_', 's1_').replace('r_', 's1r_')), on='manifest_index')
            out = out.merge(S0[['manifest_index', 'n_x', 'n_y', 'n_z', 'n_lx', 'n_ly', 'n_tilt', 'n_heading', 'n_dist', 'n_tilt_a', 'n_tilt_b']], on='manifest_index')
            out = out.merge(G[['manifest_index', 'sg_ok', 'sg_inl', 'sg_dh', 'sg_dtilt', 'sg_vx', 'sg_vy', 'sg_vz', 'sg_rot']], on='manifest_index').merge(MP[['manifest_index', 'map_ok', 'map_ta', 'map_tb']], on='manifest_index') \
                     .merge(Z[['manifest_index', 'zm_ok', 'zm_inl', 'zmap_ok', 'zmap_ta', 'zmap_tb', 'zmap_hgta', 'zmap_dh', 'zmap_dtilt', 'zmap_vx', 'zmap_vy', 'zmap_vz', 'zmap_par']], on='manifest_index')
            # per-camera tilt estimate and guide pose (build_guide.py)
            fa = np.where(out.zmap_ok == 1, out.zmap_ta, np.where(out.map_ok == 1, out.map_ta, out.n_tilt_a)); fb = np.where(out.zmap_ok == 1, out.zmap_tb, np.where(out.map_ok == 1, out.map_tb, out.n_tilt_b))
            r = (out.s1_tilt.values - (fb - fa)) / 2; out['g_ta'] = np.clip(fa - r, 0, 47); out['g_tb'] = np.clip(fb + r, 0, 47); out['g_dh'] = out.s1_heading
            Hh = out.s1_dist * out.zmap_hgta; out['g_H'] = np.where(np.isfinite(Hh) & (out.zmap_ok == 1) & (out.zm_inl > 100), np.clip(Hh, 0.75 * self.Hcam, 1.3 * self.Hcam), self.Hcam)
            if stage2:
                gd = out[['manifest_index', 'image_a', 'image_b', 'g_dh', 'g_ta', 'g_tb', 'g_H']].assign(p_lx=out.s1_lx, p_ly=out.s1_ly, p_z=out.s1_z, p_dist=out.s1_dist)
                self._imagine(gd, f'{wd}/imag', rot_only=imag_rot_only); lap('match_imagine')
                Zd = Z.set_index('manifest_index'); Md = MP.set_index('manifest_index'); U = gd.set_index('manifest_index'); jobs = []
                for i in ids:
                    g, m, z, u = Gd.loc[i], Md.loc[i], Zd.loc[i], U.loc[i]; fu = np.array([u.p_lx, u.p_ly, u.p_z]); fu = fu / (np.linalg.norm(fu) + 1e-9)
                    cands = dict(sg=(g.sg_dh, g.sg_ta, g.sg_tb, (g.sg_vx, g.sg_vy, g.sg_vz)), map=(m.map_dh, m.map_ta, m.map_tb, (m.map_vx, m.map_vy, m.map_vz)), zm=(z.zm_dh, z.zm_ta, z.zm_tb, (z.zm_vx, z.zm_vy, z.zm_vz)),
                                 zmap=(z.zmap_dh, z.zmap_ta, z.zmap_tb, (z.zmap_vx, z.zmap_vy, z.zmap_vz)), s1=(u.g_dh, u.g_ta, u.g_tb, tuple(fu)))
                    jobs.append((f('imag', i), _prior(N.loc[i]), cands))
                I = pd.DataFrame(self.pool.map(imagine_geom.work, jobs, chunksize=2)); lap('pose_imagine')
                D2 = D.merge(s1[['manifest_index', 'p_heading', 'p_tilt', 'p_dist', 'p_z', 'p_lx', 'p_ly']].rename(columns=lambda c: c.replace('p_', 's1_')), on='manifest_index').merge(I, on='manifest_index')
                s2 = fc.stage2(D2, self.st2); lap('fusion2')
                out = out.merge(s2.rename(columns=lambda c: c.replace('p_', 's2_').replace('r_', 's2r_')), on='manifest_index').merge(I[['manifest_index', 'im_ok', 'im_inl', 'imap_ok', 'im_disp_med', 'im_ntiles']], on='manifest_index')
            if stage2 and plane:   # ground-plane translation (in camera heights) with the rotation fixed to the refined estimate; see plane_servo.py
                ia = I.set_index('manifest_index').loc[out.manifest_index]; ta = np.where(ia.imap_ok.values == 1, ia.imap_ta.values, out.g_ta.values); tb = np.where(ia.imap_ok.values == 1, ia.imap_tb.values, out.g_tb.values)
                r = (out.s2_tilt.values - (tb - ta)) / 2; ta = np.clip(ta - r, 0, 47); tb = np.clip(tb + r, 0, 47)
                pw = np.array(self.pool.map(plane_servo.work, [(f('imag', i), float(h), float(a_), float(b_)) for i, h, a_, b_ in zip(out.manifest_index, out.s2_heading, ta, tb)], chunksize=2))
                out['pl_wx'], out['pl_wy'], out['pl_wz'], out['pl_inl'] = pw[:, 0], pw[:, 1], pw[:, 2], pw[:, 3]; lap('plane')
            if abs_heading:
                xa, xb = self._small(out.image_a.tolist()), self._small(out.image_b.tolist())
                for s, x, tl in (('a', xa, out.g_ta.values), ('b', xb, out.g_tb.values)):
                    out[f'v2h_{s}'], out[f'v2k_{s}'] = self._heading_v2(x); out[f'bvh_{s}'], out[f'bvk_{s}'] = self._heading_bev(x, tl)
                def comb(src, dh):
                    s = 0; c = 0
                    for p in src:
                        ta = np.deg2rad(out[f'{p}h_a']); tb = np.deg2rad(out[f'{p}h_b'] - dh); s = s + out[f'{p}k_a'] * np.sin(ta) + out[f'{p}k_b'] * np.sin(tb); c = c + out[f'{p}k_a'] * np.cos(ta) + out[f'{p}k_b'] * np.cos(tb)
                    return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)
                for pre, src in (('s1', ('v2',)),) + ((('s2', ('bv', 'v2')),) if stage2 else ()):
                    h, k = comb(src, out[f'{pre}_heading']); out[f'{pre}_absh'] = h; out[f'{pre}_absk'] = k; hr = np.deg2rad(h); lx, ly, z = out[f'{pre}_lx'], out[f'{pre}_ly'], out[f'{pre}_z']
                    x = np.cos(hr) * lx + np.sin(hr) * ly; y = -np.sin(hr) * lx + np.cos(hr) * ly; sc = (out[f'{pre}_dist'] / np.sqrt(x**2 + y**2 + z**2).clip(1e-6)) ** RESCALE[pre]
                    out[f'{pre}_x'] = x * sc; out[f'{pre}_y'] = y * sc; out[f'{pre}_zw'] = z * sc
                lap('abs_heading')
            out.attrs['timing'] = T; self.log(f'[engine] {len(df)} pairs: ' + ' '.join(f'{k}={v:.1f}s' for k, v in T.items()) + f' | {sum(T.values())/len(df):.2f} s/pair')
            return out.sort_values('manifest_index').reset_index(drop=True)
        finally:
            if keep_dir is None: shutil.rmtree(wd, ignore_errors=True)

    def close(self): self.pool.terminate()
