"""Track C: inference wrappers for the two archived absolute-heading networks (no training code).

raw-view network  : superglue_fusion/abs_heading_train2.py, 3 crops x 8 dihedral views = 24 test-time views
nadir-rectified   : stage2_imagination/bev_heading.py, 12 rotations x mirror = 24 imagined nadir views
Both functions take the cached network input (uint8 RGB, N x 256 x 448 x 3, i.e. the full image resized to 448 x 256) and
return the von Mises mean direction (degrees, clockwise from north) and concentration, exactly as online/engine.py does.
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
SG = '/root/rasr-pairuav/outputs/pairuav5d_60k/superglue'; RAW_W = f'{SG}/absheading/model_cnx_s_d8.pt'; BEV_W = f'{SG}/stage2/model_bev_s.pt'


class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.b = timm.create_model('convnext_small.fb_in22k_ft_in1k', pretrained=False, num_classes=0, drop_path_rate=0.2); s.h = nn.Linear(s.b.num_features, 3)
    def forward(s, x): o = s.h(s.b(x)); return F.normalize(o[:, :2], dim=1), F.softplus(o[:, 2]) + 1e-3


def load(path, dev='cuda'):
    n = Net().to(dev).to(memory_format=torch.channels_last); n.load_state_dict(torch.load(path, map_location=dev)); n.eval(); return n


class Heading:
    def __init__(s, raw=RAW_W, bev=BEV_W, dev='cuda', bs=48):
        import os
        cap = os.environ.get('TRACKC_MEMCAP_GB')        # hard cap of this process's PyTorch allocator (shared GPU): exceeding it fails here instead of starving other jobs
        if cap and dev == 'cuda': torch.cuda.set_per_process_memory_fraction(float(cap) * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        s.dev = dev; s.bs = bs; s.raw = load(raw, dev) if raw else None; s.bev = load(bev, dev) if bev else None
        s.mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255; s.std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
        s.amp = dict(device_type='cuda', dtype=torch.bfloat16) if dev == 'cuda' else dict(device_type='cpu', enabled=False)

    def _t(s, x): return torch.from_numpy(np.ascontiguousarray(x)).to(s.dev).permute(0, 3, 1, 2).float()

    @torch.inference_mode()
    def raw_view(s, X, tta=True, crop=96, k=0, flip=False):
        """tta=True: 24 views. tta=False: one view (centre crop by default), optionally mirrored / rotated by k*90 deg, WITHOUT undoing the transform."""
        S_, C_ = [], []
        for i in range(0, len(X), s.bs):
            x = s._t(X[i:i + s.bs]); n = len(x); sn = torch.zeros(n, device=s.dev); cs = torch.zeros(n, device=s.dev)
            if tta:
                for x0 in (0, 96, 192):
                    xc = (x[:, :, :, x0:x0 + 256] - s.mean) / s.std
                    for fl in (False, True):
                        xf = torch.flip(xc, dims=[3]) if fl else xc
                        for kk in range(4):
                            with torch.autocast(**s.amp): mu, kap = s.raw(torch.rot90(xf, kk, (2, 3)).contiguous(memory_format=torch.channels_last))
                            ang = torch.atan2(mu[:, 0].float(), mu[:, 1].float()) - kk * math.pi / 2
                            if fl: ang = -ang
                            sn += kap.float() * torch.sin(ang); cs += kap.float() * torch.cos(ang)
                sn /= 24; cs /= 24
            else:
                xc = (x[:, :, :, crop:crop + 256] - s.mean) / s.std
                if flip: xc = torch.flip(xc, dims=[3])
                with torch.autocast(**s.amp): mu, kap = s.raw(torch.rot90(xc, k, (2, 3)).contiguous(memory_format=torch.channels_last))
                sn = kap.float() * mu[:, 0].float(); cs = kap.float() * mu[:, 1].float()
            S_.append(sn.cpu()); C_.append(cs.cpu())
        sn, cs = torch.cat(S_).numpy(), torch.cat(C_).numpy(); return np.rad2deg(np.arctan2(sn, cs)), np.hypot(sn, cs)

    def _bev(s, x, tilt, delta, flip):
        """Imagined nadir view rotated by `delta` (same code as online/engine.py::_bev)."""
        dev = s.dev; B = len(x); z = torch.zeros(B, device=dev); WC, HC, S = 448, 256, 256
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

    @torch.inference_mode()
    def nadir(s, X, tilt_deg, krot=12):
        S_, C_ = [], []; tilt_deg = np.asarray(tilt_deg, np.float32)
        for i in range(0, len(X), s.bs):
            x = s._t(X[i:i + s.bs]); n = len(x); tl = torch.tensor(np.deg2rad(tilt_deg[i:i + s.bs]), dtype=torch.float32, device=s.dev); sn = torch.zeros(n, device=s.dev); cs = torch.zeros(n, device=s.dev)
            for fl in (False, True):
                for k in range(krot):
                    d = torch.full((n,), 2 * math.pi * k / krot, device=s.dev)
                    xt = ((s._bev(x, tl, d, torch.full((n,), fl, device=s.dev, dtype=torch.bool)) - s.mean) / s.std).contiguous(memory_format=torch.channels_last)
                    with torch.autocast(**s.amp): mu, kap = s.bev(xt)
                    ang = torch.atan2(mu[:, 0].float(), mu[:, 1].float()) - d
                    if fl: ang = -ang
                    sn += kap.float() * torch.sin(ang); cs += kap.float() * torch.cos(ang)
            S_.append((sn / (2 * krot)).cpu()); C_.append((cs / (2 * krot)).cpu())
        sn, cs = torch.cat(S_).numpy(), torch.cat(C_).numpy(); return np.rad2deg(np.arctan2(sn, cs)), np.hypot(sn, cs)


def combine(*hk):
    """Sum of von Mises natural parameters of several (heading_deg, kappa) estimates of the same angle."""
    s = sum(k * np.sin(np.deg2rad(h)) for h, k in hk); c = sum(k * np.cos(np.deg2rad(h)) for h, k in hk); return np.rad2deg(np.arctan2(s, c)), np.hypot(s, c)
