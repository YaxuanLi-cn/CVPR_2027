"""Full-reference metrics used in the study, with one consistent interface.

Per-frame (image) metrics : PSNR, SSIM, LPIPS (VGG and AlexNet), FLIP, ColorVideoVDP in image mode
Video metrics             : ColorVideoVDP in video mode (spatio-temporal), plus a simple temporal
                            residual diagnostic (tPSNR) that isolates frame-to-frame error changes.
All inputs are uint8 sRGB arrays (F,H,W,3). Higher is better for PSNR/SSIM/cvvdp; lower for LPIPS/FLIP.
"""
import math
import warnings
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

IMAGE_METRICS = ["psnr", "ssim", "lpips_vgg", "lpips_alex", "flip", "cvvdp_img"]
# +1: higher = better quality, -1: lower = better
SIGN = {"psnr": 1, "ssim": 1, "lpips_vgg": -1, "lpips_alex": -1, "flip": -1, "cvvdp_img": 1,
        "cvvdp_vid": 1, "tpsnr": 1}
LABEL = {"psnr": "PSNR", "ssim": "SSIM", "lpips_vgg": "LPIPS (VGG)", "lpips_alex": "LPIPS (Alex)",
         "flip": "FLIP", "cvvdp_img": "CVVDP (image)", "cvvdp_vid": "CVVDP (video)", "tpsnr": "tPSNR"}


def _flip_one(a):
    import flip_evaluator
    ref, test, ppd = a
    _, m, _ = flip_evaluator.evaluate(ref.astype(np.float32) / 255.0, test.astype(np.float32) / 255.0, "LDR",
                                      applyMagma=False, parameters={"ppd": ppd})
    return float(m)


class Metrics:
    def __init__(self, display="standard_4k", ppd=None, device="cuda", flip_workers=10):
        """display: ColorVideoVDP display name (photometry + geometry).
        ppd: if given, overrides the display geometry (pixels per visual degree) for cvvdp and FLIP."""
        import lpips
        import pycvvdp
        self.dev = device
        self.lp_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        self.lp_alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        geom = None
        if ppd is not None:
            geom = pycvvdp.vvdp_display_geometry([1920, 1080], ppd=ppd)
        self.cvvdp = pycvvdp.cvvdp(display_name=display, display_geometry=geom, quiet=True, device=torch.device(device),
                                    gpu_mem=3)  # cap the memory cvvdp plans for (GPU is shared)
        self.ppd = float(self.cvvdp.pix_per_deg if hasattr(self.cvvdp, "pix_per_deg") else
                         self.cvvdp.display_geometry.get_ppd())
        g = torch.tensor([math.exp(-((x - 5) ** 2) / (2 * 1.5 ** 2)) for x in range(11)])
        g = (g / g.sum())[:, None]
        self.win = (g @ g.t())[None, None].repeat(3, 1, 1, 1).to(device)
        self.flip_workers = flip_workers
        self._pool = None

    # ------------------------------------------------------------------ helpers
    def _ssim(self, a, b):
        w = self.win
        mu1, mu2 = F.conv2d(a, w, groups=3), F.conv2d(b, w, groups=3)
        s1 = F.conv2d(a * a, w, groups=3) - mu1 ** 2
        s2 = F.conv2d(b * b, w, groups=3) - mu2 ** 2
        s12 = F.conv2d(a * b, w, groups=3) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        return m.mean((1, 2, 3))

    def pool(self):
        if self._pool is None:
            self._pool = get_context("spawn").Pool(self.flip_workers)
        return self._pool

    # ------------------------------------------------------------------ per-frame metrics
    @torch.no_grad()
    def frames(self, test, ref, which=IMAGE_METRICS, chunk=2):
        """test, ref: uint8 (F,H,W,3). Returns dict name -> np.array(F)."""
        assert test.shape == ref.shape and test.dtype == np.uint8, (test.shape, ref.shape)
        out = {k: [] for k in which}
        flip_async = None
        if "flip" in which:
            flip_async = self.pool().map_async(_flip_one, [(ref[i], test[i], self.ppd) for i in range(len(ref))],
                                               chunksize=4)
        for a in range(0, len(test), chunk):
            t = torch.from_numpy(test[a:a + chunk]).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            r = torch.from_numpy(ref[a:a + chunk]).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
            if "psnr" in which:
                out["psnr"] += (-10 * torch.log10(((t - r) ** 2).mean((1, 2, 3)).clamp_min(1e-10))).tolist()
            if "ssim" in which:
                out["ssim"] += self._ssim(t, r).tolist()
            if "lpips_vgg" in which:
                out["lpips_vgg"] += self.lp_vgg(t * 2 - 1, r * 2 - 1).flatten().tolist()
            if "lpips_alex" in which:
                out["lpips_alex"] += self.lp_alex(t * 2 - 1, r * 2 - 1).flatten().tolist()
        if "cvvdp_img" in which:
            for i in range(len(test)):
                jod, _ = self.cvvdp.predict(torch.from_numpy(test[i]), torch.from_numpy(ref[i]), dim_order="HWC")
                out["cvvdp_img"].append(float(jod))
        if flip_async is not None:
            out["flip"] = flip_async.get()
        return {k: np.asarray(v, np.float64) for k, v in out.items()}

    # ------------------------------------------------------------------ video metrics
    @torch.no_grad()
    def video(self, test, ref, fps):
        """Spatio-temporal metrics on a whole sequence. test, ref: uint8 (F,H,W,3)."""
        jod, _ = self.cvvdp.predict(torch.from_numpy(test), torch.from_numpy(ref), dim_order="FHWC",
                                    frames_per_second=fps)
        # temporal residual: how much the error image changes between consecutive frames
        se, n = 0.0, 0
        for a in range(0, len(test) - 1, 16):
            t = torch.from_numpy(test[a:a + 17]).to(self.dev).float() / 255.0
            r = torch.from_numpy(ref[a:a + 17]).to(self.dev).float() / 255.0
            e = t - r
            de = e[1:] - e[:-1]
            se += float((de ** 2).sum())
            n += de.numel()
        return {"cvvdp_vid": float(jod), "tpsnr": -10 * math.log10(max(se / n, 1e-12))}
