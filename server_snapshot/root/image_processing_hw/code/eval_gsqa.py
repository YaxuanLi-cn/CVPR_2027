"""Part 2: run all metrics on the GS-QA videos (64 synthesized videos, 8 references)."""
import json, os, sys, time
import numpy as np
from common import ROOT, read_video
from metrics import Metrics

GS = os.path.join(ROOT, "data/gsqa")
SCENES = {"antique": "IST", "flowers": "IST", "playground2": "IST", "statue": "IST",
          "m60": "T&T", "playground": "T&T", "train": "T&T", "truck": "T&T"}
# file suffix -> method name used in dmos.xlsx
METHODS = {"3dgs": "3DGS", "eagles": "EAGLES", "lightgaussian": "LightGaussian", "mipsplatting": "Mip-Splatting",
           "octree": "Octree-GS", "scaffold": "Scaffold-GS", "sog": "SOG (w/o SH)", "sog_sh": "SOG"}
# Viewing conditions of the subjective test: FullHD monitor at 3 picture heights -> 56.5 pixels per degree
PPD = 2 * 3 * 1080 * np.tan(np.radians(0.5))

if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "results/gsqa")
    os.makedirs(out_dir, exist_ok=True)
    M = Metrics(display="standard_fhd", ppd=PPD, flip_workers=6)
    print("ppd", M.ppd, flush=True)
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1      # per-frame metrics on every stride-th frame
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(SCENES)
    for scene, grp in SCENES.items():
        if scene not in only:
            continue
        ref, fps = read_video(f"{GS}/Reference Videos/{scene}_reference.mp4")
        for suf, meth in METHODS.items():
            dst = f"{out_dir}/{scene}__{suf}.json"
            if os.path.exists(dst):
                continue
            t0 = time.time()
            test, fps2 = read_video(f"{GS}/Synthesized Videos/{grp}/{scene}/{scene}_{suf}.mp4")
            assert test.shape == ref.shape and fps == fps2, (test.shape, ref.shape, fps, fps2)
            fr = M.frames(test[::stride], ref[::stride])
            vid = M.video(test, ref, fps)
            json.dump({"scene": scene, "group": grp, "method": meth, "fps": fps, "n_frames": len(test), "stride": stride,
                       "frames": {k: v.tolist() for k, v in fr.items()}, **vid}, open(dst, "w"))
            print(f"{scene:12s} {meth:14s} F={len(test)} fps={fps:.0f} " +
                  " ".join(f"{k}={v.mean():.4f}" for k, v in fr.items()) +
                  f" cvvdp_vid={vid['cvvdp_vid']:.3f} tpsnr={vid['tpsnr']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
