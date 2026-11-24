"""Controlled temporal-artefact experiment (reference camera path of the GS-QA 'truck' scene).

Two models trained with different random seeds have (almost) the same per-frame quality but differ in
their exact errors (floater positions, fine texture). Alternating between the two renderings every k frames
produces a video whose *frames* are, by construction, exactly as good as those of the two source videos,
but which flickers at 30/(2k) Hz. Any per-frame metric averaged over time cannot see this manipulation;
a spatio-temporal metric should.
"""
import glob, json, os
import numpy as np
from common import ROOT, load_scene, load_png
from metrics import Metrics
from eval_gsqa import PPD

SCENE = "truck"
CONDS = {"views16": ("q_views16", "q_views16_s1"), "full": ("q_full", "q_full_s1")}
STEP = "step_007000"
CONDS = {k: v for k, v in CONDS.items() if k in os.environ.get("COND", "views16,full")}
STRIDE = 7   # per-frame metrics on every 7th frame (coprime with the alternation periods)
PERIODS = [1, 2, 4, 8, 15]

if __name__ == "__main__":
    sc = load_scene(SCENE)
    ref = np.stack([sc.load_image(i) for i in sc.test_idx])
    M = Metrics(display="standard_fhd", ppd=PPD, flip_workers=6)
    out = []
    for cname, (ta, tb) in CONDS.items():
        fa_, fb_ = [sorted(glob.glob(f"{ROOT}/runs/{SCENE}/{t}/{STEP}/*.png")) for t in (ta, tb)]
        if len(fa_) != len(ref) or len(fb_) != len(ref):
            print("skip", cname, len(fa_), len(fb_)); continue
        A = np.stack([load_png(f) for f in fa_])
        B = np.stack([load_png(f) for f in fb_])
        if not (len(A) == len(B) == len(ref)):
            print("skip", cname, len(A), len(B)); continue
        ix = np.arange(0, len(A), STRIDE)
        fa, fb = M.frames(A[ix], ref[ix]), M.frames(B[ix], ref[ix])
        seqs = {"seed0": (A, None), "seed1": (B, None)}
        for k in PERIODS:
            pick = (np.arange(len(A)) // k) % 2          # 0 -> frame from A, 1 -> frame from B
            seqs[f"alt{k}"] = (np.where(pick[:, None, None, None] == 0, A, B), pick)
        # how different are the two seeds from each other (visibility of the alternation)?
        between = M.frames(A[ix], B[ix], which=["psnr", "flip", "cvvdp_img"])
        for sname, (V, pick) in seqs.items():
            if pick is None:
                fr = fa if sname == "seed0" else fb
            else:   # per-frame scores of an alternating video are a mix of already computed scores
                fr = {m: np.where(pick[ix] == 0, fa[m], fb[m]) for m in fa}
            vid = M.video(V, ref, sc.fps)
            row = {"cond": cname, "seq": sname, "period": 0 if pick is None else int(sname[3:]),
                   "flicker_hz": 0.0 if pick is None else sc.fps / (2 * int(sname[3:])),
                   **{m: float(v.mean()) for m, v in fr.items()}, **vid,
                   "between_psnr": float(between["psnr"].mean()), "between_flip": float(between["flip"].mean()),
                   "between_cvvdp": float(between["cvvdp_img"].mean())}
            out.append(row)
            print(row, flush=True)
        json.dump(out, open(f"{ROOT}/results/temporal_exp_{os.environ.get("COND", "all")}.json", "w"), indent=1)
    json.dump(out, open(f"{ROOT}/results/temporal_exp_{os.environ.get("COND", "all")}.json", "w"), indent=1)
