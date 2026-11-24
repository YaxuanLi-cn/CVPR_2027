"""Part 1: metrics for every rendered condition (runs/<scene>/<tag>/step_*).
Held-out images (Mip-NeRF 360) get per-image metrics; reference camera paths (GS-QA scenes)
additionally get the video metrics. Results are cached as one JSON per condition."""
import glob, json, os, sys, time
import numpy as np
from common import ROOT, SCENES, load_scene, load_png
from metrics import Metrics
from eval_gsqa import PPD

if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "results/part1")
    os.makedirs(out_dir, exist_ok=True)
    watch = "--watch" in sys.argv
    M = Metrics(display="standard_fhd", ppd=PPD, flip_workers=6)
    refs, scenes = {}, {}


    def pending():
        todo = []
        for d in sorted(glob.glob(os.path.join(ROOT, "runs/*/*/step_*"))):
            scene, tag, step = d.split("/")[-3:]
            if tag.startswith("_") or scene not in SCENES or scene not in os.environ.get("ONLY", scene):
                continue
            dst = f"{out_dir}/{scene}__{tag}__{step}.json"
            if os.path.exists(dst):
                continue
            if scene not in scenes:
                scenes[scene] = load_scene(scene)
            files = sorted(glob.glob(d + "/*.png"))
            if len(files) != len(scenes[scene].test_idx) or time.time() - os.path.getmtime(files[-1]) < 20:
                continue
            todo.append((scene, tag, step, files, dst))
        return todo


    def queues_done():
        return all(os.path.exists(f) and "QUEUE_DONE" in open(f).read()
                   for f in [f"{ROOT}/logs/queue_gs.log", f"{ROOT}/logs/queue_nerf.log"])


    while True:
        todo = pending()
        for scene, tag, step, files, dst in todo:
            t0 = time.time()
            sc = scenes[scene]
            if scene not in refs:
                if len(refs) >= 2:
                    refs.clear()
                refs[scene] = np.stack([sc.load_image(i) for i in sc.test_idx])
            ref = refs[scene]
            test = np.stack([load_png(f) for f in files])
            res = {"scene": scene, "tag": tag, "step": int(step.split("_")[1]), "kind": sc.kind,
                   "log": json.load(open(os.path.join(os.path.dirname(files[0]), "..", "log.json")))}
            st = int(os.environ.get("STRIDE", 1)) if sc.kind == "gsqa" else 1
            res["stride"] = st
            fr = M.frames(test[::st], ref[::st])
            res["frames"] = {k: v.tolist() for k, v in fr.items()}
            if sc.kind == "gsqa":
                res.update(M.video(test, ref, sc.fps))
            json.dump(res, open(dst, "w"))
            print(f"{scene:8s} {tag:16s} {step} " + " ".join(f"{k}={v.mean():.4f}" for k, v in fr.items()) +
                  (f" cvvdp_vid={res['cvvdp_vid']:.3f} tpsnr={res['tpsnr']:.2f}" if sc.kind == "gsqa" else "") +
                  f" ({time.time()-t0:.0f}s)", flush=True)
        if not watch or (queues_done() and not pending()):
            break
        if not todo:
            time.sleep(60)
    print("DONE", flush=True)
