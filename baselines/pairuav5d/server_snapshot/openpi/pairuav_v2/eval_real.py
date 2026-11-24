#!/usr/bin/env python3
"""Zero-shot evaluation of the PairUAV-5D pi0-FAST checkpoint on real drone photographs.

Pairs: the Mezieres pairs already scored for DreamNav (/root/realdata/mezieres/pred.pkl), restricted to the
baseline/height range seen in training (dist_equiv <= 38 m). No weight is updated and no label is read by the
model. Images and inference are those of the PairUAV-5D test run (build_cache224.py / eval_v2.py): full frame,
resize_with_pad to 224x224, greedy FAST decoding, mean of the decoded 10-step chunk, batch 16.
"""
import argparse
import json
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_cache224 import decode  # noqa: E402  (absolute paths override its IMAGE_ROOT)
from eval_v2 import INDEX, PROMPT, build_policy, infer_batch  # noqa: E402

REAL = pathlib.Path("/root/realdata/mezieres")


def run(policy, cfg, images, ia, ib, batch_size):
    tokenizer = policy._output_transform.transforms[0].tokenizer
    state = np.zeros(8, dtype=np.float32)
    preds, undecodable = [], 0
    for start in range(0, len(ia), batch_size):
        rows = range(start, min(start + batch_size, len(ia)))
        obs = [{"image": images[ia[i]].copy(), "wrist_image": images[ib[i]].copy(), "state": state.copy(),
                "prompt": PROMPT} for i in rows]
        obs += [obs[-1]] * (batch_size - len(obs))  # keep one compiled batch shape
        raw, actions = infer_batch(policy, obs)
        for k in range(len(rows)):
            decoded = tokenizer.extract_actions(raw[k].astype(np.int32), cfg.model.action_horizon, cfg.model.action_dim)
            undecodable += int(not np.any(decoded))  # all-zero chunk = tokens could not be decoded
            preds.append(np.asarray(actions[k], dtype=np.float64).mean(axis=0))
        print(start + len(rows), "pairs", flush=True)
    return np.stack(preds), undecodable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi0_fast_pairuav_v2")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=HERE.parent / "checkpoints/pi0_fast_pairuav_v2/2979")
    parser.add_argument("--out", type=pathlib.Path, default=REAL / "baselines")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--check-pairuav", type=int, default=0, help="first re-run this many PairUAV-5D test pairs")
    args = parser.parse_args()

    # pred.pkl rows with dist_equiv <= 38, exported as CSV (this venv's pandas cannot unpickle pred.pkl)
    pairs = pd.read_csv(REAL / "baselines" / "pairs_in_range.csv", index_col="pair_index")
    paths = sorted(set(pairs.image_a) | set(pairs.image_b))
    slot = {p: i for i, p in enumerate(paths)}
    with Pool(8) as pool:
        images = np.stack(pool.map(decode, paths))  # [n, 224, 224, 3] uint8, same as images_224_pad.npy

    policy, cfg = build_policy(args.config, args.checkpoint.resolve())
    assert cfg.model.model_type.value == "pi0_fast"
    if args.check_pairuav:  # same code path on PairUAV-5D test pairs: must reproduce the stored test predictions
        index = np.load(INDEX / "test_index.npz")
        files = json.loads((INDEX / "paths.json").read_text())
        n = args.check_pairuav
        need = sorted(set(index["a"][:n]) | set(index["b"][:n]))
        where = {p: i for i, p in enumerate(need)}
        with Pool(8) as pool:
            test_images = np.stack(pool.map(decode, [files[i] for i in need]))
        got, _ = run(policy, cfg, test_images, [where[i] for i in index["a"][:n]], [where[i] for i in index["b"][:n]],
                     args.batch_size)
        ref = np.load(HERE / "results/pi0_fast/test_predictions.npz")["pred_xyz_tilt_heading"][:n]
        diff = np.abs(got - ref)
        diff[:, 4] = np.abs((diff[:, 4] + 180) % 360 - 180)
        print("PairUAV check on", n, "test pairs: max |diff| to stored predictions (x, y, z, tilt, heading)",
              diff.max(0).round(4).tolist(), "| rows identical", int((diff.max(1) < 1e-6).sum()), flush=True)
    started = time.time()
    # Greedy decoding is not bit-reproducible on this GPU: repeats inside one process are identical, but about a
    # third of the pairs decode to a different chunk in another process (near-tied tokens). The first run is the
    # reported prediction; the table metrics move by less than 0.1 m / 0.1 deg between processes.
    runs = [run(policy, cfg, images, [slot[p] for p in pairs.image_a], [slot[p] for p in pairs.image_b],
                args.batch_size) for _ in range(args.repeats)]
    pred, undecodable = runs[0]

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "pi0_fast_predictions.npz", pair_index=pairs.index.values,
                        a=np.array(pairs.a.tolist(), dtype=str), b=np.array(pairs.b.tolist(), dtype=str), pred_xyz_tilt_heading=pred,
                        pred_all_runs=np.stack([r[0] for r in runs]),
                        true_xyz_tilt_heading=pairs[["x", "y", "z", "tilt", "heading"]].values)
    print(json.dumps({"config": args.config, "checkpoint": str(args.checkpoint), "pairs": len(pairs),
                      "images": len(paths), "undecodable_outputs": [r[1] for r in runs], "pred_std": pred.std(0).tolist(),
                      "inference_seconds": time.time() - started}))


if __name__ == "__main__":
    main()
