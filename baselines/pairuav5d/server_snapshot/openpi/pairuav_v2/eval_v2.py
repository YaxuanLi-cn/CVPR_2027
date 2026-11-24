#!/usr/bin/env python3
"""Evaluate an openpi checkpoint on the fixed PairUAV-5D test split and save every prediction.

Works for pi0.5 (flow matching, one sample per pair, 10 integration steps, policy seed 0) and
pi0-FAST (greedy token decoding; the 10-step chunk is averaged into one relative pose).
Saves pred / true (x, y, z, tilt, heading) per row, in the test-manifest order used by all baselines.
"""
import argparse
import json
import logging
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
CACHE = HERE / "cache" / "images_224_pad.npy"
INDEX = pathlib.Path("/root/baseline/visualnav-transformer/pairuav5d_v2/cache")
PROMPT = "Move from image viewpoint to wrist_image viewpoint"


def build_policy(config_name, checkpoint_dir):
    import openpi.models.model as _model
    import openpi.policies.policy as _policy
    import openpi.shared.normalize as _normalize
    import openpi.training.config as _config
    from openpi import transforms as _transforms

    cfg = _config.get_config(config_name)
    stats_dirs = [p.parent for p in (checkpoint_dir / "assets").glob("*/norm_stats.json")]
    assert len(stats_dirs) == 1, stats_dirs
    norm_stats = _normalize.load(stats_dirs[0])
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    model = cfg.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    repack = _transforms.RepackTransform({"observation/image": "image", "observation/wrist_image": "wrist_image",
                                          "observation/state": "state", "prompt": "prompt"})
    policy = _policy.Policy(
        model,
        transforms=[repack, *data.data_transforms.inputs,
                    _transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           _transforms.Unnormalize(norm_stats, use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        metadata=cfg.policy_metadata,
        # pi0-FAST: a pose chunk is at most ~20 tokens ("Action: " + <=14 FAST tokens + "|" + EOS);
        # the default of 256 decoding steps only enlarges the key-value cache.
        sample_kwargs={"max_decoding_steps": 48} if cfg.model.model_type.value == "pi0_fast" else None,
    )
    if cfg.model.model_type.value == "pi0_fast":
        # max_decoding_steps sets array shapes, so it has to be a compile-time constant
        import openpi.shared.nnx_utils as nnx_utils
        policy._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("max_decoding_steps",))
    return policy, cfg


def stack(items):
    out = {}
    for key, value in items[0].items():
        out[key] = stack([d[key] for d in items]) if isinstance(value, dict) else np.stack([d[key] for d in items])
    return out


def infer_batch(policy, observations):
    import openpi.models.model as _model
    batch = jax.tree.map(jnp.asarray, stack([policy._input_transform(dict(o)) for o in observations]))
    policy._rng, rng = jax.random.split(policy._rng)
    raw = np.asarray(policy._sample_actions(rng, _model.Observation.from_dict(batch), **policy._sample_kwargs))
    state = np.asarray(batch["state"])
    return raw, [policy._output_transform({"state": state[i], "actions": raw[i]})["actions"] for i in range(len(observations))]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def metrics(pred, true):
    err = np.abs(pred[:, :3] - true[:, :3]).mean(0)
    dist = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(true[:, :3], axis=1))
    out = {"MAE_x_m": err[0], "MAE_y_m": err[1], "MAE_z_m": err[2], "MAE_distance_from_xyz_m": dist.mean(),
           "MAE_tilt_deg": np.abs(pred[:, 3] - true[:, 3]).mean(),
           "MAE_heading_deg_circular": np.abs(wrap(pred[:, 4] - true[:, 4])).mean(),
           "distance_success_count": int((dist <= 1.0).sum()), "distance_success_rate": (dist <= 1.0).mean()}
    return {k: (v if isinstance(v, int) else float(v)) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint", type=pathlib.Path, help="step directory containing params/ and assets/")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    policy, cfg = build_policy(args.config, args.checkpoint.resolve())
    images = np.load(CACHE, mmap_mode="r")
    index = np.load(INDEX / f"{args.split}_index.npz")
    total = len(index["a"]) if args.max_samples is None else args.max_samples
    truth = index["target"][:total].astype(np.float64)
    state = np.zeros(8, dtype=np.float32)
    is_fast = cfg.model.model_type.value == "pi0_fast"
    preds, undecodable = [], 0
    started = time.time()
    for start in tqdm(range(0, total, args.batch_size), mininterval=30):
        rows = range(start, min(start + args.batch_size, total))
        obs = [{"image": np.array(images[index["a"][i]]), "wrist_image": np.array(images[index["b"][i]]),
                "state": state.copy(), "prompt": PROMPT} for i in rows]
        if len(obs) < args.batch_size:  # keep one compiled batch shape
            obs += [obs[-1]] * (args.batch_size - len(obs))
        raw, actions = infer_batch(policy, obs)
        for k in range(len(rows)):
            chunk = np.asarray(actions[k], dtype=np.float64)  # [horizon, 5]
            if is_fast:
                # extract_actions returns an all-zero chunk when the emitted tokens cannot be decoded
                decoded = policy._output_transform.transforms[0].tokenizer.extract_actions(
                    raw[k].astype(np.int32), cfg.model.action_horizon, cfg.model.action_dim)
                undecodable += int(not np.any(decoded))
            preds.append(chunk.mean(axis=0))
    pred = np.stack(preds)
    result = {"config": args.config, "checkpoint": str(args.checkpoint), "split": args.split, "rows": int(total),
              "sampling": ("greedy FAST token decoding (temperature 0); mean of the decoded 10-step chunk" if is_fast
                           else "one flow-matching sample per pair, 10 Euler steps, policy seed 0"),
              **metrics(pred, truth), "zero_predictor": metrics(np.zeros_like(truth), truth),
              "prediction_std": pred.std(0).tolist(), "target_std": truth.std(0).tolist(),
              "prediction_target_correlation": [float(np.corrcoef(pred[:, i], truth[:, i])[0, 1]) if pred[:, i].std() > 0 else None for i in range(5)],
              "distance_definition": "abs(norm(pred_xyz) - norm(true_xyz)); success = this error <= 1 m",
              "inference_seconds": time.time() - started}
    if is_fast:
        result["undecodable_outputs"] = undecodable
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.split}_predictions.npz", pred_xyz_tilt_heading=pred, true_xyz_tilt_heading=truth)
    (args.out / f"{args.split}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
