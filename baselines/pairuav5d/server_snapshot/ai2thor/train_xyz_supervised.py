#!/usr/bin/env python3
"""Batch-train the AI2THOR pair encoder for xyz, tilt, and heading."""
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import CHECKPOINT_DIR
from data_loader import ImagePairDataLoader
from network_continuous import ActorCriticContinuousNetwork


TRAIN_DIR = "/root/baseline/_mast3r_split_data_v2/train"
TEST_DIR = "/root/baseline/_mast3r_split_data_v2/test"
IMAGE_ROOT = "/root/pairUAV-5D-temp"
SOURCE_CHECKPOINT = "/root/baseline/ai2thor/checkpoints_mast3r_split_stable/checkpoint-23844"
OUT_DIR = "/root/baseline/ai2thor/checkpoints_xyz_mast3r_split_supervised"
SCALES = np.asarray([40.0, 40.0, 40.0, 45.0, 180.0], dtype=np.float32)


def targets(batch):
    return np.column_stack((batch["xyz"], batch["tilt"], batch["heading"])).astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tf.reset_default_graph()
    model = ActorCriticContinuousNetwork(
        device="/cpu:0", network_scope="continuous_nav", scene_scopes=["shared"])
    key = model._get_key(["continuous_nav", "shared"])
    y = tf.placeholder(tf.float32, [None, 5], name="xyz_tilt_heading")
    scale = tf.constant(SCALES)
    loss = tf.losses.huber_loss(y / scale, model.pi_mean[key] / scale, delta=0.2)
    train_vars = [v for v in tf.trainable_variables()
                  if "W_policy_log_std" not in v.name
                  and "b_policy_log_std" not in v.name
                  and "W_value" not in v.name
                  and "b_value" not in v.name]
    train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, var_list=train_vars)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)
    train_loader = ImagePairDataLoader(TRAIN_DIR, batch_size=128, dataset_root=IMAGE_ROOT, shuffle=True)
    test_loader = ImagePairDataLoader(TEST_DIR, batch_size=256, dataset_root=IMAGE_ROOT, shuffle=False)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        reader = tf.train.NewCheckpointReader(SOURCE_CHECKPOINT)
        source = reader.get_variable_to_shape_map()
        assignments = []
        for var in tf.trainable_variables():
            name = var.name.split(":")[0]
            if name not in source:
                continue
            old = reader.get_tensor(name)
            shape = var.get_shape().as_list()
            if list(old.shape) == shape:
                value = old
            elif (name.endswith(("W_policy_mean", "b_policy_mean"))
                  and old.ndim == len(shape) and old.shape[-1] == 3 and shape[-1] == 5):
                value = sess.run(var)
                value[..., 3:5] = old[..., 1:3]
            else:
                continue
            assignments.append(tf.assign(var, value))
        sess.run(assignments)
        print("Transferred", len(assignments), "tensors from", SOURCE_CHECKPOINT, flush=True)

        for _ in range(train_loader.get_num_batches()):
            batch = train_loader.get_batch()
            if len(batch["heading"]) == 0:
                continue
            sess.run(train_op, feed_dict={
                model.s: batch["image_a"], model.t: batch["image_b"], y: targets(batch)})

        pred_rows, true_rows = [], []
        for _ in tqdm(range(test_loader.get_num_batches()), desc="Evaluate"):
            batch = test_loader.get_batch()
            if len(batch["heading"]) == 0:
                continue
            pred = sess.run(model.pi_mean[key], feed_dict={
                model.s: batch["image_a"], model.t: batch["image_b"]})
            pred_rows.append(pred)
            true_rows.append(targets(batch))
        pred = np.concatenate(pred_rows)
        true = np.concatenate(true_rows)
        xyz_mae = np.abs(pred[:, :3] - true[:, :3]).mean(axis=0)
        distance_mae = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(true[:, :3], axis=1)).mean()
        tilt_mae = np.abs(pred[:, 3] - true[:, 3]).mean()
        heading_error = np.abs((pred[:, 4] - true[:, 4] + 180.0) % 360.0 - 180.0)
        metrics = {
            "model": "AI2THOR xyz-supervised head",
            "training": "one batch epoch; Huber loss over normalized xyz, tilt, heading",
            "train_rows": train_loader.get_num_samples(),
            "test_rows": len(true),
            "MAE_x_m": float(xyz_mae[0]),
            "MAE_y_m": float(xyz_mae[1]),
            "MAE_z_m": float(xyz_mae[2]),
            "MAE_distance_from_xyz_m": float(distance_mae),
            "MAE_tilt_deg": float(tilt_mae),
            "MAE_heading_deg_circular": float(heading_error.mean()),
            "distance_definition": "MAE between norms of predicted and ground-truth xyz; no distance head",
        }
        saver.save(sess, os.path.join(OUT_DIR, "model"))
        np.savez_compressed(os.path.join(OUT_DIR, "test_predictions.npz"), pred=pred, true=true)
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
