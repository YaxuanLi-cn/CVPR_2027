#!/usr/bin/env python3
"""DINO-WM one-step latent dynamics and CEM pose planning for PairUAV-5D."""
import json
import os
import random
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/root/baseline/dinov3")

from models.dino import DinoV2Encoder
from models.proprio import ProprioceptiveEmbedding
from models.vit import ViTPredictor
from models.visual_world_model import VWorldModel
from train_pairuav5d import PairDataset


ROOT = Path("/root/baseline")
TRAIN_DIR = ROOT / "_mast3r_split_data_v2/train"
TEST_DIR = ROOT / "_mast3r_split_data_v2/test"
IMAGE_ROOT = Path("/root/pairUAV-5D-temp")
OUT = Path(__file__).parent / "outputs_pairuav5d"
ACTION_SCALES = torch.tensor([40.0, 40.0, 40.0, 45.0, 180.0])


class ZeroProprio(nn.Module):
    def forward(self, x):
        return x


def fixed_seed(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(device):
    encoder = DinoV2Encoder("dinov2_vits14", "x_norm_patchtokens")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    action_encoder = ProprioceptiveEmbedding(in_chans=5, emb_dim=16, num_frames=1)
    predictor = ViTPredictor(
        num_patches=196, num_frames=1, dim=400, depth=6, heads=16,
        mlp_dim=2048, dropout=0.1, emb_dropout=0.0, pool="mean")
    wm = VWorldModel(
        image_size=224, num_hist=1, num_pred=1, encoder=encoder,
        proprio_encoder=ZeroProprio(), action_encoder=action_encoder,
        decoder=None, predictor=predictor, proprio_dim=0, action_dim=16,
        concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=False, train_predictor=True, train_decoder=False)
    wm.to(device)
    wm.encoder.eval()
    return wm


def make_dataset(data_dir, transform):
    dataset = PairDataset(data_dir, IMAGE_ROOT, transform)
    return dataset


def fit(wm, loader, action_mean, action_std, device):
    params = list(wm.predictor.parameters()) + list(wm.action_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-4)
    wm.train()
    wm.encoder.eval()
    logs = []
    for image_a, image_b, _, labels in tqdm(loader, desc="DINO-WM train"):
        b = image_a.shape[0]
        visual = torch.stack((image_a, image_b), dim=1).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        actions = ((labels - action_mean) / action_std).unsqueeze(1).expand(-1, 2, -1)
        obs = {"visual": visual, "proprio": torch.empty((b, 2, 0), device=device)}
        optimizer.zero_grad(set_to_none=True)
        _, _, _, loss, parts = wm(obs, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        logs.append(float(parts["z_visual_loss"].detach()))
    return float(np.mean(logs))


@torch.inference_mode()
def plan_batch(wm, image_a, image_b, action_mean, action_std, device,
               samples=32, topk=8, steps=3):
    b = image_a.shape[0]
    visual = torch.stack((image_a, image_b), dim=1).to(device, non_blocking=True)
    obs = {"visual": visual, "proprio": torch.empty((b, 2, 0), device=device)}
    encoded = wm.encode_obs(obs)
    src = encoded["visual"][:, :1]
    goal = encoded["visual"][:, 1]
    mu = torch.zeros((b, 5), device=device)
    sigma = torch.ones_like(mu)
    for _ in range(steps):
        actions = mu[:, None] + sigma[:, None] * torch.randn((b, samples, 5), device=device)
        actions = actions.clamp(-3.0, 3.0)
        actions[:, 0] = mu
        flat_actions = actions.reshape(b * samples, 1, 5)
        act_emb = wm.encode_act(flat_actions)
        act_tiled = act_emb.unsqueeze(2).expand(-1, -1, src.shape[2], -1)
        src_tiled = src[:, None].expand(-1, samples, -1, -1, -1).reshape(b * samples, 1, src.shape[2], src.shape[3])
        z = torch.cat((src_tiled, act_tiled), dim=-1)
        prediction = wm.predict(z)[:, 0, :, :src.shape[-1]]
        target = goal[:, None].expand(-1, samples, -1, -1).reshape_as(prediction)
        loss = (prediction - target).square().mean(dim=(1, 2)).view(b, samples)
        values, indices = loss.topk(topk, largest=False, dim=1)
        selected = actions.gather(1, indices[..., None].expand(-1, -1, 5))
        mu = selected.mean(dim=1)
        sigma = selected.std(dim=1, unbiased=False).clamp_min(0.05)
    return mu * action_std + action_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-checkpoint", type=Path, default=None)
    ap.add_argument("--max-test-rows", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    fixed_seed()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_data = make_dataset(TRAIN_DIR, transform)
    test_data = make_dataset(TEST_DIR, transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8,
                              pin_memory=True, persistent_workers=True)
    labels = torch.tensor([row[3] for row in train_data.samples], dtype=torch.float32)
    action_mean = labels.mean(dim=0).to(device)
    action_std = labels.std(dim=0).clamp_min(1e-3).to(device)
    wm = make_model(device)
    if args.resume_checkpoint is not None:
        saved = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        wm.load_state_dict(saved["wm"])
        action_mean = saved["action_mean"].to(device)
        action_std = saved["action_std"].to(device)
        train_latent_mse = float(saved["train_latent_mse"])
    else:
        train_latent_mse = fit(wm, train_loader, action_mean, action_std, device)
    wm.eval()
    wm.encoder.eval()
    torch.save({"wm": wm.state_dict(), "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(), "train_latent_mse": train_latent_mse},
               out_dir / "checkpoint_pretest.pt")

    pred_rows, true_rows = [], []
    if args.max_test_rows is not None:
        test_data.samples = test_data.samples[:args.max_test_rows]
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=8,
                             pin_memory=True, persistent_workers=True)
    for image_a, image_b, _, labels in tqdm(test_loader, desc="DINO-WM CEM test"):
        pred_rows.append(plan_batch(wm, image_a, image_b, action_mean, action_std, device,
                                    samples=16, topk=4, steps=3).cpu().numpy())
        true_rows.append(labels.numpy())
    pred, true = np.concatenate(pred_rows), np.concatenate(true_rows)
    xyz_mae = np.abs(pred[:, :3] - true[:, :3]).mean(axis=0)
    distance_mae = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(true[:, :3], axis=1)).mean()
    tilt_mae = np.abs(pred[:, 3] - true[:, 3]).mean()
    heading_mae = np.abs((pred[:, 4] - true[:, 4] + 180.0) % 360.0 - 180.0).mean()
    metrics = {
        "model": "DINO-WM (official DINOv2 ViT-S/14 encoder, one-step latent predictor, CEM pose planning)",
        "train_rows": len(train_data), "test_rows": len(test_data),
        "world_model_train_latent_mse": train_latent_mse,
        "planner": {"method": "CEM", "samples": 16, "topk": 4, "iterations": 3, "horizon": 1},
        "MAE_x_m": float(xyz_mae[0]), "MAE_y_m": float(xyz_mae[1]),
        "MAE_z_m": float(xyz_mae[2]), "MAE_distance_from_xyz_m": float(distance_mae),
        "MAE_tilt_deg": float(tilt_mae), "MAE_heading_deg_circular": float(heading_mae),
        "distance_definition": "MAE between norms of planned and ground-truth xyz; no distance head",
    }
    torch.save({"wm": wm.state_dict(), "action_mean": action_mean.cpu(),
                "action_std": action_std.cpu(), "metrics": metrics}, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "test_predictions.npz", pred_xyz_tilt_heading=pred, true_xyz_tilt_heading=true)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
