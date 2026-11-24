#!/usr/bin/env python3
"""Visual-Foresight-style action-conditioned latent prediction and goal-image MPC."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, "/root/baseline/dinov3")
from train_pairuav5d import PairDataset

ROOT = Path("/root/baseline")
FRAME = sys.argv[1] if len(sys.argv) > 1 else "enu"
assert FRAME in ("enu", "cam")
sys.path.insert(0, str(Path(__file__).parent))
from splitdata import ordered, labels as frame_labels
OUT = Path("/root/rasr-pairuav/outputs/supp/trackE/runs") / f"visual_foresight_{FRAME}"
FEATURES = ROOT / "dinov3/outputs_xyz7b_mast3r_split/unique_image_features.pt"
TRAIN = ROOT / "_mast3r_split_data_v2/train"
TEST = ROOT / "_mast3r_split_data_v2/test"
IMAGE_ROOT = Path("/root/pairUAV-5D-temp")
SCALES = torch.tensor([40.0, 40.0, 40.0, 45.0])


class ActionConditionedPredictor(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 128))
        self.dynamics = nn.Sequential(
            nn.Linear(dim + 128, 512), nn.GELU(),
            nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, dim),
        )

    def forward(self, image_feature, action):
        return image_feature + self.dynamics(torch.cat((image_feature, self.action(action)), dim=-1))


def pair_indices(dataset, path_to_index, features):
    a = torch.tensor([path_to_index[str(row[0])] for row in dataset.samples], dtype=torch.long)
    b = torch.tensor([path_to_index[str(row[1])] for row in dataset.samples], dtype=torch.long)
    targets = torch.tensor([row[2] for row in dataset.samples], dtype=torch.float32)
    heading = targets[:, 4] * torch.pi
    actions = torch.cat((targets[:, :4], heading.sin().unsqueeze(1), heading.cos().unsqueeze(1)), dim=1)
    labels = torch.tensor([row[3] for row in dataset.samples], dtype=torch.float32)
    return features[a].float(), features[b].float(), actions, labels


@torch.inference_mode()
def cem_plan(model, source, goal, action_mean, action_std, device, samples=16, topk=4, steps=3):
    predictions = []
    for start in tqdm(range(0, len(source), 128), desc="Visual-Foresight CEM"):
        src = source[start:start + 128].to(device)
        target = goal[start:start + 128].to(device)
        batch = len(src)
        mu = torch.zeros((batch, 6), device=device)
        sigma = torch.ones_like(mu)
        for _ in range(steps):
            actions = (mu[:, None] + sigma[:, None] * torch.randn(batch, samples, 6, device=device)).clamp(-3, 3)
            actions[:, 0] = mu
            flat = actions.reshape(batch * samples, 6)
            repeated = src[:, None].expand(-1, samples, -1).reshape(batch * samples, -1)
            predicted_goal = model(repeated, flat).reshape(batch, samples, -1)
            scores = (predicted_goal - target[:, None]).square().mean(-1)
            _, indices = scores.topk(topk, largest=False, dim=1)
            selected = actions.gather(1, indices.unsqueeze(-1).expand(-1, -1, 6))
            mu = selected.mean(dim=1)
            sigma = selected.std(dim=1, unbiased=False).clamp_min(0.05)
        predictions.append(mu * action_std + action_mean)
    action = torch.cat(predictions)
    xyz_tilt = action[:, :4] * action.new_tensor([40.0, 40.0, 40.0, 45.0])
    heading = torch.atan2(action[:, 4], action[:, 5]) * (180.0 / torch.pi)
    return torch.cat((xyz_tilt, heading.unsqueeze(1)), dim=1).cpu()


def main():
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.load(FEATURES, map_location="cpu", weights_only=True)
    transform = lambda image: image
    train_data, test_data = PairDataset(TRAIN, IMAGE_ROOT, transform), PairDataset(TEST, IMAGE_ROOT, transform)
    order = {"train": ordered("train"), "test": ordered("test")}
    for data, name in ((train_data, "train"), (test_data, "test")):   # same samples, same order; (x, y) -> (lx, ly) when FRAME == "cam"
        L = frame_labels(order[name], FRAME)
        assert len(L) == len(data.samples) and all(str(s[0]) == o["image_a"] and str(s[1]) == o["image_b"] for s, o in zip(data.samples, order[name]))
        data.samples = [(s[0], s[1], (l[0] / 40.0, l[1] / 40.0, l[2] / 40.0, l[3] / 45.0, l[4] / 180.0), tuple(float(v) for v in l)) for s, l in zip(data.samples, L)]
    samples = train_data.samples + test_data.samples
    image_paths = sorted({str(path) for row in samples for path in row[:2]})
    if len(image_paths) != len(features):
        raise RuntimeError(f"Feature cache mismatch: images={len(image_paths)} cached={len(features)}")
    path_to_index = {path: i for i, path in enumerate(image_paths)}
    train_a, train_b, actions, train_labels = pair_indices(train_data, path_to_index, features)
    test_a, test_b, _, test_labels = pair_indices(test_data, path_to_index, features)
    action_mean = actions.mean(0)
    action_std = actions.std(0).clamp_min(1e-3)
    normalized_actions = (actions - action_mean) / action_std
    train_loader = DataLoader(TensorDataset(train_a, normalized_actions, train_b), batch_size=256, shuffle=True)
    model = ActionConditionedPredictor(features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    epochs = 5
    train_losses = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for src, action, target in tqdm(train_loader, desc=f"Visual-Foresight dynamics {epoch + 1}/{epochs}"):
            src, action, target = src.to(device), action.to(device), target.to(device)
            prediction = model(src, action)
            positive_error = (prediction - target).square().mean(dim=1)
            negative_action = action[torch.randperm(len(action), device=device)]
            negative_prediction = model(src, negative_action)
            negative_error = (negative_prediction - target).square().mean(dim=1)
            latent_loss = F.smooth_l1_loss(prediction, target, beta=0.1)
            action_ranking = F.relu(0.01 + positive_error - negative_error).mean()
            loss = latent_loss + 0.2 * action_ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        train_losses.append(float(np.mean(losses)))
    model.eval()
    pred = cem_plan(model, test_a, test_b, action_mean.to(device), action_std.to(device), device)
    truth = test_labels
    err = (pred[:, :3] - truth[:, :3]).abs().mean(0)
    distance = torch.linalg.vector_norm(pred[:, :3] - truth[:, :3], dim=1).mean()
    heading = (torch.remainder(pred[:, 4] - truth[:, 4] + 180, 360) - 180).abs().mean()
    distance_error = (torch.linalg.vector_norm(pred[:, :3], dim=1) - torch.linalg.vector_norm(truth[:, :3], dim=1)).abs()
    success_count = int((distance_error <= 1.0).sum())
    metrics = {
        "model": "Visual Foresight-style PairUAV adaptation: action-conditioned DINOv3 ViT-7B latent dynamics + goal-image CEM MPC",
        "pretrained_visual_features": "cached DINOv3 ViT-7B CLS features",
        "dynamics": "xyz/tilt/sin-cos-heading action-conditioned residual latent predictor with counterfactual action ranking, trained on PairUAV train split",
        "planner": {"method": "CEM minimizing predicted future-goal latent distance", "samples": 16, "topk": 4, "iterations": 3, "horizon": 1},
        "train_rows": len(train_data), "test_rows": len(test_data), "epochs": epochs, "train_latent_losses": train_losses,
        "MAE_x_m": float(err[0]), "MAE_y_m": float(err[1]), "MAE_z_m": float(err[2]),
        "MAE_distance_from_xyz_m": float(distance), "MAE_tilt_deg": float((pred[:, 3] - truth[:, 3]).abs().mean()),
        "MAE_heading_deg_circular": float(heading),
        "distance_success_threshold_m": 1.0, "distance_success_count": success_count,
        "distance_success_rate": float(success_count / len(truth)),
        "distance_definition": "MAE between predicted and target xyz vector norms; no distance prediction head",
        "scope_note": "Method adaptation, not an official Visual Foresight pretrained checkpoint; original method learns action-conditioned visual dynamics and plans toward image goals.",
    }
    torch.save({"model": model.cpu().state_dict(), "action_mean": action_mean, "action_std": action_std, "metrics": metrics}, OUT / "checkpoint.pt")
    np.savez_compressed(OUT / "test_predictions.npz", pred_xyz_tilt_heading=pred.numpy(), true_xyz_tilt_heading=truth.numpy(), manifest_index=np.array([o["manifest_index"] for o in order["test"]]), frame=FRAME)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
