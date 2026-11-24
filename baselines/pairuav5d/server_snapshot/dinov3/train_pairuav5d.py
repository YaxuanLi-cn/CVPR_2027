#!/usr/bin/env python3
"""Train a frozen DINOv3 visual baseline on the fixed PairUAV-5D manifests."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from tqdm import tqdm

DISTANCE_SCALE_M = 40.0
TILT_SCALE_DEG = 45.0
HEADING_SCALE_DEG = 180.0
XYZ_SCALE_M = 40.0


class PairDataset(Dataset):
    def __init__(self, json_dir: Path, image_root: Path, transform):
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        for tour_dir in sorted(path for path in json_dir.iterdir() if path.is_dir()):
            for path in sorted(tour_dir.glob("*.json")):
                row = json.loads(path.read_text())
                delta = row["delta"]
                xyz = tuple(float(delta[key]) for key in ("x_m", "y_m", "z_m"))
                tilt = float(row["tilt_num"])
                heading = float(row["heading_num"])
                self.samples.append((
                    self.image_root / row["image_a"],
                    self.image_root / row["image_b"],
                    tuple(v / XYZ_SCALE_M for v in xyz) + (tilt / TILT_SCALE_DEG, heading / HEADING_SCALE_DEG),
                    xyz + (tilt, heading),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_a_path, image_b_path, targets, labels = self.samples[index]
        with Image.open(image_a_path) as image:
            image_a = self.transform(image.convert("RGB"))
        with Image.open(image_b_path) as image:
            image_b = self.transform(image.convert("RGB"))
        return image_a, image_b, torch.tensor(targets), torch.tensor(labels)


class DINOv3PairModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.regressor = nn.Linear(self.backbone.num_features * 2, 5)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image_a, image_b):
        with torch.no_grad():
            features = torch.cat((self.backbone(image_a), self.backbone(image_b)), dim=-1)
        return self.regressor(features.float())


def validate(loader, model, device):
    model.eval()
    predictions = []
    labels_all = []
    with torch.inference_mode():
        for image_a, image_b, _, labels in tqdm(loader, desc="Validate"):
            output = model(image_a.to(device, non_blocking=True), image_b.to(device, non_blocking=True))
            decoded = output * output.new_tensor([XYZ_SCALE_M] * 3 + [TILT_SCALE_DEG, HEADING_SCALE_DEG])
            predictions.append(decoded.cpu())
            labels_all.append(labels)
    predictions = torch.cat(predictions)
    labels_all = torch.cat(labels_all)
    xyz_error = (predictions[:, :3] - labels_all[:, :3]).abs().mean(dim=0)
    distance_mae = (torch.linalg.vector_norm(predictions[:, :3], dim=1) - torch.linalg.vector_norm(labels_all[:, :3], dim=1)).abs().mean().item()
    tilt_mae = (predictions[:, 3] - labels_all[:, 3]).abs().mean().item()
    heading_error = torch.remainder(predictions[:, 4] - labels_all[:, 4] + 180.0, 360.0) - 180.0
    heading_mae = heading_error.abs().mean().item()
    return tuple(v.item() for v in xyz_error) + (distance_mae, tilt_mae, heading_mae, predictions.numpy(), labels_all.numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=Path("/root/pairUAV-5D-temp"))
    parser.add_argument("--train-dir", type=Path, default=Path("/root/baseline/_mast3r_split_data_v2/train"))
    parser.add_argument("--test-dir", type=Path, default=Path("/root/baseline/_mast3r_split_data_v2/test"))
    parser.add_argument("--model-name", default="vit_large_patch16_dinov3.lvd1689m")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_mast3r_split"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_for_config = timm.create_model(args.model_name, pretrained=False)
    data_config = resolve_model_data_config(model_for_config)
    train_transform = create_transform(**data_config, is_training=True)
    test_transform = create_transform(**data_config, is_training=False)
    train_data = PairDataset(args.train_dir, args.image_root, train_transform)
    test_data = PairDataset(args.test_dir, args.image_root, test_transform)
    logging.info("MASt3R split rows: train=%d test=%d", len(train_data), len(test_data))

    model = DINOv3PairModel(args.model_name, pretrained=args.init_checkpoint is None)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        backbone_state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
        logging.info("Loaded backbone init: missing=%d unexpected=%d", len(missing), len(unexpected))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        for image_a, image_b, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1} train"):
            image_a = image_a.to(device, non_blocking=True)
            image_b = image_b.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = F.smooth_l1_loss(model(image_a, image_b), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = validate(test_loader, model, device)
        result = {
            "epoch": epoch + 1,
            "model": args.model_name,
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "MAE_x_m": metrics[0],
            "MAE_y_m": metrics[1],
            "MAE_z_m": metrics[2],
            "MAE_distance_from_xyz_m": metrics[3],
            "MAE_tilt_deg": metrics[4],
            "MAE_heading_deg_circular": metrics[5],
            "distance_definition": "MAE between norms of predicted and ground-truth xyz; no distance head",
        }
        logging.info(json.dumps(result))
        torch.save({"model": model.state_dict(), "metrics": result}, args.output_dir / "best.pt")
        (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(args.output_dir / "test_predictions.npz", pred_xyz_tilt_heading=metrics[6], true_xyz_tilt_heading=metrics[7])


if __name__ == "__main__":
    main()
