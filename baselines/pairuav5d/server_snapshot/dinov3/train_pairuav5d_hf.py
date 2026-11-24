#!/usr/bin/env python3
"""Frozen Hugging Face DINOv3 ViT-7B features with a PairUAV xyz pose head."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
from transformers import AutoModel
from tqdm import tqdm

from train_pairuav5d import PairDataset, XYZ_SCALE_M, TILT_SCALE_DEG, HEADING_SCALE_DEG


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


def cache_features(paths, transform, backbone, device, batch_size, workers):
    loader = DataLoader(ImagePathDataset(paths, transform), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    features = None
    for images, indices in tqdm(loader, desc="Cache unique image features"):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            batch_features = backbone(pixel_values=images).last_hidden_state[:, 0].float().cpu()
        if features is None:
            features = torch.empty((len(paths), batch_features.shape[1]), dtype=torch.float16)
        features[indices] = batch_features.to(torch.float16)
    return features


def train_epoch(loader, head, optimizer, device):
    head.train()
    losses = []
    for feat_a, feat_b, targets, _ in tqdm(loader, desc="Train head"):
        features = torch.cat((feat_a, feat_b), dim=1).to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = head(features)
        loss = F.smooth_l1_loss(output, targets, beta=0.1)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def evaluate(loader, head, device):
    head.eval()
    pred_rows, true_rows = [], []
    for feat_a, feat_b, _, labels in tqdm(loader, desc="Test head"):
        features = torch.cat((feat_a, feat_b), dim=1).to(device, non_blocking=True).float()
        output = head(features)
        decoded = output.float() * output.new_tensor([XYZ_SCALE_M] * 3 + [TILT_SCALE_DEG, HEADING_SCALE_DEG])
        pred_rows.append(decoded.cpu().numpy())
        true_rows.append(labels.numpy())
    pred, true = np.concatenate(pred_rows), np.concatenate(true_rows)
    xyz_mae = np.abs(pred[:, :3] - true[:, :3]).mean(axis=0)
    distance_mae = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(true[:, :3], axis=1)).mean()
    tilt_mae = np.abs(pred[:, 3] - true[:, 3]).mean()
    heading_mae = np.abs((pred[:, 4] - true[:, 4] + 180.0) % 360.0 - 180.0).mean()
    return pred, true, {
        "model": "DINOv3 ViT-7B (facebook/dinov3-vit7b16-pretrain-lvd1689m)",
        "train_rows": 47687,
        "test_rows": len(true),
        "MAE_x_m": float(xyz_mae[0]),
        "MAE_y_m": float(xyz_mae[1]),
        "MAE_z_m": float(xyz_mae[2]),
        "MAE_distance_from_xyz_m": float(distance_mae),
        "MAE_tilt_deg": float(tilt_mae),
        "MAE_heading_deg_circular": float(heading_mae),
        "distance_definition": "MAE between norms of predicted and ground-truth xyz; no distance head",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("pretrained_vit7b"))
    ap.add_argument("--image-root", type=Path, default=Path("/root/pairUAV-5D-temp"))
    ap.add_argument("--train-dir", type=Path, default=Path("/root/baseline/_mast3r_split_data_v2/train"))
    ap.add_argument("--test-dir", type=Path, default=Path("/root/baseline/_mast3r_split_data_v2/test"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs_xyz7b_mast3r_split"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_info = json.loads((args.model_dir / "config.json").read_text())
    backbone = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    head = nn.Linear(model_info["hidden_size"] * 2, 5).to(device)

    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    crop_size = processor.crop_size or processor.size
    transform = Compose([
        Resize(processor.size.get("shortest_edge", 256)),
        CenterCrop((crop_size.get("height", 224), crop_size.get("width", 224))),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    train_data = PairDataset(args.train_dir, args.image_root, transform)
    test_data = PairDataset(args.test_dir, args.image_root, transform)
    all_samples = train_data.samples + test_data.samples
    image_paths = sorted({str(p) for row in all_samples for p in row[:2]})
    path_index = {p: i for i, p in enumerate(image_paths)}
    feature_path = args.output_dir / "unique_image_features.pt"
    if feature_path.exists():
        image_features = torch.load(feature_path, map_location="cpu", weights_only=True)
    else:
        image_features = cache_features(image_paths, transform, backbone, device, args.batch_size, args.num_workers)
        torch.save(image_features, feature_path)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def pair_tensors(samples):
        ia = torch.tensor([path_index[str(row[0])] for row in samples], dtype=torch.long)
        ib = torch.tensor([path_index[str(row[1])] for row in samples], dtype=torch.long)
        targets = torch.tensor([row[2] for row in samples], dtype=torch.float32)
        labels = torch.tensor([row[3] for row in samples], dtype=torch.float32)
        return TensorDataset(image_features[ia], image_features[ib], targets, labels)

    train_pairs = pair_tensors(train_data.samples)
    test_pairs = pair_tensors(test_data.samples)
    train_loader = DataLoader(train_pairs, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_pairs, batch_size=256, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    epoch_losses = []
    for epoch in range(args.epochs):
        epoch_losses.append(train_epoch(train_loader, head, optimizer, device))
    pred, true, metrics = evaluate(test_loader, head, device)
    metrics.update({"epochs": args.epochs, "epoch_train_loss": epoch_losses,
                    "model_config": {"hidden_size": model_info["hidden_size"], "layers": model_info["num_hidden_layers"]}})
    torch.save({"head": head.state_dict(), "metrics": metrics}, args.output_dir / "head.pt")
    np.savez_compressed(args.output_dir / "test_predictions.npz", pred_xyz_tilt_heading=pred, true_xyz_tilt_heading=true)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
