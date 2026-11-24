#!/usr/bin/env python3
"""Adapt official ViNT/NoMaD pretrained visual-navigation models to PairUAV-5D."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "train"))
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT

sys.path.insert(0, "/root/baseline/dinov3")
from train_pairuav5d import PairDataset

ROOT = Path("/root/baseline")
IMAGE_ROOT = Path("/root/pairUAV-5D-temp")
TRAIN_DIR = ROOT / "_mast3r_split_data_v2/train"
TEST_DIR = ROOT / "_mast3r_split_data_v2/test"
ABS_HEADING_PREDICTIONS = Path("/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/absheading/pred_cnx_s_d8.pkl")
XYZ_TILT_SCALES = torch.tensor([40.0, 40.0, 40.0, 45.0])


def encode_targets(target):
    heading_radians = target[:, 4] * torch.pi
    return torch.cat((target[:, :4], heading_radians.sin().unsqueeze(1),
                      heading_radians.cos().unsqueeze(1)), dim=1)


def decode_predictions(output):
    xyz_tilt = output[:, :4] * XYZ_TILT_SCALES.to(output.device)
    heading = torch.atan2(output[:, 4], output[:, 5]) * (180.0 / torch.pi)
    return torch.cat((xyz_tilt, heading.unsqueeze(1)), dim=1)


class AdaptedPolicy(nn.Module):
    def __init__(self, name, pretrained, device):
        super().__init__()
        self.name = name
        if name == "vint":
            checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
            self.encoder = checkpoint["model"]
            # ViNT's serialized TransformerEncoderLayer predates this PyTorch runtime field.
            for module in self.encoder.modules():
                if isinstance(module, nn.TransformerEncoderLayer) and not hasattr(module, "activation_relu_or_gelu"):
                    module.activation_relu_or_gelu = 2
            self.feature = None
            self.feature_dim = 32
            self.image_size = 224
        else:
            self.encoder = NoMaD_ViNT(
                context_size=3, obs_encoder="efficientnet-b0", obs_encoding_size=256,
                mha_num_attention_heads=4, mha_num_attention_layers=4,
                mha_ff_dim_factor=4,
            )
            state = torch.load(pretrained, map_location="cpu", weights_only=False)
            vision_state = {key.removeprefix("vision_encoder."): value
                            for key, value in state.items() if key.startswith("vision_encoder.")}
            missing, unexpected = self.encoder.load_state_dict(vision_state, strict=False)
            if len(missing) > 20 or unexpected:
                raise RuntimeError(f"NoMaD checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
            self.feature_dim = 256
            self.image_size = 96
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 6),
        )
        if name == "vint":
            train_modules = [self.encoder.obs_encoder, self.encoder.goal_encoder, self.encoder.decoder,
                             self.encoder.compress_obs_enc, self.encoder.compress_goal_enc]
        else:
            train_modules = [self.encoder.goal_encoder, self.encoder.sa_encoder, self.encoder.compress_obs_enc,
                             self.encoder.compress_goal_enc, self.encoder.positional_encoding]
        self.train_modules = train_modules
        for module in train_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self._device = device
        if name == "vint":
            self._captured = None
            self.encoder.decoder.register_forward_hook(self._save_feature)

    def _save_feature(self, module, args, output):
        self._captured = output

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        for module in self.train_modules:
            module.train(mode)
        return self

    def encode(self, image_a, image_b):
        if self.name == "vint":
            # Official ViNT uses six past frames, then fuses the latest frame with the goal.
            obs = image_a.repeat(1, 6, 1, 1)
            self._captured = None
            self.encoder(obs, image_b)
            return self._captured
        # Official NoMaD encoder expects context_size + 1 observations.
        obs = image_a.repeat(1, 4, 1, 1)
        mask = torch.zeros(image_a.shape[0], dtype=torch.long, device=image_a.device)
        return self.encoder(obs, image_b, input_goal_mask=mask)

    def forward(self, image_a, image_b):
        feature = self.encode(image_a, image_b)
        return self.regressor(feature)


class PairImageTransform:
    def __init__(self, image_size):
        self.height, self.width = image_size

    def __call__(self, image):
        if hasattr(image, "draft"):
            image.draft("RGB", (340, 240))
        image = image.convert("RGB")
        width, height = image.size
        aspect = 4.0 / 3.0
        if width > height:
            crop_width = int(height * aspect)
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
        else:
            crop_height = int(width / aspect)
            top = (height - crop_height) // 2
            image = image.crop((0, top, width, top + crop_height))
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        return transforms.ToTensor()(image)


class PairData(Dataset):
    def __init__(self, directory, image_size, camera_frame_targets=False):
        transform = PairImageTransform(image_size)
        base = PairDataset(directory, IMAGE_ROOT, transform)
        self.base = base
        self.camera_frame_targets = camera_frame_targets
        self.image_indices = None
        self.cached_images = None
        pair_table = pd.read_pickle("/root/rasr-pairuav/outputs/pairuav5d_60k/superglue/pairs.pkl")
        heading_by_pair = {
            (str(row.image_a), str(row.image_b)): float(row.heading_a)
            for row in pair_table.itertuples()
        }
        self.metadata = []
        for image_a, image_b, _, _ in base.samples:
            key_a = image_a.relative_to(IMAGE_ROOT).as_posix()
            key_b = image_b.relative_to(IMAGE_ROOT).as_posix()
            pair_key = (key_a, key_b)
            if pair_key not in heading_by_pair:
                raise KeyError(f"Pair metadata missing for {pair_key}")
            self.metadata.append((heading_by_pair[pair_key], key_a, key_b))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        if self.image_indices is None:
            a, b, target, labels = self.base[index]
        else:
            _, _, target, labels = self.base.samples[index]
            target = torch.as_tensor(target, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.float32)
            a_index, b_index = self.image_indices[index]
            a = self.cached_images[a_index].float()
            b = self.cached_images[b_index].float()
        heading_a, image_a_key, image_b_key = self.metadata[index]
        target = target.float().clone()
        if self.camera_frame_targets:
            angle = torch.deg2rad(torch.tensor(heading_a, dtype=target.dtype))
            x, y = target[0].clone(), target[1].clone()
            target[0] = torch.cos(angle) * x - torch.sin(angle) * y
            target[1] = torch.sin(angle) * x + torch.cos(angle) * y
        heading_a_radians = torch.deg2rad(torch.tensor(heading_a, dtype=torch.float32))
        return a, b, target, labels.float(), heading_a_radians, image_a_key, image_b_key


class ImageCacheDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image)
        return tensor.to(torch.float16), index


def cache_pair_images(train_data, test_data, output_dir, workers):
    samples = train_data.base.samples + test_data.base.samples
    paths = sorted({str(path) for row in samples for path in row[:2]})
    cache_path = output_dir / "image_cache.pt"
    path_index = {path: index for index, path in enumerate(paths)}
    if cache_path.exists():
        cached_paths, features = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached_paths != paths:
            raise RuntimeError("Cached ViNT images do not match the fixed split image list")
    else:
        loader = DataLoader(
            ImageCacheDataset(paths, train_data.base.transform), batch_size=512,
            shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        )
        chunks = []
        for images, _ in tqdm(loader, desc="Cache fixed-split images"):
            chunks.append(images.contiguous())
        features = torch.cat(chunks, dim=0)
        torch.save((paths, features), cache_path)
    for dataset in (train_data, test_data):
        dataset.cached_images = features
        dataset.image_indices = [
            (path_index[str(row[0])], path_index[str(row[1])])
            for row in dataset.base.samples
        ]
    logging.info("cached %d images, %.2f GiB", len(paths), features.numel() * features.element_size() / (1024 ** 3))


def metrics(pred, truth):
    err = (pred[:, :3] - truth[:, :3]).abs().mean(dim=0)
    dist = (torch.linalg.vector_norm(pred[:, :3], dim=1) - torch.linalg.vector_norm(truth[:, :3], dim=1)).abs().mean()
    tilt = (pred[:, 3] - truth[:, 3]).abs().mean()
    heading = (torch.remainder(pred[:, 4] - truth[:, 4] + 180, 360) - 180).abs().mean()
    return [float(x) for x in (*err, dist, tilt, heading)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["vint", "nomad"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--vision-lr", type=float, default=1e-4)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--camera-frame-targets", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--xyz-loss-weight", type=float, default=1.0)
    args = parser.parse_args()
    ckpt = REPO / "pretrained" / f"{args.model}.pth"
    out_name = ("outputs_pairuav5d_vint_camera_frame_tuned" if args.model == "vint" and args.camera_frame_targets
                else f"outputs_pairuav5d_{args.model}_adapted")
    out = REPO / out_name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")],
        force=True,
    )
    image_size = (64, 85) if args.model == "vint" else (96, 96)
    train_data = PairData(TRAIN_DIR, image_size, camera_frame_targets=args.camera_frame_targets)
    test_data = PairData(TEST_DIR, image_size, camera_frame_targets=args.camera_frame_targets)
    if args.cache_images:
        cache_pair_images(train_data, test_data, out, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptedPolicy(args.model, ckpt, device).to(device)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    trainable = list(model.regressor.parameters()) + [
        parameter for module in model.train_modules for parameter in module.parameters() if parameter.requires_grad
    ]
    visual_parameters = [p for module_name in ("obs_encoder", "goal_encoder")
                         for p in getattr(model.encoder, module_name).parameters() if p.requires_grad]
    visual_ids = {id(p) for p in visual_parameters}
    head_parameters = list(model.regressor.parameters())
    head_ids = {id(p) for p in head_parameters}
    policy_parameters = [p for p in trainable if id(p) not in visual_ids and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": visual_parameters, "lr": args.vision_lr},
        {"params": policy_parameters, "lr": args.policy_lr},
        {"params": head_parameters, "lr": args.head_lr},
    ], weight_decay=1e-4)
    epoch_losses = []
    start_epoch = 0
    if args.resume_checkpoint is not None:
        state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        epoch_losses = state.get("train_smooth_l1_by_epoch", [])
        start_epoch = int(state["epochs_completed"])
        logging.info("resuming from completed epoch %d", start_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = []
        for a, b, target, _, _, _, _ in tqdm(train_loader, desc=f"{args.model} epoch {epoch + 1}"):
            a, b, target = a.to(device, non_blocking=True), b.to(device, non_blocking=True), target.to(device, non_blocking=True)
            pred = model(a, b)
            encoded_target = encode_targets(target)
            xyz_loss = F.smooth_l1_loss(pred[:, :3], encoded_target[:, :3], beta=0.1)
            auxiliary_loss = F.smooth_l1_loss(pred[:, 3:], encoded_target[:, 3:], beta=0.1)
            loss = args.xyz_loss_weight * xyz_loss + auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            running.append(loss.item())
        epoch_losses.append(float(np.mean(running)))
        logging.info("%s epoch %d train smooth_l1=%.6f", args.model, epoch + 1, epoch_losses[-1])
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epochs_completed": epoch + 1,
            "train_smooth_l1_by_epoch": epoch_losses,
        }, out / "progress.pt")
    model.eval()
    preds, truths = [], []
    heading_table = None
    if args.camera_frame_targets:
        heading_table = pd.read_pickle(ABS_HEADING_PREDICTIONS).set_index("image")
    with torch.inference_mode():
        for a, b, _, truth, _, image_a_keys, image_b_keys in tqdm(test_loader, desc=f"{args.model} test"):
            local = decode_predictions(model(a.to(device, non_blocking=True), b.to(device, non_blocking=True)))
            if heading_table is not None:
                abs_a = heading_table.reindex(image_a_keys)
                abs_b = heading_table.reindex(image_b_keys)
                if abs_a.ah_pred.isna().any() or abs_b.ah_pred.isna().any():
                    raise RuntimeError("Absolute-heading predictions are missing for test images")
                rel = local[:, 4].cpu().numpy()
                ha = abs_a.ah_pred.to_numpy()
                hb_adjusted = abs_b.ah_pred.to_numpy() - rel
                ka = abs_a.ah_kappa.to_numpy()
                kb = abs_b.ah_kappa.to_numpy()
                sine = ka * np.sin(np.deg2rad(ha)) + kb * np.sin(np.deg2rad(hb_adjusted))
                cosine = ka * np.cos(np.deg2rad(ha)) + kb * np.cos(np.deg2rad(hb_adjusted))
                abs_heading = torch.as_tensor(np.rad2deg(np.arctan2(sine, cosine)), device=device, dtype=local.dtype)
                angle = torch.deg2rad(abs_heading)
                world_x = torch.cos(angle) * local[:, 0] + torch.sin(angle) * local[:, 1]
                world_y = -torch.sin(angle) * local[:, 0] + torch.cos(angle) * local[:, 1]
                prediction = torch.stack((world_x, world_y, local[:, 2], local[:, 3], local[:, 4]), dim=1)
            else:
                prediction = local
            preds.append(prediction.cpu())
            truths.append(truth)
    pred, truth = torch.cat(preds), torch.cat(truths)
    values = metrics(pred, truth)
    zero = torch.zeros_like(truth)
    zero_values = metrics(zero, truth)
    correlations = []
    for index in range(5):
        if pred[:, index].std() == 0 or truth[:, index].std() == 0:
            correlations.append(None)
        else:
            correlations.append(float(torch.corrcoef(torch.stack((pred[:, index], truth[:, index])))[0, 1]))
    distance_error = (torch.linalg.vector_norm(pred[:, :3], dim=1) - torch.linalg.vector_norm(truth[:, :3], dim=1)).abs()
    success_count = int((distance_error <= 1.0).sum())
    result = {
        "model": f"{args.model.upper()} official pretrained visual-navigation model, PairUAV fine-tuned fusion/transformer and xyz/tilt/heading head",
        "official_checkpoint": str(ckpt), "adaptation": ("camera-a local xyz/tilt/relative-heading supervision; convert to ENU using the saved image absolute-heading model"
                       if args.camera_frame_targets else "fine-tuned official goal/current image fusion EfficientNet, policy Transformer, and supervised output head; observation context encoder frozen"),
        "epochs": args.epochs, "train_rows": len(train_data), "test_rows": len(test_data),
        "xyz_loss_weight": args.xyz_loss_weight,
        "train_smooth_l1_by_epoch": epoch_losses,
        "MAE_x_m": values[0], "MAE_y_m": values[1], "MAE_z_m": values[2],
        "MAE_distance_from_xyz_m": values[3], "MAE_tilt_deg": values[4], "MAE_heading_deg_circular": values[5],
        "zero_predictor_MAE": {"x_m": zero_values[0], "y_m": zero_values[1], "z_m": zero_values[2],
                               "distance_from_xyz_m": zero_values[3], "tilt_deg": zero_values[4],
                               "heading_deg_circular": zero_values[5]},
        "prediction_std": pred.std(dim=0).tolist(), "target_std": truth.std(dim=0).tolist(),
        "prediction_target_correlation": correlations,
        "distance_success_threshold_m": 1.0,
        "distance_success_count": success_count,
        "distance_success_rate": float(success_count / len(truth)),
        "distance_definition": "MAE between predicted and target xyz vector norms; no distance prediction head",
        "checkpoint_selection": "final epoch; no test-set model selection",
    }
    logging.info(json.dumps(result))
    torch.save({"model": model.state_dict(), "metrics": result}, out / "best.pt")
    np.savez_compressed(out / "test_predictions.npz", pred_xyz_tilt_heading=pred.numpy(), true_xyz_tilt_heading=truth.numpy())
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
