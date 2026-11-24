#!/usr/bin/env python3
"""Fine-tune the official ViNT / NoMaD checkpoints on the fixed PairUAV-5D split.

Both official models have two heads, a distance head and an action head; both are kept.

ViNT  : official encoder + Transformer -> 32-d policy feature.
        action head  : linear, re-sized to (x, y, z, tilt, sin/cos heading), regression as in ViNT.
        distance head: the official dist_predictor, re-targeted from temporal to metric distance.
NoMaD : official vision encoder (goal un-masked) -> 256-d context.
        action head  : the official conditional 1-D U-Net diffusion policy. The action is an
                       8-step straight-line trajectory to the goal in 5-DoF pose space,
                       delta-encoded and normalised to [-1, 1] as in NoMaD; the predicted pose
                       is the final waypoint (sum of the 8 deltas) of one DDPM sample.
        distance head: the official dist_pred_net, re-targeted to metric distance.

Images come from the uint8 cache written by build_cache.py and live on the GPU.
"""
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path("/root/baseline/visualnav-transformer/pairuav5d_v2")
sys.path.insert(0, str(Path(__file__).parent))
REPO = HERE.parent
sys.path.insert(0, str(REPO / "train"))
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT  # noqa: E402
from vint_train.models.vint.vint import ViNT  # noqa: E402

CACHE = HERE / "cache"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# x, y, z, tilt -> roughly [-1, 1]; heading is encoded as sin/cos.
SCALE = torch.tensor([15.0, 15.0, 33.0, 45.0])
DISTANCE_SCALE = 20.0
NATIVE = {"vint": (85, 64), "nomad": (96, 96)}  # width, height (official configs)
HORIZON = 8


# ----------------------------------------------------------------------------- data
def load_images(size, device, chunk=4096, store=None):
    store = store or device
    width, height = size
    source = np.load(CACHE / "images_256x192.npy", mmap_mode="r")
    if source.shape[0] * 3 * height * width > 4e9:  # too large for the shared GPU: keep on the CPU
        assert (height, width) == tuple(source.shape[1:3])
        return torch.from_numpy(np.ascontiguousarray(source)).permute(0, 3, 1, 2)
    out = torch.empty((source.shape[0], 3, height, width), dtype=torch.uint8, device=store)
    for start in range(0, source.shape[0], chunk):
        block = torch.from_numpy(np.ascontiguousarray(source[start:start + chunk])).to(device)
        block = block.permute(0, 3, 1, 2).float()
        if (height, width) != tuple(block.shape[-2:]):
            block = F.interpolate(block, size=(height, width), mode="bilinear", antialias=True, align_corners=False)
        out[start:start + chunk] = block.round().clamp(0, 255).to(torch.uint8).to(store)
    return out


def load_index(name, device, frame="enu"):
    data = np.load(CACHE / f"{name}_index.npz")
    target = data["target"].copy()
    swapped = -target  # east-north frame: swapping the pair negates every component
    if frame == "cam":
        from camlabels import by_pair_id
        t = by_pair_id().loc[data["pair_id"]]
        assert np.abs(t[["x", "y", "z", "tilt", "heading"]].values - target).max() < 1e-9
        target[:, 0], target[:, 1] = t.lx.values, t.ly.values          # source-camera levelled frame
        swapped[:, 0], swapped[:, 1] = t.lx_sw.values, t.ly_sw.values  # swapped pair: frame of camera b
    return {
        "a": torch.from_numpy(data["a"]).to(device),
        "b": torch.from_numpy(data["b"]).to(device),
        "target": torch.from_numpy(target).float().to(device),
        "target_swap": torch.from_numpy(swapped).float().to(device),
        "tour": data["tour"],
        "pair_id": data["pair_id"],
    }


def encode_targets(target):
    heading = torch.deg2rad(target[:, 4])
    return torch.cat((target[:, :4] / SCALE.to(target.device), heading.sin()[:, None], heading.cos()[:, None]), dim=1)


def decode_predictions(output):
    heading = torch.rad2deg(torch.atan2(output[:, 4], output[:, 5]))
    return torch.cat((output[:, :4] * SCALE.to(output.device), heading[:, None]), dim=1)


def distance_target(target):
    return torch.linalg.vector_norm(target[:, :3], dim=1) / DISTANCE_SCALE


def take(images, rows, device):
    if images.device.type == "cpu":
        return images[rows.cpu()].to(device, non_blocking=True)
    return images[rows]


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, image_uint8, jitter=False):
        image = image_uint8.float()
        if jitter:
            batch = image.shape[0]
            contrast = 1.0 + (torch.rand(batch, 1, 1, 1, device=image.device) - 0.5) * 0.4
            brightness = (torch.rand(batch, 1, 1, 1, device=image.device) - 0.5) * 0.2 * 255.0
            mean = image.mean(dim=(1, 2, 3), keepdim=True)
            image = ((image - mean) * contrast + mean + brightness).clamp(0, 255)
        return (image - self.mean) / self.std


# --------------------------------------------------------------------------- models
def pooled(encoder, image):
    feature = encoder._avg_pooling(encoder.extract_features(image)).flatten(start_dim=1)
    return encoder._dropout(feature)


class ViNTAdapter(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        official = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
        self.net = ViNT(context_size=5, len_traj_pred=5, learn_angle=True, obs_encoder="efficientnet-b0",
                        obs_encoding_size=512, late_fusion=False, mha_num_attention_heads=4,
                        mha_num_attention_layers=4, mha_ff_dim_factor=4)
        self.net.load_state_dict(official.state_dict(), strict=True)
        del self.net.action_predictor  # shape depends on the action space; replaced by self.head
        self.head = nn.Linear(32, 6)
        # The distance head is kept, but its output unit changes from temporal steps to metres, so the
        # layer is re-initialised (the pretrained scale swamps the shared feature with gradient otherwise).
        self.net.dist_predictor[0].reset_parameters()

    def forward(self, image_a, image_b):
        net = self.net
        obs = net.compress_obs_enc(pooled(net.obs_encoder, image_a))
        goal = net.compress_goal_enc(pooled(net.goal_encoder, torch.cat((image_a, image_b), dim=1)))
        # The official model sees context_size + 1 = 6 observation frames; a single still image
        # is a static history, so all six observation tokens are the same encoding.
        tokens = torch.cat((obs[:, None].expand(-1, net.context_size + 1, -1), goal[:, None]), dim=1)
        feature = net.decoder(tokens)
        return torch.cat((self.head(feature), net.dist_predictor(feature)), dim=1)  # [action(6), distance(1)]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, kernel, groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(inp, out, kernel, padding=kernel // 2), nn.GroupNorm(groups, out), nn.Mish())

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, inp, out, cond_dim, kernel=3, groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(inp, out, kernel, groups), Conv1dBlock(out, out, kernel, groups)])
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out))
        self.residual_conv = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x) + self.cond_encoder(cond)[:, :, None]
        return self.blocks[1](out) + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Same layout and parameter names as diffusion_policy's ConditionalUnet1D used by NoMaD."""

    def __init__(self, input_dim, global_cond_dim, step_dim=256, down_dims=(64, 128, 256), kernel=3, groups=8):
        super().__init__()
        dims = [input_dim] + list(down_dims)
        cond_dim = step_dim + global_cond_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(step_dim), nn.Linear(step_dim, step_dim * 4), nn.Mish(), nn.Linear(step_dim * 4, step_dim))
        in_out = list(zip(dims[:-1], dims[1:]))
        mid = dims[-1]
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid, mid, cond_dim, kernel, groups) for _ in range(2)])
        self.down_modules = nn.ModuleList([
            nn.ModuleList([ConditionalResidualBlock1D(i, o, cond_dim, kernel, groups),
                           ConditionalResidualBlock1D(o, o, cond_dim, kernel, groups),
                           Downsample1d(o) if index < len(in_out) - 1 else nn.Identity()])
            for index, (i, o) in enumerate(in_out)])
        self.up_modules = nn.ModuleList([
            nn.ModuleList([ConditionalResidualBlock1D(o * 2, i, cond_dim, kernel, groups),
                           ConditionalResidualBlock1D(i, i, cond_dim, kernel, groups),
                           Upsample1d(i) if index < len(in_out) - 1 else nn.Identity()])
            for index, (i, o) in enumerate(reversed(in_out[1:]))])
        self.final_conv = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], kernel), nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        x = sample.transpose(1, 2)
        cond = torch.cat((self.diffusion_step_encoder(timestep.float()), global_cond), dim=-1)
        skips = []
        for first, second, down in self.down_modules:
            x = second(first(x, cond), cond)
            skips.append(x)
            x = down(x)
        for block in self.mid_modules:
            x = block(x, cond)
        for first, second, up in self.up_modules:
            x = up(second(first(torch.cat((x, skips.pop()), dim=1), cond), cond))
        return self.final_conv(x).transpose(1, 2)


class DDPM:
    """diffusers.DDPMScheduler(squaredcos_cap_v2, epsilon, clip_sample=True, fixed_small) as used by NoMaD."""

    def __init__(self, steps, device):
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = [min(1 - alpha_bar((i + 1) / steps) / alpha_bar(i / steps), 0.999) for i in range(steps)]
        self.steps = steps
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    def add_noise(self, clean, noise, timestep):
        acp = self.alphas_cumprod[timestep].view(-1, 1, 1)
        return acp.sqrt() * clean + (1 - acp).sqrt() * noise

    def step(self, eps, t, sample, generator):
        acp_t = self.alphas_cumprod[t]
        acp_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.ones((), device=sample.device)
        beta_t = 1 - acp_t / acp_prev
        clean = ((sample - (1 - acp_t).sqrt() * eps) / acp_t.sqrt()).clamp(-1, 1)
        mean = acp_prev.sqrt() * beta_t / (1 - acp_t) * clean + (1 - beta_t).sqrt() * (1 - acp_prev) / (1 - acp_t) * sample
        if t == 0:
            return mean
        variance = ((1 - acp_prev) / (1 - acp_t) * beta_t).clamp(min=1e-20)
        return mean + variance.sqrt() * torch.randn(sample.shape, device=sample.device, generator=generator)


class NoMaDAdapter(nn.Module):
    def __init__(self, checkpoint, diffusion_steps=10):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.vision_encoder = NoMaD_ViNT(context_size=3, obs_encoder="efficientnet-b0", obs_encoding_size=256,
                                         mha_num_attention_heads=4, mha_num_attention_layers=4, mha_ff_dim_factor=4)
        vision = {k.removeprefix("vision_encoder."): v for k, v in state.items() if k.startswith("vision_encoder.")}
        self.vision_encoder.load_state_dict(vision, strict=True)
        self.noise_pred_net = ConditionalUnet1D(input_dim=6, global_cond_dim=256)
        own = self.noise_pred_net.state_dict()
        policy = {k.removeprefix("noise_pred_net."): v for k, v in state.items() if k.startswith("noise_pred_net.")}
        assert set(policy) == set(own), (set(policy) ^ set(own))
        reused = {k: v for k, v in policy.items() if v.shape == own[k].shape}
        self.reinitialised = sorted(set(own) - set(reused))  # only the layers whose shape depends on the action dim
        self.noise_pred_net.load_state_dict(reused, strict=False)
        self.dist_pred_net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
        self.dist_pred_net.load_state_dict({k.removeprefix("dist_pred_net.network."): v for k, v in state.items()
                                            if k.startswith("dist_pred_net.")}, strict=True)
        self.dist_pred_net[-1].reset_parameters()  # output unit changes from temporal steps to metres
        self.diffusion_steps = diffusion_steps

    def condition(self, image_a, image_b):
        enc = self.vision_encoder
        obs = enc.compress_obs_enc(pooled(enc.obs_encoder, image_a))
        goal = enc.compress_goal_enc(pooled(enc.goal_encoder, torch.cat((image_a, image_b), dim=1)))
        tokens = torch.cat((obs[:, None].expand(-1, enc.context_size + 1, -1), goal[:, None]), dim=1)
        tokens = enc.positional_encoding(tokens)
        return enc.sa_encoder(tokens).mean(dim=1)  # goal un-masked: plain mean over tokens

    def loss(self, image_a, image_b, encoded_target, distance, ddpm, draws=1, distance_weight=0.1):
        cond = self.condition(image_a, image_b).float()
        distance_loss = F.smooth_l1_loss(self.dist_pred_net(cond)[:, 0], distance, beta=0.05)
        cond = cond.repeat(draws, 1)
        # 8 equal delta steps of a straight-line trajectory; their sum is the relative pose.
        action = encoded_target.repeat(draws, 1)[:, None, :].expand(-1, HORIZON, -1)
        noise = torch.randn_like(action)
        timestep = torch.randint(0, ddpm.steps, (action.shape[0],), device=action.device)
        predicted = self.noise_pred_net(ddpm.add_noise(action, noise, timestep), timestep, cond)
        return F.mse_loss(predicted, noise) + distance_weight * distance_loss

    @torch.no_grad()
    def sample(self, image_a, image_b, ddpm, generator, samples=1):
        cond = self.condition(image_a, image_b).float()
        distance = self.dist_pred_net(cond)[:, 0]
        cond = cond.repeat_interleave(samples, dim=0)
        action = torch.randn((cond.shape[0], HORIZON, 6), device=cond.device, generator=generator)
        for t in reversed(range(ddpm.steps)):
            timestep = torch.full((cond.shape[0],), t, device=cond.device, dtype=torch.long)
            action = ddpm.step(self.noise_pred_net(action, timestep, cond), t, action, generator)
        # mean delta * 8 / 8 = final waypoint in normalised units
        return action.mean(dim=1).view(-1, samples, 6), distance


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].lerp_(value.detach(), 1 - decay)
            else:
                self.shadow[key].copy_(value)


# ------------------------------------------------------------------------- evaluate
def metrics(pred, truth, distance=None):
    """distance=None: distance is the norm of the predicted xyz; otherwise the distance-head output (m)."""
    if distance is None:
        distance = torch.linalg.vector_norm(pred[:, :3], dim=1)
    distance_error = (distance - torch.linalg.vector_norm(truth[:, :3], dim=1)).abs()
    heading_error = (torch.remainder(pred[:, 4] - truth[:, 4] + 180, 360) - 180).abs()
    axis = (pred[:, :3] - truth[:, :3]).abs().mean(dim=0)
    return {
        "MAE_x_m": axis[0].item(), "MAE_y_m": axis[1].item(), "MAE_z_m": axis[2].item(),
        "MAE_distance_from_xyz_m": distance_error.mean().item(),
        "MAE_tilt_deg": (pred[:, 3] - truth[:, 3]).abs().mean().item(),
        "MAE_heading_deg_circular": heading_error.mean().item(),
        "distance_success_count": int((distance_error <= 1.0).sum()),
        "distance_success_rate": (distance_error <= 1.0).float().mean().item(),
    }


@torch.no_grad()
def predict(model, kind, images, index, rows, normalizer, ddpm, batch_size=256, seed=0, samples=1):
    model.eval()
    generator = torch.Generator(device=rows.device).manual_seed(seed)
    outputs, distances = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        a = normalizer(take(images, index["a"][chunk], rows.device)).contiguous(memory_format=torch.channels_last)
        b = normalizer(take(images, index["b"][chunk], rows.device)).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            if kind == "vint":
                out = model(a, b).float()
                out, distance = out[:, None, :6], out[:, 6]
            else:
                out, distance = model.sample(a, b, ddpm, generator, samples=samples)
        outputs.append(out.float())
        distances.append(distance.float())
    raw = torch.cat(outputs)  # [N, samples, 6]
    first = decode_predictions(raw[:, 0])
    mean = decode_predictions(raw.mean(dim=1))
    return first, mean, (torch.cat(distances) * DISTANCE_SCALE).clamp(min=0)


def summarise(pred, truth, distance=None):
    result = metrics(pred, truth, distance)
    correlation = []
    for i in range(5):
        correlation.append(float(torch.corrcoef(torch.stack((pred[:, i], truth[:, i])))[0, 1]))
    result["prediction_std"] = pred.std(dim=0).tolist()
    result["target_std"] = truth.std(dim=0).tolist()
    result["prediction_target_correlation"] = correlation
    return result


def short(m):
    return ("x %.2f y %.2f z %.2f dist %.2f tilt %.2f head %.2f succ %.2f%%" % (
        m["MAE_x_m"], m["MAE_y_m"], m["MAE_z_m"], m["MAE_distance_from_xyz_m"], m["MAE_tilt_deg"],
        m["MAE_heading_deg_circular"], 100 * m["distance_success_rate"]))


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["vint", "nomad"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=str, default=None, help="WxH; default is the official input size")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--val-tours", type=int, default=0, help="train tours held out for monitoring")
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--noise-draws", type=int, default=4, help="NoMaD only: diffusion noise draws per pair")
    parser.add_argument("--distance-weight", type=float, default=0.1, help="weight of the distance-head loss")
    parser.add_argument("--test", action="store_true", help="evaluate the final model on the fixed test split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="code check only: 5 steps per epoch, 512 test rows")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--micro-batches", type=int, default=1, help="NoMaD: accumulate the batch gradient over this many chunks (memory only)")
    parser.add_argument("--frame", choices=["enu", "cam"], default="enu", help="frame of the horizontal displacement target")
    parser.add_argument("--xy-scale", type=float, default=15.0, help="normalisation of the two horizontal components (15 = original)")
    parser.add_argument("--images-on-cpu", action="store_true", help="keep the uint8 image cache on the CPU (same pixels)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / "train.log")])
    logging.info("args %s", vars(args))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    size = tuple(int(v) for v in args.size.split("x")) if args.size else NATIVE[args.model]

    SCALE[0] = SCALE[1] = args.xy_scale
    images = load_images(size, device, chunk=512 if args.images_on_cpu else 4096, store=torch.device("cpu") if args.images_on_cpu else None)
    torch.cuda.empty_cache()  # release the resize workspace (shared GPU)
    train, test = load_index("train", device, args.frame), load_index("test", device, args.frame)
    logging.info("images %s on %s, train %d test %d", tuple(images.shape), images.device, len(train["a"]), len(test["a"]))

    tours = sorted(set(train["tour"].tolist()))
    rng = np.random.default_rng(1234)
    held_out = set(rng.choice(tours, size=args.val_tours, replace=False).tolist()) if args.val_tours else set()
    is_val = np.array([t in held_out for t in train["tour"]])
    train_rows = torch.from_numpy(np.flatnonzero(~is_val)).to(device)
    val_rows = torch.from_numpy(np.flatnonzero(is_val)).to(device)
    logging.info("train rows %d, held-out train-tour rows %d", len(train_rows), len(val_rows))

    checkpoint = REPO / "pretrained" / f"{args.model}.pth"
    model = ViNTAdapter(checkpoint) if args.model == "vint" else NoMaDAdapter(checkpoint)
    if args.no_pretrained:
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    if args.model == "nomad":
        logging.info("NoMaD policy layers re-initialised for the new action dim: %s", model.reinitialised)
    model = model.to(device).to(memory_format=torch.channels_last)
    normalizer = Normalizer().to(device)
    ddpm = DDPM(10, device) if args.model == "nomad" else None

    head_names = (("head.", "net.dist_predictor.") if args.model == "vint" else
                  tuple(f"noise_pred_net.{k}" for k in model.reinitialised) + ("dist_pred_net.4.",))
    head = [p for n, p in model.named_parameters() if n.startswith(head_names)]
    body = [p for n, p in model.named_parameters() if not n.startswith(head_names)]
    optimizer = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr or args.lr}],
                                  weight_decay=args.weight_decay)
    steps_per_epoch = len(train_rows) // args.batch_size
    if args.smoke:
        steps_per_epoch, val_rows = 5, val_rows[:512]
    total = steps_per_epoch * args.epochs
    warmup = int(steps_per_epoch * args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup if s < warmup else
                                                  0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model)
    eval_model = ViNTAdapter(checkpoint) if args.model == "vint" else NoMaDAdapter(checkpoint)
    eval_model = eval_model.to(device).to(memory_format=torch.channels_last)

    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows), device=device)]
        running = 0.0
        for step in range(steps_per_epoch):
            rows = order[step * args.batch_size:(step + 1) * args.batch_size]
            ia, ib, target = train["a"][rows], train["b"][rows], train["target"][rows]
            if not args.no_swap:  # the label is antisymmetric: swapping the pair negates every component
                swap = torch.rand(len(rows), device=device) < 0.5
                ia, ib = torch.where(swap, ib, ia), torch.where(swap, ia, ib)
                target = torch.where(swap[:, None], train["target_swap"][rows], target)
            a = normalizer(take(images, ia, device), jitter=not args.no_jitter).contiguous(memory_format=torch.channels_last)
            b = normalizer(take(images, ib, device), jitter=not args.no_jitter).contiguous(memory_format=torch.channels_last)
            encoded = encode_targets(target)
            with torch.autocast("cuda", dtype=torch.float16):
                if args.model == "vint":
                    output = model(a, b).float()
                    loss = (F.smooth_l1_loss(output[:, :6], encoded, beta=0.05)
                            + args.distance_weight * F.smooth_l1_loss(output[:, 6], distance_target(target), beta=0.05))
                elif args.micro_batches <= 1:
                    loss = model.loss(a, b, encoded, distance_target(target), ddpm, draws=args.noise_draws,
                                      distance_weight=args.distance_weight)
            optimizer.zero_grad(set_to_none=True)
            if args.model == "nomad" and args.micro_batches > 1:
                loss = torch.zeros((), device=device)
                for part in torch.arange(len(rows), device=device).chunk(args.micro_batches):
                    with torch.autocast("cuda", dtype=torch.float16):
                        part_loss = model.loss(a[part].contiguous(memory_format=torch.channels_last), b[part].contiguous(memory_format=torch.channels_last), encoded[part], distance_target(target[part]), ddpm,
                                               draws=args.noise_draws, distance_weight=args.distance_weight) * (len(part) / len(rows))
                    scaler.scale(part_loss).backward()
                    loss = loss + part_loss.detach()
            else:
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running += loss.item()
        entry = {"epoch": epoch + 1, "train_loss": running / steps_per_epoch, "minutes": (time.time() - started) / 60}
        message = "epoch %d loss %.5f (%.1f min)" % (epoch + 1, entry["train_loss"], entry["minutes"])
        if len(val_rows) and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            eval_model.load_state_dict(ema.shadow)
            first, _, distance = predict(eval_model, args.model, images, train, val_rows, normalizer, ddpm)
            entry["val"] = metrics(first, train["target"][val_rows], distance)
            entry["val_distance_from_xyz"] = metrics(first, train["target"][val_rows])
            message += " | val " + short(entry["val"]) + " | xyz-norm dist %.2f succ %.2f%%" % (
                entry["val_distance_from_xyz"]["MAE_distance_from_xyz_m"],
                100 * entry["val_distance_from_xyz"]["distance_success_rate"])
        history.append(entry)
        logging.info(message)

    eval_model.load_state_dict(ema.shadow)
    torch.save({"model": ema.shadow, "args": {k: str(v) for k, v in vars(args).items()}, "size": size}, args.out / "final.pt")
    result = {
        "model": args.model, "official_checkpoint": str(checkpoint), "input_size_wh": list(size),
        "epochs": args.epochs, "train_rows": int(len(train_rows)), "held_out_train_tour_rows": int(len(val_rows)),
        "history": history, "checkpoint_selection": "final epoch (EMA weights); no test-set model selection",
        "distance_success_threshold_m": 1.0, "frame": args.frame, "xy_scale": args.xy_scale,
    }
    if args.test:
        rows = torch.arange(512 if args.smoke else len(test["a"]), device=device)
        truth = test["target"][rows]
        first, mean, distance = predict(eval_model, args.model, images, test, rows, normalizer, ddpm,
                                        samples=8 if args.model == "nomad" else 1)
        result["test_rows"] = len(rows)
        result.update(summarise(first, truth, distance))
        result["MAE_distance_m"] = result.pop("MAE_distance_from_xyz_m")
        result["distance_definition"] = ("official distance head re-targeted to metric distance: "
                                         "abs(distance_head - norm(true_xyz)); success = this error <= 1 m")
        result["distance_from_xyz_norm"] = {k: v for k, v in metrics(first, truth).items() if "distance" in k}
        result["zero_predictor"] = metrics(torch.zeros_like(truth), truth)
        if args.model == "nomad":
            result["prediction"] = "first of 8 DDPM samples (as in NoMaD deployment), 10 denoising steps, seed 0"
            result["mean_of_8_samples"] = summarise(mean, truth, distance)
        logging.info("TEST %s | xyz-norm %s", short({**result, "MAE_distance_from_xyz_m": result["MAE_distance_m"]}),
                     result["distance_from_xyz_norm"])
        np.savez_compressed(args.out / "test_predictions.npz", pred_xyz_tilt_heading=first.cpu().numpy(),
                            pred_distance_head_m=distance.cpu().numpy(), true_xyz_tilt_heading=truth.cpu().numpy(),
                            pred_mean_xyz_tilt_heading=mean.cpu().numpy(), pair_id=test["pair_id"][:len(rows)], frame=args.frame)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
