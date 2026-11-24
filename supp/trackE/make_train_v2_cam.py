#!/usr/bin/env python3
"""Generate supp/trackE/train_v2_cam.py from the reported ViNT / NoMaD recipe (pairuav5d_v2/train_v2.py).

The copy is the original file with the textual edits listed below and nothing else, so the recipe diff is exactly this list.
--frame enu reproduces the original behaviour (swap label = negated label, xy scale 15)."""
from pathlib import Path
SRC = Path('/root/baseline/visualnav-transformer/pairuav5d_v2/train_v2.py'); DST = Path(__file__).parent / 'train_v2_cam.py'
s = SRC.read_text()
EDITS = [
    # 1. the copy lives elsewhere: keep pointing at the original cache / pretrained weights; import the label helper
    ('HERE = Path(__file__).parent\n',
     'HERE = Path("/root/baseline/visualnav-transformer/pairuav5d_v2")\nsys.path.insert(0, str(Path(__file__).parent))\n'),
    # 2. images may be stored on the CPU (shared GPU); pixels are unchanged (same GPU resize, then moved)
    ('def load_images(size, device, chunk=4096):\n    width, height = size\n',
     'def load_images(size, device, chunk=4096, store=None):\n    store = store or device\n    width, height = size\n'),
    ('    out = torch.empty((source.shape[0], 3, height, width), dtype=torch.uint8, device=device)\n',
     '    out = torch.empty((source.shape[0], 3, height, width), dtype=torch.uint8, device=store)\n'),
    ('        out[start:start + chunk] = block.round().clamp(0, 255).to(torch.uint8)\n',
     '        out[start:start + chunk] = block.round().clamp(0, 255).to(torch.uint8).to(store)\n'),
    # 3. labels: camera-frame (lx, ly) and the properly swapped label
    ('def load_index(name, device):\n    data = np.load(CACHE / f"{name}_index.npz")\n    return {\n        "a": torch.from_numpy(data["a"]).to(device),\n        "b": torch.from_numpy(data["b"]).to(device),\n        "target": torch.from_numpy(data["target"]).float().to(device),\n        "tour": data["tour"],\n    }\n',
     'def load_index(name, device, frame="enu"):\n    data = np.load(CACHE / f"{name}_index.npz")\n    target = data["target"].copy()\n    swapped = -target  # east-north frame: swapping the pair negates every component\n'
     '    if frame == "cam":\n        from camlabels import by_pair_id\n        t = by_pair_id().loc[data["pair_id"]]\n        assert np.abs(t[["x", "y", "z", "tilt", "heading"]].values - target).max() < 1e-9\n'
     '        target[:, 0], target[:, 1] = t.lx.values, t.ly.values          # source-camera levelled frame\n        swapped[:, 0], swapped[:, 1] = t.lx_sw.values, t.ly_sw.values  # swapped pair: frame of camera b\n'
     '    return {\n        "a": torch.from_numpy(data["a"]).to(device),\n        "b": torch.from_numpy(data["b"]).to(device),\n        "target": torch.from_numpy(target).float().to(device),\n        "target_swap": torch.from_numpy(swapped).float().to(device),\n        "tour": data["tour"],\n        "pair_id": data["pair_id"],\n    }\n'),
    ('                target = torch.where(swap[:, None], -target, target)\n',
     '                target = torch.where(swap[:, None], train["target_swap"][rows], target)\n'),
    # 4. new switches
    ('    parser.add_argument("--eval-every", type=int, default=5)\n',
     '    parser.add_argument("--eval-every", type=int, default=5)\n    parser.add_argument("--frame", choices=["enu", "cam"], default="enu", help="frame of the horizontal displacement target")\n'
     '    parser.add_argument("--xy-scale", type=float, default=15.0, help="normalisation of the two horizontal components (15 = original)")\n'
     '    parser.add_argument("--images-on-cpu", action="store_true", help="keep the uint8 image cache on the CPU (same pixels)")\n'),
    ('    images = load_images(size, device)\n    train, test = load_index("train", device), load_index("test", device)\n',
     '    SCALE[0] = SCALE[1] = args.xy_scale\n    images = load_images(size, device, chunk=512 if args.images_on_cpu else 4096, store=torch.device("cpu") if args.images_on_cpu else None)\n    torch.cuda.empty_cache()  # release the resize workspace (shared GPU)\n    train, test = load_index("train", device, args.frame), load_index("test", device, args.frame)\n'),
    # 5. bookkeeping in the outputs
    ('        "distance_success_threshold_m": 1.0,\n    }\n    if args.test:\n',
     '        "distance_success_threshold_m": 1.0, "frame": args.frame, "xy_scale": args.xy_scale,\n    }\n    if args.test:\n'),
    ('                            pred_distance_head_m=distance.cpu().numpy(), true_xyz_tilt_heading=truth.cpu().numpy())\n',
     '                            pred_distance_head_m=distance.cpu().numpy(), true_xyz_tilt_heading=truth.cpu().numpy(),\n                            pred_mean_xyz_tilt_heading=mean.cpu().numpy(), pair_id=test["pair_id"][:len(rows)], frame=args.frame)\n'),
    # 6. NoMaD only, memory only: accumulate the batch-256 gradient over micro-batches. NoMaD's encoders and U-Net use GroupNorm /
    #    LayerNorm (no batch statistics), so the weighted sum of the chunk losses is the full-batch loss; only the order of the
    #    random draws (diffusion noise, timesteps, dropout) differs.
    ('                else:\n                    loss = model.loss(a, b, encoded, distance_target(target), ddpm, draws=args.noise_draws,\n                                      distance_weight=args.distance_weight)\n            optimizer.zero_grad(set_to_none=True)\n            scaler.scale(loss).backward()\n',
     '                elif args.micro_batches <= 1:\n                    loss = model.loss(a, b, encoded, distance_target(target), ddpm, draws=args.noise_draws,\n                                      distance_weight=args.distance_weight)\n            optimizer.zero_grad(set_to_none=True)\n'
     '            if args.model == "nomad" and args.micro_batches > 1:\n                loss = torch.zeros((), device=device)\n                for part in torch.arange(len(rows), device=device).chunk(args.micro_batches):\n'
     '                    with torch.autocast("cuda", dtype=torch.float16):\n                        part_loss = model.loss(a[part].contiguous(memory_format=torch.channels_last), b[part].contiguous(memory_format=torch.channels_last), encoded[part], distance_target(target[part]), ddpm,\n                                               draws=args.noise_draws, distance_weight=args.distance_weight) * (len(part) / len(rows))\n'
     '                    scaler.scale(part_loss).backward()\n                    loss = loss + part_loss.detach()\n            else:\n                scaler.scale(loss).backward()\n'),
    ('    parser.add_argument("--eval-every", type=int, default=5)\n    parser.add_argument("--frame"',
     '    parser.add_argument("--eval-every", type=int, default=5)\n    parser.add_argument("--micro-batches", type=int, default=1, help="NoMaD: accumulate the batch gradient over this many chunks (memory only)")\n    parser.add_argument("--frame"'),
]
for old, new in EDITS:
    assert s.count(old) == 1, (s.count(old), old[:80]); s = s.replace(old, new)
DST.write_text(s); print('wrote', DST, '| edits', len(EDITS))
