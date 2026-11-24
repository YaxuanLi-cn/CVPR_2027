#!/usr/bin/env python3
"""DINOv3 ViT-7B baseline (frozen CLS features of both images -> one linear layer), re-trained with camera-frame targets.

Recipe of /root/baseline/dinov3/train_pairuav5d_hf.py (the run reported in Table 2), unchanged: cached CLS features,
nn.Linear(2*4096, 5), targets (x, y, z)/40, tilt/45, heading/180, smooth-L1 (beta 0.1), AdamW lr 1e-3, batch 256, 1 epoch.
The original run had no seed; we run several seeds in both frames (the east-north control shows the run-to-run spread).
Usage: dinov3_head_cam.py [n_seeds]
"""
import sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from splitdata import ordered, labels
OUT = Path('/root/rasr-pairuav/outputs/supp/trackE/runs/dinov3_vit7b'); OUT.mkdir(parents=True, exist_ok=True)
FEATURES = '/root/baseline/dinov3/outputs_xyz7b_mast3r_split/unique_image_features.pt'
SCALE = torch.tensor([40.0, 40.0, 40.0, 45.0, 180.0]); nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5; dev = torch.device('cuda')
tr, te = ordered('train'), ordered('test'); paths = sorted({p for r in tr + te for p in (r['image_a'], r['image_b'])}); index = {p: i for i, p in enumerate(paths)}
feats = torch.load(FEATURES, map_location='cpu', weights_only=True); assert len(feats) == len(paths), (len(feats), len(paths))
feats = feats.to(dev)                                    # float16 CLS features, as cached
ia = lambda rows: torch.tensor([index[r['image_a']] for r in rows], device=dev); ib = lambda rows: torch.tensor([index[r['image_b']] for r in rows], device=dev)
tra, trb, tea, teb = ia(tr), ib(tr), ia(te), ib(te); res = {}
for frame in ('enu', 'cam'):
    Ltr = torch.tensor(labels(tr, frame), device=dev); Lte = labels(te, frame); Ytr = Ltr / SCALE.to(dev)
    for seed in range(nseeds):
        torch.manual_seed(seed); head = nn.Linear(feats.shape[1] * 2, 5).to(dev); opt = torch.optim.AdamW(head.parameters(), lr=1e-3); head.train(); losses = []
        perm = torch.randperm(len(tr), device=dev)
        for s in range(0, len(tr), 256):                 # DataLoader(batch_size=256, shuffle=True), last batch kept
            rows = perm[s:s + 256]; x = torch.cat((feats[tra[rows]], feats[trb[rows]]), dim=1).float()
            loss = F.smooth_l1_loss(head(x), Ytr[rows], beta=0.1); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        head.eval(); pred = []
        with torch.inference_mode():
            for s in range(0, len(te), 256): pred.append((head(torch.cat((feats[tea[s:s + 256]], feats[teb[s:s + 256]]), dim=1).float()) * SCALE.to(dev)).cpu().numpy())
        pred = np.concatenate(pred); e = np.abs(pred[:, :3] - Lte[:, :3]).mean(0); d = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(Lte[:, :3], axis=1))
        res[f'{frame}/seed{seed}'] = dict(train_loss=float(np.mean(losses)), mae_xyz=e.tolist(), d_mae=float(d.mean()), sr_d=float((d <= 1).mean()), tilt=float(np.abs(pred[:, 3] - Lte[:, 3]).mean()), heading=float(np.abs((pred[:, 4] - Lte[:, 4] + 180) % 360 - 180).mean()))
        print(frame, seed, {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 4)) for k, v in res[f'{frame}/seed{seed}'].items()}, flush=True)
        np.savez_compressed(OUT / f'test_predictions_{frame}_seed{seed}.npz', pred_xyz_tilt_heading=pred, true_xyz_tilt_heading=Lte, manifest_index=np.array([r['manifest_index'] for r in te]), frame=frame)
        if seed == 0: torch.save(head.state_dict(), OUT / f'head_{frame}_seed0.pt')
json.dump(res, open(OUT / 'runs.json', 'w'), indent=1)
