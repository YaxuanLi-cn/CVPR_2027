#!/usr/bin/env python3
"""Sample4Geo baseline (frozen University-1652 ConvNeXt-B, linear head on [f(A), f(B)]) in the east-north and the camera frame.

The reported recipe (/root/baseline/sample4geo/train.py with --freeze_backbone, 1 epoch) trains only nn.Linear(2048, 5) on
backbone features of photometrically augmented images. Because the backbone is frozen and has no batch statistics, one pass
over the training set gives the exact feature stream that script would see; the head update is then replayed on that stream
for BOTH target frames (identical batches, identical augmentation draws, identical initial head), which removes the
run-to-run spread from the comparison. Everything else follows train.py: train_drone_transforms / val_transforms at 384 px,
targets (x, y, z)/40, tilt/45, heading/180, smooth-L1 (beta 0.1), SGD momentum 0.9, lr 1e-3, weight decay 1e-10, one epoch.
  pass 1: features of the shuffled, augmented training stream and of the test set        -> runs/sample4geo/features.npz
  pass 2: head replay per (frame, batch size, seed): seed 0 = recorded stream order; other seeds = other shuffles of the
          same augmented samples and another head initialisation (run-to-run spread of the head)
Usage: s4g_cam.py extract [workers] | s4g_cam.py heads
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); S4G = '/root/baseline/sample4geo'; sys.path.insert(0, S4G)
OUT = Path('/root/rasr-pairuav/outputs/supp/trackE/runs/sample4geo'); OUT.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT = '/root/pairUAV-5D-temp'; SPLIT = '/root/baseline/_mast3r_split_data_v2'; CKPT = f'{S4G}/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
SCALE = np.array([40.0, 40.0, 40.0, 45.0, 180.0], np.float32)


def _one_cv_thread(_):
    import cv2; cv2.setNumThreads(1)      # CPU budget on the shared machine; does not change the pixels


def extract(workers=4, batch=8, seed=0):
    os.chdir(S4G); import train as T                      # the original script: dataset class, model class
    from torch.utils.data import DataLoader
    from sample4geo.dataset.university import get_transforms
    torch.manual_seed(seed); np.random.seed(seed); dev = torch.device('cuda'); t0 = time.time()
    model = T.EndToEndModel('convnext_base.fb_in22k_ft_in1k_384', pretrained=True, img_size=384)
    missing = model.backbone.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False); print('checkpoint loaded; missing/unexpected:', missing, flush=True)
    for p in model.backbone.parameters(): p.requires_grad = False
    assert not any(getattr(m, 'drop_prob', 0.0) for m in model.backbone.modules()), 'stochastic depth would make train-mode features random'
    model.to(dev); cfg = model.backbone.get_config(); val_tf, _, train_tf = get_transforms((384, 384), mean=cfg['mean'], std=cfg['std'])
    tr = T.TourFrameImageDataset(f'{SPLIT}/train', IMAGE_ROOT, transforms=train_tf); te = T.TourFrameImageDataset(f'{SPLIT}/test', IMAGE_ROOT, transforms=val_tf)
    res = {}
    for name, ds, shuffle in (('train', tr, True), ('test', te, False)):
        model.train() if name == 'train' else model.eval()   # as in train.py: train() during the epoch, eval() in validate()
        dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True, worker_init_fn=_one_cv_thread); fa, fb, paths = [], [], []
        with torch.no_grad():
            for i, (a, b, _, _, jp) in enumerate(dl):
                fa.append(model.backbone(a.to(dev, non_blocking=True)).float().cpu().numpy()); fb.append(model.backbone(b.to(dev, non_blocking=True)).float().cpu().numpy()); paths += list(jp)
                if i % 500 == 0: print(name, i, len(dl), f'{(time.time() - t0) / 60:.1f} min', flush=True)
        mi = np.array([json.load(open(p))['manifest_index'] for p in paths]); res[name] = (np.concatenate(fa), np.concatenate(fb), mi); print(name, 'done', res[name][0].shape, f'{(time.time() - t0) / 60:.1f} min', flush=True)
    np.savez(OUT / 'features.npz', train_a=res['train'][0], train_b=res['train'][1], train_manifest=res['train'][2], test_a=res['test'][0], test_b=res['test'][1], test_manifest=res['test'][2], stream_batch=batch, seed=seed, minutes=(time.time() - t0) / 60)


def heads(nseeds=5):
    from camlabels import table
    d = np.load(OUT / 'features.npz'); tab = table(); dev = torch.device('cuda')
    Xtr = torch.tensor(np.concatenate([d['train_a'], d['train_b']], 1), device=dev); Xte = torch.tensor(np.concatenate([d['test_a'], d['test_b']], 1), device=dev)
    def lab(mi, frame):
        t = tab.loc[mi]; return np.c_[t.lx.values if frame == 'cam' else t.x.values, t.ly.values if frame == 'cam' else t.y.values, t.z.values, t.tilt.values, t.heading.values].astype(np.float32)
    runs = {}
    for bs in (8, 32):
        for seed in range(nseeds):
            g = torch.Generator().manual_seed(1000 + seed); order = torch.arange(len(Xtr)) if seed == 0 else torch.randperm(len(Xtr), generator=g)
            torch.manual_seed(seed); init = nn.Linear(2048, 5).state_dict()
            for frame in ('enu', 'cam'):
                head = nn.Linear(2048, 5).to(dev); head.load_state_dict(init); opt = torch.optim.SGD(head.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-10)
                Y = torch.tensor(lab(d['train_manifest'], frame) / SCALE, device=dev); o = order.to(dev)
                for s in range(0, len(Xtr), bs):
                    rows = o[s:s + bs]; loss = F.smooth_l1_loss(head(Xtr[rows]), Y[rows], beta=0.1, reduction='mean'); opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad(): pred = (head(Xte).cpu().numpy() * SCALE)
                L = lab(d['test_manifest'], frame); dd = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(L[:, :3], axis=1)); e = np.abs(pred[:, :3] - L[:, :3]).mean(0)
                r = dict(mae_xyz=e.tolist(), d_mae=float(dd.mean()), sr_d=float((dd <= 1).mean()), tilt=float(np.abs(pred[:, 3] - L[:, 3]).mean()), heading=float(np.abs((pred[:, 4] - L[:, 4] + 180) % 360 - 180).mean()), pred_std=pred.std(0).tolist())
                runs[f'bs{bs}/{frame}/seed{seed}'] = r; print(f'bs{bs} {frame} seed{seed}', {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 4)) for k, v in r.items()}, flush=True)
                np.savez_compressed(OUT / f'test_predictions_bs{bs}_{frame}_seed{seed}.npz', pred_xyz_tilt_heading=pred, true_xyz_tilt_heading=L, manifest_index=d['test_manifest'], frame=frame)
    json.dump(dict(runs=runs, feature_pass_minutes=float(d['minutes']), stream_batch=int(d['stream_batch'])), open(OUT / 'runs.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'extract': extract(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else: heads()
