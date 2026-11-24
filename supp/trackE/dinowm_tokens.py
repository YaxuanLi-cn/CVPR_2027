#!/usr/bin/env python3
"""DINO-WM adaptation of Table 2 (frozen DINOv2 ViT-S/14 patch tokens, one-step latent predictor, CEM pose planning) in the
east-north and the camera frame, from encoder tokens computed once.

/root/baseline/gaoyuezhou-dino_wm/pairuav_baseline.py encodes both images of every pair with a FROZEN encoder in eval mode
and uses no augmentation, so the tokens of an image never change. We compute them once per unique image (same transform:
Resize(224), CenterCrop(224), Normalize(0.5), then the world model's own Resize(196)), keep them in RAM (float32) and run the
original training / planning functions on them: same model constructor, optimiser, loss, batch size, one epoch, CEM settings
and seed (17). Only the image->token step is replaced by a lookup, which makes both frames and several seeds affordable.
Usage: dinowm_tokens.py <workers> <frame:seed,frame:seed,...>      e.g.  4 cam:17,enu:17,cam:18,enu:18
Outputs: runs/dino_wm_<frame>[_seed<k>]/{test_predictions.npz, metrics.json}
"""
import sys, os, json, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; DWM = '/root/baseline/gaoyuezhou-dino_wm'; sys.path.insert(0, str(HERE)); sys.path.insert(0, '/root/baseline/dinov3'); sys.path.insert(0, DWM)
RUNS = Path('/root/rasr-pairuav/outputs/supp/trackE/runs')
from splitdata import ordered, labels as frame_labels
os.chdir(DWM); import pairuav_baseline as PB
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
dev = torch.device('cuda'); tr, te = ordered('train'), ordered('test')
SMOKE = bool(os.environ.get('SMOKE'))
if SMOKE: tr, te = tr[:320], te[:64]; RUNS = RUNS / '_smoke'; RUNS.mkdir(parents=True, exist_ok=True)   # code check only
paths = sorted({p for r in tr + te for p in (r['image_a'], r['image_b'])}); index = {p: i for i, p in enumerate(paths)}


def tokens(workers):
    t0 = time.time(); transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])   # as in pairuav_baseline.main
    class DS(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            with Image.open(paths[i]) as im: return transform(im.convert('RGB')), i
    PB.fixed_seed(); wm = PB.make_model(dev); wm.eval(); wm.encoder.eval(); Z = np.empty((len(paths), 196, 384), np.float32)
    with torch.inference_mode():
        for k, (x, idx) in enumerate(DataLoader(DS(), batch_size=32, shuffle=False, num_workers=workers)):
            Z[idx.numpy()] = wm.encoder.forward(wm.encoder_transform(x.to(dev))).float().cpu().numpy()       # exactly VWorldModel.encode_obs
            if k % 200 == 0: print('tokens', k * 32, len(paths), f'{(time.time() - t0) / 60:.1f} min', flush=True)
    del wm; torch.cuda.empty_cache(); m = (time.time() - t0) / 60; json.dump(dict(images=len(paths), minutes=m), open(RUNS / 'dino_wm_tokens.json', 'w')); print('tokens done', f'{m:.1f} min', flush=True); return Z


def run(Z, frame, seed):
    t0 = time.time(); out_dir = RUNS / (f'dino_wm_{frame}' + ('' if seed == 17 else f'_seed{seed}')); out_dir.mkdir(parents=True, exist_ok=True)
    class Pairs(Dataset):                                   # same sample order as dinov3.train_pairuav5d.PairDataset
        def __init__(self, rows): self.rows = rows; self.L = frame_labels(rows, frame); self.ia = [index[r['image_a']] for r in rows]; self.ib = [index[r['image_b']] for r in rows]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return torch.from_numpy(Z[self.ia[i]]), torch.from_numpy(Z[self.ib[i]]), 0, torch.from_numpy(self.L[i])
    PB.fixed_seed(seed); train_data, test_data = Pairs(tr), Pairs(te)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=0)
    labels = torch.tensor(train_data.L, dtype=torch.float32); action_mean = labels.mean(dim=0).to(dev); action_std = labels.std(dim=0).clamp_min(1e-3).to(dev)
    wm = PB.make_model(dev)
    wm.encode_obs = lambda obs: {'visual': obs['visual'], 'proprio': wm.encode_proprio(obs['proprio'])}   # 'visual' already holds the encoder tokens
    train_latent_mse = PB.fit(wm, train_loader, action_mean, action_std, dev); wm.eval(); wm.encoder.eval(); t_train = (time.time() - t0) / 60
    pred_rows, true_rows = [], []
    for za, zb, _, lab in DataLoader(test_data, batch_size=16, shuffle=False, num_workers=0):
        pred_rows.append(PB.plan_batch(wm, za, zb, action_mean, action_std, dev, samples=16, topk=4, steps=3).cpu().numpy()); true_rows.append(lab.numpy())
    pred, true = np.concatenate(pred_rows), np.concatenate(true_rows); d = np.abs(np.linalg.norm(pred[:, :3], axis=1) - np.linalg.norm(true[:, :3], axis=1)); e = np.abs(pred[:, :3] - true[:, :3]).mean(0)
    m = dict(frame=frame, seed=seed, train_rows=len(tr), test_rows=len(te), world_model_train_latent_mse=train_latent_mse, MAE_x_m=float(e[0]), MAE_y_m=float(e[1]), MAE_z_m=float(e[2]), MAE_distance_from_xyz_m=float(d.mean()),
             distance_success_rate=float((d <= 1).mean()), MAE_tilt_deg=float(np.abs(pred[:, 3] - true[:, 3]).mean()), MAE_heading_deg_circular=float(np.abs((pred[:, 4] - true[:, 4] + 180) % 360 - 180).mean()), train_minutes=t_train, total_minutes=(time.time() - t0) / 60)
    np.savez_compressed(out_dir / 'test_predictions.npz', pred_xyz_tilt_heading=pred, true_xyz_tilt_heading=true, manifest_index=np.array([r['manifest_index'] for r in te]), frame=frame)
    (out_dir / 'metrics.json').write_text(json.dumps(m, indent=2) + '\n'); print(json.dumps(m), flush=True); del wm; torch.cuda.empty_cache()


def wait_gpu(mib):
    """Shared GPU: block until `mib` MiB are free on two checks 15 s apart."""
    while True:
        f1 = torch.cuda.mem_get_info()[0] / 2**20; time.sleep(15); f2 = torch.cuda.mem_get_info()[0] / 2**20
        if min(f1, f2) >= mib: return
        time.sleep(30)


if __name__ == '__main__':
    Z = tokens(int(sys.argv[1]))
    for cfg in sys.argv[2].split(','):
        frame, seed = cfg.split(':')
        for attempt in range(20):                        # the token array lives in this process: never lose it to a transient out-of-memory
            try:
                wait_gpu(3000); run(Z, frame, int(seed)); break
            except torch.OutOfMemoryError:
                print('out of GPU memory, retrying', cfg, attempt, flush=True); torch.cuda.empty_cache(); time.sleep(120)
