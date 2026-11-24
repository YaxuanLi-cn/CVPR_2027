#!/usr/bin/env python3
"""Why the two world-model baselines cannot gain from a better action frame: how much does the predicted goal latent depend on
the action at all? For the trained dynamics models we compare, on test pairs, the planning cost (latent MSE to the true goal)
obtained with the TRUE relative pose, with the pose of a random other test pair, and with the mean action; plus the cost of
predicting "no change" (source latent). If the true action is not clearly better than a wrong one, CEM has nothing to optimise.
  Visual Foresight adaptation : archived east-north checkpoint and our camera-frame checkpoint, all common test pairs
  DINO-WM adaptation          : archived east-north checkpoint (Table 2 run), 1,000 test pairs
Output: outputs/supp/trackE/action_sensitivity.json
"""
import sys, os, json
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, '/root/rasr-pairuav/supp'); sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, '/root/baseline/dinov3'); sys.path.insert(0, '/root/baseline/gaoyuezhou-dino_wm')
from splitdata import ordered, labels as frame_labels
O = Path('/root/rasr-pairuav/outputs/supp/trackE'); dev = torch.device('cuda'); res = {}; g = torch.Generator().manual_seed(0)
def comp_stats(true, comp):
    """comp: {name: cost with ONLY that component group taken from a random other pair}."""
    t = true.double().cpu().numpy(); out = {}
    for k, v in comp.items():
        v = v.double().cpu().numpy(); out[k] = dict(cost=float(v.mean()), rel_increase=float(v.mean() / t.mean() - 1), frac_pairs_true_better=float((t < v).mean()))
    return out


def stats(true, shuf, mean, ident):
    t, s, m, i = (x.double().cpu().numpy() for x in (true, shuf, mean, ident))
    return dict(n=int(len(t)), cost_true_action=float(t.mean()), cost_random_other_action=float(s.mean()), cost_mean_action=float(m.mean()), cost_no_change=float(i.mean()),
                rel_gain_true_vs_random=float(1 - t.mean() / s.mean()), frac_pairs_true_better_than_random=float((t < s).mean()), frac_pairs_true_better_than_mean=float((t < m).mean()))

# ---------------- Visual Foresight adaptation
te = ordered('test'); tr = ordered('train'); paths = sorted({p for r in tr + te for p in (r['image_a'], r['image_b'])}); ix = {p: i for i, p in enumerate(paths)}
F_ = torch.load('/root/baseline/dinov3/outputs_xyz7b_mast3r_split/unique_image_features.pt', map_location='cpu', weights_only=True)
sys.path.insert(0, '/root/baseline/visualnav-transformer'); os.chdir('/root/baseline/visualnav-transformer')
import importlib.util
spec = importlib.util.spec_from_file_location('vf', '/root/baseline/visualnav-transformer/visual_foresight_pairuav.py'); vf = importlib.util.module_from_spec(spec); spec.loader.exec_module(vf)
A = F_[[ix[r['image_a']] for r in te]].float(); B = F_[[ix[r['image_b']] for r in te]].float(); perm = torch.randperm(len(te), generator=g)
for frame, ck in (('enu', '/root/baseline/visualnav-transformer/outputs_pairuav5d_visual_foresight_v2/checkpoint.pt'), ('cam', str(O / 'runs/visual_foresight_cam/checkpoint.pt'))):
    c = torch.load(ck, map_location='cpu', weights_only=False); m = vf.ActionConditionedPredictor(F_.shape[1]); m.load_state_dict(c['model']); m.to(dev).eval()
    L = torch.tensor(frame_labels(te, frame)); h = torch.deg2rad(L[:, 4]); act = torch.cat((L[:, :3] / 40.0, L[:, 3:4] / 45.0, h.sin()[:, None], h.cos()[:, None]), 1); act = (act - c['action_mean']) / c['action_std']
    out = [[], [], [], []]; GR = {'horizontal displacement': [0, 1], 'vertical displacement': [2], 'tilt change': [3], 'heading change': [4, 5]}; comp = {k: [] for k in GR}
    with torch.inference_mode():
        for s in range(0, len(te), 512):
            a, b = A[s:s + 512].to(dev), B[s:s + 512].to(dev); mse = lambda u: (m(a, u.to(dev)) - b).square().mean(-1)
            out[0].append(mse(act[s:s + 512])); out[1].append(mse(act[perm[s:s + 512]])); out[2].append(mse(torch.zeros_like(act[s:s + 512]))); out[3].append((a - b).square().mean(-1))
            for k, cols in GR.items(): u = act[s:s + 512].clone(); u[:, cols] = act[perm[s:s + 512]][:, cols]; comp[k].append(mse(u))
    key = f'Visual Foresight adaptation ({frame}-trained dynamics)'; res[key] = stats(*[torch.cat(o) for o in out]); res[key]['one_component_from_another_pair'] = comp_stats(torch.cat(out[0]), {k: torch.cat(v) for k, v in comp.items()}); print(frame, res[key], flush=True)
del F_, A, B

# ---------------- DINO-WM adaptation (archived Table 2 checkpoint)
os.chdir('/root/baseline/gaoyuezhou-dino_wm'); import pairuav_baseline as PB
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
N = 1000; rows = [te[i] for i in torch.randperm(len(te), generator=g)[:N].tolist()]
tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
class DS(Dataset):
    def __len__(self): return N
    def __getitem__(self, i): return tf(Image.open(rows[i]['image_a']).convert('RGB')), tf(Image.open(rows[i]['image_b']).convert('RGB'))
PB.fixed_seed(); wm = PB.make_model(dev); c = torch.load('outputs_pairuav5d/checkpoint.pt', map_location='cpu', weights_only=False); wm.load_state_dict(c['wm']); wm.eval(); wm.encoder.eval(); am, asd = c['action_mean'].to(dev), c['action_std'].to(dev)
L = torch.tensor(frame_labels(rows, 'enu'), device=dev); act = (L - am) / asd; perm = torch.randperm(N, generator=g).to(dev); out = [[], [], [], []]; k = 0; GR = {'horizontal displacement': [0, 1], 'vertical displacement': [2], 'tilt change': [3], 'heading change': [4]}; comp = {q: [] for q in GR}
with torch.inference_mode():
    for ia, ib in DataLoader(DS(), batch_size=16, num_workers=2):
        b = len(ia); obs = {'visual': torch.stack((ia, ib), 1).to(dev), 'proprio': torch.empty((b, 2, 0), device=dev)}; enc = wm.encode_obs(obs); src, goal = enc['visual'][:, :1], enc['visual'][:, 1]
        def cost(a):
            e = wm.encode_act(a.reshape(b, 1, 5)); z = torch.cat((src, e.unsqueeze(2).expand(-1, -1, src.shape[2], -1)), dim=-1); return (wm.predict(z)[:, 0, :, :src.shape[-1]] - goal).square().mean(dim=(1, 2))
        idx = torch.arange(k, k + b, device=dev); out[0].append(cost(act[idx])); out[1].append(cost(act[perm[idx]])); out[2].append(cost(torch.zeros_like(act[idx]))); out[3].append((src[:, 0] - goal).square().mean(dim=(1, 2)))
        for q, cols in GR.items(): u = act[idx].clone(); u[:, cols] = act[perm[idx]][:, cols]; comp[q].append(cost(u))
        k += b
key = 'DINO-WM adaptation (Table 2 checkpoint, enu-trained)'; res[key] = stats(*[torch.cat(o) for o in out]); res[key]['one_component_from_another_pair'] = comp_stats(torch.cat(out[0]), {q: torch.cat(v) for q, v in comp.items()}); print(res[key])
json.dump(res, open(O / 'action_sensitivity.json', 'w'), indent=1)
