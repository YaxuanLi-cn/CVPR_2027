"""Equivalence check for dinowm_tokens.py: with the ARCHIVED DINO-WM checkpoint, CEM planning from cached tokens must give the
same result as planning from images through the original code path (same random draws)."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/root/rasr-pairuav/supp/trackE'); sys.path.insert(0, '/root/baseline/dinov3'); sys.path.insert(0, '/root/baseline/gaoyuezhou-dino_wm'); os.chdir('/root/baseline/gaoyuezhou-dino_wm')
import pairuav_baseline as PB
from torchvision import transforms
from PIL import Image
from splitdata import ordered
dev = torch.device('cuda'); te = ordered('test')[:32]
tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
load = lambda p: tf(Image.open(p).convert('RGB'))
A = torch.stack([load(r['image_a']) for r in te]); B = torch.stack([load(r['image_b']) for r in te])
PB.fixed_seed(); wm = PB.make_model(dev); ck = torch.load('outputs_pairuav5d/checkpoint.pt', map_location='cpu', weights_only=False); wm.load_state_dict(ck['wm']); wm.eval(); wm.encoder.eval()
am, asd = ck['action_mean'].to(dev), ck['action_std'].to(dev)
torch.manual_seed(0); torch.cuda.manual_seed_all(0); p_img = torch.cat([PB.plan_batch(wm, A[i:i + 16], B[i:i + 16], am, asd, dev, samples=16, topk=4, steps=3) for i in (0, 16)]).cpu().numpy()
with torch.inference_mode(): ZA = wm.encoder.forward(wm.encoder_transform(A.to(dev))).float().cpu(); ZB = wm.encoder.forward(wm.encoder_transform(B.to(dev))).float().cpu()
orig = wm.encode_obs; wm.encode_obs = lambda obs: {'visual': obs['visual'], 'proprio': wm.encode_proprio(obs['proprio'])}
torch.manual_seed(0); torch.cuda.manual_seed_all(0); p_tok = torch.cat([PB.plan_batch(wm, ZA[i:i + 16], ZB[i:i + 16], am, asd, dev, samples=16, topk=4, steps=3) for i in (0, 16)]).cpu().numpy()
print('max |plan(images) - plan(tokens)| per output (x, y, z, tilt, heading):', np.abs(p_img - p_tok).max(0).round(5), '| typical magnitude', np.abs(p_img).mean(0).round(2))
