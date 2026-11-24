#!/usr/bin/env python3
"""Generate supp/trackE/vf_cam.py from /root/baseline/visualnav-transformer/visual_foresight_pairuav.py (the Visual Foresight
adaptation reported in Table 2). Only edits: output directory, a --frame switch that replaces (x, y) by the camera-frame
(lx, ly) in the action / label, and the manifest index saved next to the predictions."""
from pathlib import Path
SRC = Path('/root/baseline/visualnav-transformer/visual_foresight_pairuav.py'); DST = Path(__file__).parent / 'vf_cam.py'; s = SRC.read_text()
EDITS = [
    ('OUT = Path(__file__).parent / "outputs_pairuav5d_visual_foresight_v2"\n',
     'FRAME = sys.argv[1] if len(sys.argv) > 1 else "enu"\nassert FRAME in ("enu", "cam")\nsys.path.insert(0, str(Path(__file__).parent))\nfrom splitdata import ordered, labels as frame_labels\n'
     'OUT = Path("/root/rasr-pairuav/outputs/supp/trackE/runs") / f"visual_foresight_{FRAME}"\n'),
    ('    train_data, test_data = PairDataset(TRAIN, IMAGE_ROOT, transform), PairDataset(TEST, IMAGE_ROOT, transform)\n',
     '    train_data, test_data = PairDataset(TRAIN, IMAGE_ROOT, transform), PairDataset(TEST, IMAGE_ROOT, transform)\n'
     '    order = {"train": ordered("train"), "test": ordered("test")}\n'
     '    for data, name in ((train_data, "train"), (test_data, "test")):   # same samples, same order; (x, y) -> (lx, ly) when FRAME == "cam"\n'
     '        L = frame_labels(order[name], FRAME)\n        assert len(L) == len(data.samples) and all(str(s[0]) == o["image_a"] and str(s[1]) == o["image_b"] for s, o in zip(data.samples, order[name]))\n'
     '        data.samples = [(s[0], s[1], (l[0] / 40.0, l[1] / 40.0, l[2] / 40.0, l[3] / 45.0, l[4] / 180.0), tuple(float(v) for v in l)) for s, l in zip(data.samples, L)]\n'),
    ('    np.savez_compressed(OUT / "test_predictions.npz", pred_xyz_tilt_heading=pred.numpy(), true_xyz_tilt_heading=truth.numpy())\n',
     '    np.savez_compressed(OUT / "test_predictions.npz", pred_xyz_tilt_heading=pred.numpy(), true_xyz_tilt_heading=truth.numpy(), manifest_index=np.array([o["manifest_index"] for o in order["test"]]), frame=FRAME)\n'),
]
for old, new in EDITS:
    assert s.count(old) == 1, (s.count(old), old[:70]); s = s.replace(old, new)
DST.write_text(s); print('wrote', DST)
