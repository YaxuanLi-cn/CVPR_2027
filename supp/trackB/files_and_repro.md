## Files

| what | where |
|---|---|
| all PairUAV rows (every matcher x variant x scale, CIs, thresholds, constants, timings) | `outputs/supp/trackB/pairuav.json` (+ `pairuav.md`) |
| all rows on the real photographs | `outputs/supp/trackB/real.json` (+ `real.md`) |
| paired comparison of the strongest baseline with DreamNav (tails, per-bin success, scale diagnostics) | `outputs/supp/trackB/paired_sift.json` / `.md` (and `paired_<matcher>_tta.*` where generated) |
| SIFT baseline on the test set of Table 2 (all pairs processed so far), camera frame and ENU | `outputs/supp/trackB/sift_full.json` / `.md` |
| matches of every method (per pair, 4 rotations, original pixel coordinates, 1/4-px quantisation) | `outputs/supp/trackB/matches/<matcher>/<set>/<id>.npz` |
| essential-matrix poses per pair x rotation x threshold | `outputs/supp/trackB/geom/<matcher>_<set>.pkl` |
| MASt3R on the real pairs: PnP rows, source depth maps | `outputs/supp/trackB/mast3r/` |
| logs | `outputs/supp/trackB/logs/` |
| code | `supp/trackB/` |

## Reproduce

```bash
source /venv/main/bin/activate; cd /root/rasr-pairuav
# matching (GPU); <set> in train (300 calibration pairs) / test (1000) / real_in / real_out
python supp/trackB/run_matchers.py <set> splg,aliked,loftr,dedode        # SHARD=i/n to split
python supp/trackB/run_matchers.py <set> roma [n_pairs]                   # first n pairs of the fixed random order
python supp/trackB/run_sift.py <set> [procs]                              # CPU; <set> may also be test_full (6,364 pairs)
python supp/trackB/run_sg_real.py real_in                                 # SuperGlue (plain and the paper's rotation-aware pass) on the real pairs
python supp/trackB/run_mast3r.py <set>                                    # MASt3R matches (+ PnP rows and depth on the real sets)
# geometry (CPU)
python supp/trackB/tb_geom.py <matcher> train 0.2,0.4,0.75,1.5,3,6,12     # calibration pairs, all candidate thresholds
python supp/trackB/choose_thr.py                                          # -> geom/chosen_thr.json
bash supp/trackB/geom_set.sh test; bash supp/trackB/geom_set.sh real_in   # chosen thresholds only
# evaluation
python supp/trackB/eval_pairuav.py; python supp/trackB/eval_real.py; python supp/trackB/compare_best.py sift
python supp/trackB/tb_geom.py sift test_full <thr>; python supp/trackB/eval_sift_full.py
python supp/trackB/make_report.py
```

Packages added to `/venv/main` for this track (torch 2.11.0+cu128, torchvision, OpenCV 5.0.0 and numpy were not touched): `kornia 0.9.2`, `kornia-rs 0.2.0`, `lightglue` (github.com/cvg/LightGlue @ 5a650c4, installed with `--no-deps`), `romatch 0.1.2` (`--no-deps`), `loguru 0.8.0`. Checkpoints downloaded to `~/.cache/torch/hub/checkpoints` (about 1.9 GB): `superpoint_v1.pth`, `superpoint_lightglue_v0-1_arxiv.pth`, `aliked-n16.pth`, `aliked_lightglue_v0-1_arxiv.pth`, `loftr_outdoor.ckpt`, `dedode_detector_SO2.pth`, `B_SO2_Spread_descriptor_setting_C.pth`, `roma_outdoor.pth`, `dinov2_vitl14_pretrain.pth`.
