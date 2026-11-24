# PairUAV-5D Stage 2 Reproduction Bundle

This bundle collects the existing `superglue_fusion` and Stage 2 imagination
implementation, matching feature caches, real model weights/predictions, split
manifests, logs, and the saved evaluation JSON. It does not contain RGB images,
pixelwise depth/maps, point clouds, or synthetic/replacement predictions.

## Provenance and Coverage

The included artifacts come from `/root/rasr-pairuav/outputs/pairuav5d_60k`.
`features/pairs.pkl` has 60,000 rows (47,687 train, 12,313 test; 1,412 tours).
The MASt3R summary table and both stage-0 predictions also have 60,000 rows.
The SuperGlue half/zoom geometry and `half_zoom_pred.pkl` have 31,273 rows
(24,909 train, 6,364 test). The saved Stage 2 fusion output has 18,364 rows
(12,000 train, 6,364 test); Stage 2 final evaluation reports 1,183 validation
and 6,364 test pairs after joining absolute-heading predictions.

The requested `predictions (2).jsonl` and `stage2_imagination.zip` were not
present in the searched workspace locations. Therefore these caches cannot be
certified as the same model version, exact sample set, or split as that JSONL.
`superglue_fusion/predictions.jsonl` is the distinct file that exists in the
source tree; do not assume it is the missing JSONL. The 500-pair pilot mapping
must be built from the actual `predictions (2).jsonl` and checked against
`manifest_index`, `pair_id`, and image paths before selecting rows.

## Contents

- `superglue_fusion/`: complete current source tree, including `fusion_lib.py`,
  `pose_from_matches.py`, `map_pose.py`, `geom.py`, stage-0/1, MASt3R feature
  loader, absolute-heading model/cache/training code, and geometry extraction.
- `data_processing/03_extract_features.py`, `run_pairuav5d_60k_features.sh`,
  `build_pairuav5d_manifests.py`, `scripts/attach_xyz_targets.py`: the existing
  production feature extraction, manifest, and target/feature preparation code.
- `stage2_imagination/`: current Stage 2 source and assets. These files were
  copied from the working tree, including its uncommitted edits at packaging
  time.
- `features/`: real pair table, 422-dimensional MASt3R summary table, Stage 0
  outputs, Stage 1/geometry/MAP/zoom caches and Stage 2 guide/imagination/output
  caches. Join rows by `manifest_index`; pair and prediction caches retain
  `split` and/or `tour` as applicable.
- `checkpoints/`: actual CNN v2 checkpoint `model_cnx_s_d8.pt`, its
  `pred_cnx_s_d8.pkl`, BEV checkpoint `model_bev_s.pt`, BEV predictions, and
  `images.pkl`/`images2.pkl` metadata. `images2.pkl` contains additional images
  from training tours, not test tours. RGB arrays are intentionally omitted.
- `manifests/`: source train/test row manifests plus `tour_splits.csv` listing
  each pair tour and the absolute-heading val/test parts represented in cache.
- `results/`: final Stage 2 metrics JSON files.
- `logs/`: stage-1 half-zoom, Stage-2 fusion, BEV, and final-evaluation logs.
- `SUPERGLUE_EXPERIMENTS.md`, `STAGE2_IMAGINATION.md`: experiment notes and
  coordinate/algorithm documentation from the repository.

All required fusion training and test rows are present in the saved caches;
`stage2_fusion.py` uses five-fold `GroupKFold` by `tour` for train OOF and fits
the full training portion to predict test rows. Stage 2's saved 12,000 training
rows are a subset; the 6,364 test rows match the documented half-result set.

## Coordinates and Inputs

`pairs.pkl` fields `x`, `y`, `z` are relative displacement in meters and
`heading`, `tilt` are relative angles in degrees, from the pair JSON labels.
The convention documented by the source is `heading = heading_b - heading_a`
with circular wrapping to `[-180, 180)`, and `tilt = tilt_b - tilt_a`.
Absolute heading is clockwise from north in `[0, 360)`; camera tilt is from
nadir, nominally `[0, 45]` degrees, with zero roll. `lx`, `ly` are computed in
camera a's leveled horizontal frame (right, forward):

```text
lx = cos(heading_a) * x - sin(heading_a) * y
ly = sin(heading_a) * x + cos(heading_a) * y
```

World coordinates are reconstructed by rotating `(lx, ly)` with absolute
heading of camera a. Pair pose is `X_b = R X_a + t` (relative pose a-to-b).
For Stage 2 imagined-view matching, `homography_b2a` warps image b into image
a's canvas: a is the reference/target canvas and b is the warped source; the
SuperGlue call then matches image0=a against image1=b. Training labels/KML pose
are supervision and offline evaluation only. Inference features use RGB, Stage 1 predictions, and estimated
SuperGlue/MAP/MASt3R/BEV/CNN features. True absolute heading and GT tilt appear
only in separately labeled upper-bound or training-supervision branches; they
are not the headline inference inputs.

## Saved Results

`results/final_s2_bev_s.json` is the screenshot-associated final evaluation
JSON in the source notes' naming. It selects `stage 2: A + B, abs-heading =
BEV + CNN v2 ensemble` on validation. Test MAE is x=0.9922 m, y=0.9784 m,
z=0.5444 m, tilt=0.2006 deg, relative heading=0.0657 deg, and XYZ distance
0.6772 m after validation-selected norm rescaling. The true-absolute-heading
upper bound is separately reported (x=0.4855 m, y=0.4969 m); it uses GT and is
not a deployable prediction branch. The log and JSON also distinguish CNN-only,
BEV-only, and ensemble results.

The core sequence recorded in `stage2_imagination/run_stage2.sh` is:

1. Build guide rows from `pairs.pkl`, half-zoom prediction, geometry, MAP, and
   Stage-0 caches.
2. Generate imagined-view SuperGlue matches and `geom_imag.pkl`.
3. Fit the Stage-2 fusion model and write `s2_pred.pkl` / `s2_metrics.json`.
4. Build BEV items, run the BEV heading model, then run `stage2_eval.py` to
   produce final JSON.

The original scripts use absolute paths under `/root/rasr-pairuav` and
`/root/pairUAV-5D-temp`. For the requested cluster path
`/n/data2/hms/dbmi/kyu/lab/dyh127/zjr102/dreamnav_project`, point those roots at
the extracted bundle/data location before running. This packaging operation
does not modify algorithm or evaluation code. Run controls should cap workers
at three per GPU and set the requested physical devices (`0,1,3`) through
`CUDA_VISIBLE_DEVICES`; the checked-in original defaults are not yet configured
for that cluster limit. Use persistent logs and skip existing non-empty outputs
for resumable execution.

## Dependencies

`requirements.txt` records versions observed in the packaging Python 3.12.15
environment, not a historical lockfile for the November 15, 2026 experiment.
The original repository `requirements.source.txt` contains lower bounds, not
resolved versions. Historical exact Python/PyTorch/CUDA/OpenCV/timm/LightGBM
versions were not recorded in the experiment logs. PyTorch was observed as
2.11.0+cu128 (CUDA runtime 12.8); target GPUs are A100, and the host CUDA
driver must be checked on the destination node. MASt3R extraction additionally
requires the upstream MASt3R checkout/weights; the frozen summary cache means
that extraction is not needed for the cached fusion run.

Public upstream model sources/identifiers used by the code:

- SuperPoint/SuperGlue: `https://github.com/magicleap/SuperGluePretrainedNetwork`;
  matching scripts configure SuperGlue's `outdoor` weights, 20 Sinkhorn
  iterations, and match threshold 0.2. The upstream checkout commit is not
  recorded in this workspace and is not vendored here.
- MASt3R: `https://github.com/naver/mast3r`, model identifier
  `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` (the feature script expects
  `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`). Runtime checkout and
  self-downloaded public weights are not bundled; the frozen summary cache
  avoids needing them for a cached fusion run.
- CNN v2 encoder: timm `convnext_small.fb_in22k_ft_in1k`; self-trained weights
  are included as `checkpoints/model_cnx_s_d8.pt`.
- BEV encoder: same ConvNeXt-Small configuration; self-trained weights are
  included as `checkpoints/model_bev_s.pt`.

## Missing / Unverified

- The actual `predictions (2).jsonl`; exact row/model/split correspondence and
  selection of its best-error 500 pairs cannot be verified yet.
- Original RGB dataset path, required for rerunning matching or regenerating
  omitted `images_448x256.npy` / `images2_448x256.npy` arrays.
- The external SuperGlue source checkout and MASt3R runtime/weights; source
  scripts and exact model identifiers are provided above, but the upstream
  source commit and downloaded public checkpoint hashes were not recorded.
- MASt3R dense outputs/point maps (not required when using `mast3r_feats.pkl`).
- The 500-row pilot manifest keyed to the missing JSONL.
- A single self-contained, path-configured launcher with per-GPU worker caps;
  current source scripts contain historical absolute paths and worker defaults.

No synthetic features, random predictions, or test labels were used to fill any
missing artifact.
