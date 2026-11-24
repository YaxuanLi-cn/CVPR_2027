# PairUAV-5D baseline results on the MASt3R/RASR split

This report uses the same fixed split as `mast3r_runtime`/RASR:

- train: 47,687 pairs from 1,122 tours
- test: 12,313 pairs from 290 tours
- train manifest SHA256: `59189fac75dbec189209d0f69518142661c9398fba67621be56b76d158130c4e`
- test manifest SHA256: `0b4185a4593cb6bbaf884c24700e5ab1213edbe0b8d06ae98a1ce395985d0e62`

The target definitions are exactly:

- `distance`: `sqrt(delta.x_m^2 + delta.y_m^2 + delta.z_m^2)` in meters
- `heading`: `delta.heading_deg` in degrees, evaluated with circular absolute error
- `tilt`: `delta.tilt_deg` in degrees

There is no `range` target in this experiment. Older files in this directory that
used `range` or a different split are not valid for this report.

| Baseline | Test distance MAE (m) | Test heading circular MAE (deg) | Test tilt MAE (deg) | Status |
|---|---:|---:|---:|---|
| Sample4Geo | 31.67 | 87.79 | 17.57 | complete |
| DINOv3 ViT-L | 6.2125 | 86.3967 | 4.7844 | complete |
| AI2THOR | 18.9731 | 85.9638 | 20.0772 | complete, stable rerun |
| OpenPI / pi05 | 7.09 | 83.64 | 6.97 | complete |

Result files:

- Sample4Geo: `sample4geo/test_results_mast3r_split.log`
- DINOv3: `dinov3/outputs_mast3r_split/metrics.json`
- AI2THOR: `ai2thor/checkpoints_mast3r_split_stable/evaluation_log.txt`
- OpenPI metrics: `openpi/inference_mast3r_split_metrics.json`
- OpenPI checkpoint: `openpi/checkpoints/pi05_mast3r_split/5959/`
