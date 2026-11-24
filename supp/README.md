# Supplementary experiments (CVPR supplementary material)

Code for the experiments that answer the reviews. One sub-folder per track; shared definitions in `common.py`.
Outputs go to `outputs/supp/<track>/` (results JSON, `REPORT.md`, logs).

Conventions
- Environment: `source /venv/main/bin/activate`. Nothing here touches the archived runs under `outputs/pairuav5d_60k/`
  or the original scripts in `superglue_fusion/`, `stage2_imagination/`, `online/`, `reviewer_baselines/` (copy and adapt instead).
- Metrics: only through `supp/common.py` (`errors`, `summarize`, `boot_indices`, `ci`). Range error d, endpoint error EPE,
  SR_d@1m (Table 2's SR), SR_EPE@1m, 5-DoF success, direction error, rotation-only pose AUC.
- Common test frame: `common.load_test()` (6,364 test pairs, 290 buildings, archived DreamNav predictions).
  1000-pair subset for expensive third-party methods: `reviewer_baselines/rb_recon.subset(1000, split='test')`,
  scale calibration on `subset(300, split='train')`.
- One GPU (RTX 5090, 32 GB) and a ~30-core CPU quota are shared by all tracks: call `bash supp/gpu_wait.sh <GB>` before a GPU job,
  respect the per-track budgets, run CPU-heavy work under `nice -n 10`, launch background jobs with `& echo $! > x.pid`
  and wait on `kill -0 $(cat x.pid)`.
