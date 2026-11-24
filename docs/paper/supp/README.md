# Supplementary material (CVPR 2027 submission)

`supp.pdf` is built from `supp.tex` with the official CVPR 2027 author kit (`cvpr.sty`, `ieeenat_fullname.bst`).

```bash
source /venv/main/bin/activate
python /root/rasr-pairuav/supp/make_tables.py          # result JSONs -> tables/*.tex and numbers.tex
for f in fig_dataset fig_imagination fig_heading; do python /root/rasr-pairuav/supp/figs/$f.py; done   # figures/*.pdf
pdflatex supp && bibtex supp && pdflatex supp && pdflatex supp
```

## Layout

| Path | Content |
|---|---|
| `supp.tex` | main file: packages, numbering (sections I, II, ...; tables / figures / equations S1, S2, ...) |
| `sec/*.tex` | one file per section; prose only |
| `tables/*.tex` | tables. All but `cands.tex` and `baselines.tex` are **generated**: do not edit by hand |
| `numbers.tex` | **generated** macros for every number quoted in the prose (`\SiiCamSRe`, ...). A number that has no result file is printed as a red `??` and listed by `make_tables.py` |
| `figures/` | figures (PDF + PNG), produced by `supp/figs/*.py` |
| `supp.bib` | bibliography; entries verified against arXiv / Crossref / CVF (`outputs/supp/related_work_verified.md`) |

## Where the numbers come from

Every number is read from a JSON file under `/root/rasr-pairuav/outputs/supp/` that an experiment script wrote
(`supp/make_tables.py` and `supp/make_tables_*.py` do the reading). Code is under `/root/rasr-pairuav/supp/`;
metric definitions are in `supp/common.py` and nowhere else.

| Section | Result files | Scripts |
|---|---|---|
| I.8 running time | `runtime/runtime.json` | `supp/runtime_bench.py` |
| I.9 baseline protocols | `baseline_protocols.md` (read from code, configs and logs) | - |
| II metrics | `metrics/main.json` | `supp/metrics_main.py` |
| III geometric baselines | `trackB/pairuav.json`, `sift_full.json`, `paired_sift.json`, `paired_roma_tta.json`, `unscreened.json`; `trackA/oracle_distance.json` | `supp/trackB/` (report: `outputs/supp/trackB/REPORT.md`) |
| III.4 SIFT candidates in the fusion | `trackA/sift_fusion.json` | `supp/trackA/sift_pairs.py`, `stage2_fusion_ctrl.py --sift`, `eval_sift_fusion.py`, `run_sift_fusion.sh` |
| IV.1 component inventory | `trackA/ablation_full.json`, `trackA/homography.json` | `supp/trackA/ablation_full.py`, `homography_baseline.py`, `sg_plain_full.py` |
| IV.2 controls | `trackA/controls.json`, `trackA/plane_vs_rot.json` | `supp/trackA/run_ctrl.sh`, `run_ctrl2.sh`, `run_fusion_cpu.sh`, `stage2_fusion_ctrl.py`, `eval_ctrl.py`, `plane_vs_rot.py` |
| IV.3 iteration, VII closed loop, IX unscreened pairs and dataset statistics | `trackD/iteration.json`, `closed_loop.json`, `onestep_gap.json`, `unfiltered.json`, `dataset_stats.json` | `supp/trackD/` (report: `outputs/supp/trackD/REPORT.md`) |
| V absolute heading | `trackC/heading.json`, `trackC/retrain_arrays.npz`, `trackA/site_geometry.json` | `supp/trackC/` (`python supp/trackC/report.py` prints the full report), `supp/trackA/site_geometry.py`, `supp/figs/fig_heading.py` |
| V.5 baselines in the camera frame | `trackE/camframe.json` | `supp/trackE/` (report: `outputs/supp/trackE/REPORT.md`) |
| VI planarity | `trackA/planarity.json`, `trackA/fig_imagination.json` | `supp/trackA/planarity.py`, `supp/figs/fig_imagination.py` |
| VII.5 classical servoing | `trackA/servo.json` | `supp/trackA/servo_table.py` (re-scores `outputs/reviewer_baselines/servo_log.pkl`) |
| VIII real photographs | `trackA/real.json`, `trackB/real.json` | `supp/trackA/real_table.py`, `supp/trackB/` |

Red notes of the form **[Authors: ...]** in the PDF mark information that only the authors can provide
(annotation protocol, release details, description of the split) or statements they should confirm before submission.
`README_zh.md` is a guide to the document in Chinese.
