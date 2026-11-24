# PairUAV-5D Baseline Runs

This directory is a portable snapshot of PairUAV-specific baseline scripts
and lightweight result artifacts. See `../README.md` for upstream revisions
and the list of artifacts intentionally omitted from Git.

## Prepared data

Run `python tools/prepare_pairuav5d.py --dataset /path/to/pairuav5d --output /path/to/prepared`
to materialize the legacy JSON layout. The script selects tours with at
least 900 saved SuperGlue match files, assigns the last completed tour to the
test split, and uses earlier completed tours for training. It does not copy
images or match files.

Labels are mapped as follows:

- `heading_num` = `delta.heading_deg`
- `range_num` = horizontal displacement, `hypot(delta.x_m, delta.y_m)`
- `image_a` and `image_b` reference files in the source image dataset

At the time of this run, tours `0000` through `0005` produced 6,000 training
pairs and tour `0006` produced 1,000 test pairs. This is a snapshot of tours
whose SuperGlue results were already available, not the full 1.4M-pair set.
To prepare a newer snapshot, choose a new empty `--output` directory.

## Entry points

- Sample4Geo: `cd sample4geo && bash run.sh`
- Sample4Geo + matches: `cd new_sample4geo && bash run_enhanced.sh`
- Ablations: run `run_ablation_no_match.sh` or
  `run_ablation_no_match_no_cossin.sh` from `new_sample4geo/`
- AI2THOR: run `python train_xyz_supervised.py` from `ai2thor/`
- OpenPI: use the scripts in `openpi/pairuav_v2/`
- DINOv3: `cd dinov3 && bash run.sh`

Required datasets and checkpoints are not included in this snapshot. Download
them from their original sources and pass their paths to the scripts; some
historical runner scripts still contain server-specific default paths.

OpenPI uses Pi05 LoRA fine-tuning to fit the RTX 5090 memory budget. The full
fine-tuning configuration exceeded available GPU memory during initialization.
