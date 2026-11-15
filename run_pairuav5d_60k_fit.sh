#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate
export PYTHONPATH=/root/rasr-pairuav
ROOT=/root/rasr-pairuav
OUT=$ROOT/outputs/pairuav5d_60k
FEAT=$OUT/features
MODELS=$OUT/models
mkdir -p "$MODELS"
python "$ROOT/scripts/attach_xyz_targets.py" \
  --features "$FEAT/train.csv" --manifest "$OUT/train_manifest.txt" \
  --data-root /root/pairUAV-5D-temp --output "$FEAT/train_xyz.csv"
python "$ROOT/scripts/attach_xyz_targets.py" \
  --features "$FEAT/test.csv" --manifest "$OUT/test_manifest.txt" \
  --data-root /root/pairUAV-5D-temp --output "$FEAT/test_xyz.csv"
for AXIS in x y z; do
  HEAD="coordinate_head_${AXIS}"
  if [[ ! -s "$MODELS/${HEAD}.pt" ]]; then
    python "$ROOT/models_src/train_head.py" --config "$ROOT/configs/heads.yaml" \
      --head-name "$HEAD" --train-features "$FEAT/train_xyz.csv" \
      --output "$MODELS/${HEAD}.pt" --target-column "${AXIS}_num" --device cuda:0
  fi
done
if [[ ! -s "$MODELS/tilt_head.pt" ]]; then
  python "$ROOT/models_src/train_head.py" --config "$ROOT/configs/heads.yaml" \
    --head-name tilt_head --train-features "$FEAT/train_xyz.csv" \
    --output "$MODELS/tilt_head.pt" --target-column tilt_num --device cuda:0
fi
python "$ROOT/models_src/heading_baseline.py" fit \
  --features "$FEAT/train_xyz.csv" --output "$MODELS/heading_baseline.json" \
  --target-column heading_num
python "$ROOT/evaluate_pairuav5d_full.py" \
  --test-features "$FEAT/test_xyz.csv" --model-dir "$MODELS" \
  --output "$OUT/metrics.json" --device cuda:0
