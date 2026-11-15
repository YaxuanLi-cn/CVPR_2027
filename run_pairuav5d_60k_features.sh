#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate
export PYTHONPATH=/root/rasr-pairuav
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/root/rasr-pairuav
DATA=/root/pairUAV-5D-temp
RUNTIME=/root/mast3r_runtime
MODEL=/root/mast3r_runtime/models/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
OUT=$ROOT/outputs/pairuav5d_60k/features
mkdir -p "$OUT"
exec > >(tee -a "$OUT/features.log") 2>&1
python "$ROOT/data_processing/03_extract_features.py" \
  --mode production --data-root "$DATA" \
  --manifest "$ROOT/outputs/pairuav5d_60k/train_manifest.txt" \
  --json-dir "$DATA" --image-dir "$DATA/images" \
  --runtime-root "$RUNTIME" --model-path "$MODEL" \
  --device cuda:0 --dtype fp16 --image-size 512 \
  --pair-batch-size 4 --batched-pair-forward --flush-every 4 --resume \
  --output "$OUT/train.csv"
python "$ROOT/data_processing/03_extract_features.py" \
  --mode production --data-root "$DATA" \
  --manifest "$ROOT/outputs/pairuav5d_60k/test_manifest.txt" \
  --json-dir "$DATA" --image-dir "$DATA/images" \
  --runtime-root "$RUNTIME" --model-path "$MODEL" \
  --device cuda:0 --dtype fp16 --image-size 512 \
  --pair-batch-size 4 --batched-pair-forward --flush-every 4 --resume \
  --output "$OUT/test.csv"
