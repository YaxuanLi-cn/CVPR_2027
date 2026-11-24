#!/bin/bash

PYTHON="/root/baseline/.venv/bin/python"
"${PYTHON}" train_pairuav5d.py \
    --image-root /root/pairUAV-5D-temp \
    --train-dir /root/baseline/_mast3r_split_data_v2/train \
    --test-dir /root/baseline/_mast3r_split_data_v2/test \
    --model-name vit_large_patch16_dinov3.lvd1689m \
    --batch-size 16 \
    --epochs 1 \
    --output-dir outputs_mast3r_split
