#!/bin/bash

IMAGE_ROOT="/root/pairUAV-5D-temp"
TRAIN_DIR="/root/baseline/_pairuav5d_data/train"
TEST_DIR="/root/baseline/_pairuav5d_data/test"
MATCH_DIR="/root/pairUAV-5D-temp/matches_superglue"

# Optional: Sample4Geo pretrained checkpoint
# Uncomment and set the path if you have the checkpoint
# CHECKPOINT="/root/dreamNav/baseline/sample4geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth"
CHECKPOINT="/root/baseline/sample4geo/pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth"

OUTPUT_FILE="test_results_enhanced.log"
SAVE_DIR="checkpoints_enhanced"

PYTHON="/root/baseline/.venv/bin/python"
"${PYTHON}" train_enhanced.py \
    --image_root ${IMAGE_ROOT} \
    --train_dir ${TRAIN_DIR} \
    --test_dir ${TEST_DIR} \
    --match_dir ${MATCH_DIR} \
    --checkpoint "${CHECKPOINT}" \
    --model_name convnext_base_384_in22ft1k \
    --img_size 384 \
    --batch_size 8 \
    --epochs 1 \
    --lr_backbone 5e-5 \
    --lr_match_encoder 1e-3 \
    --lr_regressor 1e-3 \
    --warmup_epochs 0 \
    --wd 1e-4 \
    --num_workers 4 \
    --match_feat_dim 256 \
    --output_file ${OUTPUT_FILE} \
    --save_dir ${SAVE_DIR}
