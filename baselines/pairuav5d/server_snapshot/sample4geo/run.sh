#!/bin/bash

IMAGE_ROOT="/root/pairUAV-5D-temp"
TRAIN_DIR="/root/baseline/_mast3r_split_data_v2/train"
TEST_DIR="/root/baseline/_mast3r_split_data_v2/test"
CHECKPOINT="pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth"

OUTPUT_FILE="test_results_mast3r_split.log"

PYTHON="/root/baseline/.venv/bin/python"
"${PYTHON}" train.py \
    --image_root ${IMAGE_ROOT} \
    --train_dir ${TRAIN_DIR} \
    --test_dir ${TEST_DIR} \
    --checkpoint ${CHECKPOINT} \
    --model_name convnext_base.fb_in22k_ft_in1k_384 \
    --img_size 384 \
    --batch_size 8 \
    --epochs 1 \
    --lr_backbone 1e-5 \
    --lr_regressor 1e-3 \
    --warmup_epochs 0 \
    --freeze_backbone \
    --output_file ${OUTPUT_FILE}
