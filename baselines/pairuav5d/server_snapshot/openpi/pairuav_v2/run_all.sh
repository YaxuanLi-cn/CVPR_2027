#!/bin/bash
# PairUAV-5D v2 runs for pi0.5 and pi0-FAST: cached 224x224 images, no geometric augmentation,
# 1 epoch over the 47,687 training pairs at batch 16, then one evaluation on the 12,313 test pairs.
set -eo pipefail
cd /root/baseline/openpi
export OPENPI_NO_GEOM_AUG=1 WANDB_MODE=disabled PAIRUAV_V2_EPOCHS=1 PAIRUAV_V2_BATCH=16
PY=.venv/bin/python
R=pairuav_v2/results

# 0. the existing 1-epoch pi0.5 checkpoint (trained with openpi's default crop/rotate augmentation): save all 5 outputs
if [ ! -f $R/pi05_old/test_metrics.json ]; then
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 $PY pairuav_v2/eval_v2.py pi05_pairuav_v2 checkpoints/pi05_xyz_mast3r_split/5959 --out $R/pi05_old > pairuav_v2/eval_pi05_old.log 2>&1
fi
for name in pi0_fast; do  # pi0.5 is run by train command + finish_pi05.sh (three-camera layout)
  cfg=${name}_pairuav_v2
  # this checkout writes checkpoints to checkpoints/<exp-name>/<step>
  if [ -z "$(ls checkpoints/$cfg 2>/dev/null | grep -E '^[0-9]+$')" ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 $PY scripts/train.py $cfg --exp-name=$cfg --overwrite > pairuav_v2/train_$name.log 2>&1
  fi
  step=$(ls checkpoints/$cfg | grep -E '^[0-9]+$' | sort -n | tail -1)
  if [ ! -f $R/$name/test_metrics.json ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 $PY pairuav_v2/eval_v2.py $cfg checkpoints/$cfg/$step --out $R/$name --batch-size 16 > pairuav_v2/eval_$name.log 2>&1
  fi
done
echo ALL_DONE
