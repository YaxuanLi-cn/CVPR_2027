#!/bin/bash
# waits for the pi0.5 training process, then evaluates its final checkpoint once on the test split
cd /root/baseline/openpi
while kill -0 $(cat pairuav_v2/train_pi05.pid) 2>/dev/null; do sleep 20; done
step=$(ls checkpoints/pi05_pairuav_v2 | grep -E '^[0-9]+$' | sort -n | tail -1)
[ "$step" = "2979" ] || { echo "training did not finish (last step: $step)"; exit 1; }
OPENPI_NO_GEOM_AUG=1 PAIRUAV_V2_EPOCHS=1 PAIRUAV_V2_BATCH=16 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python pairuav_v2/eval_v2.py pi05_pairuav_v2 checkpoints/pi05_pairuav_v2/$step --out pairuav_v2/results/pi05 --batch-size 32 > pairuav_v2/eval_pi05.log 2>&1
echo PI05_DONE
