#!/bin/bash
# NoMaD camera-frame run. The reported recipe at batch 256 needs ~9 GB of GPU memory; the gradient is therefore accumulated over
# 4 micro-batches of 64 (exact for this model: GroupNorm / LayerNorm only), which keeps the run inside this track's GPU share.
O=/root/rasr-pairuav/outputs/supp/trackE; cd /root/rasr-pairuav/supp/trackE; source /venv/main/bin/activate
while kill -0 $(cat $O/logs/vint_camframe.pid) 2>/dev/null; do sleep 15; done
bash ../gpu_wait.sh 4
echo "launching NoMaD at $(date -u), free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)" > $O/logs/nomad_launch.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python train_v2_cam.py nomad --out $O/runs/nomad_camframe --val-tours 60 --eval-every 2 --test --frame cam --xy-scale 21.2132 --images-on-cpu --micro-batches 4 > $O/logs/nomad_camframe.log 2>&1
echo "NoMaD finished at $(date -u), exit $?" >> $O/logs/nomad_launch.txt
