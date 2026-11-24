#!/bin/bash
# DINO-WM (both frames, seeds 17 and 18) from encoder tokens. Starts when the Sample4Geo feature pass has released its CPU workers.
O=/root/rasr-pairuav/outputs/supp/trackE; cd /root/rasr-pairuav/supp/trackE; source /venv/main/bin/activate
while kill -0 $(cat $O/logs/s4g_extract.pid) 2>/dev/null; do sleep 15; done
bash ../gpu_wait.sh 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 python dinowm_tokens.py 4 cam:17,enu:17,cam:18,enu:18 > $O/logs/dinowm.log 2>&1
echo "dinowm finished $(date -u) exit $?" > $O/logs/dinowm_done.txt
