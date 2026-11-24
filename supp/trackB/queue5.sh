#!/bin/bash
# extension: RoMa turned out to be one of the two strongest baselines, so its random subsets are enlarged (same fixed random order)
source /venv/main/bin/activate; cd /root/rasr-pairuav; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; L=outputs/supp/trackB/logs
until grep -q "QUEUE DONE" $L/queue.status; do sleep 10; done
retry() { need=$1; log=$2; shift 2; for i in $(seq 1 25); do bash supp/gpu_wait.sh $need; if nice -n 10 "$@" >> $log 2>&1; then return 0; fi; echo "[queue] attempt $i failed" >> $log; sleep 20; done; return 1; }
echo "queue5 start $(date)" >> $L/queue.status
retry 8 $L/roma_test.log python supp/trackB/run_matchers.py test roma 200;     echo "roma test 200 done $(date)" >> $L/queue.status
retry 8 $L/roma_real_in.log python supp/trackB/run_matchers.py real_in roma 80; echo "roma real_in 80 done $(date)" >> $L/queue.status
echo "QUEUE5 DONE $(date)" >> $L/queue.status
