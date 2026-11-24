#!/bin/bash
# Track B job queue, last part (GPU heavily shared: random subsets). MASt3R on the real pairs, MASt3R re-matching on PairUAV, then RoMa.
source /venv/main/bin/activate; cd /root/rasr-pairuav; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; L=outputs/supp/trackB/logs
retry() {   # retry <need_GB> <logfile> <cmd...>
  need=$1; log=$2; shift 2
  for i in $(seq 1 25); do
    bash supp/gpu_wait.sh $need
    if nice -n 10 "$@" >> $log 2>&1; then return 0; fi
    echo "[queue] attempt $i of '$*' failed, retrying" >> $log; sleep 20
  done; return 1
}
echo "queue4 start $(date)" >> $L/queue.status
retry 5 $L/mast3r_real_in.log python supp/trackB/run_mast3r.py real_in 100; echo "mast3r real_in done $(date)" >> $L/queue.status
retry 5 $L/mast3r_train.log python supp/trackB/run_mast3r.py train 80; echo "mast3r train done $(date)" >> $L/queue.status
retry 5 $L/mast3r_test.log python supp/trackB/run_mast3r.py test 200; echo "mast3r test done $(date)" >> $L/queue.status
retry 8 $L/roma_train.log python supp/trackB/run_matchers.py train roma 40;   echo "roma train done $(date)" >> $L/queue.status
retry 8 $L/roma_test.log python supp/trackB/run_matchers.py test roma 100;     echo "roma test done $(date)" >> $L/queue.status
retry 8 $L/roma_real_in.log python supp/trackB/run_matchers.py real_in roma 40; echo "roma real_in done $(date)" >> $L/queue.status
echo "QUEUE DONE $(date)" >> $L/queue.status
