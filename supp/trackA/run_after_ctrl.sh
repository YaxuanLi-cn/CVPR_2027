#!/bin/bash
# queued behind run_ctrl.sh: plain SuperGlue on the full test set
source /venv/main/bin/activate; cd /root/rasr-pairuav; O=outputs/supp/trackA
while kill -0 $(cat $O/logs/run_ctrl.pid) 2>/dev/null; do sleep 30; done
bash supp/gpu_wait.sh 3
nice -n 5 python supp/trackA/sg_plain_full.py > $O/logs/sg_plain_full.log 2>&1
echo "sg_plain_full done $(date +%T)" >> $O/logs/chain.log
