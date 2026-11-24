#!/bin/bash
# Wait for the last checkpoint of a raw-view training run, stop the process (its built-in end-of-training evaluation repeats the
# 24-view pass four times) and write the predictions with one pass.   Usage: after_train.sh <tag>
tag=$1; O=/root/rasr-pairuav/outputs/supp/trackC; log=$O/logs/train_$tag.log; pid=$(cat $O/logs/train_$tag.pid); t0=$(stat -c %Y $O/logs/train_$tag.pid)
until grep -q "== epoch 9 " $log; do sleep 20; done
sleep 20                                   # the checkpoint is written right after the epoch line
t1=$(stat -c %Y $O/model_$tag.pt); kill $pid 2>/dev/null; sleep 5
source /venv/main/bin/activate; cd /root/rasr-pairuav; bash supp/gpu_wait.sh 2
python supp/trackC/predict_raw.py $tag $((t1 - t0)) > $O/logs/predict_$tag.log 2>&1
echo done > $O/logs/predict_$tag.done
