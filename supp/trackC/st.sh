#!/bin/bash
# one-line status of the Track C jobs; blocks up to $1 seconds (default 160) or until the chain is done
O=/root/rasr-pairuav/outputs/supp/trackC; t=${1:-160}; e=$(( $(date +%s) + t ))
while [ $(date +%s) -lt $e ] && [ ! -f $O/logs/chain.done ] && ! grep -qs "Traceback" $O/logs/*.log; do sleep 10; done
a=$(grep -o "step [0-9]*/[0-9]*" $O/logs/train_raw_sitedisjoint.log | tail -1)
echo "$(date +%H:%M) B:$a ep$(grep -c '== epoch' $O/logs/train_raw_sitedisjoint.log) | sens raw:$(grep -c 'MAE' $O/logs/sens_main_raw.log 2>/dev/null) bev:$(grep -c 'MAE' $O/logs/sens_main_bev.log 2>/dev/null) Aheld:$(grep -c 'MAE' $O/logs/sens_Aheld.log 2>/dev/null) Bheld:$(grep -c 'MAE' $O/logs/sens_Bheld.log 2>/dev/null) | done:$(ls $O/logs/*.done 2>/dev/null | wc -l) tb:$(grep -ls Traceback $O/logs/*.log | tr '\n' ' ')"
