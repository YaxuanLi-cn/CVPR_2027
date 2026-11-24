#!/bin/bash
# Track D GPU queue, second version: closed-loop seeds after the unfiltered run ($1 = its pid).
# One simulator run at a time by default (it reserves ~7.6 GB of the 8 GB budget). A further seed is started next to a running one
# only when at least 17 GB of GPU memory are free anyway, i.e. when the other tracks have finished (leaves > 9 GB untouched).
set -o pipefail
cd /root/rasr-pairuav; source /venv/main/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
O=outputs/supp/trackD; L=$O/logs; UPID=$1; shift; pending=("$@"); declare -A pid
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
start() { local s=$1
  nice -n 10 python supp/trackD/sim_nav_seeds.py --seeds $s --out $O/sim_seed$s --step0-cache $O/cache/step0 --workers 4 --loader-workers 4 > $L/sim_seed$s.log 2>&1 &
  pid[$s]=$!; echo "$(date) started seed $s (pid ${pid[$s]}, free before start $(free_mb) MiB)"; }
while true; do
  for s in "${!pid[@]}"; do if ! kill -0 ${pid[$s]} 2>/dev/null; then echo "$(date) seed $s ended ($([ -f $O/sim_seed$s/config.json ] && echo complete || echo INCOMPLETE))"; unset "pid[$s]"; fi; done
  n=${#pid[@]}; busy=$n; kill -0 $UPID 2>/dev/null && busy=$((busy + 1))
  if [ ${#pending[@]} -eq 0 ] && [ $n -eq 0 ]; then break; fi
  if [ ${#pending[@]} -gt 0 ]; then
    s=${pending[0]}
    if [ -f $O/sim_seed$s/config.json ]; then pending=("${pending[@]:1}"); continue; fi
    f1=$(free_mb); sleep 20; f2=$(free_mb)
    if [ $busy -eq 0 ] && [ "$f1" -ge 8192 ] && [ "$f2" -ge 8192 ]; then start $s; pending=("${pending[@]:1}"); sleep 240
    elif [ $busy -ge 1 ] && [ $busy -le 2 ] && [ "$f1" -ge 17408 ] && [ "$f2" -ge 17408 ]; then start $s; pending=("${pending[@]:1}"); sleep 240; fi
  fi
  sleep 30
done
while [ -e $O/iter/.iteration_started ] && [ ! -e $O/iter/.iteration_done ]; do sleep 20; done
touch $O/.queue_done; echo "$(date) queue done"
