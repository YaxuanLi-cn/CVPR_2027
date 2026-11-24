#!/bin/bash
# Track D GPU queue. One engine job at a time (the engine alone reserves ~7.6 GB of the 8 GB budget):
#   unfiltered pairs (already running, pid in $1) -> closed-loop seed 1 -> internal iteration (unless already started) -> seeds 2, 3.
# A side loop starts the light iteration job (two SuperGlue matchers, ~1.4 GB each) earlier, next to a running engine job,
# but only when at least 6 GB of GPU memory are free anyway (i.e. the other tracks have released theirs).
set -uo pipefail
cd /root/rasr-pairuav; source /venv/main/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
O=outputs/supp/trackD; L=$O/logs; LOCK=$O/iter/.iteration_started
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
run_iter() { WORKERS=2 GPU_GB=2 bash supp/trackD/run_iterate.sh >> $L/iterate.log 2>&1; touch $O/iter/.iteration_done; }
sim() {
  local s=$1
  if [ -f $O/sim_seed$s/config.json ]; then echo "seed $s already done"; return; fi
  bash supp/gpu_wait.sh 8
  nice -n 10 python supp/trackD/sim_nav_seeds.py --seeds $s --out $O/sim_seed$s --step0-cache $O/cache/step0 --workers 6 --loader-workers 6 > $L/sim_seed$s.log 2>&1 &
  echo $! > $L/engine_job.pid; wait $!
}
( # opportunistic side loop
  while [ ! -e $LOCK ] && [ ! -e $O/.queue_done ]; do
    sleep 45
    if kill -0 $(cat $L/engine_job.pid 2>/dev/null) 2>/dev/null; then
      f1=$(free_gb); sleep 20; f2=$(free_gb)
      if [ "$f1" -ge 6144 ] && [ "$f2" -ge 6144 ] && mkdir $LOCK 2>/dev/null; then echo "$(date) side loop starts the iteration job (free $f2 MiB)"; run_iter; fi
    fi
  done ) &
echo $1 > $L/engine_job.pid; while kill -0 $1 2>/dev/null; do sleep 10; done; echo "$(date) unfiltered run finished"
sim 1; echo "$(date) seed 1 finished"
if mkdir $LOCK 2>/dev/null; then echo "$(date) queue starts the iteration job"; bash supp/gpu_wait.sh 3; run_iter; echo "$(date) iteration finished"; fi
sim 2; echo "$(date) seed 2 finished"
sim 3; echo "$(date) seed 3 finished"
while [ -e $LOCK ] && [ ! -e $O/iter/.iteration_done ]; do sleep 20; done
touch $O/.queue_done; echo "$(date) queue done"
