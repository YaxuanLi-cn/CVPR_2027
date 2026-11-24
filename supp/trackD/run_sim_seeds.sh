#!/bin/bash
# Track D item 1: closed-loop runs with new actuation-noise seeds, one seed after the other (each run = all 8 sites, 160 episodes).
# usage: run_sim_seeds.sh "<pids to wait for>" seed [seed ...]
set -uo pipefail
cd /root/rasr-pairuav; source /venv/main/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for p in $1; do while kill -0 $p 2>/dev/null; do sleep 10; done; done; shift
for s in "$@"; do
  if [ -f outputs/supp/trackD/sim_seed$s/config.json ]; then echo "seed $s already done"; continue; fi
  bash supp/gpu_wait.sh 8
  nice -n 10 python supp/trackD/sim_nav_seeds.py --seeds $s --out outputs/supp/trackD/sim_seed$s --step0-cache outputs/supp/trackD/cache/step0 --workers 6 --loader-workers 6 > outputs/supp/trackD/logs/sim_seed$s.log 2>&1
done
