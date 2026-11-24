#!/bin/bash
# Coordinator's instruction (18:03): seeds in parallel, hard deadline 19:45 container clock. At the deadline (or as soon as the
# three new seeds are complete) closed_loop.json / closed_loop.md are written from the COMPLETE seeds only; runs still going are stopped.
cd /root/rasr-pairuav; source /venv/main/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
O=outputs/supp/trackD; target=$(date -d "today 19:45:00" +%s)
while [ $(date +%s) -lt $target ]; do
  if [ -f $O/sim_seed1/config.json ] && [ -f $O/sim_seed2/config.json ] && [ -f $O/sim_seed3/config.json ]; then break; fi
  sleep 5
done
echo "$(date) deadline script fires"
for s in 1 2 3; do
  if [ ! -f $O/sim_seed$s/config.json ]; then p=$(cat $O/logs/sim_seed$s.pid); pkill -P $p 2>/dev/null; kill $p 2>/dev/null; sleep 2; pkill -9 -P $p 2>/dev/null; kill -9 $p 2>/dev/null; rm -rf /dev/shm/simnav_trackD_$p; echo "seed $s INCOMPLETE at the deadline: stopped (pid $p), not used"; fi
done
dirs=""; used=""; for s in 1 2 3; do if [ -f $O/sim_seed$s/config.json ]; then dirs="$dirs $O/sim_seed$s"; used="$used $s"; fi; done
echo "complete new seeds:$used"
nice -n 5 python supp/trackD/closed_loop_report.py $dirs > $O/logs/closed_loop_report.log 2>&1 && echo "closed_loop.json written"
python supp/trackD/make_report.py >> $O/logs/closed_loop_report.log 2>&1
echo "$used" > $O/.closed_loop_final; echo "$(date) done"
