#!/bin/bash
# Control variants of the refinement stage: 3,000 test pairs + 3,000 fusion-training pairs (ids_ctrl.npy), resumable.
set -uo pipefail
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; S2=$ROOT/outputs/pairuav5d_60k/superglue/stage2; O=$ROOT/outputs/supp/trackA; cd $ROOT
run () {  # name, number of shards, extra args...
  name=$1; ns=$2; shift 2
  bash supp/gpu_wait.sh 3
  for i in $(seq 0 $((ns - 1))); do
    nice -n 5 python supp/trackA/imagine_match_ctrl.py "$@" --out $O/matches_$name --workers 3 --shard $i $ns > $O/logs/match_${name}_$i.log 2>&1 &
  done; wait
  GEOM_PROCS=6 nice -n 10 python supp/trackA/geom_ctrl.py $O/matches_$name $O/geom_$name.pkl ${GUIDE:-} > $O/logs/geom_$name.log 2>&1
  echo "$name done $(date +%T)" >> $O/logs/chain.log
}
run rot_tiles   2 --ids $O/ids_ctrl.npy --scale 1.0 --max-grid 3 --rot-only
run plane_1tile 1 --ids $O/ids_ctrl.npy --scale 1.0 --max-grid 1
run plane_512   1 --canvas 512 288 --ids $O/ids_ctrl.npy --scale 1.0 --max-grid 1
GUIDE=$O/guide_oracle.pkl run oracle 2 --guide $O/guide_oracle.pkl --ids $O/ids_oracle.npy --scale 1.0 --max-grid 3
echo "ALL DONE $(date +%T)" >> $O/logs/chain.log
