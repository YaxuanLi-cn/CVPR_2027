#!/bin/bash
# Extra control: tile-wise second pass with NO pose-guided alignment (only the k*90deg rotation of the coarse search).
# Raw geometry only, on the 1,000 test pairs of the oracle control (the shared GPU does not leave room for the 6,000 pairs a fusion fit needs).
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; O=$ROOT/outputs/supp/trackA; cd $ROOT
bash supp/gpu_wait.sh 3
for i in 0 1; do
  nice -n 5 python supp/trackA/imagine_match_ctrl.py --guide $O/guide_rot90.pkl --ids $O/ids_oracle.npy --scale 1.0 --max-grid 3 --rot-only --out $O/matches_rot90_tiles --workers 3 --shard $i 2 > $O/logs/match_rot90_tiles_$i.log 2>&1 &
done; wait
mkdir -p $O/matches_rot90_sub; python - <<'PY'
import numpy as np, os, shutil
O='/root/rasr-pairuav/outputs/supp/trackA'
for i in np.load(f'{O}/ids_oracle.npy'):
    f=f'{O}/matches_rot90_tiles/{int(i)}.npz'
    if os.path.exists(f): shutil.copyfile(f, f'{O}/matches_rot90_sub/{int(i)}.npz')
PY
GEOM_PROCS=6 nice -n 10 python supp/trackA/geom_ctrl.py $O/matches_rot90_sub $O/geom_rot90_tiles.pkl > $O/logs/geom_rot90_tiles.log 2>&1
nice -n 10 python supp/trackA/eval_ctrl.py > $O/logs/eval_ctrl.log 2>&1
echo "rot90_tiles raw geometry done $(date +%T)" >> $O/logs/chain_fusion.log
