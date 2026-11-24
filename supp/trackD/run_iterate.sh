#!/bin/bash
# Track D item 2: two extra rounds of internal refinement on 1000 fixed test pairs (see iterate.py).
set -euo pipefail
cd /root/rasr-pairuav; source /venv/main/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
W=outputs/supp/trackD/iter; I=supp/trackD/iterate.py; M=stage2_imagination/imagine_match.py; WK=${WORKERS:-2}
python $I check
python $I guide 2 A; python $I guide 2 B
bash supp/gpu_wait.sh ${GPU_GB:-2}; nice -n 10 python $M --guide $W/guide_r2A.pkl --out $W/matches_r2 --scale 1.0 --max-grid 3 --workers $WK
for p in A B; do nice -n 10 python $I geom 2 $p; python $I fuse 2 $p; python $I guide 3 $p; done
python $I report
bash supp/gpu_wait.sh 3
nice -n 10 python $M --guide $W/guide_r3A.pkl --out $W/matches_r3A --scale 1.0 --max-grid 3 --workers $WK &
nice -n 10 python $M --guide $W/guide_r3B.pkl --out $W/matches_r3B --scale 1.0 --max-grid 3 --workers $WK &
wait
for p in A B; do nice -n 10 python $I geom 3 $p; python $I fuse 3 $p; done
python $I report
