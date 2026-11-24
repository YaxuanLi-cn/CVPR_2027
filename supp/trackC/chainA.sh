#!/bin/bash
# Track C: inference-only studies with the archived networks, two capped processes at a time (GPU budget), while the site-disjoint training continues.
O=/root/rasr-pairuav/outputs/supp/trackC; cd /root/rasr-pairuav; source /venv/main/bin/activate; S=supp/trackC; export TRACKC_MEMCAP_GB=1.1
nice -n 5 python $S/sens.py --net bev --tag main --n 2000 --bs 24 > $O/logs/sens_main_bev.log 2>&1 &
while kill -0 $(cat $O/logs/sens_main_raw.pid) 2>/dev/null; do sleep 15; done          # the raw-view run was started by hand
nice -n 5 python $S/real.py --bs 24 > $O/logs/real.log 2>&1
nice -n 5 python $S/sens.py --net raw --tag Aheld --n 1500 --bs 24 --subset heldout --conds focus > $O/logs/sens_Aheld.log 2>&1
wait
python $S/sens_combine.py main part_sens.json > $O/logs/sens_combine_main.log 2>&1
python $S/sens_combine.py Aheld part_sens_A_heldout.json > $O/logs/sens_combine_Aheld.log 2>&1
echo done > $O/logs/chainA.done
