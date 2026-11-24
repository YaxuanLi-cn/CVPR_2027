#!/bin/bash
# Track C: after the site-disjoint training: evaluation, sensitivity of the site-disjoint network on the held-out sites, figures, report.
O=/root/rasr-pairuav/outputs/supp/trackC; cd /root/rasr-pairuav; source /venv/main/bin/activate; S=supp/trackC
until [ -f $O/logs/predict_raw_sitedisjoint.done ]; do sleep 15; done
nice -n 10 python $S/eval_retrain.py > $O/logs/eval_retrain.log 2>&1
bash supp/gpu_wait.sh 2
TRACKC_MEMCAP_GB=1.1 nice -n 5 python $S/sens.py --net raw --tag Bheld --n 1500 --bs 24 --subset heldout --conds focus --raw-weights $O/model_raw_sitedisjoint.pt > $O/logs/sens_Bheld.log 2>&1
until [ -f $O/logs/chainA.done ]; do sleep 15; done
python $S/sens_combine.py Bheld part_sens_B_heldout.json > $O/logs/sens_combine_Bheld.log 2>&1
nice -n 10 python $S/figs.py > $O/logs/figs.log 2>&1
echo done > $O/logs/chain.done
