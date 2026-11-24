#!/bin/bash
# when the light matchers and SIFT have finished the test pairs: geometry at the chosen thresholds (+ two extra thresholds for SIFT as a sensitivity check), then the evaluation
source /venv/main/bin/activate; cd /root/rasr-pairuav; L=outputs/supp/trackB/logs
until grep -q "light test done" $L/queue.status; do sleep 20; done
until [ $(ls outputs/supp/trackB/matches/sift/test | wc -l) -ge 1000 ]; do sleep 20; done
GEOM_PROCS=4 bash supp/trackB/geom_set.sh test sift splg aliked loftr dedode
GEOM_PROCS=4 nice -n 10 python supp/trackB/tb_geom.py sift test 0.75,3 2>&1 | grep -v -i warn | tail -3
nice -n 10 python supp/trackB/eval_pairuav.py > $L/eval_pairuav_pass2.log 2>&1
echo "AFTER_LIGHT_TEST_DONE $(date)"
