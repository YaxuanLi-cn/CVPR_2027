#!/bin/bash
# when SuperGlue and MASt3R have finished the real pairs: geometry for every matcher on the real pairs (recomputed from scratch so that the
# MASt3R source depth is available for the native-scale variant), then the evaluation
source /venv/main/bin/activate; cd /root/rasr-pairuav; L=outputs/supp/trackB/logs; export GEOM_PROCS=4
until grep -q "mast3r real_in done" $L/queue.status; do sleep 20; done
mkdir -p outputs/supp/trackB/geom_nodepth; mv outputs/supp/trackB/geom/*_real_in.pkl outputs/supp/trackB/geom_nodepth/ 2>/dev/null
bash supp/trackB/geom_set.sh real_in sift sgplain sgrot splg aliked loftr dedode
nice -n 10 python supp/trackB/tb_geom.py mast3r real_in 0.2,0.4,0.75,1.5,3,6,12 2>&1 | grep -v -i warn | tail -7
if [ -d outputs/supp/trackB/matches/sift/real_out ]; then nice -n 10 python supp/trackB/tb_geom.py sift real_out 0.2 2>&1 | grep -v -i warn | tail -1; fi
nice -n 10 python supp/trackB/eval_real.py > $L/eval_real_pass2.log 2>&1
echo "AFTER_REAL_DONE $(date)"
