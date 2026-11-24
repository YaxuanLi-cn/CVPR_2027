#!/bin/bash
# calibration pairs: every matcher, all candidate thresholds (CPU only)
source /venv/main/bin/activate; cd /root/rasr-pairuav; export GEOM_PROCS=3
nice -n 10 python supp/trackB/tb_geom.py sgrot train 3 2>&1 | grep -v -i warn | tail -1
for m in sift sgplain splg aliked loftr dedode; do nice -n 10 python supp/trackB/tb_geom.py $m train 0.2,0.4,0.75,1.5,3,6,12 2>&1 | grep -v -i warn | tail -1; done
nice -n 10 python supp/trackB/tb_geom.py sgrot test 3 2>&1 | grep -v -i warn | tail -1; nice -n 10 python supp/trackB/tb_geom.py sgplain test 0.4 2>&1 | grep -v -i warn | tail -1
echo GEOM_TRAIN_ALL_DONE
