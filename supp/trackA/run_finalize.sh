#!/bin/bash
# As each control matching variant finishes (line "<name> done" in chain.log), fit its fusion, then re-score everything.
source /venv/main/bin/activate; cd /root/rasr-pairuav; O=outputs/supp/trackA
for name in rot_tiles plane_1tile plane_512; do
  until grep -q "^$name done" $O/logs/chain.log 2>/dev/null; do sleep 30; done
  nice -n 10 python supp/trackA/stage2_fusion_ctrl.py --tag $name --jobs 1 --imag $O/geom_$name.pkl > $O/logs/fusion_$name.log 2>&1
  nice -n 10 python supp/trackA/eval_ctrl.py > $O/logs/eval_ctrl.log 2>&1
  echo "fusion+eval $name done $(date +%T)" >> $O/logs/chain_fusion.log
done
until grep -q "^oracle done" $O/logs/chain.log 2>/dev/null; do sleep 30; done
nice -n 10 python supp/trackA/eval_ctrl.py > $O/logs/eval_ctrl.log 2>&1; echo "eval with oracle done $(date +%T)" >> $O/logs/chain_fusion.log
until grep -q "^sg_plain_full done" $O/logs/chain.log 2>/dev/null; do sleep 30; done
nice -n 10 python supp/trackA/ablation_full.py > $O/logs/ablation_full.log 2>&1; echo "ablation_full with G0 done $(date +%T)" >> $O/logs/chain_fusion.log
