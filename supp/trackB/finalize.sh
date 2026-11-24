#!/bin/bash
# geometry for whatever MASt3R / RoMa / unscreened matches exist, threshold choice, all evaluations, paired comparisons, report
source /venv/main/bin/activate; cd /root/rasr-pairuav; export GEOM_PROCS=${GEOM_PROCS:-4}; L=outputs/supp/trackB/logs
for m in mast3r roma; do
  if [ -d outputs/supp/trackB/matches/$m/train ]; then nice -n 10 python supp/trackB/tb_geom.py $m train 0.2,0.4,0.75,1.5,3,6,12 2>&1 | grep -v -i warn | tail -1; fi
done
python supp/trackB/choose_thr.py 2>&1 | grep -v -i warn
thr() { python -c "import json; d=json.load(open('outputs/supp/trackB/geom/chosen_thr.json')); print(','.join(str(x) for x in sorted(set(d.get('$1', [3.0])) | ({3.0} if '$1' == 'mast3r' else set()))))"; }
for m in mast3r roma; do
  for s in test real_in; do
    if [ -d outputs/supp/trackB/matches/$m/$s ] && [ -f outputs/supp/trackB/geom/${m}_train.pkl ]; then nice -n 10 python supp/trackB/tb_geom.py $m $s $(thr $m) 2>&1 | grep -v -i warn | tail -2; fi
  done
done
for m in sift dedode; do
  if [ -d outputs/supp/trackB/matches/$m/unscreened ]; then nice -n 10 python supp/trackB/tb_geom.py $m unscreened $(thr $m) 2>&1 | grep -v -i warn | tail -2; fi
done
nice -n 10 python supp/trackB/eval_pairuav.py > $L/eval_pairuav_final.log 2>&1; echo "eval_pairuav exit $?"
nice -n 10 python supp/trackB/eval_real.py > $L/eval_real_final.log 2>&1; echo "eval_real exit $?"
nice -n 10 python supp/trackB/compare_best.py sift > $L/compare_sift.log 2>&1; nice -n 10 python supp/trackB/compare_best.py dedode tta > $L/compare_dedode.log 2>&1; if [ -f outputs/supp/trackB/geom/roma_test.pkl ]; then nice -n 10 python supp/trackB/compare_best.py roma tta > $L/compare_roma.log 2>&1; fi
if [ -d outputs/supp/trackB/matches/sift/test_full ]; then nice -n 10 python supp/trackB/tb_geom.py sift test_full 0.2 2>&1 | grep -v -i warn | tail -1; nice -n 10 python supp/trackB/eval_sift_full.py > $L/eval_sift_full.log 2>&1; echo "eval_sift_full exit $?"; fi
nice -n 10 python supp/trackB/eval_unscreened.py > $L/eval_unscreened.log 2>&1; echo "eval_unscreened exit $?"
python supp/trackB/make_report.py; echo "FINALIZE_DONE $(date)"
