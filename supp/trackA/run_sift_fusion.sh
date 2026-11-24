#!/bin/bash
# SIFT pose and scale cues as additional candidates of the two fusion stages (control subset: 3,000 train + 3,000 test pairs).
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; O=$ROOT/outputs/supp/trackA; cd $ROOT
until [ ! -e /proc/$(cat $O/logs/sift_match.pid) ] && [ ! -e /proc/$(cat $O/logs/sift_match_rev.pid) ]; do sleep 5; done
nice -n 19 python supp/trackA/sift_pairs.py match 8 > $O/logs/sift_match_final.log 2>&1          # picks up anything still missing (also test pairs)
nice -n 10 python supp/trackA/sift_pairs.py geom 10 > $O/logs/sift_geom.log 2>&1
for v in "sift_full" "sift_coarse --no-imag-cands"; do set -- $v; tag=$1; shift
  nice -n 10 python supp/trackA/stage2_fusion_ctrl.py --tag $tag --jobs 2 --ids $O/ids_ctrl.npy --sift $O/geom_sift.pkl "$@" > $O/logs/fusion_$tag.log 2>&1 &
done; wait
nice -n 10 python supp/trackA/eval_sift_fusion.py > $O/logs/eval_sift_fusion.log 2>&1
echo "sift fusion done $(date +%T)" >> $O/logs/chain_fusion.log
