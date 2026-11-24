#!/bin/bash
# Fusion variants that need no new matching, fitted on the 3,000 training pairs of ids_ctrl.npy (archived imagination geometry).
set -uo pipefail
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; O=$ROOT/outputs/supp/trackA; cd $ROOT
f () { tag=$1; shift; nice -n 10 python supp/trackA/stage2_fusion_ctrl.py --tag $tag --jobs 1 --ids $O/ids_ctrl.npy "$@" > $O/logs/fusion_$tag.log 2>&1; echo "$tag done $(date +%T)" >> $O/logs/chain_fusion.log; }
f full
f fusion_only --no-imag-cands
f no_ver --no-ver
f no_scale --no-scale
f uniform_nores --uniform --no-residual
f learned_nores --no-residual
echo "ALL DONE(3000) $(date +%T)" >> $O/logs/chain_fusion.log
