#!/bin/bash
# geometry (CPU) for one pair set at the thresholds chosen on the calibration pairs; incremental, can be re-run while matching is in progress
# usage: geom_set.sh <test|real_in|real_out> [matcher ...]
source /venv/main/bin/activate; cd /root/rasr-pairuav; export GEOM_PROCS=${GEOM_PROCS:-3}; S=$1; shift
MS=${@:-sift sgplain sgrot splg aliked loftr dedode roma mast3r}
for m in $MS; do
  T=$(python -c "import json; d=json.load(open('outputs/supp/trackB/geom/chosen_thr.json')); print(','.join(str(x) for x in d.get('$m', [3.0])))")
  if [ "$m" = "sgplain" ] || [ "$m" = "sgrot" ] || [ -d outputs/supp/trackB/matches/$m/$S ]; then nice -n 10 python supp/trackB/tb_geom.py $m $S $T 2>&1 | grep -v -i warn | tail -2; fi
done; echo "GEOM_SET_DONE $S $(date)"
