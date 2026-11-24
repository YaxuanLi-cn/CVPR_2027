set -euo pipefail
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; S2=$ROOT/outputs/pairuav5d_60k/superglue/stage2; cd "$ROOT/stage2_imagination"; mkdir -p "$S2/logs"

python build_guide.py

for ids in ids_small ids_more; do
  for i in 0 1 2; do python imagine_match.py --ids "$S2/$ids.npy" --out "$S2/matches_imag" --scale 1.0 --max-grid 3 --workers 5 --shard $i 3 > "$S2/logs/imag_${ids}_$i.log" 2>&1 & done; wait
done
GEOM_PROCS=20 python imagine_geom.py "$S2/matches_imag" "$S2/geom_imag.pkl"
python stage2_fusion.py --tag s2 --jobs 4

python build_items.py
python bev_heading.py --tag bev_s

python stage2_eval.py --tag s2 --bev bev_s
