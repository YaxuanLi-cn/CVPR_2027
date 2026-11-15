#!/usr/bin/env bash
# Stage 2 (imagination refinement) on top of the stage-1 SuperGlue fusion. See STAGE2_IMAGINATION.md.
# Prerequisite: the stage-1 result set (superglue_fusion/run_half_zoom.sh): half_zoom_pred.pkl, geom_half.pkl, map_half.pkl, geom_zoom_half.pkl, absheading/.
# NOTE: the container CPU quota is ~30 cores; worker counts are sized for that.
set -euo pipefail
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; S2=$ROOT/outputs/pairuav5d_60k/superglue/stage2; cd "$ROOT/stage2_imagination"; mkdir -p "$S2/logs"

# --- module A: imagine the other view from the stage-1 pose, re-match imagination vs reality with SuperGlue (tiled, native resolution) ---
python build_guide.py                                   # stage-1 pose per pair (OOF for train rows) -> guide.pkl
# subset used for the reported numbers: all 6,364 test pairs of the stage-1 result set + 12,000 random train pairs (ids_small.npy + ids_more.npy)
for ids in ids_small ids_more; do
  for i in 0 1 2; do python imagine_match.py --ids "$S2/$ids.npy" --out "$S2/matches_imag" --scale 1.0 --max-grid 3 --workers 5 --shard $i 3 > "$S2/logs/imag_${ids}_$i.log" 2>&1 & done; wait
done
GEOM_PROCS=20 python imagine_geom.py "$S2/matches_imag" "$S2/geom_imag.pkl"      # pose from imagination matches + candidate verification scores
python stage2_fusion.py --tag s2 --jobs 4                                         # stage-1 prediction + imagination candidates -> stage-2 relative pose

# --- module B: absolute heading from the imagined canonical bird's-eye view (tilt from the SuperGlue geometry) ---
python build_items.py
python bev_heading.py --tag bev_s

# --- final assembly + ablation table (abs-heading source and norm-rescale exponent are chosen on the validation tours) ---
python stage2_eval.py --tag s2 --bev bev_s
