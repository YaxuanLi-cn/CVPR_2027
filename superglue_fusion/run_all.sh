#!/usr/bin/env bash
# End-to-end pipeline for the SuperGlue-fusion result on the fixed 60k PairUAV-5D split (see SUPERGLUE_EXPERIMENTS.md).
# Prerequisite: MASt3R feature tables outputs/pairuav5d_60k/features/{train,test}_xyz.csv from the existing baseline pipeline.
# The reported result used the first 31,273 pairs (a random half); this script processes whatever is in matches_rot/
# (all 60,000 pass-1 match files already exist, so step 1 is a no-op unless they are deleted).
# NOTE: the container CPU quota is ~30 cores even though nproc reports 128; worker counts are sized for that.
set -euo pipefail
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav; SGD=$ROOT/outputs/pairuav5d_60k/superglue; cd "$ROOT"; mkdir -p "$SGD/logs"

# 0. pair table (labels + KML poses; KML poses of TRAIN tours are auxiliary training targets, never used at inference)
[[ -s $SGD/pairs.pkl ]] || python superglue_fusion/build_pair_table.py

# 1. pass 1: rotation-aware, multi-scale SuperGlue (5 shards on one GPU; resumable, skips existing files)
for i in 0 1 2 3 4; do
  python superglue_fusion/run_superglue_rot.py --out "$SGD/matches_rot" --fine 1280 720 --shuffle --shard $i 5 > "$SGD/logs/sg_rot_$i.log" 2>&1 &
done; wait

# 2. MASt3R-feature-only controls / priors with tour-grouped OOF
[[ -s $SGD/stage0.pkl ]] || python superglue_fusion/stage0_mast3r.py
[[ -s $SGD/stage0_mlp.pkl ]] || python superglue_fusion/stage0_mlp.py

# 3. pass-1 geometry: essential matrix pose, then MAP pose with MLP tilt/heading priors
GEOM_PROCS=24 python superglue_fusion/extract_geom.py "$SGD/matches_rot" "$SGD/geom_rot.pkl"
GEOM_PROCS=24 python superglue_fusion/map_pose.py "$SGD/matches_rot" "$SGD/geom_rot.pkl" "$SGD/map_rot.pkl"

# 4. pass 2: pose-guided zoom re-matching on every pair (resumable), then its geometry
for i in 0 1 2 3 4; do
  python superglue_fusion/run_superglue_zoom.py --geom "$SGD/geom_rot.pkl" --map "$SGD/map_rot.pkl" --out "$SGD/matches_zoom" \
    --max-inl 1000000 --workers 4 --shard $i 5 > "$SGD/logs/sg_zoom_$i.log" 2>&1 &
done; wait
GEOM_PROCS=24 python superglue_fusion/extract_geom_zoom.py "$SGD/matches_zoom" "$SGD/geom_zoom.pkl"

# 5. fusion of SuperGlue geometry with the MASt3R predictions
python superglue_fusion/stage1_fusion.py --geom "$SGD/geom_rot.pkl" --map "$SGD/map_rot.pkl" --zoom "$SGD/geom_zoom.pkl" --partial --tag fusion --jobs 4

# 6. single-image absolute heading (needed for world-frame x/y), then final assembly + the six MAEs
[[ -s $SGD/absheading/images.pkl ]] || python superglue_fusion/abs_heading_cache.py
[[ -s $SGD/absheading/images2.pkl ]] || python superglue_fusion/abs_heading_cache2.py
[[ -s $SGD/absheading/pred_cnx_s_d8.pkl ]] || python superglue_fusion/abs_heading_train2.py --tag cnx_s_d8
python superglue_fusion/final_eval.py --tag fusion --abs cnx_s_d8
