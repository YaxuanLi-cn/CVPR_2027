#!/usr/bin/env bash
# Result-set pipeline with the zoom pass: wait for zoom matching, extract zoom geometry, fuse, evaluate.
set -euo pipefail
source /venv/main/bin/activate; cd /root/rasr-pairuav; SGD=outputs/pairuav5d_60k/superglue
while pgrep -f "run_superglue_zoom.py --geom" > /dev/null; do sleep 10; done
echo "zoom pass finished: $(ls $SGD/matches_zoom | wc -l) files at $(date)"
GEOM_PROCS=24 python superglue_fusion/extract_geom_zoom.py $SGD/matches_zoom $SGD/geom_zoom_half.pkl
echo "zoom geometry done at $(date)"
python superglue_fusion/stage1_fusion.py --geom $SGD/geom_half_k2.pkl --map $SGD/map_half_k2.pkl --zoom $SGD/geom_zoom_half.pkl --partial --tag half_zoom --jobs 4
echo "fusion done at $(date)"
python superglue_fusion/final_eval.py --tag half_zoom --abs cnx_s_d8 --out $SGD/final_metrics_half_zoom.json
echo "ALL DONE at $(date)"
