#!/bin/bash
set -euo pipefail
ulimit -c 0
export DISPLAY=:1
export PYTHONUNBUFFERED=1
cd /root/pairUAV_img/dataset_code/build_kml_pair
exec .venv/bin/python group_batch_pipeline.py --start "${KML_BATCH_START:-1}" --end "${KML_BATCH_END:-50}" --render-chunk 10 --render-retries 4 "$@"
