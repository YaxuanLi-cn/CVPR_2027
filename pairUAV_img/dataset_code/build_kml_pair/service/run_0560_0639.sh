#!/bin/bash
set -euo pipefail
export DISPLAY=:1
export PYTHONUNBUFFERED=1
cd /root/pairUAV_img/dataset_code/build_kml_pair
exec .venv/bin/python build_uniform_range.py --start 560 --end 639
