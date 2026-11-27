#!/bin/bash
set -euo pipefail
cd /root/pairUAV_img/dataset_code/build_kml_pair
export PYTHONUNBUFFERED=1
export DISPLAY=:1
exec .venv/bin/python build_single_tour.py --tour 0026 --uniform-image-tilt --publish-standard --kml-root /root/pairUAV_img/processed_kml_uniform_tilt --image-root /root/pairUAV_img/processed_images_uniform_tilt --json-root /root/pairUAV_img/processed_paired_json_uniform_tilt
