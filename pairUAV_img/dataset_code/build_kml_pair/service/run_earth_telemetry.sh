#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
exec /usr/bin/python3 /root/pairUAV_img/dataset_code/build_kml_pair/telemetry_server.py
