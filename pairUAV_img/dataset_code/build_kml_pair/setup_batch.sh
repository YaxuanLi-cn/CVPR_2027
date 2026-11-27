#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if ! command -v google-earth-pro >/dev/null; then
  bash ../gen_img/install_google_earth.sh
fi
apt-get update
apt-get install -y -o Dpkg::Options::=--force-confold python3-venv python3-pil x11-utils util-linux
if [[ ! -x .venv/bin/python ]]; then /usr/bin/python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements-batch.txt
bash ../gen_img/setup_google_earth_runtime.sh
install -m 755 service/run_earth_telemetry.sh /opt/supervisor-scripts/earth-camera-telemetry.sh
cp service/earth-camera-telemetry.conf service/kml-group-batch.conf /etc/supervisor/conf.d/
supervisorctl reread
supervisorctl update
supervisorctl status earth-camera-telemetry
# Start is explicit so installing cannot unexpectedly launch a dataset job.
printf '%s\n' 'Ready: supervisorctl start kml-group-batch'
