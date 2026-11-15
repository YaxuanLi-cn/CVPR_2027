#!/usr/bin/env bash
set -euo pipefail
source /venv/main/bin/activate
ROOT=/root/rasr-pairuav
OUT=$ROOT/outputs/pairuav5d_60k
LOG=$OUT/features/features.log
mkdir -p "$OUT"
exec > >(tee -a "$OUT/full_fit.log") 2>&1
while true; do
  status="$(supervisorctl status pairuav5d_60k_features 2>/dev/null || true)"
  if [[ "$status" == *"RUNNING"* || "$status" == *"STARTING"* || "$status" == *"BACKOFF"* ]]; then
    sleep 30
    continue
  fi
  if [[ "$status" == *"EXITED"* ]] && grep -Fq '"output": "/root/rasr-pairuav/outputs/pairuav5d_60k/features/train.csv"' "$LOG" && grep -Fq '"output": "/root/rasr-pairuav/outputs/pairuav5d_60k/features/test.csv"' "$LOG"; then
    break
  fi
  echo "Feature job did not complete successfully: $status"
  exit 1
done
source /venv/main/bin/activate
exec "$ROOT/run_pairuav5d_60k_fit.sh"
