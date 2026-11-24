#!/bin/bash
# Block until at least $1 GB of GPU memory is free on two checks 15 s apart (the single RTX 5090 is shared by several tracks).
need=${1:-4}; sleep $((RANDOM % 7))
while true; do
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); sleep 15
  f2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$f1" -ge $((need * 1024)) ] && [ "$f2" -ge $((need * 1024)) ]; then exit 0; fi
  sleep 20
done
