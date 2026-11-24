#!/bin/bash
# wait for orphaned first-wave jobs, mark them done, then start the queues with bounded concurrency
cd /root/image_processing_hw
source .venv/bin/activate
for pid in 553341 553343 559732; do while kill -0 $pid 2>/dev/null; do sleep 10; done; done
for t in gs_full gs_views32 gs_views16; do
  if [ "$(tail -n 1 runs/garden/$t.log)" = "DONE" ]; then echo ok > runs/garden/$t/DONE; fi
done
nohup python code/run_queue.py logs/jobs_gs.txt 2 > logs/queue_gs.log 2>&1 &
nohup python code/run_queue.py logs/jobs_nerf.txt 1 > logs/queue_nerf.log 2>&1 &
wait
