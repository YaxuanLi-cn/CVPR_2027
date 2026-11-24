"""Tiny GPU job queue: run shell commands from a file with a fixed number of parallel workers.
Each line: <done_marker_path>\t<command>. Jobs whose marker exists are skipped."""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
jobs = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
workers = int(sys.argv[2])
def run(j):
    marker, cmd = j
    if os.path.exists(marker):
        return
    t = time.time()
    log = marker.replace("/DONE", "") + ".log"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    for attempt in range(4):
        r = subprocess.run(cmd, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
        if r.returncode == 0 or "OutOfMemory" not in open(log).read():
            break
        time.sleep(120)  # GPU memory contention: wait and retry
    if r.returncode == 0:
        open(marker, "w").write("ok")
    print(f"{'OK  ' if r.returncode == 0 else 'FAIL'} {time.time()-t:6.0f}s {cmd}", flush=True)
with ThreadPoolExecutor(workers) as ex:
    list(ex.map(run, jobs))
print("QUEUE_DONE", flush=True)
