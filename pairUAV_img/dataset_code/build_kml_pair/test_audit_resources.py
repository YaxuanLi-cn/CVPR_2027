"""Exercise recovery with a disposable real process, never the Earth GUI."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import audit_resources as r


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r.JOURNAL = root / 'journal.json'
        r.LOCK_PATH = root / 'lock'
        r.EARTH_COMM = 'audit-test'
        child = subprocess.Popen([sys.executable, '-c', '''
import ctypes,time,threading
ctypes.CDLL(None).prctl(15,b'audit-test',0,0,0)
while True: time.sleep(.1)
'''])
        try:
            proc = Path('/proc') / str(child.pid)
            for _ in range(100):
                if (proc/'comm').read_text().strip() == r.EARTH_COMM: break
                time.sleep(.01)
            original = os.sched_getaffinity(child.pid)
            try:
                with r.audit_resources():
                    assert len(os.sched_getaffinity(child.pid)) <= 2
                    assert r.JOURNAL.exists()
                    raise RuntimeError('test exception')
            except RuntimeError: pass
            assert os.sched_getaffinity(child.pid) == original
            assert not r.JOURNAL.exists()
            worker = subprocess.Popen([sys.executable, '-c', f'''
import pathlib,time,audit_resources as r
r.JOURNAL=pathlib.Path({str(r.JOURNAL)!r})
r.LOCK_PATH=pathlib.Path({str(r.LOCK_PATH)!r})
r.EARTH_COMM='audit-test'
with r.audit_resources():
 pathlib.Path({str(root/'ready')!r}).touch()
 time.sleep(30)
'''], cwd=Path(__file__).resolve().parent)
            try:
                for _ in range(300):
                    if (root/'ready').exists():break
                    time.sleep(.01)
                assert (root/'ready').exists()
                worker.kill(); worker.wait()
                assert len(os.sched_getaffinity(child.pid)) <= 2
                with r.LOCK_PATH.open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    r.recover_affinity()
                assert os.sched_getaffinity(child.pid) == original
                assert not r.JOURNAL.exists()
                # A stale PID identity must never change the current process.
                r.JOURNAL.write_text(json.dumps({'processes':[dict(pid=child.pid,
                    start_time='invalid',threads={})]}))
                r.recover_affinity()
                assert os.sched_getaffinity(child.pid) == original
            finally:
                if worker.poll() is None: worker.kill(); worker.wait()
        finally:
            child.terminate(); child.wait()
    print('PASS: normal exception, SIGKILL recovery, and stale process identity')

if __name__ == '__main__': main()
