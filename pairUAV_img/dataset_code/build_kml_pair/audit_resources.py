"""Temporarily limit idle Earth CPU use; recover even after audit SIGKILL.

Every caller of recover_affinity must hold gen_img/export.lock. A journal binds
process/thread IDs to Linux start times, so a reused PID is never modified.
"""
import fcntl
import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOCK_PATH = BASE.parent / 'gen_img/export.lock'
JOURNAL = BASE / '.audit_affinity.json'
EARTH_COMM = 'googleearth-bin'


def start_time(path):
    return (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def recover_affinity():
    """Restore the journal under the export lock; retain it on restoration error."""
    if not JOURNAL.exists():
        return
    state = json.loads(JOURNAL.read_text())
    for row in state['processes']:
        proc = Path('/proc') / str(row['pid'])
        try:
            if start_time(proc) != row['start_time']:
                continue
            for task in (proc / 'task').iterdir():
                try:
                    old = row['threads'].get(task.name)
                    if old and old['start_time'] == start_time(task):
                        affinity = old['affinity']
                    else:
                        # Threads created during the audit may inherit its cap.
                        current = os.sched_getaffinity(int(task.name))
                        if not current <= set(row['restricted_cpus']):
                            continue
                        affinity = row['default_affinity']
                    os.sched_setaffinity(int(task.name), set(affinity))
                except (FileNotFoundError, ProcessLookupError):
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
    JOURNAL.unlink()
    print('Visual audit: restored Earth CPU affinity', flush=True)


@contextmanager
def audit_resources():
    lock = LOCK_PATH.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    handlers = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        recover_affinity()
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        processes = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                if (proc / 'comm').read_text().strip() != EARTH_COMM:
                    continue
                default = sorted(os.sched_getaffinity(int(proc.name)))
                row = dict(pid=int(proc.name), start_time=start_time(proc),
                           default_affinity=default, restricted_cpus=default[:2], threads={})
                for task in (proc / 'task').iterdir():
                    try:
                        row['threads'][task.name] = dict(start_time=start_time(task),
                            affinity=sorted(os.sched_getaffinity(int(task.name))))
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                processes.append(row)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if processes:
            temp = JOURNAL.with_suffix('.tmp')
            with temp.open('w') as f:
                json.dump({'processes': processes}, f)
                f.flush()
                os.fsync(f.fileno())
            temp.replace(JOURNAL)
            for row in processes:
                for tid, old in row['threads'].items():
                    try:
                        task = Path('/proc') / str(row['pid']) / 'task' / tid
                        if start_time(task) == old['start_time']:
                            allowed = set(old['affinity']) & set(row['restricted_cpus'])
                            os.sched_setaffinity(int(tid), allowed or set(old['affinity'][:2]))
                    except (FileNotFoundError, ProcessLookupError):
                        continue
            print(f'Visual audit: limited {sum(len(r["threads"]) for r in processes)} idle Earth threads', flush=True)
        yield
    finally:
        try:
            recover_affinity()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            lock.close()
