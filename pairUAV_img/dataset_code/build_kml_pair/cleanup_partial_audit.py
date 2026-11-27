"""Remove a partial cache only after proving full records contain it verbatim."""
import fcntl
import hashlib
import json
from pathlib import Path
from common import sha


def _prune(out):
    out=Path(out)
    if not (out/'partial_summary.json').exists():return 0
    with (out/'partial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        full=json.loads((out/'summary.json').read_text())
        partial=json.loads((out/'partial_summary.json').read_text())
        for key in ('schema_version','views_sha256','thresholds'):
            if full[key]!=partial[key]:return 0
        if any(full['image_sha256'].get(k)!=v for k,v in partial['image_sha256'].items()):return 0
        matches=out/'matches.jsonl';cached=out/'partial_matches.jsonl'
        if not full.get('matches_sha256') or sha(matches)!=full['matches_sha256']:return 0
        if partial.get('matches_sha256') and sha(cached)!=partial['matches_sha256']:return 0
        def fingerprint(row):
            return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        known={}
        for line in matches.open():
            row=json.loads(line);key=(row['a'],row['b'])
            if key in known:return 0
            known[key]=fingerprint(row)
        if len(known)!=full['candidate_pairs']:return 0
        seen=set()
        for line in cached.open():
            row=json.loads(line);key=(row['a'],row['b'])
            if key in seen or known.get(key)!=fingerprint(row):return 0
            seen.add(key)
        if len(seen)!=partial['checked_edges']:return 0
        size=cached.stat().st_size
        # Removing metadata first makes a mid-cleanup interruption a cache miss.
        (out/'partial_summary.json').unlink()
        cached.unlink()
        print(f'Pruned {size} duplicate partial-audit bytes; full evidence retained',flush=True)
        return size


def prune(out):
    """Optional cleanup must not invalidate an otherwise complete audit.

    Keep uncertain cache files; the caller still verifies the full audit.
    """
    try:
        return _prune(out)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f'Skipped partial-cache cleanup at {out}: {type(exc).__name__}: {exc}', flush=True)
        return 0
