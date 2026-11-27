#!/usr/bin/env python3
"""Resumable five-tour balancing, with geometry preflight and verified artifacts."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
PYTHON = BASE / '.venv/bin/python'
EXPORTER = BASE.parent / 'gen_img/export_google_earth_pro.py'
THRESHOLDS = dict(min_inliers=20, min_inlier_ratio=.25, min_hull_fraction=.01, min_grid_cells=4)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic_json(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, default=str) + '\n')
    tmp.replace(p)


def load(p, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


_jpeg_validation_cache = {}


def valid_jpeg(path):
    # Hash bytes on every call; only reuse decoding when content is identical.
    # The bounded in-process cache never trusts timestamps or persisted state.
    try:
        import io
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in _jpeg_validation_cache:
            return _jpeg_validation_cache[digest]
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            valid = im.format == 'JPEG' and im.size == (3686, 2073)
        if len(_jpeg_validation_cache) >= 2048:
            _jpeg_validation_cache.clear()
        _jpeg_validation_cache[digest] = valid
        return valid
    except (OSError, ValueError, SyntaxError):
        return False


def sync_exports(staging, idir, kdir, quality):
    """Only adopt cropped, hash-verified exports from completed manifest entries."""
    receipt_path = idir / 'render_receipt.json'
    receipt = load(receipt_path, {'images': {}})
    verified_stems = set()
    for manifest in sorted((staging / 'runs').glob('*/manifest.json')):
        d = load(manifest, {})
        for e in d.get('images', []):
            stem = Path(e['source']).stem
            src, dest, kml = Path(e['output']), idir / (stem + '.jpg'), kdir / (stem + '.kml')
            if stem in verified_stems:
                continue
            if not kml.is_file() or e['source_sha256'] != sha(kml):
                continue
            prior = receipt['images'].get(stem, {})
            if valid_jpeg(dest) and prior.get('source_sha256') == sha(kml) and prior.get('sha256') == sha(dest):
                verified_stems.add(stem)
                continue
            if not valid_jpeg(src) or sha(src) != e['sha256']:
                continue
            tmp = dest.with_suffix('.tmp')
            with Image.open(src) as im:
                im.save(tmp, 'JPEG', quality=quality, optimize=True, subsampling=0)
            tmp.replace(dest)
            receipt['images'][stem] = {**e, 'export_sha256': e['sha256'], 'output': str(dest), 'sha256': sha(dest), 'jpeg_quality': quality}
            verified_stems.add(stem)
    atomic_json(receipt_path, receipt)
    return receipt


def verified_images(idir, kdir, count):
    receipt = load(idir / 'render_receipt.json', {'images': {}})['images']
    # Historical complete tours carry the image and KML hashes in their audit.
    return {i for i in range(count) if (e := receipt.get(f'{i:04d}'))
            and valid_jpeg(p := idir / f'{i:04d}.jpg')
            and e.get('sha256') == sha(p)
            and e.get('source_sha256') == sha(kdir / f'{i:04d}.kml')}


def audit_valid(audit, idir, kdir, count):
    s = load(audit / 'summary.json', {})
    if s.get('images') != count or s.get('views_sha256') != sha(kdir / 'views.json'):
        return False
    if any(s.get('thresholds', {}).get(k) != v for k, v in THRESHOLDS.items()):
        return False
    for i in range(count):
        p = idir / f'{i:04d}.jpg'
        if not valid_jpeg(p) or s.get('image_sha256', {}).get(p.name) != sha(p):
            return False
    try:
        matches = audit / 'matches.jsonl'
        if s.get('matches_sha256') and s['matches_sha256'] != sha(matches):
            return False
        design = load(kdir / 'views.json', {})
        candidates = {(e['a'], e['b']): e['delta'] for e in design['candidate_edges']}
        seen = set()
        accepted = 0
        with matches.open() as f:
            for line in f:
                row = json.loads(line)
                edge = (row['a'], row['b'])
                if edge in seen or edge not in candidates or row['delta'] != candidates[edge]:
                    return False
                if type(row['passed']) is not bool:
                    return False
                seen.add(edge)
                accepted += row['passed']
        return (len(seen) == s.get('candidate_pairs') == len(candidates)
                and accepted == s.get('accepted_pairs'))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def invoke(command, **kw):
    command = list(map(str, command))
    print('+', ' '.join(command), flush=True)
    timeout = kw.pop('timeout', None)
    # Own a process group so timeouts cannot leave OCR/dialog helpers behind.
    child = subprocess.Popen(command, start_new_session=True, **kw)
    def terminate():
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait()
    def interrupted(signum, frame):
        terminate()
        raise SystemExit(128 + signum)
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        code = child.wait(timeout=timeout)
        if code: raise subprocess.CalledProcessError(code, command)
    except BaseException:
        terminate()
        raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def recover_ui():
    invoke(['/usr/bin/python3', BASE.parent / 'gen_img/recover_google_earth.py'], timeout=120)


def audit_checkpoints(total):
    """Reserve unrendered slots for repair, including near the image limit."""
    return sorted({n for n in (50, 100, total-25, total-10) if 0 < n < total})


def render_count(missing, chunk, checkpoints, completed):
    next_checkpoint = min((n for n in checkpoints if n > completed), default=completed+chunk)
    limit = min(chunk, next_checkpoint-completed)
    count = 1
    while count < limit and missing[0]+count in missing:
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name, default in [('start', 1), ('end', 50), ('group-size', 5), ('seed', 49), ('images', 150), ('per-tour', 1000), ('bins', 5), ('jpeg-quality', 80), ('render-chunk', 10), ('render-retries', 4)]:
        ap.add_argument('--' + name, type=int, default=default)
    ap.add_argument('--min-free-gb', type=float, default=5)
    ap.add_argument('--input-root', type=Path, default=Path('/root/pairUAV_img/tours_splited_kml'))
    ap.add_argument('--kml-root', type=Path, default=Path('/root/pairUAV_img/processed_kml_relaxed_batch'))
    ap.add_argument('--image-root', type=Path, default=Path('/root/pairUAV_img/processed_images_relaxed_batch'))
    ap.add_argument('--json-root', type=Path, default=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'))
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    if a.group_size != 5 or a.bins < 5 or a.per_tour < 1000 or a.per_tour * 5 % a.bins or not 1 <= a.images <= 150 or (a.end-a.start+1) % 5:
        ap.error('require complete groups of five, >=5 bins, >=1000 pairs, divisible group quota, <=150 images')
    if a.render_chunk < 1 or a.render_retries < 1:
        ap.error('chunk and retries must be positive')
    for p in (a.kml_root, a.image_root, a.json_root):
        p.mkdir(parents=True, exist_ok=True)
    lock = (a.json_root / 'group_batch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = a.json_root / 'group_batch_manifest.json'
    state = load(manifest, {'groups': []})
    groups = {r['group_id']: r for r in state['groups']}

    def status(gid, tours, stage, **extra):
        groups[gid] = {'group_id': gid, 'tours': tours, 'status': stage, 'updated_at': time.time(), **extra}
        atomic_json(manifest, {'configuration': vars(a), 'groups': sorted(groups.values(), key=lambda r:r['group_id'])})
        print(gid, stage, extra, flush=True)

    def disk_guard():
        free = shutil.disk_usage(a.json_root).free / 1e9
        if free < a.min_free_gb:
            raise RuntimeError(f'Low disk: {free:.2f} GB free; stopped before further rendering')

    env = {**os.environ, 'DISPLAY': ':1'}
    for first in range(a.start, a.end+1, 5):
        tours = [f'{i:04d}' for i in range(first, first+5)]
        gid = f'{tours[0]}_{tours[-1]}'
        report = load(a.json_root / f'group_{gid}.json', {})
        if (report.get('status') == 'complete' and report.get('bins') == a.bins
                and all(report.get('per_tour', {}).get(t, 0) >= a.per_tour for t in tours)):
            invoke([PYTHON, BASE/'validate_group.py', '--group', gid, '--json-root', a.json_root, '--kml-root', a.kml_root, '--image-root', a.image_root])
            status(gid, tours, 'complete', reused=True)
            continue
        try:
            disk_guard()
            status(gid, tours, 'preflight')
            pre = a.json_root / f'group_{gid}_preflight.json'
            invoke([PYTHON, BASE/'preflight_group.py', '--tours', *tours, '--seed', a.seed, '--images', a.images, '--per-tour', a.per_tour, '--bins', a.bins, '--input-root', a.input_root, '--kml-root', a.kml_root, '--output', pre, '--image-root', a.image_root])
            if not load(pre, {}).get('geometry_feasible'):
                raise RuntimeError('Geometry preflight infeasible: no images rendered for this group')
            if a.preflight_only:
                status(gid, tours, 'geometry_feasible')
                continue
            from group_execution import execute_group
            execute_group(a, tours, gid, status, disk_guard, env)
            status(gid, tours, 'selecting_pairs')
            invoke([PYTHON, BASE/'select_group_visual_pairs.py', '--tours', *tours, '--per-tour', a.per_tour, '--bins', a.bins, '--tilt-limit', 45, '--kml-root', a.kml_root, '--audit-root', a.json_root, '--image-root', a.image_root, '--output-root', a.json_root, '--time-limit', 600])
            invoke([PYTHON, BASE/'validate_group.py', '--group', gid, '--json-root', a.json_root, '--kml-root', a.kml_root, '--image-root', a.image_root])
            status(gid, tours, 'complete')
        except Exception as exc:
            status(gid, tours, 'needs_attention', error=repr(exc))
            # Do not quietly skip failed groups and consume another 750 renders.
            raise


if __name__ == '__main__':
    main()
