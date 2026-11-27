#!/usr/bin/env python3
import datetime
import hashlib
import json
import os
import re
import time
import traceback
import urllib.request
from pathlib import Path

BASE = Path('/root/pairUAV_img/hf_upload_staging')
PAYLOAD = BASE / 'payload'
REPORTS = BASE / 'reports'
REPO = 'YaxuanLi/pairUAV-5D-temp'
GROUP = os.environ['HF_UPLOAD_GROUP']  # e.g. 0640_0644
assert re.fullmatch(r'\d{4}_\d{4}', GROUP), GROUP
os.environ.setdefault('HF_HOME', str(BASE / 'hub_cache'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '0')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_url
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()


def emit(event, **fields):
    row = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': event, **fields}
    print(json.dumps(row, ensure_ascii=False), flush=True)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def metadata(file):
    return {'size': file.size, 'blob_id': file.blob_id,
            'lfs_sha256': file.lfs.sha256 if file.lfs else None}


def matches(remote, local):
    if remote['size'] != local['size']:
        return False
    if remote['lfs_sha256']:
        return remote['lfs_sha256'] == local['sha256']
    return remote['blob_id'] == local['git_blob_sha1']


def retry(function, *args, **kwargs):
    for attempt in range(10):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            response = getattr(exc, 'response', None)
            if getattr(response, 'status_code', None) != 429 or attempt == 9:
                raise
            match = re.search(r'Retry after (\d+) seconds', str(exc))
            header = response.headers.get('retry-after', '') if response else ''
            delay = int(header) if header.isdigit() else (int(match.group(1)) + 3 if match else 60)
            emit('rate_limit_wait', operation=getattr(function, '__name__', 'request'), seconds=delay)
            while delay:
                step = min(delay, 30)
                time.sleep(step)
                delay -= step


def exact_paths(api, revision, paths):
    found = {}
    for offset in range(0, len(paths), 500):
        batch = paths[offset:offset + 500]
        for file in retry(api.get_paths_info, REPO, batch, revision=revision, repo_type='dataset'):
            if file.path in batch:
                found[file.path] = metadata(file)
    return found


def source_snapshot(paths):
    snapshot = {}
    for raw in paths:
        path = Path(raw)
        stat = path.stat()
        assert path.is_file() and not path.is_symlink(), path
        snapshot[raw] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return snapshot


def verify_snapshot(snapshot):
    for raw, old in snapshot.items():
        stat = Path(raw).stat()
        assert [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] == old, raw


def main(token):
    manifest = json.loads((REPORTS / GROUP / 'manifest.json').read_text())
    assert len(manifest) == 1505 and len({entry['path'] for entry in manifest}) == 1505
    paths = {entry['path'] for entry in manifest}
    assert all(re.fullmatch(r'(?:images/063[5-9]/\d{4}\.jpg|kml/063[5-9]/\d{4}\.kml|pairs/063[5-9]\.jsonl)', path) for path in paths)
    assert sum(path.startswith('images/') for path in paths) == 750
    assert sum(path.startswith('kml/') for path in paths) == 750
    assert sum(path.startswith('pairs/') for path in paths) == 5

    validation = json.loads((Path('/root/pairUAV_img/processed_paired_json_uniform_tilt') / f'group_{GROUP}_validation.json').read_text())
    assert validation['status'] == 'validated' and validation['full_image_hash_check'] is True
    assert validation['pairs'] == 5000 and len(validation['validated_artifact_sha256']) == 781
    for raw, expected in validation['validated_artifact_sha256'].items():
        assert sha_file(Path(raw)) == expected, raw

    source_paths = set(validation['validated_artifact_sha256'])
    for entry in manifest:
        path = PAYLOAD / entry['path']
        data_hash = sha_file(path)
        assert path.is_file() and path.stat().st_size == entry['size'] and data_hash == entry['sha256'], path
        if entry['path'].endswith(('.kml', '.jsonl')):
            data = path.read_bytes()
            entry['git_blob_sha1'] = git_sha(data)
        source_paths.add(str(path))
    pair_data = {}
    for tour in range(int(GROUP[:4]), int(GROUP[:4]) + 5):
        name = f'{tour:04d}'
        data = (PAYLOAD / f'pairs/{name}.jsonl').read_bytes()
        assert hashlib.sha256(data).hexdigest() == next(e['sha256'] for e in manifest if e['path'] == f'pairs/{name}.jsonl')
        rows = [json.loads(line) for line in data.splitlines()]
        assert len(rows) == 1000 and all(row['tour_id'] == name for row in rows)
        pair_data[name] = data

    snapshot = source_snapshot(source_paths)
    write_json(REPORTS / 'source_snapshot_upload.json', snapshot)
    api = HfApi(token=token)
    current = retry(api.repo_info, REPO, repo_type='dataset', files_metadata=False)
    before = exact_paths(api, current.sha, sorted(paths))
    emit('preflight_complete', revision=current.sha, files=len(manifest), existing=len(before))

    for entry in manifest:
        remote = before.get(entry['path'])
        if remote:
            assert matches(remote, entry), f'Conflicting existing path: {entry["path"]}'

    pending = [entry for entry in manifest if entry['path'] not in before]
    if pending:
        operations = [CommitOperationAdd(path_in_repo=entry['path'], path_or_fileobj=str(PAYLOAD / entry['path'])) for entry in pending]
        retry(api.preupload_lfs_files, REPO, additions=operations, repo_type='dataset', revision='main', num_threads=4)
        assert not any(operation._should_ignore for operation in operations)
        committed = False
        for attempt in range(12):
            head = retry(api.repo_info, REPO, repo_type='dataset', files_metadata=False)
            latest = exact_paths(api, head.sha, sorted(paths))
            for entry in manifest:
                remote = latest.get(entry['path'])
                if remote:
                    assert matches(remote, entry), f'Conflicting concurrent path: {entry["path"]}'
            still_missing = [entry for entry in manifest if entry['path'] not in latest]
            if not still_missing:
                committed = True
                break
            ops = operations if len(still_missing) == len(pending) else [CommitOperationAdd(path_in_repo=e['path'], path_or_fileobj=str(PAYLOAD / e['path'])) for e in still_missing]
            if ops is not operations:
                retry(api.preupload_lfs_files, REPO, additions=ops, repo_type='dataset', revision='main', num_threads=4)
            try:
                commit = retry(api.create_commit, REPO, operations=ops, repo_type='dataset', revision='main', parent_commit=head.sha, num_threads=4,
                               commit_message=f'Add validated tours {GROUP}',
                               commit_description='Add only final selected pairs and their referenced images and KML files. One JSONL file per tour. No files are deleted.')
                visible = exact_paths(api, commit.oid, sorted(paths))
                if all(entry['path'] in visible and matches(visible[entry['path']], entry) for entry in manifest):
                    write_json(REPORTS / 'commit.json', {'revision': commit.oid, 'url': commit.commit_url, 'files': len(ops)})
                    emit('group_committed', revision=commit.oid, files=len(ops))
                    committed = True
                    break
            except Exception as exc:
                latest_head = retry(api.repo_info, REPO, repo_type='dataset', files_metadata=False)
                if latest_head.sha == head.sha and attempt == 11:
                    raise
                emit('commit_retry', attempt=attempt + 1, reason=type(exc).__name__)
                time.sleep(5)
        assert committed, 'Could not confirm group commit'
    else:
        emit('group_already_present')

    final = retry(api.repo_info, REPO, repo_type='dataset', files_metadata=False)
    remote = exact_paths(api, final.sha, sorted(paths))
    assert len(remote) == 1505
    for entry in manifest:
        assert entry['path'] in remote and matches(remote[entry['path']], entry), entry['path']

    pair_checks = []
    for tour, data in pair_data.items():
        request = urllib.request.Request(hf_hub_url(REPO, f'pairs/{tour}.jsonl', repo_type='dataset', revision=final.sha), headers={'Authorization': f'Bearer {token}'})
        with urllib.request.urlopen(request, timeout=180) as response:
            remote_data = response.read()
        assert hashlib.sha256(remote_data).hexdigest() == hashlib.sha256(data).hexdigest(), tour
        pair_checks.append(tour)

    verify_snapshot(snapshot)
    result = {'status': 'uploaded_and_verified', 'repo': REPO, 'group': GROUP, 'revision': final.sha,
              'tours': 5, 'pairs': 5000, 'images': 750, 'kml': 750, 'pair_jsonl_files': 5,
              'files': len(manifest), 'remote_hash_verified_files': len(remote),
              'remote_pair_jsonl_verified': len(pair_checks), 'source_files_unchanged': len(snapshot),
              'deletions': 0}
    write_json(REPORTS / 'upload_result.json', result)
    emit('UPLOAD_COMPLETE', **result)


if __name__ == '__main__':
    token = Path(os.environ['HF_UPLOAD_TOKEN_FILE']).read_text().strip()
    try:
        main(token)
    except BaseException:
        print(traceback.format_exc().replace(token, '[REDACTED]'), flush=True)
        raise
