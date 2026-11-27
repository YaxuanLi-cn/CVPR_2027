"""Repair unrendered slots while preserving rendered KML and visual evidence."""
import copy
import json
import re
from pathlib import Path
from common import sha
from balance import candidates


def image_snapshot(image_root, tour, count):
    root = image_root / f'{tour}_redesigned'
    files = {}
    frozen = set()
    for p in root.rglob('*.jpg'):
        if p.name in ('preview_montage.jpg', 'failure.jpg'):
            continue
        files[str(p)] = sha(p)
        match = re.fullmatch(r'(?:tour\d+_)?(\d{4})(?:\.[a-f0-9]+)?', p.stem)
        if not match or int(match[1]) >= count:
            # Unknown image ownership cannot authorize changing a camera.
            frozen.update(range(count))
        else:
            frozen.add(int(match[1]))
    return frozen, files


def preserve_frozen(original, revised, frozen):
    revised = copy.deepcopy(revised)
    for i in frozen:
        revised['views'][i] = copy.deepcopy(original['views'][i])
    revised['candidate_edges'] = candidates(revised['views'])
    revised['capacity_repair']['frozen_view_indices'] = sorted(frozen)
    return revised


def rebound_summary(audit, old_dir, new_dir, image_dir, frozen):
    """Rebind only after verifying original cache and every cached endpoint."""
    import audit_visual_pairs as visual
    from partial_visual_audit import read_cache
    from group_batch_pipeline import THRESHOLDS
    thresholds = {**visual.THRESHOLDS, **THRESHOLDS}
    old = json.loads((old_dir/'views.json').read_text())
    new = json.loads((new_dir/'views.json').read_text())
    paths = [image_dir/(v['id']+'.jpg') for v in old['views']]
    records = read_cache(audit, old_dir/'views.json', paths, thresholds)
    meta = json.loads((audit/'partial_summary.json').read_text())
    if not records or len(records) != meta['checked_edges']:
        raise ValueError('Partial repair requires an intact, image-bound cache')
    for i in frozen:
        if old['views'][i] != new['views'][i]:
            raise ValueError('Frozen camera metadata changed')
        name=old['views'][i]['id']+'.kml'
        if sha(old_dir/name) != sha(new_dir/name):
            raise ValueError('Frozen KML bytes changed')
    edges = {(e['a'],e['b']):e['delta'] for e in new['candidate_edges']}
    for row in records.values():
        for i in (row['a'],row['b']):
            if i not in frozen or old['views'][i] != new['views'][i]:
                raise ValueError('Attempt to change an audited endpoint')
            name = old['views'][i]['id']+'.kml'
            if sha(old_dir/name) != sha(new_dir/name):
                raise ValueError('Rendered KML bytes changed')
        if edges.get((row['a'],row['b'])) != row['delta']:
            raise ValueError('Audited pair geometry changed')
    for name, digest in meta['image_sha256'].items():
        i = next(i for i,v in enumerate(old['views']) if v['id']+'.jpg' == name)
        if i not in frozen or old['views'][i] != new['views'][i] or sha(image_dir/name) != digest:
            raise ValueError('Audited image binding changed')
    result = copy.deepcopy(meta)
    result['views_sha256'] = sha(new_dir/'views.json')
    result['matches_sha256'] = sha(audit/'partial_matches.jsonl')
    result['geometry_rebinding'] = {
        'previous_views_sha256':sha(old_dir/'views.json'),
        'frozen_view_indices':sorted(frozen),
        'reason':'Only unrendered cameras revised; all cached endpoints unchanged'}
    return result
