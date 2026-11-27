"""Offline interrupted/tampered cache regression checks."""
import json,tempfile
from pathlib import Path
from common import sha
import audit_visual_pairs as visual
from partial_visual_audit import read_cache

with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);views=p/'views.json';images=[p/'0.jpg',p/'1.jpg']
    for im in images:im.write_bytes(b'image bytes')
    edge={'a':0,'b':1,'delta':{'x_m':1},'passed':True}
    views.write_text(json.dumps({'candidate_edges':[edge]}))
    matches=p/'partial_matches.jsonl';matches.write_text(json.dumps(edge)+'\n')
    meta=dict(schema_version=visual.VERSION,thresholds=visual.THRESHOLDS,views_sha256=sha(views),
        image_sha256={im.name:sha(im) for im in images},checked_edges=1,accepted_edges=1,matches_sha256=sha(matches))
    def write_meta(): (p/'partial_summary.json').write_text(json.dumps(meta))
    write_meta();assert len(read_cache(p,views,images,visual.THRESHOLDS))==1
    matches.write_text(json.dumps({**edge,'passed':False})+'\n')
    assert not read_cache(p,views,images,visual.THRESHOLDS)
    matches.write_text(json.dumps(edge)+'\n{')
    assert not read_cache(p,views,images,visual.THRESHOLDS)
    matches.write_text(json.dumps(edge)+'\n');images[0].write_bytes(b'changed')
    assert not read_cache(p,views,images,visual.THRESHOLDS)
    images[0].write_bytes(b'image bytes');meta.pop('matches_sha256');write_meta()
    assert len(read_cache(p,views,images,visual.THRESHOLDS))==1
    matches.write_text((json.dumps(edge)+'\n')*2)
    assert not read_cache(p,views,images,visual.THRESHOLDS)
print('PASS: intact legacy/current caches; tamper, truncation, changed image and duplicate rejection')
