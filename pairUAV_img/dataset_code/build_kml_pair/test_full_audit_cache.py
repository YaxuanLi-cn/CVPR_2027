"""Offline full-audit rejection checks using disposable JPEGs and records."""
import json,tempfile
from pathlib import Path
from PIL import Image
import group_batch_pipeline as g

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);kd=root/'k';ad=root/'a';im=root/'i'
    for p in (kd,ad,im):p.mkdir()
    edge=dict(a=0,b=1,delta={'x_m':1},passed=True)
    g.atomic_json(kd/'views.json',{'candidate_edges':[edge]})
    for i in range(2):Image.new('RGB',(3686,2073),(30+i,50,70)).save(im/f'{i:04d}.jpg')
    matches=ad/'matches.jsonl';matches.write_text(json.dumps(edge)+'\n')
    summary=dict(images=2,views_sha256=g.sha(kd/'views.json'),thresholds=g.THRESHOLDS,
                 image_sha256={p.name:g.sha(p) for p in im.glob('*.jpg')},candidate_pairs=1,accepted_pairs=1,
                 matches_sha256=g.sha(matches))
    g.atomic_json(ad/'summary.json',summary)
    assert g.audit_valid(ad,im,kd,2)
    matches.write_text(json.dumps({**edge,'delta':{'x_m':2}})+'\n')
    assert not g.audit_valid(ad,im,kd,2)
    summary.pop('matches_sha256');g.atomic_json(ad/'summary.json',summary)
    assert not g.audit_valid(ad,im,kd,2)
    matches.write_text((json.dumps(edge)+'\n')*2);summary['candidate_pairs']=2
    g.atomic_json(ad/'summary.json',summary);assert not g.audit_valid(ad,im,kd,2)
print('PASS: full-audit cache rejects modified geometry, hash mismatch and duplicate edges')
