"""Evidence-preserving cleanup: matching subset deletes, discrepancies retain."""
import tempfile,json
from pathlib import Path
from common import sha,save_json
from cleanup_partial_audit import prune
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);full=p/'matches.jsonl';part=p/'partial_matches.jsonl'
 row={'a':0,'b':1,'passed':True,'points_a':[[1.0,2.0]]}
 full.write_text(json.dumps(row)+'\n');part.write_text(json.dumps(row)+'\n')
 common=dict(schema_version='test',views_sha256='geometry',thresholds={},image_sha256={'0.jpg':'a','1.jpg':'b'})
 save_json(p/'summary.json',{**common,'matches_sha256':sha(full),'candidate_pairs':1})
 save_json(p/'partial_summary.json',{**common,'matches_sha256':sha(part),'checked_edges':1})
 original=full.read_bytes();size=part.stat().st_size
 assert prune(p)==size and full.read_bytes()==original and not part.exists()
 part.write_text(json.dumps({**row,'passed':False})+'\n')
 save_json(p/'partial_summary.json',{**common,'matches_sha256':sha(part),'checked_edges':1})
 assert prune(p)==0 and part.exists()
print('PASS: cleanup retains full evidence; conflicting partial records are not deleted')

with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp);full=p/'matches.jsonl';part=p/'partial_matches.jsonl'
 full.write_text('full evidence retained\n');part.write_text('partial evidence retained\n')
 before=(full.read_bytes(),part.read_bytes())
 for bad in ('{', 'null', '{}', '[]'):
  (p/'partial_summary.json').write_text(bad)
  (p/'summary.json').write_text('{}')
  assert prune(p)==0
  assert (full.read_bytes(),part.read_bytes())==before
 (p/'summary.json').unlink()
 assert prune(p)==0 and (full.read_bytes(),part.read_bytes())==before
print('PASS: malformed or missing cleanup metadata cannot abort audit or delete evidence')
