"""Offline range runner checks; does not touch GUI or production artifacts."""
import tempfile,json,sys
from pathlib import Path
from unittest.mock import patch
import build_uniform_range as runner


def run(failure=False):
 with tempfile.TemporaryDirectory() as tmp:
  r=Path(tmp);source=r/'source';out=r/'out';calls=[]
  for i in range(160,240):
   p=source/f'{i:04d}';p.mkdir(parents=True);(p/'0000.kml').write_text('<kml/>')
  def invoke(cmd):
   tours=cmd[cmd.index('--tours')+1:cmd.index('--input-root')]
   assert len(tours)==5
   calls.append(tours)
   if failure and tours[0]=='0165':raise RuntimeError('simulated render failure')
   gid=f'{tours[0]}_{tours[-1]}'
   (out/f'group_{gid}_validation.json').write_text(json.dumps({'status':'validated','pairs':5000,'per_tour':{t:1000 for t in tours}}))
  argv=['build_uniform_range.py','--start','160','--end','239','--input-root',str(source),'--json-root',str(out),'--image-root',str(r/'images'),'--kml-root',str(r/'kml')]
  with patch.object(sys,'argv',argv),patch.object(runner,'invoke',invoke):
   if failure:
    try:runner.main()
    except RuntimeError:pass
    else:raise AssertionError('Failure was skipped')
   else:runner.main()
  state=json.loads((out/'range_0160_0239_progress.json').read_text())
  if failure:
   assert len(calls)==2 and state['status']=='needs_attention'
   assert state['groups']['0160_0164']['status']=='complete'
  else:
   assert len(calls)==16 and [t for g in calls for t in g]==[f'{i:04d}' for i in range(160,240)]
   assert state['total_pairs']==80000 and state['status']=='complete'
run();run(True)
print('PASS: exact authorized range, complete group checks, fail-stop with completed evidence retained')
