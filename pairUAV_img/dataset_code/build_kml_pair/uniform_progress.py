#!/usr/bin/env python3
"""Read-only progress from manifests; use validation reports for acceptance."""
import json
from pathlib import Path
r=Path('/root/pairUAV_img/processed_paired_json_uniform_tilt');ids=['0026','0027','0028','0029','0030']
state_path=r/'group_0026_0030_progress.json'
state=json.loads(state_path.read_text()) if state_path.exists() else {}
result={'status':state.get('status'),'active_tour':state.get('tour'),'stage_target':state.get('stage_target'),'tours':{}}
for t in ids:
    im=Path('/root/pairUAV_img/processed_images_uniform_tilt')/f'{t}_redesigned'
    ids_done=set()
    receipt=im/'render_receipt.json'
    if receipt.exists():ids_done.update(json.loads(receipt.read_text()).get('images',{}))
    for p in (im/'earth_pro_export/runs').glob('*/manifest.json'):
        try:ids_done.update(Path(e['source']).stem for e in json.loads(p.read_text()).get('images',[]))
        except (OSError,ValueError):pass
    info={'exports_recorded':len(ids_done)}
    for name in ('summary.json','partial_summary.json'):
        p=r/t/'visual_audit_redesigned'/name
        if p.exists():
            d=json.loads(p.read_text());info['audit_images']=d.get('images',d.get('verified_images'));info['visual_pass']=d.get('accepted_pairs',d.get('accepted_edges'));break
    result['tours'][t]=info
if state.get('error'):result['error']=state['error']
print(json.dumps(result,ensure_ascii=False))
