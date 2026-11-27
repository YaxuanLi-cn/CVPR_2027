"""Integration against an existing 0012 partial audit; uses disposable copies only.
Run while that partial snapshot exists; full audit cleanup removes it.
"""
import sys,json,tempfile,shutil
from pathlib import Path
from types import SimpleNamespace
base=Path('/root/pairUAV_img/dataset_code/build_kml_pair');sys.path.insert(0,str(base))
import repair_capacity as r
from common import sha
from partial_visual_audit import read_cache
import audit_visual_pairs as visual
from group_batch_pipeline import THRESHOLDS
with tempfile.TemporaryDirectory(prefix='partial_repair_integration_') as tmp:
 root=Path(tmp);kd=root/'kml';im=root/'images';ad=root/'audit';tools=root/'tools';tools.mkdir();(root/'gen_img').mkdir()
 for name in ['.venv','preflight_group.py']:(tools/name).symlink_to(base/name)
 r.BASE=tools
 tours=['0011','0012','0013','0014','0015']
 for t in tours:shutil.copytree(Path('/root/pairUAV_img/processed_kml_relaxed_batch')/t,kd/t)
 ad.mkdir();im.mkdir()
 (im/'0011_redesigned').symlink_to(Path('/root/pairUAV_img/processed_images_relaxed_batch/0011_redesigned'))
 (ad/'0011').symlink_to(Path('/root/pairUAV_img/processed_paired_json_relaxed_batch/0011'))
 target=ad/'0012'/'visual_audit_redesigned';target.mkdir(parents=True)
 source=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch/0012/visual_audit_redesigned')
 for name in ['partial_summary.json','partial_matches.jsonl']:shutil.copy2(source/name,target/name)
 meta=json.loads((target/'partial_summary.json').read_text());images=im/'0012_redesigned';images.mkdir(parents=True)
 for name in meta['image_sha256']:(images/name).symlink_to(Path('/root/pairUAV_img/processed_images_relaxed_batch/0012_redesigned')/name)
 frozen={int(Path(n).stem) for n in meta['image_sha256']}
 before={i:sha(kd/'0012'/f'{i:04d}.kml') for i in frozen};matches=sha(target/'partial_matches.jsonl')
 a=SimpleNamespace(tours=tours,audited_tours=['0011'],partial_tours=['0012'],image_root=im,kml_root=kd,audit_root=ad,seed=49,images=150,per_tour=1000,bins=5,tilt_limit=45,max_pairs_per_image=150,input_root=Path('/root/pairUAV_img/tours_splited_kml'))
 result=r.repair(a,{'capacity_diagnostics':{'shortages':[]}})
 assert result and result['geometry_feasible']
 assert all(sha(kd/'0012'/f'{i:04d}.kml')==h for i,h in before.items())
 assert matches==sha(target/'partial_matches.jsonl')
 d=json.loads((kd/'0012'/'views.json').read_text());paths=[images/(v['id']+'.jpg') for v in d['views']]
 assert len(read_cache(target,kd/'0012'/'views.json',paths,{**visual.THRESHOLDS,**THRESHOLDS}))==meta['checked_edges']
 result['integration_checks']={'frozen_kml_count':len(frozen),'preserved_visual_records':meta['checked_edges'],'real_outputs_modified':False}
 (base/'diagnostics/partial_repair_integration_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('PASS partial repair integration',result['integration_checks'],flush=True)
