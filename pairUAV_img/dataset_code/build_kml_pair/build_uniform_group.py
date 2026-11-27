#!/usr/bin/env python3
"""Resume five-tour groups with uniform absolute image tilts and visual pairing."""
import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from group_batch_pipeline import BASE, PYTHON, invoke, load, atomic_json
from build_single_tour import design_matches
from image_tilt import validate_image_tilt
from group_execution import execute_group
from publish_group import publish


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tours',nargs=5,required=True)
    p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml_uniform_tilt'))
    p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images_uniform_tilt'))
    p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json_uniform_tilt'))
    p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'))
    p.add_argument('--seed',type=int,default=49)
    a=p.parse_args();tours=[f'{int(t):04d}' for t in a.tours]
    if tours != [f'{i:04d}' for i in range(int(tours[0]),int(tours[0])+5)]:
        p.error('Require five distinct consecutive tours in order')
    gid=f'{tours[0]}_{tours[-1]}'
    a.json_root.mkdir(parents=True,exist_ok=True)
    lock=(a.json_root/'uniform_group.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(group,ts,stage,**extra):
        d={'group_id':group,'tours':ts,'status':stage,'updated_at':time.time(),**extra}
        atomic_json(a.json_root/f'group_{group}_progress.json',d)
        print(group,stage,extra,flush=True)
    def disk_guard():
        if shutil.disk_usage(a.json_root).free<5e9:raise RuntimeError('Less than 5GB free')
    try:
        for t in tours:
            disk_guard();vp=a.kml_root/t/'views.json'
            if not vp.exists():
                status(gid,tours,'design_and_geometry_preflight',tour=t)
                invoke([PYTHON,BASE/'design_views.py','--tour',t,'--seed',a.seed+int(t),
                    '--images',150,'--uniform-image-tilt','--max-camera-range',256,
                    '--camera-height',220,'--position-radius',15,'--tilt-min',0,'--tilt-max',45,
                    '--paired-tilt-groups','--tilt-group-count',15,'--edge-fraction',.8,'--hfov',90,
                    '--input-root',a.input_root,'--kml-root',a.kml_root])
            design=load(vp,{})
            if not design_matches(design,t,a.seed,150,256,220,15):
                raise ValueError(f'{t}: incompatible design; refusing to overwrite')
            if not design.get('image_tilt_constraint'):raise ValueError(f'{t}: no uniform image tilt constraint')
            validate_image_tilt(design)
        pre=a.json_root/f'group_{gid}_preflight.json'
        status(gid,tours,'group_geometry_preflight')
        invoke([PYTHON,BASE/'preflight_group.py','--tours',*tours,'--seed',a.seed,
            '--images',150,'--per-tour',1000,'--bins',5,'--input-root',a.input_root,
            '--kml-root',a.kml_root,'--image-root',a.image_root,'--output',pre])
        if not load(pre,{}).get('geometry_feasible'):raise RuntimeError('Group geometry not feasible; no new images rendered')
        args=SimpleNamespace(kml_root=a.kml_root,image_root=a.image_root,json_root=a.json_root,
            input_root=a.input_root,images=150,bins=5,per_tour=1000,seed=a.seed,
            jpeg_quality=80,render_chunk=10,render_retries=4)
        execute_group(args,tours,gid,status,disk_guard,{**os.environ,'DISPLAY':':1'})
        status(gid,tours,'selecting_pairs')
        invoke([PYTHON,BASE/'select_group_visual_pairs.py','--tours',*tours,'--per-tour',1000,
            '--bins',5,'--tilt-limit',45,'--kml-root',a.kml_root,'--audit-root',a.json_root,
            '--image-root',a.image_root,'--output-root',a.json_root,'--time-limit',600])
        status(gid,tours,'validating')
        invoke([PYTHON,BASE/'validate_group.py','--group',gid,'--json-root',a.json_root,
            '--kml-root',a.kml_root,'--image-root',a.image_root])
        for t in tours:
            invoke([PYTHON,BASE/'plot_image_tilt.py','--views',a.kml_root/t/'views.json',
                '--output',a.json_root/f'{t}_absolute_image_tilt.png'])
            invoke([PYTHON,BASE/'tilt_contact_sheet.py','--views',a.kml_root/t/'views.json',
                '--images',a.image_root/f'{t}_redesigned','--output',a.json_root/f'{t}_tilt_examples.jpg'])
        for bins in (10,20):
            invoke([PYTHON,BASE/'group_fine_distributions.py','--group',gid,
                '--json-root',a.json_root,'--bins',bins])
        invoke([PYTHON,BASE/'compare_candidate_selected.py','--group',gid,'--json-root',a.json_root])
        invoke([PYTHON,BASE/'plot_group_acceptance.py','--group',gid,'--json-root',a.json_root,'--kml-root',a.kml_root])
        publish(a.json_root,a.kml_root,a.image_root,gid,replace=True)
        status(gid,tours,'complete',pairs=5000)
    except Exception as exc:
        status(gid,tours,'needs_attention',error=repr(exc));raise

if __name__=='__main__':main()
