#!/usr/bin/env python3
"""Process an explicit range of complete five-tour groups, resuming durable outputs."""
import argparse,fcntl,json,os,time
from pathlib import Path
from group_batch_pipeline import BASE,PYTHON,invoke,atomic_json,load

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True)
    p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml_uniform_tilt'))
    p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images_uniform_tilt'))
    p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json_uniform_tilt'))
    p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'))
    a=p.parse_args()
    if a.start<0 or a.end<a.start or (a.end-a.start+1)%5:p.error('Range must contain complete groups of five')
    for i in range(a.start,a.end+1):
        if not (a.input_root/f'{i:04d}'/'0000.kml').is_file():raise FileNotFoundError(f'Missing source {i:04d}')
    a.json_root.mkdir(parents=True,exist_ok=True)
    lock=(a.json_root/'uniform_range.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=a.json_root/f'range_{a.start:04d}_{a.end:04d}_progress.json'
    state=load(manifest,{})
    state.update(start=a.start,end=a.end,total_tours=a.end-a.start+1,pid=os.getpid(),status='running')
    state.setdefault('groups',{})
    def save():state['updated_at']=time.time();atomic_json(manifest,state)
    for start in range(a.start,a.end+1,5):
        tours=[f'{i:04d}' for i in range(start,start+5)];gid=f'{tours[0]}_{tours[-1]}'
        state.update(active_group=gid,status='running');state['groups'][gid]={'status':'running'};save()
        try:
            invoke([PYTHON,BASE/'build_uniform_group.py','--tours',*tours,
                '--input-root',a.input_root,'--kml-root',a.kml_root,
                '--image-root',a.image_root,'--json-root',a.json_root])
            v=load(a.json_root/f'group_{gid}_validation.json',{})
            if v.get('status')!='validated' or v.get('pairs')!=5000 or set(v.get('per_tour',{}))!=set(tours):
                raise RuntimeError(f'Group {gid} lacks completed validation')
            state['groups'][gid]={'status':'complete','pairs':5000,'tours':tours,'validated_at':time.time()}
            save()
        except Exception as exc:
            state.update(status='needs_attention',error=repr(exc));state['groups'][gid]={'status':'needs_attention','error':repr(exc)};save();raise
    state.update(status='complete',active_group=None,total_pairs=(a.end-a.start+1)*1000);save()
    print(json.dumps(state,indent=2),flush=True)
if __name__=='__main__':main()
