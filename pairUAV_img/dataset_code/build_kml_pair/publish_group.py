#!/usr/bin/env python3
"""Expose validated group artifacts at the requested roots without copying images."""
import argparse,json,hashlib,os,uuid,time
from pathlib import Path

def link(target, dest, replace=False):
 target=target.resolve(strict=True);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.is_symlink():
  if dest.resolve()==target:return
  if not replace:raise FileExistsError(f'Refusing to replace existing link: {dest}')
  temporary=dest.with_name('.'+dest.name+'.'+uuid.uuid4().hex)
  temporary.symlink_to(target,target_is_directory=target.is_dir())
  os.replace(temporary,dest)
  return
 if dest.exists():raise FileExistsError(f'Refusing to replace existing output: {dest}')
 dest.symlink_to(target,target_is_directory=target.is_dir())

def publish(root,kroot,iroot,gid,dest=Path('/root/pairUAV_img'),replace=False):
 report=json.loads((root/f'group_{gid}.json').read_text())
 validation=json.loads((root/f'group_{gid}_validation.json').read_text())
 if validation.get('status')!='validated' or validation['pairs']!=report['pairs']:raise ValueError('Group validation required')
 if not validation.get('full_image_hash_check') or not validation.get('validated_artifact_sha256'):raise ValueError('Full hash-bound validation required')
 for name,digest in validation['validated_artifact_sha256'].items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Artifact changed after validation: {name}')
 if replace:
  previous={}
  for t in report['tours']:
   for path in (dest/'processed_kml'/t,dest/'processed_images'/t,dest/'processed_paired_json'/f'{t}.json',dest/'processed_paired_json'/f'{t}.jsonl'):
    if path.is_symlink():previous[str(path)]=os.readlink(path)
    elif path.exists():raise FileExistsError(f'Refusing to replace real output directory/file: {path}')
  history=root/'publication_history';history.mkdir(exist_ok=True)
  (history/f'{time.time_ns()}.json').write_text(json.dumps(previous,indent=2)+'\n')
 for t in report['tours']:
  link(kroot/t,dest/'processed_kml'/t,replace)
  link(iroot/f'{t}_redesigned',dest/'processed_images'/t,replace)
  for suffix in ('.json','.jsonl'):
   link(root/f'{t}_group_{gid}{suffix}',dest/'processed_paired_json'/f'{t}{suffix}',replace)
 for suffix in ('.json','_validation.json','_distributions.png'):
  link(root/f'group_{gid}{suffix}',dest/'processed_paired_json'/f'group_{gid}{suffix}',replace)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args()
 publish(Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'),Path('/root/pairUAV_img/processed_kml_relaxed_batch'),Path('/root/pairUAV_img/processed_images_relaxed_batch'),a.group)
 print('Published validated group',a.group)
