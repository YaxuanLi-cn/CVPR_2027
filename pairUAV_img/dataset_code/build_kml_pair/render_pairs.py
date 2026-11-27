#!/usr/bin/env python3
"""Render generated KML views through Earth Pro and publish pair JSON after checks."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml'));p.add_argument('--output-root',type=Path,default=Path('/root/pairUAV_img/processed_images'));p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));p.add_argument('--settle',type=float,default=20);p.add_argument('--timeout',type=float,default=180);p.add_argument('--limit',type=int);a=p.parse_args();tour=str(a.tour).zfill(4);kroot=a.kml_root/tour;views=sorted(kroot.glob('[0-9][0-9][0-9][0-9].kml')); n=min(len(views),a.limit) if a.limit else len(views)
 if not 1<=n<=150:p.error(f'expected 1..150 generated KMLs, found {len(views)}')
 planned=json.loads((kroot/'views.json').read_text());out=a.output_root/tour;staging=out/'earth_pro_export';out.mkdir(parents=True,exist_ok=True);staging.mkdir(exist_ok=True)
 setup=subprocess.run(['bash','/root/pairUAV_img/dataset_code/gen_img/setup_google_earth_runtime.sh'],capture_output=True,text=True);(out/'setup.log').write_text(setup.stdout+setup.stderr)
 cmd=['/usr/bin/python3','/root/pairUAV_img/dataset_code/gen_img/export_google_earth_pro.py','--input',str(kroot),'--output',str(staging),'--limit',str(n),'--settle',str(a.settle),'--timeout',str(a.timeout)]
 if planned.get('hfov_deg'):cmd.extend(['--hfov',str(planned['hfov_deg'])])
 env=os.environ.copy();env.update(DISPLAY=':1',HOME='/root/pairUAV_img/earth_runtime',XDG_CONFIG_HOME='/root/pairUAV_img/earth_runtime/.config');log=out/'render.log'
 r=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write_text(r.stdout);print(r.stdout,end='')
 if r.returncode:raise SystemExit(r.returncode)
 rendered=[]
 for i in range(n):
  src=staging/f'tour{kroot.name}_{i:04d}.jpg'
  if not src.exists():raise RuntimeError(f'missing native exporter output: {src}')
  dst=out/f'{i:04d}.jpg';shutil.copyfile(src,dst);rendered.append(dst)
 # Publish only after every requested view passed the exporter.
 records=[]
 for pair in planned['planned_pairs']:
  if pair['a']>=n or pair['b']>=n:continue
  q=pair.copy();q.update(pair_id=f'{tour}_{len(records):06d}',tour_id=tour,kml_a=str(kroot/f'{pair["a"]:04d}.kml'),kml_b=str(kroot/f'{pair["b"]:04d}.kml'),image_a=str(out/f'{pair["a"]:04d}.jpg'),image_b=str(out/f'{pair["b"]:04d}.jpg'),anchor=planned['anchor'],seed=planned['seed_sha256']);records.append(q)
 final={'schema_version':'2.0','status':'complete','tour_id':tour,'seed_sha256':planned['seed_sha256'],'images_rendered':n,'pairs':len(records),'records':records,'distribution':planned['planned_histograms'],'views_manifest':str(kroot/'views.json'),'render_manifest':str(staging/'manifest.json'),'image_root':str(out),'note':'Pairs use predicted common first-key visibility; final visual inspection is still required for publication.'}
 a.json_root.mkdir(parents=True,exist_ok=True);(a.json_root/f'{tour}.json').write_text(json.dumps(final,indent=2));(a.json_root/f'{tour}.jsonl').write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in records)+'\n');print(json.dumps({'tour':tour,'images':n,'pairs':len(records),'output':str(out),'json':str(a.json_root/f'{tour}.json')},indent=2))
if __name__=='__main__':main()
