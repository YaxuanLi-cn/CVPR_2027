#!/usr/bin/env python3
"""Publish processed pair JSON after all 150 unique Earth Pro images exist."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml'));p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images'));p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));a=p.parse_args();tour=str(a.tour).zfill(4);k=a.kml_root/tour;d=json.loads((k/'views.json').read_text());out=a.image_root/tour;staging=out/'earth_pro_export';files=[]
 for i in range(len(d['views'])):
  src=staging/f'tour{tour}_{i:04d}.jpg';dst=out/f'{i:04d}.jpg'
  if not src.exists():raise RuntimeError(f'missing {src}')
  shutil.copyfile(src,dst)
  with Image.open(dst) as im:
   if im.size!=(3686,2073):raise RuntimeError(f'{dst} wrong size {im.size}')
  files.append(dst)
 records=[]
 for i,pair in enumerate(d['planned_pairs']):
  q=dict(pair);q.update(pair_id=f'{tour}_{i:06d}',tour_id=tour,kml_a=str(k/f'{pair["a"]:04d}.kml'),kml_b=str(k/f'{pair["b"]:04d}.kml'),image_a=str(out/f'{pair["a"]:04d}.jpg'),image_b=str(out/f'{pair["b"]:04d}.jpg'),anchor=d['anchor'],seed=d['seed_sha256']);records.append(q)
 final={'schema_version':'2.0','status':'complete','tour_id':tour,'seed_sha256':d['seed_sha256'],'images_rendered':len(files),'pairs':len(records),'records':records,'distribution':d['planned_histograms'],'views_manifest':str(k/'views.json'),'image_root':str(out),'note':'Earth Pro native 3D images; pair visibility is geometry-screened and should be reviewed for publication.'}
 a.json_root.mkdir(parents=True,exist_ok=True);(a.json_root/f'{tour}.json').write_text(json.dumps(final,indent=2));(a.json_root/f'{tour}.jsonl').write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in records)+'\n');print(json.dumps({'tour':tour,'images':len(files),'pairs':len(records),'json':str(a.json_root/f'{tour}.json')},indent=2))
if __name__=='__main__':main()
