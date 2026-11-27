#!/usr/bin/env python3
"""Design, render, visually screen, and balance reproducible high-overlap pairs."""
from __future__ import annotations
import argparse,os,shutil,subprocess,sys
from pathlib import Path

def run(cmd,env=None):
 print('+',' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True,env=env)
def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--images',type=int,default=150);p.add_argument('--pairs',type=int,default=1200);p.add_argument('--bins',type=int,default=12);p.add_argument('--min-tilt-delta',type=float,default=45);p.add_argument('--settle',type=float,default=0);p.add_argument('--timeout',type=float,default=180);p.add_argument('--camera-height',type=float,default=500);p.add_argument('--hfov',type=float,default=90);p.add_argument('--position-radius',type=float,default=15);p.add_argument('--tilt-min',type=float,default=0);p.add_argument('--tilt-max',type=float,default=60);p.add_argument('--tilt-group-count',type=int,default=15);p.add_argument('--edge-fraction',type=float,default=.8);p.add_argument('--min-inliers',type=int,default=40);p.add_argument('--min-inlier-ratio',type=float,default=.45);p.add_argument('--min-hull-fraction',type=float,default=.04);p.add_argument('--min-grid-cells',type=int,default=6);p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'));p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml_redesigned'));p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images'));p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));p.add_argument('--jpeg-quality',type=int,default=95);a=p.parse_args();tour=a.tour.zfill(4);b=Path(__file__).resolve().parent;py=b/'.venv/bin/python';kdir=a.kml_root/tour;idir=a.image_root/f'{tour}_redesigned';staging=idir/'earth_pro_export';audit=a.json_root/tour/'visual_audit_redesigned'
 if kdir.exists() or idir.exists():raise RuntimeError(f'Refusing to overwrite existing redesign output. Choose a new root: {kdir} or {idir}')
 run([py,b/'design_views.py','--tour',tour,'--seed',a.seed,'--images',a.images,'--input-root',a.input_root,'--kml-root',a.kml_root,'--camera-height',a.camera_height,'--position-radius',a.position_radius,'--tilt-min',a.tilt_min,'--tilt-max',a.tilt_max,'--paired-tilt-groups','--tilt-group-count',a.tilt_group_count,'--edge-fraction',a.edge_fraction,'--hfov',a.hfov])
 staging.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(DISPLAY=':1',HOME='/root/pairUAV_img/earth_runtime',XDG_CONFIG_HOME='/root/pairUAV_img/earth_runtime/.config')
 run(['/usr/bin/python3','/root/pairUAV_img/dataset_code/gen_img/export_google_earth_pro.py','--input',kdir,'--output',staging,'--limit',a.images,'--settle',a.settle,'--timeout',a.timeout,'--hfov',a.hfov],env)
 for i in range(a.images):
  src=staging/f'tour{tour}_{i:04d}.jpg';dst=idir/f'{i:04d}.jpg';dst.parent.mkdir(parents=True,exist_ok=True)
  if not src.exists():raise FileNotFoundError(src)
  if a.jpeg_quality >= 95:
   shutil.copyfile(src,dst)
  else:
   from PIL import Image
   with Image.open(src) as im:
    im.save(dst,'JPEG',quality=a.jpeg_quality,optimize=True,subsampling=0)
 if a.jpeg_quality < 95 and staging.exists(): shutil.rmtree(staging)
 run([py,b/'audit_visual_pairs.py','--tour',tour,'--kml-root',a.kml_root,'--image-root',idir,'--image-pattern','{id}.jpg','--output-dir',audit,'--workers',8,'--min-inliers',a.min_inliers,'--min-inlier-ratio',a.min_inlier_ratio,'--min-hull-fraction',a.min_hull_fraction,'--min-grid-cells',a.min_grid_cells])
 if (audit/'features').exists(): shutil.rmtree(audit/'features')
 run([py,b/'select_visual_pairs.py','--tour',tour,'--count',a.pairs,'--bins',a.bins,'--minimum-tilt-limit',a.min_tilt_delta,'--step',1,'--kml-root',a.kml_root,'--image-dir',idir,'--json-root',a.json_root,'--audit-dir',audit,'--output-json',a.json_root/f'{tour}.json'])
 # Published JSON records use the separate redesigned image directory.
 run([py,b/'validate_pairs.py','--tour',tour,'--json-root',a.json_root,'--image-root',a.image_root,'--image-dir',idir])
 run([py,b/'plot_distributions.py','--tour',tour,'--json-root',a.json_root])
if __name__=='__main__':main()
