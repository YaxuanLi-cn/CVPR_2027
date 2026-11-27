#!/usr/bin/python3
"""Generate at most 150 views, then select >1000 distinct, balanced pairs."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from common import *
from balance import candidates,select

def lhs(rng,n,lo,hi):
 x=lo+(np.arange(n)+rng.random(n))*(hi-lo)/n;rng.shuffle(x);return x

def edge_positions(rng,n,radius):
 centers=np.array([-.9167,-.55,-.1833,.1833,.55,.9167])*radius
 counts=np.floor(np.array([.24,.13,.13,.13,.13,.24])*n).astype(int)
 for i in range(n-counts.sum()):counts[i%6]+=1
 x=np.concatenate([rng.uniform(c-.012*radius,c+.012*radius,k) for c,k in zip(centers,counts)])
 rng.shuffle(x);return x

def build(tour,seed,n,files,pair_count,bins):
 source=view_from_file(files[0])
 if source['type']!='LookAt':raise ValueError('First key must have LookAt')
 anchor={'longitude':source['longitude'],'latitude':source['latitude'],'altitude':source.get('altitude',0.0),'altitude_msl':source.get('altitude',0.0),'terrain_altitude_msl':None,'feature_key':'first-key','source_kml':str(files[0]),'source_sha256':sha(files[0]),'semantic_note':'all cameras point at the first-key LookAt building region'}
 digest=seed_for(tour,seed);target=pair_count//bins;chosen_attempt=None;frame=Frame(anchor['longitude'],anchor['latitude'],0.0)
 for attempt in range(100):
  rng=np.random.default_rng(int(digest[16:32],16)+attempt);pos=np.c_[rng.uniform(-10,10,n),rng.uniform(-10,10,n),rng.uniform(1,34,n)];views=[]
  for i,(x,y,z) in enumerate(pos):
   horizontal=float(np.hypot(x,y));heading=float(np.degrees(np.arctan2(-x,-y))%360);tilt=float(np.degrees(np.arctan2(horizontal,max(1e-6,z))));rngv=float(np.hypot(horizontal,z));lon,lat,alt=frame.to_geo([x,y,z])
   views.append({'id':f'{i:04d}','cluster':0,'longitude':float(lon),'latitude':float(lat),'altitude':float(z),'heading':heading,'tilt':tilt,'range':rngv,'roll':0.,'position_enu_m':[float(x),float(y),float(z)],'east':float(x),'north':float(y),'up':float(z),'anchor_pixel_native':[1920,1080],'anchor_pixel_crop':[1843,1080],'visible_grid':[True]})
  edges=candidates(views)
  if len(edges)<pair_count:continue
  mins=[]
  for k,(lo,hi) in AXES.items():
   counts=[0]*bins;axis=list(AXES).index(k)
   for e in edges:counts[bin_ids(e['delta'],bins)[axis]]+=1
   mins.append(min(counts))
  if min(mins)<target:continue
  chosen_attempt=(views,edges,attempt);break
 if chosen_attempt is None:raise RuntimeError('Could not find deterministic camera set with enough candidates in every requested bin')
 views,edges,attempt=chosen_attempt;chosen,counts,msg=select(edges,views,pair_count,bins,int(digest[32:40],16))
 return {'schema_version':'5.0','status':'planned','tour_id':tour,'seed':seed,'seed_sha256':digest,'sampling_attempt':attempt,'source_hashes':{str(p):sha(p) for p in files},'images':n,'requested_pairs':pair_count,'bins':bins,'hfov_deg':60.,'native_size':[3840,2160],'crop_box':crop_box(),'anchor':anchor,'views':views,'candidate_edges':edges,'planned_pairs':chosen,'planned_histograms':counts,'solver':msg,'coordinate_convention':{'translation':'B-A actual Camera position in fixed first-key ENU east/north/up meters','heading':'wrap(B-A) in [-180,180)','tilt':'B-A degrees','range':'camera-to-first-key distance','anchor_visibility':'each Camera orientation is analytically pointed at first-key target','pair_uniqueness':'unordered image pair, one chosen orientation'}}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tour',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--images',type=int,default=150);p.add_argument('--pairs',type=int,default=1200);p.add_argument('--bins',type=int,default=6);p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'));p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml'));p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));p.add_argument('--dry-run',action='store_true');a=p.parse_args();tour=a.tour.zfill(4)
 if not a.tour.isdecimal() or not 75<=a.images<=150 or a.pairs<=1000 or a.pairs%a.bins:p.error('Require numeric tour ID, 75..150 images, pairs>1000 divisible by bins')
 files=source_files(a.input_root,tour);data=build(tour,a.seed,a.images,files,a.pairs,a.bins)
 if a.dry_run:print(json.dumps({'images':len(data['views']),'candidate_edges':len(data['candidate_edges']),'histograms':data['planned_histograms']},indent=2));return
 out=a.kml_root/tour;old=out/'views.json'
 if old.exists() and json.loads(old.read_text()).get('seed_sha256')!=data['seed_sha256']:raise RuntimeError('Different seed already exists. Use another --kml-root/--json-root, preserving the existing dataset.')
 out.mkdir(parents=True,exist_ok=True)
 for v in data['views']:emit_kml(out/f'{v["id"]}.kml',v,data['anchor'],60)
 save_json(out/'views.json',data)
 save_json(a.json_root/f'{tour}.json',{'status':'planned','tour_id':tour,'images':a.images,'pairs':0,'planned_pairs':a.pairs,'seed_sha256':data['seed_sha256'],'views_manifest':str(out/'views.json'),'note':'Final pairs are published only after rendering and image-overlap verification.'})
 print(json.dumps({'tour':tour,'images':a.images,'planned_unique_pairs':a.pairs,'candidate_edges':len(data['candidate_edges']),'kml_root':str(out),'histograms':data['planned_histograms']},indent=2),flush=True)
if __name__=='__main__':main()
