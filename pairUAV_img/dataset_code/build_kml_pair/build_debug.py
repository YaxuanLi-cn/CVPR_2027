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
 anchor={'longitude':source['longitude'],'latitude':source['latitude'],'altitude':source.get('altitude',0.0),'altitude_msl':source.get('altitude',0.0),'terrain_altitude_msl':None,'feature_key':'first-key','source_kml':str(files[0]),'source_sha256':sha(files[0]),'semantic_note':'all views target the first LookAt building region'}
 frame=Frame(anchor['longitude'],anchor['latitude'],0.0);digest=seed_for(tour,seed);rng=np.random.default_rng(int(digest[:16],16));views=[]
 hs=np.linspace(0,360,n,endpoint=False)+rng.uniform(-0.4,0.4,n);ts=rng.uniform(5,65,n);rs=rng.uniform(20,40,n);rng.shuffle(hs);rng.shuffle(ts);rng.shuffle(rs)
 grid=np.column_stack(np.meshgrid(np.linspace(-15,15,21),np.linspace(-15,15,21),np.zeros((21,21))).reshape(3,-1).T) if False else np.zeros((1,3))
 for i,(h,t,rngv) in enumerate(zip(hs,ts,rs)):
  hr,tr=np.radians([h,t]);pos=np.array([-np.sin(hr)*rngv*np.sin(tr),-np.cos(hr)*rngv*np.sin(tr),rngv*np.cos(tr)])
  lon,lat,alt=frame.to_geo(pos);precise=frame.from_geo([lon,lat,alt]);cam={'id':f'{i:04d}','longitude':anchor['longitude'],'latitude':anchor['latitude'],'altitude':anchor['altitude'],'heading':float(h%360),'tilt':float(t),'range':float(rngv),'roll':0.,'position_enu_m':precise.tolist(),'east':float(precise[0]),'north':float(precise[1]),'up':float(precise[2]),'cluster':0,'anchor_pixel_native':[1920,1080],'anchor_pixel_crop':[1843,1080],'visible_grid':[True]}
  views.append(cam)
 edges=candidates(views);print('DEBUG edges',len(edges),flush=True);
 from collections import Counter
 print('bins', [Counter(bin_ids(e['delta'],bins)[q] for e in edges) for q in range(5)],flush=True);chosen,counts,msg=select(edges,views,pair_count,bins,int(digest[16:24],16))
 return {'schema_version':'3.0','status':'planned','tour_id':tour,'seed':seed,'seed_sha256':digest,'source_hashes':{str(p):sha(p) for p in files},'images':n,'requested_pairs':pair_count,'bins':bins,'hfov_deg':60.,'native_size':[3840,2160],'crop_box':crop_box(),'anchor':anchor,'views':views,'candidate_edges':edges,'planned_pairs':chosen,'planned_histograms':counts,'solver':msg,'coordinate_convention':{'translation':'B-A in fixed first-key ENU east/north/up meters','heading':'wrap(B-A) in [-180,180)','tilt':'B-A degrees','range':'LookAt range meters','pair_uniqueness':'unordered image pair, one chosen orientation'}}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tour',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--images',type=int,default=150);p.add_argument('--pairs',type=int,default=1200);p.add_argument('--bins',type=int,default=6);p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'));p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml'));p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));p.add_argument('--dry-run',action='store_true');a=p.parse_args();tour=a.tour.zfill(4)
 if not a.tour.isdecimal() or not 75<=a.images<=150 or a.pairs<=1000 or a.pairs%a.bins:p.error('Require numeric tour ID, 75..150 images, pairs>1000 divisible by bins')
 files=source_files(a.input_root,tour);data=build(tour,a.seed,a.images,files,a.pairs,a.bins)
 if a.dry_run:print(json.dumps({'images':len(data['views']),'candidate_edges':len(data['candidate_edges']),'histograms':data['planned_histograms']},indent=2));return
 out=a.kml_root/tour;old=out/'views.json'
 if old.exists() and json.loads(old.read_text()).get('seed_sha256')!=data['seed_sha256']:raise RuntimeError('Different seed already exists. Use another --kml-root/--json-root, preserving the existing dataset.')
 out.mkdir(parents=True,exist_ok=True)
 for v in data['views']:emit_lookat_kml(out/f'{v["id"]}.kml',v,data['anchor'])
 save_json(out/'views.json',data)
 save_json(a.json_root/f'{tour}.json',{'status':'planned','tour_id':tour,'images':a.images,'pairs':0,'planned_pairs':a.pairs,'seed_sha256':data['seed_sha256'],'views_manifest':str(out/'views.json'),'note':'Final pairs are published only after rendering and image-overlap verification.'})
 print(json.dumps({'tour':tour,'images':a.images,'planned_unique_pairs':a.pairs,'candidate_edges':len(data['candidate_edges']),'kml_root':str(out),'histograms':data['planned_histograms']},indent=2),flush=True)
if __name__=='__main__':main()
