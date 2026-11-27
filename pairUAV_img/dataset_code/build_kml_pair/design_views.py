#!/usr/bin/env python3
"""Design reproducible, high-overlap nadir camera views for an image-audited tour."""
from __future__ import annotations
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from common import NS,GNS,source_files,view_from_file,Frame,seed_for,sha,save_json,AXES,delta,crop_box
from balance import candidates

def lhs(rng,n,lo,hi):
 x=lo+(np.arange(n)+rng.random(n))*(hi-lo)/n;rng.shuffle(x);return x

def emit(path,v,anchor,hfov):
 root=ET.Element(NS+'kml');doc=ET.SubElement(root,NS+'Document');ET.SubElement(doc,NS+'name').text=v['id'];tour=ET.SubElement(doc,GNS+'Tour');ET.SubElement(tour,NS+'name').text=v['id'];pl=ET.SubElement(tour,GNS+'Playlist');fly=ET.SubElement(pl,GNS+'FlyTo');ET.SubElement(fly,GNS+'duration').text='0';ET.SubElement(fly,GNS+'flyToMode').text='bounce';cam=ET.SubElement(fly,NS+'Camera')
 for k in ('longitude','latitude','altitude','heading','tilt','roll'):ET.SubElement(cam,NS+k).text=f'{v[k]:.12f}'
 ET.SubElement(cam,NS+'altitudeMode').text='absolute';ET.SubElement(cam,GNS+'horizFov').text=str(hfov);ET.SubElement(ET.SubElement(pl,GNS+'Wait'),GNS+'duration').text='0.3';ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--images',type=int,default=150);p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'));p.add_argument('--kml-root',type=Path,default=Path('/tmp/redesign_kml'));p.add_argument('--pairs',type=int,default=1200);p.add_argument('--bins',type=int,default=12);p.add_argument('--camera-height',type=float,default=300);p.add_argument('--max-camera-range',type=float,default=None);p.add_argument('--position-radius',type=float,default=7.5);p.add_argument('--tilt-min',type=float,default=0);p.add_argument('--tilt-max',type=float,default=60);p.add_argument('--structured-orientation',action='store_true');p.add_argument('--paired-tilt-groups',action='store_true');p.add_argument('--tilt-group-count',type=int,default=15);p.add_argument('--edge-fraction',type=float,default=.4);p.add_argument('--uniform-image-tilt',action='store_true');p.add_argument('--hfov',type=float,default=90);a=p.parse_args();tour=a.tour.zfill(4);files=source_files(a.input_root,tour);src=view_from_file(files[0]);
 if src['type']!='LookAt':raise ValueError('first source key needs LookAt')
 calibration=Path(__file__).resolve().parent/'calibration'/f'{tour}.json'
 if not calibration.exists() or json.loads(calibration.read_text()).get('source_sha256')!=sha(files[0]):
  for attempt in range(3):
   try:
    subprocess.run(['/usr/bin/python3',str(Path(__file__).resolve().parent/'calibrate_earth.py'),'--source',str(files[0]),'--output',str(calibration)],check=True,timeout=180)
    break
   except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
    if attempt==2:raise
    print(f'{tour}: terrain calibration retry {attempt+1}; recovering Earth UI',flush=True)
    subprocess.run(['/usr/bin/python3',str(Path(__file__).resolve().parent.parent/'gen_img/recover_google_earth.py')],check=True,timeout=120)
 terrain=json.loads(calibration.read_text())
 if a.max_camera_range is not None:
  if a.max_camera_range <= a.position_radius:
   raise ValueError('--max-camera-range must exceed --position-radius')
  max_height=(a.max_camera_range**2-2*a.position_radius**2)**0.5
  if a.camera_height+16.5 > max_height:
   raise ValueError(f'camera height range exceeds max camera range: upper={a.camera_height+16.5:.2f}m, limit={max_height:.2f}m')
 anchor={'longitude':src['longitude'],'latitude':src['latitude'],'altitude':src['altitude'],'terrain_altitude_msl':terrain['terrain_altitude_msl'],'terrain_calibration':terrain,'feature_key':'first-key','source_kml':str(files[0]),'source_sha256':sha(files[0]),'semantic_note':'first-key camera target region; not a semantic building polygon'}
 digest=seed_for(tour,a.seed);rng=np.random.default_rng(int(digest[:16],16));frame=Frame(anchor['longitude'],anchor['latitude'],anchor['terrain_altitude_msl']);n=a.images
 def tail_values(lo,hi,edge_fraction=.35):
  edge=max(1,int(n*edge_fraction/2));mid=n-2*edge
  vals=np.r_[lhs(rng,edge,lo,lo+(hi-lo)*.12),lhs(rng,edge,hi-(hi-lo)*.12,hi),lhs(rng,mid,lo+(hi-lo)*.25,lo+(hi-lo)*.75)]
  rng.shuffle(vals);return vals
 xs=tail_values(-a.position_radius,a.position_radius);ys=tail_values(-a.position_radius,a.position_radius);zs=tail_values(a.camera_height-16.5,a.camera_height+16.5);head=lhs(rng,n,0,360);tilt=lhs(rng,n,a.tilt_min,a.tilt_max)
 if a.structured_orientation:
  groups=12; order=rng.permutation(n); head=np.empty(n);tilt=np.empty(n); split=np.array_split(order,groups)
  for gi,ids in enumerate(split):
   count=len(ids); low=max(2,count//4);high=max(2,count//4);mid=count-low-high
   vals=np.r_[rng.uniform(0,6,low),rng.uniform(54,60,high),rng.uniform(12,48,mid)]
   rng.shuffle(vals);tilt[ids]=vals;head[ids]=(gi*360/groups+rng.uniform(-3,3,count))%360
 if a.paired_tilt_groups:
  groups=a.tilt_group_count; order=rng.permutation(n);head=np.empty(n);tilt=np.empty(n); split=np.array_split(order,groups)
  # Each heading/spatial group spans the complete tilt range. Repeated low,
  # middle and high tilts create many true-image opportunities for large tilt deltas.
  for gi,ids in enumerate(split):
   count=len(ids); edge=max(1,int(count*a.edge_fraction/2));middle=max(0,count-2*edge)
   vals=np.r_[rng.uniform(a.tilt_min,a.tilt_min+5,edge),rng.uniform(a.tilt_max-5,a.tilt_max,edge),rng.uniform(a.tilt_min+8,a.tilt_max-8,middle)]
   rng.shuffle(vals);tilt[ids]=vals;head[ids]=(gi*360/groups+rng.uniform(-4,4,count))%360
 if a.uniform_image_tilt:
  from image_tilt import uniform_tilts
  tilt=uniform_tilts(rng,n,a.tilt_min,a.tilt_max)
 roll=np.zeros(n)
 # Correlate no axes by assigning independent Latin hypercube marginals.
 views=[]
 for i in range(n):
  lon,lat,alt=frame.to_geo([xs[i],ys[i],zs[i]])
  pos=[float(xs[i]),float(ys[i]),float(zs[i])]
  view={'id':f'{i:04d}','longitude':float(lon),'latitude':float(lat),'altitude':float(alt),'heading':float(head[i]),'tilt':float(tilt[i]),'roll':0.,'position_enu_m':pos,'east':pos[0],'north':pos[1],'up':pos[2],'camera_range_m':float(np.linalg.norm(pos)),'cluster':0,'anchor_pixel_native':None,'anchor_pixel_crop':None,'visible_grid':None}
  if a.max_camera_range is not None and view['camera_range_m'] > a.max_camera_range + 1e-8:
   raise ValueError(f'generated camera exceeds max range: {view["camera_range_m"]:.3f}m')
  views.append(view)
 edges=candidates(views);out=a.kml_root/tour;out.mkdir(parents=True,exist_ok=True)
 for v in views:emit(out/f'{v["id"]}.kml',v,anchor,a.hfov)
 data={'schema_version':'6.1-camera-range-limited','status':'planned_for_visual_audit','tour_id':tour,'seed':a.seed,'seed_sha256':digest,'images':n,'requested_pairs':a.pairs,'bins':a.bins,'hfov_deg':a.hfov,'tilt_camera_range_deg':[a.tilt_min,a.tilt_max],'structured_orientation':a.structured_orientation,'paired_tilt_groups':a.paired_tilt_groups,'tilt_edge_fraction':a.edge_fraction,'orientation_groups':12 if a.structured_orientation else (a.tilt_group_count if a.paired_tilt_groups else None),'position_radius_m':a.position_radius,'camera_height_center_m':a.camera_height,'max_camera_range_m':a.max_camera_range,'native_size':[3840,2160],'crop_box':crop_box(),'anchor':anchor,'views':views,'candidate_edges':edges,'sampling':'deterministic camera-view design with a bounded camera-to-first-key ENU range; image audit validates actual shared features before pair selection','coordinate_convention':{'translation':'B-A actual camera positions, first-key ENU east/north/up meters','heading':'wrapped camera-heading difference','tilt':'camera-tilt difference','camera_tilt':'KML Camera tilt from nadir'}}
 if a.uniform_image_tilt:
  from image_tilt import validate_image_tilt
  data['image_tilt_constraint']={'sampling':'stratified-uniform-v1','range_deg':[a.tilt_min,a.tilt_max],'bins':10}
  data['image_tilt_validation']=validate_image_tilt(data)
 save_json(out/'views.json',data);print(json.dumps({'tour':tour,'images':n,'geometric_candidates':len(edges),'kml_root':str(out),'seed_sha256':digest},indent=2))
if __name__=='__main__':main()
