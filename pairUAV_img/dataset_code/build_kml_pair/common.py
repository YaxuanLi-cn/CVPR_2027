#!/usr/bin/python3
"""Geometry shared by generation, rendering and independent validation."""
from __future__ import annotations
import hashlib, json, math, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from pyproj import Transformer
KML='http://www.opengis.net/kml/2.2'; GX='http://www.google.com/kml/ext/2.2'
NS='{'+KML+'}'; GNS='{'+GX+'}'
ET.register_namespace('',KML);ET.register_namespace('gx',GX)
BASE=Path(__file__).resolve().parent
AXES={'x_m':(-15.,15.),'y_m':(-15.,15.),'z_m':(-33.,33.),'tilt_deg':(-60.,60.),'heading_deg':(-180.,180.)}
FORWARD=Transformer.from_crs(4979,4978,always_xy=True)
INVERSE=Transformer.from_crs(4978,4979,always_xy=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def seed_for(tour,seed):return hashlib.sha256(f'kml-pair-v2:{tour}:{seed}'.encode()).hexdigest()
def emit_lookat_kml(path,camera,anchor,hfov=60):
    root=ET.Element(NS+'kml');doc=ET.SubElement(root,NS+'Document');ET.SubElement(doc,NS+'name').text=camera['id'];tour=ET.SubElement(doc,GNS+'Tour');ET.SubElement(tour,NS+'name').text=camera['id'];pl=ET.SubElement(tour,GNS+'Playlist');fly=ET.SubElement(pl,GNS+'FlyTo');ET.SubElement(fly,GNS+'duration').text='0';ET.SubElement(fly,GNS+'flyToMode').text='bounce';look=ET.SubElement(fly,NS+'LookAt')
    for k in ('longitude','latitude','altitude','heading','tilt','range'):
        val=anchor[k] if k in ('longitude','latitude','altitude') else camera[k];ET.SubElement(look,NS+k).text=f'{val:.12f}'
    ET.SubElement(look,GNS+'altitudeMode').text='relativeToSeaFloor';ET.SubElement(look,GNS+'horizFov').text=str(hfov);ET.SubElement(ET.SubElement(pl,GNS+'Wait'),GNS+'duration').text='0.5';data=ET.SubElement(doc,NS+'ExtendedData')
    for k,v in {'first_key_longitude':anchor['longitude'],'first_key_latitude':anchor['latitude'],'first_key_altitude_msl':anchor['altitude_msl']}.items():ET.SubElement(ET.SubElement(data,NS+'Data',name=k),NS+'value').text=str(v)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def source_files(root,tour):
 for d in (Path(root)/tour,Path('/root/pairUAV_img/tours_split_kml')/tour,Path('/root/pairUAV_img/tours_splited_kml')/tour):
  fs=sorted(d.glob('*.kml'))
  if fs:return fs
 raise ValueError(f'No KMLs for tour {tour}')
def view_from_file(p):
 v=next((e for e in ET.parse(p).getroot().iter() if e.tag in (NS+'LookAt',NS+'Camera')),None)
 if v is None:raise ValueError(f'No camera: {p}')
 d={e.tag.split('}')[-1]:float(e.text) for e in v if e.tag.split('}')[-1] in ('longitude','latitude','altitude','heading','tilt','range','roll','horizFov')}
 d['type']=v.tag.split('}')[-1]
 am=next((e.text for e in v if e.tag.split('}')[-1]=='altitudeMode'),'clampToGround');d['altitudeMode']=am
 return d
class Frame:
 """One fixed ENU frame per tour; no mixing body-frame and world displacements."""
 def __init__(self,lon,lat,alt):
  self.origin=np.array(FORWARD.transform(lon,lat,alt));self.lon=lon;self.lat=lat;self.alt=alt
  lo,la=np.radians([lon,lat]);self.R=np.array([[-np.sin(lo),np.cos(lo),0],[-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)],[np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),np.sin(la)]])
 def to_geo(self,p):return INVERSE.transform(*(self.origin+self.R.T@np.asarray(p)))
 def from_geo(self,p):return self.R@(np.array(FORWARD.transform(*p))-self.origin)
def rotation(heading,tilt):
 h,t=np.radians([heading,tilt]);right=np.array([np.cos(h),-np.sin(h),0]);up=np.array([np.sin(h)*np.cos(t),np.cos(h)*np.cos(t),np.sin(t)]);f=np.array([np.sin(h)*np.sin(t),np.cos(h)*np.sin(t),-np.cos(t)])
 return np.stack((right,-up,f)) # camera x right, y down, z forward

def intrinsic(hfov=120,w=3840,h=2160):
 f=w/(2*np.tan(np.radians(hfov)/2));return np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
def project(points,C,R,K):
 p=(R@(np.asarray(points)-C).T).T;uv=p[:,:2]/p[:,2:3];return uv@K[:2,:2].T+K[:2,2],p[:,2]
def crop_box():return [77,0,3763,2073]
def visibility(uv,z,margin=0):
 return (z>0)&(uv[:,0]>=77+margin)&(uv[:,0]<3763-margin)&(uv[:,1]>=margin)&(uv[:,1]<2073-margin)
def delta(a,b):
 p=np.array(b['position_enu_m'])-a['position_enu_m'];h=(b['heading']-a['heading']+180)%360-180
 return dict(zip(AXES,[float(p[0]),float(p[1]),float(p[2]),b['tilt']-a['tilt'],float(h)]))
def bin_ids(d,bins):
 return [min(bins-1,max(0,int((d[k]-lo)/(hi-lo)*bins))) for k,(lo,hi) in AXES.items()]
def eligible(d):return all(lo-1e-8<=d[k]<=hi+1e-8 for k,(lo,hi) in AXES.items())
def camera_element(camera,hfov):
 c=ET.Element(NS+'Camera')
 for k in ('longitude','latitude','altitude','heading','tilt','roll'):ET.SubElement(c,NS+k).text=f'{camera[k]:.12f}'
 ET.SubElement(c,GNS+'altitudeMode').text='relativeToSeaFloor';ET.SubElement(c,GNS+'horizFov').text=str(hfov);return c

def emit_kml(path,camera,anchor,hfov=120):
 # A zero-duration tour applies gx:horizFov in Earth Pro (a standalone
 # Placemark Camera may silently keep the previous field of view).
 root=ET.Element(NS+'kml');doc=ET.SubElement(root,NS+'Document');ET.SubElement(doc,NS+'name').text=camera['id']
 tour=ET.SubElement(doc,GNS+'Tour');ET.SubElement(tour,NS+'name').text=camera['id'];pl=ET.SubElement(tour,GNS+'Playlist');fly=ET.SubElement(pl,GNS+'FlyTo');ET.SubElement(fly,GNS+'duration').text='0';ET.SubElement(fly,GNS+'flyToMode').text='bounce';fly.append(camera_element(camera,hfov));ET.SubElement(ET.SubElement(pl,GNS+'Wait'),GNS+'duration').text='0.5'
 data=ET.SubElement(doc,NS+'ExtendedData')
 for k,v in {'first_key_longitude':anchor['longitude'],'first_key_latitude':anchor['latitude'],'first_key_altitude_msl':anchor['altitude_msl']}.items():ET.SubElement(ET.SubElement(data,NS+'Data',name=k),NS+'value').text=str(v)
 ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
