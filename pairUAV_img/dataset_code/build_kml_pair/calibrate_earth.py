#!/usr/bin/python3
"""Measure first-key terrain through the local Google Earth NetworkLink, without exporting images."""
import argparse,fcntl,json,os,sys,time,uuid,xml.etree.ElementTree as ET
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'gen_img'))
import export_google_earth_pro as earth
K='{http://www.opengis.net/kml/2.2}'

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 src=ET.parse(a.source);v=next(e for e in src.iter() if e.tag in (K+'LookAt',K+'Camera'));lon=float(v.findtext(K+'longitude'));lat=float(v.findtext(K+'latitude'))
 os.environ['DISPLAY']=':1'
 lock=(BASE.parent/'gen_img/export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 from audit_resources import recover_affinity
 recover_affinity()
 work=a.output.parent/'calibration_ui'/uuid.uuid4().hex[:10];work.mkdir(parents=True,exist_ok=True)
 app=earth.Earth(work)
 live=False
 try:live=time.time()-json.loads((BASE/'.telemetry/latest.json').read_text())['time']<10
 except (OSError,ValueError,KeyError):pass
 if not live:app.open_view(BASE/'telemetry.kml','telemetry',3,45)
 root=ET.Element(K+'kml');doc=ET.SubElement(root,K+'Document');cam=ET.SubElement(doc,K+'Camera')
 for k,val in dict(longitude=lon,latitude=lat,altitude=20000,heading=0,tilt=0,roll=0,altitudeMode='absolute').items():ET.SubElement(cam,K+k).text=str(val)
 target=work/'terrain_probe.kml';ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)
 prepared=work/'terrain_probe_prepared.kml';earth.prepare(target,prepared,'TERRAINPROBE',0)
 start=time.time();app.open_view(prepared,'TERRAINPROBE',4,60)
 deadline=time.monotonic()+40
 while time.monotonic()<deadline:
  try:
   d=json.loads((BASE/'.telemetry/latest.json').read_text())
   if d['time']>start and abs(d['cameraLon']-lon)<.0001 and abs(d['cameraLat']-lat)<.0001 and abs(d['lookatTerrainLon']-lon)<.0002 and abs(d['lookatTerrainLat']-lat)<.0002:
    result={'source_kml':str(a.source),'source_sha256':earth.sha(a.source),'terrain_altitude_msl':d['lookatTerrainAlt'],'method':'Earth Pro NetworkLink nadir probe','telemetry':d}
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(a.output);print(json.dumps(result));return
  except (OSError,ValueError,KeyError):pass
  time.sleep(1)
 raise RuntimeError('No fresh telemetry at requested first-key coordinates')
if __name__=='__main__':main()
