#!/usr/bin/python3
"""Calibrate first-key terrain elevation from Google Earth telemetry."""
from __future__ import annotations
import argparse,json,os,re,subprocess,time
from pathlib import Path
from common import source_files,view_from_file,sha,BASE,save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--input-root',type=Path,default=Path('/root/pairUAV_img/tours_splited_kml'));p.add_argument('--wait',type=float,default=10);p.add_argument('--terrain-altitude',type=float);a=p.parse_args();tour=str(a.tour).zfill(4);f=source_files(a.input_root,tour)[0];v=view_from_file(f);out=BASE/'calibration';out.mkdir(exist_ok=True)
 if a.terrain_altitude is not None:terrain=float(a.terrain_altitude);method='manual'
 else:
  # Existing calibration server is loopback only. Google Earth sends absolute
  # camera/terrain telemetry after the telemetry KML is loaded by the operator.
  p.error('Automatic Earth telemetry needs an interactive calibration session; pass --terrain-altitude for a reproducible run')
 save_json(out/f'{tour}.json',{'tour_id':tour,'source_kml':str(f),'source_sha256':sha(f),'terrain_altitude_msl':terrain,'method':method,'coordinate_note':'Earth Pro telemetry terrainAltitude is used; confirm units are meters above sea level'});print(json.dumps(json.loads((out/f'{tour}.json').read_text()),indent=2))
if __name__=='__main__':main()
