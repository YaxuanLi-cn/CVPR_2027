#!/usr/bin/python3
"""Loopback-only Google Earth NetworkLink camera telemetry (no external port)."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import json,time,os,threading
WRITE_LOCK=threading.Lock()
DEST=Path(os.environ.get('EARTH_TELEMETRY','/root/pairUAV_img/dataset_code/build_kml_pair/.telemetry')); DEST.mkdir(parents=True,exist_ok=True)
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  q={k:v[-1] for k,v in parse_qs(urlparse(self.path).query).items()};q['time']=time.time()
  for k,v in list(q.items()):
   try:q[k]=float(v)
   except (TypeError,ValueError):pass
  if 'cameraLon' in q:
   with WRITE_LOCK:
    tmp=DEST/'latest.tmp';tmp.write_text(json.dumps(q));tmp.replace(DEST/'latest.json')
    with (DEST/'history.jsonl').open('a') as f:f.write(json.dumps(q)+'\n')
  data=b'<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Telemetry</name></Document></kml>'
  self.send_response(200);self.send_header('Content-Type','application/vnd.google-earth.kml+xml');self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(data)
 def log_message(self,*a):pass
ThreadingHTTPServer(('127.0.0.1',18765),Handler).serve_forever()
