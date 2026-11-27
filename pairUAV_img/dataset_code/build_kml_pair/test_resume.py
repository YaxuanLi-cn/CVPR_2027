#!/usr/bin/env python3
"""Offline regression checks for artifact reuse; never touches Earth or real outputs."""
import tempfile
from pathlib import Path
from PIL import Image
import group_batch_pipeline as g
from balance import candidates

def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);k=root/'k';i=root/'i';st=root/'st'
  k.mkdir();i.mkdir();(st/'runs/r').mkdir(parents=True)
  source=root/'crop.jpg';Image.new('RGB',(3686,2073),(120,130,140)).save(source)
  (k/'0000.kml').write_text('original source')
  entry={'source':str(k/'0000.kml'),'source_sha256':g.sha(k/'0000.kml'),'output':str(source),'sha256':g.sha(source)}
  manifest=st/'runs/r/manifest.json';g.atomic_json(manifest,{'images':[entry]})
  g.sync_exports(st,i,k,80);assert g.verified_images(i,k,1)=={0}
  before=(i/'0000.jpg').stat().st_mtime_ns;g.sync_exports(st,i,k,80);assert (i/'0000.jpg').stat().st_mtime_ns==before
  (i/'0000.jpg').write_bytes(b'broken');assert not g.verified_images(i,k,1)
  g.sync_exports(st,i,k,80);assert g.verified_images(i,k,1)=={0}
  (k/'0000.kml').write_text('changed source');assert not g.verified_images(i,k,1)
  # Raw native exports must never be adopted as cropped final images.
  (i/'0000.jpg').unlink();Image.new('RGB',(3840,2160)).save(source)
  entry.update(source_sha256=g.sha(k/'0000.kml'),sha256=g.sha(source));g.atomic_json(manifest,{'images':[entry]})
  g.sync_exports(st,i,k,80);assert not g.verified_images(i,k,1)
  manifest.write_text('{partial');g.sync_exports(st,i,k,80);assert not g.verified_images(i,k,1)
  a=dict(position_enu_m=[0,0,0],heading=0,tilt=0,visible_grid=None)
  b=dict(position_enu_m=[10,10,20],heading=45,tilt=44,visible_grid=None)
  edges=candidates([a,b]);assert len(edges)==1 and edges[0]['predicted_overlap'] is None
 print('PASS: reuse without rewrite; corrupt-image recovery; stale-KML, raw-native and partial-manifest rejection; unknown visibility supported')
if __name__=='__main__':main()
