#!/usr/bin/python3
"""Verify the current manifest, native JPEGs, crop geometry and source cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

NS = '{http://www.opengis.net/kml/2.2}'
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree(e): return (e.tag, e.attrib, (e.text or '').strip(), [tree(c) for c in e])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path, help='e.g. /root/pairUAV_img/images/0003')
    args=ap.parse_args()
    m=json.loads((args.output/'manifest.json').read_text())
    if m['status']!='complete': raise ValueError(f'Run is {m["status"]}: {m.get("error", "")}')
    entries=m['images']
    if not entries: raise ValueError('No images in manifest')
    for e in entries:
        source=Path(e['source'])
        assert digest(source)==e['source_sha256'], f'{source}: source changed'
        views=[v for v in ET.parse(source).getroot().iter() if v.tag in (NS+'LookAt', NS+'Camera')]
        original=views[e['camera']['selected_view']]
        prepared=next(v for v in ET.parse(e['prepared_kml']).getroot().iter() if v.tag in (NS+'LookAt', NS+'Camera'))
        assert tree(original)==tree(prepared)==tree(ET.fromstring(e['camera']['xml'])), f'{source}: camera was modified'
        for field,hashfield,dims in [('native','native_sha256','native_dimensions'),('output','sha256','dimensions')]:
            path=Path(e[field]); assert digest(path)==e[hashfield], f'{path}: hash mismatch'
            with Image.open(path) as im:
                assert im.format=='JPEG' and list(im.size)==e[dims], f'{path}: wrong image dimensions/format'
                im.verify()
        box=e['crop_box_native_pixels']
        if box is not None:
            assert [box[2]-box[0],box[3]-box[1]]==e['dimensions']
        print(f'{source.name}: native {e["native_dimensions"]}, output {e["dimensions"]}; hashes and camera XML OK')
    print(f'PASS: {len(entries)} images. This verifies files and copied camera parameters; inspect the preview for rendered content.')

if __name__=='__main__': main()
