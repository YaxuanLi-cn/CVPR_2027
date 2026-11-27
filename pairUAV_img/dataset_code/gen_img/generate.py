#!/usr/bin/python3
"""Render a tour with Google Earth Pro: ./generate.py --tour 0003 --limit 10."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tour', required=True, help='Tour ID, e.g. 0003 (3 is also accepted)')
    count = parser.add_mutually_exclusive_group()
    count.add_argument('--limit', type=int, default=10, help='First X KML files (default: 10)')
    count.add_argument('--all', action='store_true', help='Render all KML files')
    parser.add_argument('--input-root', type=Path, help='Root containing tour directories; autodetected by default')
    parser.add_argument('--output-root', type=Path, default=Path('/root/pairUAV_img/images'))
    parser.add_argument('--start', type=int, default=0, help='Skip this many sorted files')
    parser.add_argument('--settle', type=float, default=30)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--check', action='store_true', help='Validate selection and print paths without installing/rendering')
    args = parser.parse_args()
    if not args.tour.isdecimal(): parser.error('--tour must be a numeric ID')
    tour = args.tour.zfill(4)
    if args.limit < 1 or args.start < 0: parser.error('Require limit >= 1 and start >= 0')
    if not 0 <= args.settle < args.timeout: parser.error('Require 0 <= settle < timeout')
    roots = [args.input_root] if args.input_root else [Path('/root/pairUAV_img/tours_split_kml'), Path('/root/pairUAV_img/tours_splited_kml')]
    source = next((r / tour for r in roots if (r / tour).is_dir()), None)
    if source is None: parser.error(f'Tour {tour} not found in {roots}; set --input-root')
    files = sorted(source.glob('*.kml'))[args.start:]
    if not args.all: files = files[:args.limit]
    if not files: parser.error(f'No KML files selected in {source}')
    # Reject invalid KML before changing environment or launching the GUI.
    import xml.etree.ElementTree as ET
    for path in files:
        root = ET.parse(path).getroot()
        if not any(e.tag in ('{http://www.opengis.net/kml/2.2}LookAt', '{http://www.opengis.net/kml/2.2}Camera') for e in root.iter()):
            parser.error(f'No camera in {path}')
    output = args.output_root.resolve() / tour
    print(f'Tour: {tour}\nInput: {source}\nOutput: {output}\nSelected: {len(files)} ({files[0].name} .. {files[-1].name})', flush=True)
    if args.check: return 0
    with open(BASE / 'generate.lock', 'w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: parser.error('Another export is running; the shared Earth GUI runs one job at a time')
        commands = ['google-earth-pro', 'Xvfb', 'xdotool', 'wmctrl', 'openbox', 'tesseract', 'supervisorctl', 'xdpyinfo', 'glxinfo', 'nvidia-smi']
        pillow = subprocess.run(['/usr/bin/python3', '-c', 'from PIL import ImageGrab'], capture_output=True).returncode == 0
        if not pillow or any(shutil.which(c) is None for c in commands):
            print('Installing dependencies; reproducible log: ' + str(BASE / 'install.log'), flush=True)
            with open(BASE / 'install.log', 'a') as log:
                result = subprocess.run(['bash', str(BASE / 'install_google_earth.sh')], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Installation failed; see {BASE / "install.log"}')
        subprocess.run(['bash', str(BASE / 'setup_google_earth_runtime.sh')], check=True)
        cmd = ['/usr/bin/python3', str(BASE / 'export_google_earth_pro.py'), '--input', str(source), '--output', str(output), '--start', str(args.start), '--settle', str(args.settle), '--timeout', str(args.timeout)]
        if not args.all: cmd += ['--limit', str(args.limit)]
        output.mkdir(parents=True, exist_ok=True)
        # Keep full stdout/stderr for failures, even when the terminal disconnects.
        with open(output / 'export.log', 'a') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = proc.wait()
            if code: return code
            return subprocess.call(['/usr/bin/python3', str(BASE / 'verify_outputs.py'), str(output)])

if __name__ == '__main__':
    sys.exit(main())
