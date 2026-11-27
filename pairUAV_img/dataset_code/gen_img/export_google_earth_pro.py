#!/usr/bin/env python3
"""Render arbitrary tour directories through the installed Google Earth Pro GUI.

Only source KML is needed. Each complete LookAt/Camera is copied without loss
into one invisible Point Placemark and opened independently, so Places ordering
and expansion state never determine the camera. Native exports are retained.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from PIL import Image, ImageChops, ImageDraw, ImageGrab, ImageStat

BASE = Path(__file__).resolve().parent
KML = 'http://www.opengis.net/kml/2.2'
GX = 'http://www.google.com/kml/ext/2.2'
ET.register_namespace('', KML)
ET.register_namespace('gx', GX)

def run(*args, check=True):
    return subprocess.run(list(map(str,args)), check=check, text=True, capture_output=True, timeout=20)

def xdo(*args):
    return run('xdotool', *args).stdout.strip()

def key(*keys):
    xdo('key', '--clearmodifiers', *keys)

def click(x,y):
    xdo('mousemove',str(round(x)),str(round(y)),'click','1')
    time.sleep(.35)

def grab():
    return ImageGrab.grab()

def words(box, psm=11):
    # OCR is restricted to UI text, never used to synthesize image pixels.
    im=grab().crop(box).resize(((box[2]-box[0])*2,(box[3]-box[1])*2))
    b=io.BytesIO(); im.save(b,format='PNG')
    p=subprocess.run(['tesseract','stdin','stdout','--psm',str(psm),'tsv'],input=b.getvalue(),capture_output=True,check=True,timeout=20,env={**os.environ,'OMP_THREAD_LIMIT':'1'})
    result=[]
    for r in csv.DictReader(io.StringIO(p.stdout.decode()),delimiter='\t', quoting=csv.QUOTE_NONE):
        if r['text'].strip():
            result.append({'text':r['text'].strip(), 'x':box[0]+int(r['left'])/2,
                'y':box[1]+int(r['top'])/2, 'w':int(r['width'])/2, 'h':int(r['height'])/2})
    return result

def find(text, box, timeout=10, psm=11):
    until=time.monotonic()+timeout
    while True:
        ws=words(box,psm)
        candidates=[w for w in ws if text.casefold() in w['text'].casefold()]
        if not candidates and ' ' in text:
            parts=text.casefold().split()
            for i in range(len(ws)-len(parts)+1):
                group=ws[i:i+len(parts)]
                if [w['text'].casefold() for w in group]==parts and max(w['y'] for w in group)-min(w['y'] for w in group)<10:
                    candidates=[dict(group[0],w=group[-1]['x']+group[-1]['w']-group[0]['x'])]; break
        if candidates:return candidates[0]
        if time.monotonic()>until:raise RuntimeError(f'UI text {text!r} missing in {box}: {[w["text"] for w in ws]}')
        time.sleep(.5)

def click_word(w): click(w['x']+w['w']/2,w['y']+w['h']/2)

def find_joined(parts, box, timeout=10):
    """Find a UI label when OCR splits it across adjacent words."""
    wanted=[p.casefold().rstrip(':') for p in parts]
    until=time.monotonic()+timeout
    while True:
        ws=words(box,psm=6)
        for w in ws:
            compact=w['text'].casefold().replace(':','')
            wanted_compact=''.join(wanted)
            if wanted_compact in compact or compact in wanted_compact and len(compact) >= 5:
                return w
        for i in range(len(ws)-len(wanted)+1):
            group=ws[i:i+len(wanted)]
            got=[w['text'].casefold().rstrip(':') for w in group]
            if got != wanted or max(w['y'] for w in group)-min(w['y'] for w in group) >= 12:
                continue
            x=min(w['x'] for w in group); y=min(w['y'] for w in group)
            right=max(w['x']+w['w'] for w in group); bottom=max(w['y']+w['h'] for w in group)
            return {'x':x,'y':y,'w':right-x,'h':bottom-y,'text':' '.join(w['text'] for w in group)}
        if time.monotonic()>until:
            raise RuntimeError(f'UI text {" ".join(parts)!r} missing in {box}: {[w["text"] for w in ws]}')
        time.sleep(.5)

def visible_window(pattern):
    p=run('xdotool','search','--onlyvisible','--name',pattern,check=False)
    return p.stdout.strip().splitlines()[-1] if p.returncode==0 and p.stdout.strip() else None

def wait_window(pattern, present=True, timeout=60):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        w=visible_window(pattern)
        if bool(w)==present:return w
        time.sleep(.3)
    raise RuntimeError(f'Window {pattern!r} did not become {"visible" if present else "closed"}')

def focus(w):
    xdo('windowactivate','--sync',w)
    xdo('windowfocus','--sync',w)
    time.sleep(.2)

def fill_dialog(pattern,path):
    path=Path(path).resolve()
    if pattern == r'^Open$' and not path.is_file():
        raise FileNotFoundError(path)
    for attempt in range(3):
        w=wait_window(pattern)
        focus(w)
        key('alt+n');key('ctrl+a')
        xdo('type','--clearmodifiers','--delay','10',str(path))
        time.sleep(.7)
        key('Return')
        try:
            wait_window(pattern,False,timeout=5)
            return
        except RuntimeError:
            # Some Qt versions consume Return to accept path completion.
            # Repeat the full filename, then activate the explicit button.
            w=visible_window(pattern)
            if not w:return
            focus(w)
            key('alt+n');key('ctrl+a')
            xdo('type','--clearmodifiers','--delay','15',str(path))
            time.sleep(.7)
            key('alt+o' if pattern == r'^Open$' else 'alt+s')
            try:
                wait_window(pattern,False,timeout=5)
                return
            except RuntimeError:
                pass
    raise RuntimeError(f'File dialog failed to accept {path}; no image exported')

def view_from_kml(path,view_index=0):
    root=ET.parse(path).getroot()
    views=[e for e in root.iter() if e.tag in (f'{{{KML}}}LookAt',f'{{{KML}}}Camera')]
    if not views:raise ValueError(f'{path}: no KML LookAt/Camera')
    if view_index>=len(views):raise ValueError(f'{path}: view-index {view_index} out of range ({len(views)} views)')
    view=views[view_index]
    for tag in ('longitude','latitude'):
        v=view.find(f'{{{KML}}}{tag}')
        if v is None or v.text is None:raise ValueError(f'{path}: missing {tag}')
        float(v.text)
    return copy.deepcopy(view),len(views)

def prepare(source,dest,name,view_index,hfov_override=None):
    view,count=view_from_kml(source,view_index)
    root=ET.Element(f'{{{KML}}}kml')
    doc=ET.SubElement(root,f'{{{KML}}}Document')
    ET.SubElement(doc,f'{{{KML}}}name').text=name
    p=ET.SubElement(doc,f'{{{KML}}}Placemark')
    ET.SubElement(p,f'{{{KML}}}name').text=name
    p.append(view)
    # Earth Pro only applies a Placemark LookAt while loading it when the
    # Placemark has a geographic feature.  The source split files contain a
    # Tour/FlyTo, so add an invisible Point at the same target coordinate.
    # Label/icon scale zero keeps the helper Placemark out of the export.
    style=ET.SubElement(p,f'{{{KML}}}Style')
    label_style=ET.SubElement(style,f'{{{KML}}}LabelStyle')
    ET.SubElement(label_style,f'{{{KML}}}scale').text='0'
    icon_style=ET.SubElement(style,f'{{{KML}}}IconStyle')
    ET.SubElement(icon_style,f'{{{KML}}}scale').text='0'
    point=ET.SubElement(p,f'{{{KML}}}Point')
    lon=view.find(f'{{{KML}}}longitude').text
    lat=view.find(f'{{{KML}}}latitude').text
    ET.SubElement(point,f'{{{KML}}}coordinates').text=f'{lon},{lat},0'
    if hfov_override is not None:
        node=view.find(f'{{{GX}}}horizFov')
        if node is None:node=ET.SubElement(view,f'{{{GX}}}horizFov')
        node.text=str(hfov_override)
    ET.ElementTree(root).write(dest,encoding='UTF-8',xml_declaration=True)
    return {'type':view.tag.rsplit('}',1)[-1], 'xml':ET.tostring(view,encoding='unicode'),
        'values':{e.tag:e.text for e in view}, 'available_views':count,'selected_view':view_index}

class Earth:
    def __init__(self,diagnostics):
        self.diagnostics=diagnostics
        # A WM is essential: without it Qt dialogs can be visible but unfocused.
        if run('wmctrl','-m',check=False).returncode:raise RuntimeError('No window manager. Run setup_google_earth_runtime.sh first.')
        ids=run('wmctrl','-l').stdout.splitlines()
        candidates=[l.split()[0] for l in ids if 'Google Earth Pro' in l]
        if not candidates:raise RuntimeError('Google Earth Pro is not running. Run setup_google_earth_runtime.sh first.')
        # Main window is the largest Earth window, not a hidden tip/popup.
        self.window=max(candidates,key=self.area)
        focus(self.window)
        run('wmctrl','-ir',self.window,'-b','add,maximized_vert,maximized_horz')
        time.sleep(.5)
        self.width,self.height=grab().size
        if (self.width,self.height)!=(3840,2160):raise RuntimeError('Use the reproducible 3840x2160 virtual display (setup script).')
        self.toolbar=(320,20,1400,160)
        for pattern in (r'^Open$',r'^Save As$'):
            modal=visible_window(pattern)
            if modal:focus(modal);key('Escape');wait_window(pattern,False,timeout=5)
        if visible_window(r'^Google Earth$'):
            self.dismiss_trust_prompt()
            modal=visible_window(r'^Open$')
            if modal:focus(modal);key('Escape')
        self.dismiss_startup_popups()

    def dismiss_startup_popups(self):
        """Close first-run tips/survey windows that can intercept GUI input."""
        tips=visible_window(r'^Start-up Tips$')
        if tips:
            focus(tips)
            try:
                click_word(find('Close',(0,100,1100,800),timeout=3))
            except RuntimeError:
                g=xdo('getwindowgeometry','--shell',tips)
                d=dict(line.split('=',1) for line in g.splitlines() if '=' in line)
                # The Qt tip dialog has a Close button at its lower-right
                # corner; use the inner button area for scaled Qt windows.
                click(int(d['X'])+int(d['WIDTH'])*93//100,
                      int(d['Y'])+int(d['HEIGHT'])*90//100)
            wait_window(r'^Start-up Tips$',False,timeout=5)
        # Inspect only small Earth-owned dialogs. The large transient Earth
        # window is the 3D viewport, not a survey: OCR there is slow/noisy.
        for line in run('wmctrl','-l').stdout.splitlines():
            if 'Google Earth' not in line:continue
            wid=line.split()[0]
            if int(wid,16)==int(self.window,16):continue
            info=run('xwininfo','-id',wid,check=False)
            if info.returncode:continue
            def coord(label):
                m=re.search(re.escape(label)+r'\s*:\s*(-?\d+)',info.stdout)
                return int(m.group(1)) if m else 0
            x,y=coord('Absolute upper-left X'),coord('Absolute upper-left Y')
            w,h=coord('Width'),coord('Height')
            if not (100<=w<=1600 and 80<=h<=1000):continue
            # Current Earth Pro first-run web promotion is an owned 368x461
            # dialog titled "Google Earth". Its left action dismisses the
            # promotion; the right action launches a browser and is not used.
            if 300<=w<=500 and 350<=h<=550:
                click(x+round(w*.42),y+h-round(h*.11))
                time.sleep(.5)
                if not visible_window(r'^Google Earth$'):
                    continue
            try:
                click_word(find('No thanks',(x,y,x+w,y+h),timeout=1,psm=6))
            except (RuntimeError,subprocess.TimeoutExpired):
                pass

    @staticmethod
    def area(w):
        g=xdo('getwindowgeometry','--shell',str(int(w,16)))
        d=dict(line.split('=',1) for line in g.splitlines() if '=' in line)
        return int(d['WIDTH'])*int(d['HEIGHT'])

    def configure(self,native):
        focus(self.window)
        # Earth Pro's View > Grid menu label is OCR-unstable across releases.
        # The fresh dedicated renderer starts with graticule disabled; avoid
        # blind Ctrl+L toggles, which can turn it on for exported imagery.
        # Locate Layers dynamically; turn off all base overlays, then enable 3D.
        panel=(0,100,320,2050)
        try:
            layers=find('Layers',panel)
            lb=(0,int(layers['y']),320,2050)
        except RuntimeError:
            # OCR on the scaled Qt layer tree can read this heading as
            # "Lavers" or omit it; the tree is always in the lower-left pane.
            lb=(0,1250,420,2050)
        # Keep Primary Database enabled: it provides the satellite imagery and
        # terrain textures. Disable only annotation/overlay children below.
        # Turning this parent off produces a flat gray Save Image export.
        ws=words(lb)
        for token in ('Survey','Borders','Places','Photos','Roads','Weather','Gallery','More'):
            try:w=find(token,lb,timeout=1)
            except RuntimeError:continue
            cy=round(w['y']+w['h']/2)
            if self.dark((32,cy-4,40,cy+4),threshold=180)>=3:click(36,cy)
        building=next((w for w in ws if any(t in w['text'].lower() for t in ('building','buitaing','buitding','suildings','suidings'))),None)
        if building is None:
            alternate=words(lb,psm=6)
            building=next((w for w in alternate if 'building' in w['text'].casefold()),None)
        if building is None:raise RuntimeError(f'3D Buildings not visible: {[w["text"] for w in ws]}')
        cy=round(building['y']+building['h']/2)
        if self.dark((32,cy-4,40,cy+4),threshold=180)<3:click(36,cy)
        terrain=find('Terrain',lb)
        cy=round(terrain['y']+terrain['h']/2)
        if self.dark((32,cy-4,40,cy+4),threshold=180)<3:click(36,cy)
        try:
            options=find('Options',self.toolbar,timeout=1)
        except RuntimeError:
            # The options row is part of Earth Pro's Save Image panel.
            key('ctrl+alt+s');time.sleep(1)
            options=find_joined(['Options'],self.toolbar,timeout=8)
        click_word(options);time.sleep(.5)
        menu=(320,80,680,600)
        for name in ('Title','Legend','Scale','Compass','HTML'):
            w=find(name,menu)
            cy=round(w['y']+w['h']/2)
            checkbox_x=round(w['x'])-14
            if self.dark((checkbox_x-4,cy-4,checkbox_x+4,cy+5),threshold=100)>=3:
                click(checkbox_x,cy)
        grab().crop(menu).save(self.diagnostics/'map_options.png')
        key('Escape')
        self.expected=(8192,4320) if native=='8k' else (3840,2160)
        # Reuse the currently active native resolution when it already matches.
        # Earth Pro reports it in the toolbar (e.g. `3840x2160 (4K UHD)`),
        # avoiding a brittle menu interaction on already-configured sessions.
        current=' '.join(w['text'] for w in words(self.toolbar,psm=6)).casefold()
        already=('8k' in current) if native=='8k' else ('4k' in current or '3840x2160' in current)
        if not already:
            resolution=find_joined(['Resolution:'],self.toolbar,timeout=8)
            click_word(resolution);time.sleep(.8)
            choice=find('8K' if native=='8k' else '4K',(460,90,690,280),psm=6)
            click_word(choice)
        self.save_button=find('Save',self.toolbar)
        grab().resize((1920,1080)).save(self.diagnostics/'configured.jpg')

    @staticmethod
    def dark(box,threshold=100):
        im=grab().crop(box)
        return sum(max(im.getpixel((x,y))[:3])<threshold for y in range(im.height) for x in range(im.width))

    @staticmethod
    def dismiss_trust_prompt(timeout=10):
        end=time.monotonic()+timeout
        last_text=''
        while time.monotonic()<end:
            modal=visible_window(r'^Google Earth$')
            if not modal:return
            # Qt can map, move and dismiss transient windows before a capture.
            # Re-read the actual screen coordinates after the compositor settles.
            time.sleep(.4)
            if visible_window(r'^Google Earth$')!=modal:continue
            info=run('xwininfo','-id',modal,check=False)
            if info.returncode:continue
            def coord(label):
                return int(re.search(re.escape(label)+r'\s*:\s*(-?\d+)',info.stdout).group(1))
            x,y=coord('Absolute upper-left X'),coord('Absolute upper-left Y')
            w,h=coord('Width'),coord('Height')
            ws=words((max(0,x),max(0,y),x+w,y+h),psm=6)
            if visible_window(r'^Google Earth$')!=modal:continue
            last_text=' '.join(z['text'] for z in ws).lower()
            button=None
            if 'reload file' in last_text and 'edits' in last_text and any(path in last_text for path in ('/root/pairuav_img/', 'dataset_code/build_kml_pair/', '/processed_images', 'dataset_code/gen_img/')):button='Yes'
            elif 'telemetry' in last_text and 'edits' in last_text:button='Yes'
            elif 'sources' in last_text and 'trust' in last_text:button='Open'
            if button=='Yes':
                # Qt's standard Yes button has the Alt+Y mnemonic. Restrict
                # this action to the recognized reload-generated-file prompt.
                focus(modal)
                key('alt+y')
                time.sleep(.4)
                if visible_window(r'^Google Earth$')!=modal:continue
                if 495<=w<=505 and 82<=h<=106 and 'no' in last_text and 'always' in last_text:
                    click(x+w*.416,y+h*.78)
                    time.sleep(.4)
                    continue
            if button:
                choices=[z for z in ws if (z['text']==button or (button=='Open' and z['text'].casefold() in ('oren','0pen'))) and z['y']>y+h*.55]
                if choices:
                    click_word(max(choices,key=lambda z:z['y']))
                    time.sleep(.3)
                    continue
            # The untrusted-source prompt is rendered as a tiny 217x138 Qt
            # dialog at 4K; its text is often too small for OCR. Its Open
            # button is stable in the lower-left button row.
            if 205 <= w <= 230 and 125 <= h <= 150:
                focus(modal)
                click(x + w * .44, y + h * .90)
                time.sleep(.4)
                continue
            time.sleep(.3)
        if visible_window(r'^Google Earth$'):
            raise RuntimeError(f'Unrecognized persistent Google Earth modal: {last_text}')

    def open_view(self,path,name,settle,timeout):
        started=time.monotonic()
        # Normalize stale file dialogs left by interrupted exports.
        for pattern in (r'^Open$',r'^Save As$'):
            modal=visible_window(pattern)
            if modal:focus(modal);key('Escape');wait_window(pattern,False,timeout=5)
        self.dismiss_trust_prompt()
        focus(self.window);key('ctrl+o')
        end=time.monotonic()+20
        while time.monotonic()<end:
            if visible_window(r'^Google Earth$'):
                self.dismiss_trust_prompt()
            if visible_window(r'^Open$'):break
            time.sleep(.2)
        dialog_ready=time.monotonic()
        fill_dialog(r'^Open$',path)
        file_opened=time.monotonic()
        self.dismiss_trust_prompt()
        focus(self.window)
        scene_started=time.monotonic()
        stability=self.wait_stable(max(settle,3),timeout)
        self.last_open_timing={'dialog_ready':round(dialog_ready-started,3),'file_dialog':round(file_opened-dialog_ready,3),'trust_and_focus':round(scene_started-file_opened,3),'scene':stability}

    def verify_camera(self, source, opened_at, timeout=25):
        """Require fresh Earth telemetry at the intended absolute Camera pose."""
        view,_=view_from_kml(source,0)
        if view.tag.rsplit('}',1)[-1]!='Camera':return None
        expected={e.tag.rsplit('}',1)[-1]:e.text for e in view}
        if expected.get('altitudeMode')!='absolute':return None
        path=BASE.parent/'build_kml_pair/.telemetry/latest.json'
        until=time.monotonic()+timeout
        observed=None
        while time.monotonic()<until:
            try:
                observed=json.loads(path.read_text())
                fresh=observed['time']>=opened_at
                close=(abs(observed['cameraLon']-float(expected['longitude']))<1e-6
                       and abs(observed['cameraLat']-float(expected['latitude']))<1e-6
                       and abs(observed['cameraAlt']-float(expected['altitude']))<.15)
                heading=abs((observed['lookatHeading']-float(expected['heading'])+180)%360-180)
                # Earth reports look-at tilt at the surface; curvature adds a
                # small difference from the Camera tilt at the camera position.
                tilt=abs(observed['lookatTilt']-float(expected['tilt']))
                if fresh and close and heading<.2 and tilt<.2:return observed
            except (OSError,ValueError,KeyError,TypeError):pass
            time.sleep(.5)
        raise RuntimeError(f'Camera telemetry does not match requested KML: {observed}')

    def verify_selection(self,name):
        until=time.monotonic()+20
        observed=[]
        while time.monotonic()<until:
            im=grab();bands=[]
            for y in range(100,1900):
                r,g,b=im.getpixel((300,y))[:3]
                if b>120 and g>80 and r<100:
                    if not bands or y>bands[-1][-1]+1:bands.append([])
                    bands[-1].append(y)
            for band in bands:
                if len(band)<8:continue
                label=im.crop((65,band[0]-2,315,band[-1]+3))
                label=label.resize((label.width*6,label.height*6))
                stream=io.BytesIO();label.save(stream,format='PNG')
                result=subprocess.run(['tesseract','stdin','stdout','--psm','7'],input=stream.getvalue(),capture_output=True,check=True).stdout.decode().strip()
                observed.append(result)
                if name in result:return
            time.sleep(.5)
        raise RuntimeError(f'Imported view name {name} not selected: {observed[-6:]}')

    def wait_stable(self,minimum,timeout):
        start=time.monotonic();previous=None;stable=0;delta=None
        while time.monotonic()-start<timeout:
            current=grab().crop((350,160,3800,2070)).resize((640,360)).convert('RGB')
            if previous is not None:
                delta=sum(ImageStat.Stat(ImageChops.difference(current,previous)).mean)/3
                stable=stable+1 if delta<.3 else 0
            elapsed=time.monotonic()-start
            if elapsed>=minimum and stable>=3:return {'elapsed_seconds':round(elapsed,2),'final_mean_pixel_delta':delta}
            previous=current
            time.sleep(.6)
        raise RuntimeError(f'Scene did not settle in {timeout}s (mean pixel difference={delta})')

    def save(self,path):
        # New UUID path cannot be confused with a stale previous output.
        if path.exists():raise RuntimeError(f'Temporary export unexpectedly exists: {path}')
        focus(self.window);click_word(self.save_button)
        fill_dialog('^Save As$',path)
        until=time.monotonic()+120
        while time.monotonic()<until:
            if path.exists():
                try:
                    with Image.open(path) as im:
                        im.load()
                        if im.size!=self.expected:raise RuntimeError(f'Native resolution is {im.size}, expected {self.expected}')
                        center=im.crop((im.width//4,im.height//4,3*im.width//4,3*im.height//4)).convert('RGB')
                        if max(ImageStat.Stat(center).stddev)<5:
                            raise RuntimeError('Native export is blank/flat; refusing to publish it')
                    return
                except (OSError,SyntaxError):pass
            time.sleep(1)
        raise RuntimeError(f'No complete new image exported: {path}')

def clean_4k(source,destination):
    # Crop native pixels without enlargement. Allocate the vertical margin
    # to the bottom: Earth Pro's footer/logo is taller than the old 2% crop.
    with Image.open(source) as im:
        w,h=im.size
        margin=round(w*.02)
        left,right=margin,w-margin
        target_h=round((right-left)*9/16)
        box=(left,0,right,min(h,target_h))
        im.crop(box).save(destination,quality=98,subsampling=0)
    return list(box)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def montage(files,path):
    out=Image.new('RGB',(1280,390*((len(files)+1)//2)),'white')
    for i,p in enumerate(files):
        with Image.open(p) as im:
            im.thumbnail((640,360));out.paste(im,((i%2)*640,(i//2)*390))
        ImageDraw.Draw(out).text(((i%2)*640+8,(i//2)*390+365),p.name,fill='black')
    out.save(path,quality=90)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='Any tour directory containing KML files')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--limit',type=int,help='Omit to render every KML in the tour')
    ap.add_argument('--start',type=int,default=0,help='Sorted file offset')
    ap.add_argument('--view-index',type=int,default=0,help='View within each KML (normally each split file has one)')
    ap.add_argument('--settle',type=float,default=30,help='Minimum wait for camera and imagery, then require stable frames')
    ap.add_argument('--timeout',type=float,default=180)
    ap.add_argument('--native-resolution',choices=['8k','4k'],default='4k')
    ap.add_argument('--hfov',type=float,help='Override KML horizontal field of view (degrees)')
    ap.add_argument('--keep-attribution',action='store_true',help='Deliver native image without edge cropping')
    ap.add_argument('--display',default=':1')
    args=ap.parse_args()
    if args.start<0 or args.view_index<0 or (args.limit is not None and args.limit<1):ap.error('Invalid indices/limit')
    if args.settle<0 or args.timeout<=args.settle:ap.error('Require 0 <= settle < timeout')
    os.environ['DISPLAY']=args.display
    lock=open(BASE/'export.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0, str(BASE.parent/'build_kml_pair'))
    from audit_resources import recover_affinity
    recover_affinity()  # lock held; recover an audit killed before its finally block
    source=args.input.resolve()
    files=sorted(source.glob('*.kml'))[args.start:]
    if args.limit is not None:files=files[:args.limit]
    if not files:ap.error(f'No KML files selected in {source}')
    output=(args.output or Path('/root/pairUAV_img/images')/args.input.name).resolve()
    output.mkdir(parents=True,exist_ok=True)
    run_id=time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    work=output/'runs'/run_id;work.mkdir(parents=True)
    native=output/'native';native.mkdir(exist_ok=True)
    manifest={'exporter_sha256':sha(Path(__file__)), 'status':'running','input':str(source),'engine':'Google Earth Pro native Save Image','run_id':run_id,
        'command':sys.argv,'crop_changes_framing':not args.keep_attribution,'images':[]}
    manifest_path=work/'manifest.json'
    def persist():
        data=json.dumps(manifest,indent=2)
        run_tmp=manifest_path.with_suffix(".tmp")
        run_tmp.write_text(data);run_tmp.replace(manifest_path)
        tmp=output/'manifest.json.tmp'
        tmp.write_text(data); tmp.replace(output/'manifest.json')
    persist()
    prepared=[]
    for i,p in enumerate(files):
        # A unique alphabetic UI label avoids OCR ambiguity in four-digit names.
        label='CAPTURE'+''.join(chr(65+(i//(26**j))%26) for j in (3,2,1,0))
        k=work/(p.stem+'.kml');camera=prepare(p,k,label,args.view_index,args.hfov)
        prepared.append((p,k,label,camera))
    try:
        earth=Earth(work)
        telemetry=BASE.parent/'build_kml_pair/telemetry.kml'
        telemetry_live=False
        try:telemetry_live=time.time()-json.loads((telemetry.parent/'.telemetry/latest.json').read_text())['time']<10
        except (OSError,ValueError,KeyError):pass
        if telemetry.exists() and not telemetry_live:earth.open_view(telemetry,'telemetry',3,45)
        earth.configure(args.native_resolution)
        outputs=[]
        for i,(p,k,label,camera) in enumerate(prepared):
            print(f'[{i+1}/{len(files)}] {args.input.name}/{p.name}: loading exact KML view',flush=True)
            opened_at=time.time()
            step_started=time.monotonic()
            earth.open_view(k,label,args.settle,args.timeout)
            opened_done=time.monotonic()
            observed=earth.verify_camera(k,opened_at) if telemetry.exists() else None
            verified_done=time.monotonic()
            temp=native/(p.stem+'.'+uuid.uuid4().hex+'.jpg')
            earth.save(temp)
            saved_done=time.monotonic()
            raw=native/f'tour{args.input.name}_{p.stem}.jpg';temp.replace(raw)
            final=output/f'tour{args.input.name}_{p.stem}.jpg'
            if args.keep_attribution:
                import shutil
                shutil.copyfile(raw,final);crop=None
            else:crop=clean_4k(raw,final)
            with Image.open(final) as im:dimensions=im.size;im.verify()
            entry={'source':str(p),'source_sha256':sha(p),'prepared_kml':str(k),'camera':camera,
                'observed_camera_telemetry':observed,
                'open_detail':getattr(earth,'last_open_timing',None),
                'earth_service_config_sha256':sha(Path('/etc/supervisor/conf.d/earth-pro.conf')) if Path('/etc/supervisor/conf.d/earth-pro.conf').exists() else None,
                'timing_seconds':{'open_and_settle':round(opened_done-step_started,3),'verify_pose':round(verified_done-opened_done,3),'native_save':round(saved_done-verified_done,3),'crop_and_encode':round(time.monotonic()-saved_done,3)},
                'native':str(raw),'native_dimensions':earth.expected,'native_sha256':sha(raw),
                'output':str(final),'dimensions':dimensions,'crop_box_native_pixels':crop,'sha256':sha(final)}
            manifest['images'].append(entry);persist();outputs.append(final)
            print(f'  OK {final} {dimensions}',flush=True)
        montage(outputs[:20],output/'preview_montage.jpg')
        manifest['status']='complete';persist()
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(f'Completed {len(outputs)} images; manifest: {output / "manifest.json"}',flush=True)
        return 0
    except Exception as exc:
        manifest['status']='failed';manifest['error']=str(exc);persist()
        grab().resize((1920,1080)).save(work/'failure.jpg')
        raise

if __name__=='__main__':raise SystemExit(main())
