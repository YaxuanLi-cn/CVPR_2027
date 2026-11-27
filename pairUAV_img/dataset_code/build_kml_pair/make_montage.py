#!/usr/bin/env python3
from pathlib import Path
from PIL import Image,ImageDraw
import argparse
p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--count',type=int,default=50);a=p.parse_args();d=Path('/root/pairUAV_img/processed_images')/str(a.tour).zfill(4);files=sorted(d.glob('[0-9][0-9][0-9][0-9].jpg'))[:a.count];cols=5;w,h=320,180;out=Image.new('RGB',(cols*w,((len(files)+cols-1)//cols)*(h+22)),'white');dr=ImageDraw.Draw(out)
for i,p in enumerate(files):
 im=Image.open(p);im.thumbnail((w,h));x=(i%cols)*w;y=(i//cols)*(h+22);out.paste(im,(x,y));dr.text((x+2,y+h+2),p.name,fill='black')
out.save(d/'preview_all.jpg',quality=90)
