#!/usr/bin/env python3
"""Sample real rendered images across absolute tilt bins, with labels below."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

def main():
    p=argparse.ArgumentParser();p.add_argument('--views',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.views.read_text());views=data['views']
    candidates=[v for v in views if (a.images/(v['id']+'.jpg')).is_file()]
    if len(candidates)!=len(views):raise ValueError('Wait for every final image')
    # Two samples per 9-degree band; retain exact numeric angles on the sheet.
    chosen=[]
    for lo in range(0,45,9):
        bucket=[v for v in candidates if lo<=v['tilt']<lo+9]
        for target in (lo+2.25,lo+6.75):
            v=min((v for v in bucket if v not in chosen),key=lambda v:abs(v['tilt']-target));chosen.append(v)
    sheet=Image.new('RGB',(1280,5*400),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for i,v in enumerate(chosen):
        x=(i%2)*640;y=(i//2)*400
        with Image.open(a.images/(v['id']+'.jpg')) as im:
            im.thumbnail((640,360));sheet.paste(im,(x,y))
        draw.text((x+10,y+365),f"{v['id']} | tilt {v['tilt']:.2f} deg | heading {v['heading']:.1f} deg",font=font,fill='black')
    a.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(a.output,quality=92)
    a.output.with_suffix('.json').write_text(json.dumps([{'id':v['id'],'tilt_deg':v['tilt'],'heading_deg':v['heading']} for v in chosen],indent=2)+'\n')
    print(a.output)
if __name__=='__main__':main()
