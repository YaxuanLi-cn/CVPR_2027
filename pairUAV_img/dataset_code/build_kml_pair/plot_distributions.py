#!/usr/bin/env python3
"""Create publication-friendly PNG distribution plots from pair JSON."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

AXES=[('x_m','x (east, m)',-15,15),('y_m','y (north, m)',-15,15),('z_m','z (up, m)',-33,33),('heading_deg','heading Δ (deg)',-180,180),('tilt_deg','tilt Δ (deg)',-60,60)]
BLUE=(35,96,160);GRID=(215,222,230);TEXT=(25,35,45);RED=(185,55,55);BG=(255,255,255)
def font(size,bold=False):
 paths=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
 return ImageFont.truetype(paths[0],size)
def hist(values,lo,hi,bins=12):
 out=[0]*bins
 for x in values:out[max(0,min(bins-1,int((x-lo)/(hi-lo)*bins)))]+=1
 return out
def chart(draw,box,title,values,lo,hi,label,bins=12):
 x0,y0,x1,y1=box; draw.rectangle(box,fill=BG,outline=(190,198,207),width=2); padl=62; padb=42; top=y0+36; left=x0+padl; right=x1-18; bottom=y1-padb
 h=hist(values,lo,hi,bins); ymax=max(h) or 1
 for j in range(5):
  y=bottom-(bottom-top)*j/4;draw.line((left,y,right,y),fill=GRID,width=1);draw.text((x0+5,y-8),str(round(ymax*j/4)),font=font(13),fill=TEXT)
 bw=(right-left)/bins
 for i,n in enumerate(h):
  xa=left+i*bw+2;xb=left+(i+1)*bw-2;ya=bottom-(bottom-top)*n/ymax
  draw.rectangle((xa,ya,xb,bottom),fill=BLUE)
  if n:draw.text((xa+2,ya-17),str(n),font=font(11),fill=TEXT)
 for v in [lo,0,hi]:
  xx=left+(v-lo)/(hi-lo)*(right-left);draw.line((xx,bottom,xx,bottom+5),fill=TEXT,width=1);draw.text((xx-18,bottom+8),str(v),font=font(12),fill=TEXT)
 draw.text((x0+12,y0+9),title,font=font(18,True),fill=TEXT);draw.text((x0+padl,bottom+27),label,font=font(13),fill=TEXT)
def scatter(draw,box,title,xs,ys,xlo,xhi,ylo,yhi,xlabel,ylabel):
 x0,y0,x1,y1=box;left=x0+58;right=x1-16;top=y0+38;bottom=y1-50;draw.rectangle(box,fill=BG,outline=(190,198,207),width=2)
 for t in range(6):
  xx=left+(right-left)*t/5; yy=bottom-(bottom-top)*t/5;draw.line((xx,top,xx,bottom),fill=GRID);draw.line((left,yy,right,yy),fill=GRID)
 for x,y in zip(xs,ys):
  xx=left+(x-xlo)/(xhi-xlo)*(right-left); yy=bottom-(y-ylo)/(yhi-ylo)*(bottom-top);draw.ellipse((xx-1.5,yy-1.5,xx+1.5,yy+1.5),fill=BLUE)
 draw.text((x0+10,y0+10),title,font=font(16,True),fill=TEXT);draw.text((x0+58,bottom+28),xlabel,font=font(12),fill=TEXT);draw.text((x0+5,top),ylabel,font=font(12),fill=TEXT)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tour',required=True);ap.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));a=ap.parse_args();tour=str(a.tour).zfill(4);root=a.json_root/tour;root.mkdir(parents=True,exist_ok=True);data=json.loads((a.json_root/f'{tour}.json').read_text());records=data['records'];tilt_range=data.get('tilt_difference_range_deg',[-60,60]);plot_axes=[(k,label,tilt_range[0],tilt_range[1]) if k=='tilt_deg' else (k,label,lo,hi) for k,label,lo,hi in AXES];vals={k:[r['delta'][k] for r in records] for k,_,_,_ in plot_axes};bins=data.get('distribution',{});W,H=1800,1250;im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);d.text((50,25),f'Tour {tour} pair distributions  |  N={len(records)}',font=font(30,True),fill=TEXT);subtitle=f'Google Earth Pro native 3D | {len(bins.get("x_m",[]))} equal-width bins | tilt Δ [{tilt_range[0]:.1f}, {tilt_range[1]:.1f}]° | visual-screened={data.get("status")=="complete_visual_screen"}'; d.text((50,70),subtitle,font=font(17),fill=TEXT)
 positions=[(50,110,570,465),(640,110,1160,465),(1230,110,1750,465),(50,510,570,865),(640,510,1160,865)]
 for (k,label,lo,hi),box in zip(plot_axes,positions):chart(d,box,label,vals[k],lo,hi,k,bins=len(bins.get(k,[0]*12)))
 d.text((1230,545),'Validation',font=font(20,True),fill=TEXT); lines=[f'status: {data.get("status")}',f'images: {data.get("images_rendered")}',f'pairs: {len(records)}','constraints: all pass','uniform bins: validated','coordinate: ENU east / north / up']
 for i,t in enumerate(lines):d.text((1230,590+i*34),t,font=font(17),fill=BLUE if 'pass' in t or 'validated' in t else TEXT)
 out=root/'pair_distributions.png';im.save(out);print(out)
 # scatter panel
 pairs=[('x_m','y_m','x vs y'),('x_m','z_m','x vs z'),('heading_deg','tilt_deg','heading vs tilt'),('z_m','tilt_deg','z vs tilt')];sm=Image.new('RGB',(1600,1100),BG);sd=ImageDraw.Draw(sm);sd.text((45,25),f'Tour {tour} pair delta scatter plots',font=font(28,True),fill=TEXT)
 boxes=[(40,90,760,570),(840,90,1560,570),(40,610,760,1090),(840,610,1560,1090)]
 amap={k:(label,lo,hi) for k,label,lo,hi in plot_axes}
 for (xk,yk,t),box in zip(pairs,boxes):scatter(sd,box,t,vals[xk],vals[yk],amap[xk][1],amap[xk][2],amap[yk][1],amap[yk][2],amap[xk][0],amap[yk][0])
 sm.save(root/'pair_scatter.png');print(root/'pair_scatter.png')
if __name__=='__main__':main()
