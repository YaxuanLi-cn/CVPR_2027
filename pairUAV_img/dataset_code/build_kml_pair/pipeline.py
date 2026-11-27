#!/usr/bin/python3
"""Generate adjusted KML, render 150 images, and publish only validated pairs."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def run(cmd): print('+',' '.join(map(str,cmd)),flush=True);subprocess.run(cmd,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--tour',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--images',type=int,default=150);p.add_argument('--pairs',type=int,default=1200);p.add_argument('--bins',type=int,default=12);p.add_argument('--settle',type=float,default=20);p.add_argument('--timeout',type=float,default=180);a=p.parse_args();b=Path('/root/pairUAV_img/dataset_code/build_kml_pair');py=str(b/'.venv/bin/python');
 run([py,str(b/'build_pairs.py'),'--tour',a.tour,'--seed',str(a.seed),'--images',str(a.images),'--pairs',str(a.pairs),'--bins',str(a.bins)])
 run([py,str(b/'render_pairs.py'),'--tour',a.tour,'--settle',str(a.settle),'--timeout',str(a.timeout)])
 run([py,str(b/'audit_visual_pairs.py'),'--tour',a.tour])
 run([py,str(b/'select_visual_pairs.py'),'--tour',a.tour,'--count',str(a.pairs),'--bins',str(a.bins),'--step','0.25'])
 run([py,str(b/'validate_pairs.py'),'--tour',a.tour,'--require-images'])
 run([py,str(b/'plot_distributions.py'),'--tour',a.tour])
if __name__=='__main__':main()
