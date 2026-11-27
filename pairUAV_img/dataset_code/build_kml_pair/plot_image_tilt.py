#!/usr/bin/env python3
"""Compare absolute camera tilts, counting each image exactly once."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_tilt import validate_image_tilt


def plot(current, output, previous=None):
    d=json.loads(current.read_text());check=validate_image_tilt(d)
    if check is None:raise ValueError('No absolute image tilt constraint')
    edges=np.asarray(check['edges_deg']);centers=(edges[1:]+edges[:-1])/2
    fig,axs=plt.subplots(1,2 if previous else 1,figsize=(13 if previous else 7,4.6),squeeze=False)
    data=[]
    if previous:data.append(('Previous design',json.loads(previous.read_text())))
    data.append(('Uniform absolute image tilt',d))
    all_counts=[]
    for label,design in data:
        counts,_=np.histogram([v['tilt'] for v in design['views']],edges);all_counts.append(counts)
    ymax=max(int(c.max()) for c in all_counts)*1.18
    for ax,(label,design),counts in zip(axs[0],data,all_counts):
        ax.bar(centers,counts,width=np.diff(edges)*.9)
        for x,n in zip(centers,counts):ax.text(x,n+.6,str(n),ha='center',fontsize=9)
        ax.axhline(len(design['views'])/len(counts),color='red',linestyle='--',label='Equal-count target')
        ax.set(title=label,xlabel='Absolute Camera tilt from nadir (degrees)',ylabel='Image count',ylim=(0,ymax),xticks=edges)
        ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle(f'Tour {d["tour_id"]}: absolute image angles (not pair differences)')
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(check,indent=2)+'\n')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--views',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path);a=p.parse_args();plot(a.views,a.output,a.previous)
