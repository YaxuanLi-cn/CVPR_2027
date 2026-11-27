#!/usr/bin/env python3
"""Show absolute image tilt and pair deltas on separate, labelled axes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import view_from_file,AXES


def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--json-root',type=Path,required=True);p.add_argument('--kml-root',type=Path,required=True);a=p.parse_args()
 report=json.loads((a.json_root/f'group_{a.group}.json').read_text())
 validation=json.loads((a.json_root/f'group_{a.group}_validation.json').read_text())
 if validation['status']!='validated':raise ValueError('Validated group required')
 tours=report['tours'];data={};angles={}
 for t in tours:
  data[t]=json.loads((a.json_root/f'{t}_group_{a.group}.json').read_text())['records']
  angles[t]=[view_from_file(k)['tilt'] for k in sorted((a.kml_root/t).glob('*.kml'))]
 fig,axs=plt.subplots(2,3,figsize=(16,9));colors=plt.get_cmap('tab10')
 specs=[('absolute_image_tilt',0,45,10)]+[(k,*([-45,45] if k=='tilt_deg' else limits),report['bins']) for k,limits in AXES.items()]
 for ax,(key,lo,hi,bins) in zip(axs.flat,specs):
  edges=np.linspace(lo,hi,bins+1);bottom=np.zeros(bins,dtype=int)
  for i,t in enumerate(tours):
   values=angles[t] if key=='absolute_image_tilt' else [r['delta'][key] for r in data[t]]
   h,_=np.histogram(values,edges)
   ax.bar((edges[:-1]+edges[1:])/2,h,bottom=bottom,width=np.diff(edges)*.9,color=colors(i),label=t);bottom+=h
  if not np.all(bottom==bottom[0]):raise ValueError(f'Unexpected histogram: {key} {bottom}')
  ax.set(title='Absolute image tilt (one count per image)' if key=='absolute_image_tilt' else f'Pair delta: {key}',ylabel='Images' if key=='absolute_image_tilt' else 'Pairs',xticks=edges)
  ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
 handles,labels=axs.flat[0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=len(tours))
 fig.suptitle(f'{a.group}: image-angle and selected-pair constraints, counted separately')
 fig.tight_layout(rect=(0,.05,1,.95));out=a.json_root/f'group_{a.group}_acceptance.png';fig.savefig(out,dpi=170);plt.close(fig);print(out)

if __name__=='__main__':main()
