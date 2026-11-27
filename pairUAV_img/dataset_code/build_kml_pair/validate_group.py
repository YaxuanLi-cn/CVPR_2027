#!/usr/bin/env python3
"""Independently verify selected records against KML geometry and visual evidence."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from common import AXES,Frame,delta,view_from_file,save_json
from group_batch_pipeline import audit_valid,THRESHOLDS,sha

def validate(root,kroot,iroot,gid,full=True,plots=True):
 report=json.loads((root/f'group_{gid}.json').read_text()); B=report['bins']
 if not __debug__: raise RuntimeError('Validation must not run with Python -O')
 first,last=map(int,gid.split('_')); expected_tours=[f'{i:04d}' for i in range(first,last+1)]
 assert len(expected_tours) in (1,5) and report['tours']==expected_tours
 assert report['status']=='complete' and B>=5
 assert report['tilt_difference_range_deg']==[-45,45]
 bindings={str(root/f'group_{gid}.json'):sha(root/f'group_{gid}.json')}
 hist={k:np.zeros(B,dtype=int) for k in AXES}; records=[]; counts={}; image_tilt_reports={}
 for tour in report['tours']:
  kd=kroot/tour; im=iroot/f'{tour}_redesigned'; ad=root/tour/'visual_audit_redesigned'
  pair_path=root/f'{tour}_group_{gid}.json'
  data=json.loads(pair_path.read_text()); rr=data['records']
  jsonl=root/f'{tour}_group_{gid}.jsonl'
  assert [json.loads(line) for line in jsonl.read_text().splitlines()]==rr
  for path in (pair_path,jsonl,kd/'views.json',ad/'summary.json',ad/'matches.jsonl'):
   bindings[str(path)]=sha(path)
  vd=json.loads((kd/'views.json').read_text()); views=vd['views']; assert len(views)<=150
  if full: assert audit_valid(ad,im,kd,len(views)),f'{tour}: audit hashes invalid'
  passed={}
  for r in map(json.loads,(ad/'matches.jsonl').open()):
   if r['passed']:passed[tuple(sorted((r['a'],r['b'])))]= {k:v for k,v in r.items() if k not in ('points_a','points_b')}
  frame=Frame(vd['anchor']['longitude'],vd['anchor']['latitude'],vd['anchor']['terrain_altitude_msl'])
  actual=[]
  for v in views:
   path=kd/(v['id']+'.kml')
   c=view_from_file(path)
   assert c['type']=='Camera' and c['altitudeMode']=='absolute',str(path)
   for key in ('longitude','latitude','altitude','heading','tilt','roll'):
    assert abs(c.get(key,0)-v.get(key,0))<1e-8,(str(path),key,'KML differs from audited view')
   c['position_enu_m']=frame.from_geo([c[k] for k in ('longitude','latitude','altitude')]).tolist()
   assert np.max(np.abs(np.array(c['position_enu_m'])-v['position_enu_m']))<1e-5,(str(path),'ENU position differs')
   bindings[str(path)]=sha(path)
   actual.append(c)
  from image_tilt import validate_image_tilt
  tilt_check=validate_image_tilt(vd,actual)
  if tilt_check is not None:
   image_tilt_reports[tour]=tilt_check
   if full:
    receipt_path=im/'render_receipt.json'
    receipt=json.loads(receipt_path.read_text())['images']; errors=[]; observed=[]
    for v,c in zip(views,actual):
     entry=receipt[v['id']]
     assert entry['source_sha256']==sha(kd/(v['id']+'.kml'))
     assert entry['sha256']==sha(im/(v['id']+'.jpg'))
     angle=entry['observed_camera_telemetry']['lookatTilt']
     error=abs(angle-c['tilt']);assert error<.2,'Rendered camera tilt differs from KML'
     errors.append(error);observed.append(angle)
    bindings[str(receipt_path)]=sha(receipt_path)
    tilt_check['earth_telemetry_max_tilt_error_deg']=max(errors)
    tilt_check['earth_telemetry_counts']=np.histogram(observed,tilt_check['edges_deg'])[0].tolist()
  if vd.get('max_camera_range_m') is not None:
   assert all(np.linalg.norm(c['position_enu_m'])<=vd['max_camera_range_m']+1e-5 for c in actual)
  seen=set(); degrees=Counter()
  for r in rr:
   assert r['tour_id']==tour
   a,b=r['a'],r['b']; assert 0<=a<len(views) and 0<=b<len(views)
   edge=tuple(sorted((a,b)));assert a!=b and edge not in seen
   seen.add(edge);degrees.update((a,b));assert edge in passed
   ev=passed[edge]
   for k,threshold in [('inliers',20),('inlier_ratio',.25),('hull_fraction',.01),('grid_cells',4)]:
    value=ev[k]; assert min(value if isinstance(value,list) else [value])>=threshold
   expected=delta(actual[a],actual[b])
   for k,(lo,hi) in AXES.items():
    if k=='tilt_deg':lo,hi=report['tilt_difference_range_deg']
    v=r['delta'][k];assert abs(v-expected[k])<1e-5,(tour,a,b,k,v,expected[k])
    assert lo-1e-7<=v<=hi+1e-7
    hist[k][min(B-1,max(0,int((v-lo)/(hi-lo)*B)))]+=1
   for suffix,idx in [('a',a),('b',b)]:
    assert Path(r['image_'+suffix])==im/f'{idx:04d}.jpg'
    assert Path(r['kml_'+suffix])==kd/f'{idx:04d}.kml'
   records.append(r)
  assert len(rr)>=1000 and len(rr)==report['per_tour'][tour]
  assert max(degrees.values())<=150
  if tilt_check is not None:
   assert set(degrees)==set(range(len(views))), 'Uniform image angles require every image to be used by a selected pair'
   tilt_check['unique_images_used_in_pairs']=len(degrees)
  counts[tour]=len(rr)
 for k,h in hist.items():assert h.tolist()==report['group_distribution'][k] and np.all(h==len(records)//B),(k,h)
 assert len(records)==report['pairs']
 result={'status':'validated','group_id':gid,'pairs':len(records),'per_tour':counts,'bins':B,'group_distribution':{k:v.tolist() for k,v in hist.items()},'full_image_hash_check':full,'limitations':'Equal marginal bin counts, not joint or continuous uniformity. Feature matches do not prove semantic building identity or unique metric pose.'}
 result['absolute_image_tilt']=image_tilt_reports
 result['validated_artifact_sha256']=bindings
 if plots:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axs=plt.subplots(2,3,figsize=(15,8));colors=plt.get_cmap('tab10')
  for ax,(k,(lo,hi)) in zip(axs.flat,AXES.items()):
   if k=='tilt_deg':lo,hi=report['tilt_difference_range_deg']
   edges=np.linspace(lo,hi,B+1);bottom=np.zeros(B)
   for i,tour in enumerate(report['tours']):
    h,_=np.histogram([r['delta'][k] for r in records if r['tour_id']==tour],edges)
    ax.bar((edges[:-1]+edges[1:])/2,h,width=np.diff(edges)*.92,bottom=bottom,label=tour,color=colors(i));bottom+=h
   ax.set_title(k);ax.set_xticks(edges);ax.set_ylabel('Pair count');ax.grid(axis='y',alpha=.2)
  axs.flat[5].axis('off');handles,labels=axs.flat[0].get_legend_handles_labels();axs.flat[5].legend(handles,labels,title='Contribution by tour',loc='upper left')
  axs.flat[5].text(0,.35,f'{len(records)} unique unordered pairs\n{B} equal-width marginal bins\nVisual screening: 20 / .25 / .01 / 4\nNot a joint-uniformity guarantee',transform=axs.flat[5].transAxes)
  fig.suptitle(f'Group {gid}: selected visual pairs');fig.tight_layout();fig.savefig(root/f'group_{gid}_distributions.png',dpi=160);plt.close(fig)
 save_json(root/f'group_{gid}_validation.json',result)
 if full and plots and root.resolve()==Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'):
  from publish_group import publish
  publish(root,kroot,iroot,gid)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--json-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'));p.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml_relaxed_batch'));p.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images_relaxed_batch'));a=p.parse_args();print(json.dumps(validate(a.json_root,a.kml_root,a.image_root,a.group),indent=2))
if __name__=='__main__':main()
