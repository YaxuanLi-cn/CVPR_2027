#!/usr/bin/env python3
"""Audit actual exported images before distribution balancing. Never certifies semantic identity or metric observability."""
from __future__ import annotations
import argparse, json, hashlib, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from common import AXES, bin_ids, save_json, sha

VERSION='sift-mutual-ratio-v1'
THRESHOLDS={'ratio_test':0.75,'min_inliers':40,'min_inlier_ratio':0.45,'min_hull_fraction':0.04,'min_grid_cells':6,'ransac_pixels':1.5}

def features(path, cache):
    key=hashlib.sha256((VERSION+sha(path)).encode()).hexdigest()
    dest=cache/(key+'.npz')
    if dest.exists():
        d=np.load(dest);return d['points'],d['descriptors']
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError(f'Cannot read {path}')
    image=cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
    kp,desc=cv2.SIFT_create(nfeatures=4000).detectAndCompute(image,None)
    pts=np.array([p.pt for p in kp],dtype=np.float32).reshape(-1,2)
    if desc is None:desc=np.empty((0,128),np.float32)
    np.savez_compressed(dest,points=pts,descriptors=desc)
    return pts,desc

def match(edge, fs):
    a,b=edge['a'],edge['b'];pa,da=fs[a];pb,db=fs[b]
    result={'a':a,'b':b,'delta':edge['delta'],'passed':False,'matches':0,'inliers':0,'inlier_ratio':0.,'hull_fraction':0.,'grid_cells':0,'model':None,'points_a':[],'points_b':[]}
    if min(len(da),len(db))<8:return result
    # BF is deterministic; reciprocal Lowe-ratio matching suppresses repeated windows.
    bf=cv2.BFMatcher(cv2.NORM_L2)
    ab={m.queryIdx:m.trainIdx for pair in bf.knnMatch(da,db,k=2) if len(pair)==2 for m,n in [pair] if m.distance<THRESHOLDS['ratio_test']*n.distance}
    ba={m.queryIdx:m.trainIdx for pair in bf.knnMatch(db,da,k=2) if len(pair)==2 for m,n in [pair] if m.distance<THRESHOLDS['ratio_test']*n.distance}
    ids=[(i,j) for i,j in ab.items() if ba.get(j)==i];result['matches']=len(ids)
    if len(ids)<8:return result
    p=np.float32([pa[i] for i,j in ids]);q=np.float32([pb[j] for i,j in ids])
    cv2.setRNGSeed(a*1000+b)
    try:
        F,m=cv2.findFundamentalMat(p,q,cv2.USAC_MAGSAC,THRESHOLDS['ransac_pixels'],.999,10000)
    except cv2.error:
        return result
    if m is None:return result
    keep=m.ravel().astype(bool);p=p[keep];q=q[keep];n=len(p)
    if n<3:return result
    hull=min(cv2.contourArea(cv2.convexHull(p)),cv2.contourArea(cv2.convexHull(q)))/(1280*720)
    cells=lambda pts: len(set((int(x//320),int(y//180)) for x,y in pts))
    grid=min(cells(p),cells(q));ratio=n/len(ids)
    result.update(inliers=n,inlier_ratio=ratio,hull_fraction=hull,grid_cells=grid,model='fundamental',points_a=p.tolist(),points_b=q.tolist())
    result['passed']=bool(n>=THRESHOLDS['min_inliers'] and ratio>=THRESHOLDS['min_inlier_ratio'] and hull>=THRESHOLDS['min_hull_fraction'] and grid>=THRESHOLDS['min_grid_cells'])
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tour',required=True);ap.add_argument('--min-inliers',type=int,default=THRESHOLDS['min_inliers']);ap.add_argument('--min-inlier-ratio',type=float,default=THRESHOLDS['min_inlier_ratio']);ap.add_argument('--min-hull-fraction',type=float,default=THRESHOLDS['min_hull_fraction']);ap.add_argument('--min-grid-cells',type=int,default=THRESHOLDS['min_grid_cells']);ap.add_argument('--kml-root',type=Path,default=Path('/root/pairUAV_img/processed_kml'));ap.add_argument('--image-root',type=Path,default=Path('/root/pairUAV_img/processed_images'));ap.add_argument('--output-root',type=Path,default=Path('/root/pairUAV_img/processed_paired_json'));ap.add_argument('--workers',type=int,default=8);ap.add_argument('--output-dir',type=Path);ap.add_argument('--image-pattern',default='{id}.jpg')
    ap.add_argument('--bins',type=int,default=5);ap.add_argument('--tilt-limit',type=float,default=45);ap.add_argument('--requested-pairs',type=int,default=1000)
    args=ap.parse_args(); THRESHOLDS.update(min_inliers=args.min_inliers,min_inlier_ratio=args.min_inlier_ratio,min_hull_fraction=args.min_hull_fraction,min_grid_cells=args.min_grid_cells); tour=args.tour.zfill(4);kroot=args.kml_root/tour; d=json.loads((kroot/'views.json').read_text());out=args.output_dir or (args.output_root/tour/'visual_audit');out.mkdir(parents=True,exist_ok=True);cache=out/'features';cache.mkdir(exist_ok=True)
    cv2.setNumThreads(1);paths=[(args.image_root/tour/args.image_pattern.format(id=v['id']) if args.output_dir is None else args.image_root/args.image_pattern.format(id=v['id'])) for v in d['views']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:fs=list(pool.map(lambda p:features(p,cache),paths))
    from partial_visual_audit import read_cache
    prior=read_cache(out,kroot/'views.json',paths,THRESHOLDS)
    print(f'Reusing {len(prior)} matching partial-audit records',flush=True)
    print(f'Extracted SIFT: {len(fs)} images, {len(d["candidate_edges"])} candidates',flush=True)
    records=[]
    with (out/'matches.jsonl.tmp').open('w') as log, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,r in enumerate(pool.map(lambda e:prior[(e['a'],e['b'])] if (e['a'],e['b']) in prior else match(e,fs),d['candidate_edges'])):
            log.write(json.dumps(r,separators=(',',':'))+'\n');records.append(r)
            if (i+1)%100==0:log.flush();print(f'{i+1}/{len(d["candidate_edges"])} checked; {sum(x["passed"] for x in records)} passed',flush=True)
    (out/'matches.jsonl.tmp').replace(out/'matches.jsonl')
    bins=args.bins;accepted=[r for r in records if r['passed']];hist={k:[0]*bins for k in AXES};oriented={k:[0]*bins for k in AXES}
    ranges={**AXES,'tilt_deg':(-args.tilt_limit,args.tilt_limit)}
    eligible=[r for r in accepted if all(lo-1e-9<=r['delta'][k]<=hi+1e-9 for k,(lo,hi) in ranges.items())]
    for r in eligible:
        for k,(lo,hi) in ranges.items():
            for direction in (1,-1):
                j=min(bins-1,max(0,int((direction*r['delta'][k]-lo)/(hi-lo)*bins)))
                oriented[k][j]+=1
                if direction==1:hist[k][j]+=1
    # Optimistic single-tour bound; group selection does not require each
    # individual tour to have a uniform histogram.
    bound=min(len(eligible),min(min(v)*bins for v in oriented.values()))
    summary={'schema_version':VERSION,'tour_id':tour,'status':'visual_screen_only','images':len(paths),'candidate_pairs':len(records),'accepted_pairs':len(accepted),'thresholds':THRESHOLDS,'bins':bins,'accepted_histograms':hist,'available_oriented_histograms':oriented,'uniform_pair_count_upper_bound':bound,'requested_pairs':args.requested_pairs,'axis_ranges':ranges,'eligible_visual_pairs':len(eligible),'matches_sha256':sha(out/'matches.jsonl'),'histogram_scope':'Single-tour candidate supply, not a per-tour balance requirement; final acceptance is at group level','image_sha256':{p.name:sha(p) for p in paths},'views_sha256':sha(kroot/'views.json'),'limitations':['Feature correspondences are a necessary screening signal, not proof of unique pose or metric scale.','No semantic first-key building segmentation or occlusion validation is supplied.','Fundamental-matrix inliers may be degenerate on planar scenes; pose observability requires separate evaluation.','Uniform marginal histograms do not imply a uniform joint distribution.']}
    save_json(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='image_sha256'},indent=2))
    plot(summary,out)
    montage(records,paths,out)
    from cleanup_partial_audit import prune
    prune(out)

def plot(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(16,9));bins=s['bins']
    for ax,(k,(lo,hi)) in zip(axs.flat,s.get('axis_ranges',AXES).items()):
        edges=np.linspace(lo,hi,bins+1);x=(edges[:-1]+edges[1:])/2
        ax.bar(x,s['available_oriented_histograms'][k],width=(hi-lo)/bins*.9,label='Available after visual screen (both orientations)')
        ax.axhline(s['requested_pairs']/bins,color='red',linestyle='--',label='Equal-share guide (not per-tour requirement)');ax.set(xlabel=k,ylabel='Candidate orientations',xlim=(lo,hi));ax.legend(fontsize=7)
    axs.flat[-1].axis('off');axs.flat[-1].text(0,1,f'Tour {s["tour_id"]}\nImages: {s["images"]}\nCandidates: {s["candidate_pairs"]}\nVisual pass: {s["accepted_pairs"]}\nSingle-tour optimistic bound: {s["uniform_pair_count_upper_bound"]}\n\nVisual screening is not semantic or metric-pose proof.',va='top',fontsize=11)
    fig.suptitle('Actual-image visual audit: availability before balancing');fig.tight_layout();fig.savefig(out/'visual_distribution.png',dpi=180);fig.savefig(out/'visual_distribution.pdf');plt.close(fig)

def montage(records,paths,out):
    from PIL import Image,ImageDraw
    good=sorted((r for r in records if r['passed']),key=lambda r:abs(r['delta']['tilt_deg']),reverse=True)[:4]
    bad=sorted((r for r in records if not r['passed']),key=lambda r:abs(r['delta']['tilt_deg']),reverse=True)[:4]
    for label,rows in [('accepted',good),('rejected',bad)]:
        canvas=Image.new('RGB',(1280,400*len(rows)),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows):
            for j,idx in enumerate((r['a'],r['b'])):
                with Image.open(paths[idx]) as im:canvas.paste(im.resize((640,360)),(j*640,i*400))
            for p,q in list(zip(r['points_a'],r['points_b']))[::max(1,r['inliers']//25)]:
                draw.line((p[0]/2,i*400+p[1]/2,640+q[0]/2,i*400+q[1]/2),fill=(70,255,70),width=1)
            draw.text((8,i*400+365),f'{label}: {r["a"]:04d} -> {r["b"]:04d}; inliers {r["inliers"]}; tilt {r["delta"]["tilt_deg"]:.1f}; heading {r["delta"]["heading_deg"]:.1f}',fill='black')
        if rows:canvas.save(out/f'{label}_examples.jpg',quality=92)

if __name__=='__main__':
    from audit_resources import audit_resources
    with audit_resources():
        main()
