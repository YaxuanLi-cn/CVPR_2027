#!/usr/bin/env python3
"""Tour-disjoint RASR-style smoke training/evaluation on pairUAV-5D-temp.

Uses the public RASR repository's deterministic 14-value-per-image smoke
features, then fits distance/tilt MLPs and circular sin/cos heading ridge heads.
This deliberately does not claim the production 422-D MASt3R feature path.
"""
from __future__ import annotations
import hashlib, json, math, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path('/root/pairUAV-5D-temp')
OUT=Path('/root/rasr-pairuav/outputs/pairuav5d_smoke')
SEED=2026
PER_TOUR=3
MAX_TOURS=600
TEST_FRACTION=.2


def image_stats(path: Path):
    with Image.open(path) as im:
        # The public smoke path computes RGB summary stats; downsample first to
        # bound cost on the high-resolution 5D-temp imagery.
        im=im.convert('RGB')
        im.thumbnail((256,256),Image.Resampling.BILINEAR)
        arr=np.asarray(im,dtype=np.float32)/255.0
    flat=arr.reshape(-1,3)
    return np.asarray([arr.shape[1],arr.shape[0],*flat.mean(0),*flat.std(0),*np.percentile(flat,10,axis=0),*np.percentile(flat,90,axis=0)],dtype=np.float32)

def split(tour):
    h=int(hashlib.sha256(('split'+str(SEED)+tour).encode()).hexdigest()[:8],16)/0xffffffff
    return 'test' if h<TEST_FRACTION else 'train'

def angle_error(a,b):
    return np.abs((a-b+180.0)%360.0-180.0)

def main():
    rng=random.Random(SEED)
    records=[]
    files=sorted((ROOT/'pairs').glob('*.jsonl'))
    files=sorted(files,key=lambda p:hashlib.sha256((str(SEED)+p.stem).encode()).hexdigest())[:MAX_TOURS]
    for p in files:
        tour=p.stem
        # Reservoir sample exactly up to PER_TOUR valid, labeled pairs per tour.
        reservoir=[]; seen=0
        with p.open() as f:
            for line in f:
                try: r=json.loads(line)
                except Exception: continue
                d=r.get('delta') or {}
                vals=[d.get(k) for k in ('x_m','y_m','z_m','tilt_deg','heading_deg')]
                if any(v is None for v in vals): continue
                try: vals=[float(v) for v in vals]
                except (ValueError,TypeError): continue
                if not all(math.isfinite(v) for v in vals): continue
                seen+=1
                if len(reservoir)<PER_TOUR: reservoir.append(r)
                else:
                    j=rng.randrange(seen)
                    if j<PER_TOUR: reservoir[j]=r
        records.extend((tour,r) for r in reservoir)
    if not records: raise RuntimeError('No labeled pairs found')
    cache={}
    xs=[]; targets=[]; splits=[]; metadata=[]
    for i,(tour,r) in enumerate(records,1):
        paths=[ROOT/'images'/tour/Path(r[k]).name for k in ('image_a','image_b')]
        stats=[]
        for p in paths:
            key=str(p)
            if key not in cache: cache[key]=image_stats(p)
            stats.append(cache[key])
        a,b=stats
        x=np.concatenate([a,b,np.abs(a-b)])
        d=r['delta']; dx,dy,dz=(float(d[k]) for k in ('x_m','y_m','z_m'))
        distance=math.sqrt(dx*dx+dy*dy+dz*dz)
        xs.append(x)
        targets.append([distance,float(d['tilt_deg']),float(d['heading_deg'])])
        splits.append(split(tour))
        metadata.append({'tour_id':tour,'pair_id':r.get('pair_id',''),'image_a':r['image_a'],'image_b':r['image_b']})
        if i%500==0: print(f'features {i}/{len(records)} unique_images={len(cache)}',flush=True)
    X=np.asarray(xs,dtype=np.float32); Y=np.asarray(targets,dtype=np.float64); S=np.asarray(splits)
    tr=S=='train'; te=S=='test'
    if tr.sum()<100 or te.sum()<100: raise RuntimeError(f'insufficient split: train={tr.sum()} test={te.sum()}')
    scaler=StandardScaler().fit(X[tr]); Xtr=scaler.transform(X[tr]); Xte=scaler.transform(X[te])
    models={}
    # Same compact MLP-head family as the released implementation; target-specific heads.
    for j,name in enumerate(('distance','tilt')):
        model=MLPRegressor(hidden_layer_sizes=(256,128),activation='relu',solver='adam',alpha=1e-4,batch_size=128,learning_rate_init=3e-4,max_iter=120,early_stopping=True,validation_fraction=.1,n_iter_no_change=12,random_state=SEED+j)
        model.fit(Xtr,Y[tr,j]); models[name]=model
    # The released public heading baseline fits sin/cos with ridge regression.
    theta=np.deg2rad(Y[tr,2]); hm_s=Ridge(alpha=1e-2).fit(Xtr,np.sin(theta)); hm_c=Ridge(alpha=1e-2).fit(Xtr,np.cos(theta))
    pred_d=models['distance'].predict(Xte)
    pred_t=models['tilt'].predict(Xte)
    pred_h=(np.rad2deg(np.arctan2(hm_s.predict(Xte),hm_c.predict(Xte)))+360)%360
    metrics={
      'MAE_distance_m':float(np.mean(np.abs(pred_d-Y[te,0]))),
      'MAE_tilt_deg':float(np.mean(np.abs(pred_t-Y[te,1]))),
      'MAE_heading_deg_circular':float(np.mean(angle_error(pred_h,Y[te,2]))),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump({'feature_scaler':scaler,'distance':models['distance'],'tilt':models['tilt'],'heading_sin':hm_s,'heading_cos':hm_c,'seed':SEED},OUT/'model.joblib')
    report={'method':'RASR public smoke-feature path adapted to pairUAV-5D-temp','seed':SEED,'sampling':f'{PER_TOUR} uniformly reservoir-sampled labeled pairs per tour from {MAX_TOURS} seeded tours','split':'tour-disjoint SHA256 split with independent salt; 20% test by design','rows_total':len(records),'rows_train':int(tr.sum()),'rows_test':int(te.sum()),'train_tours':len(set(m['tour_id'] for m,s in zip(metadata,S) if s=='train')),'test_tours':len(set(m['tour_id'] for m,s in zip(metadata,S) if s=='test')),'feature_dim':int(X.shape[1]),'feature_definition':'14 RGB/size image statistics per image, RGB stats computed after 256x256 max-side downsampling; [A,B,abs(A-B)]','targets':'distance=sqrt(x_m^2+y_m^2+z_m^2), tilt_deg, heading_deg','heading_metric':'circular absolute angular error in degrees','metrics':metrics,'limitations':'Smoke run only: sampled rows and public smoke image-statistics features; production RASR uses frozen MASt3R 422-D features and PairUAV protocol-specific calibration.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
