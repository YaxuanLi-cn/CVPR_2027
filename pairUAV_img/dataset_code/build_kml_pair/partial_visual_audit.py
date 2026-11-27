#!/usr/bin/env python3
"""Audit available verified images, keeping unknown edges optimistic for capacity checks."""
import argparse,json,fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import audit_visual_pairs as visual
from common import sha,save_json
from group_batch_pipeline import verified_images,THRESHOLDS


def read_cache(path, views_path, paths, thresholds):
    """Reject incomplete snapshots, stale bytes, changed geometry and duplicate edges."""
    try:
        meta=json.loads((path/'partial_summary.json').read_text())
        if meta['schema_version']!=visual.VERSION or meta['thresholds']!=thresholds or meta['views_sha256']!=sha(views_path):return {}
        matches=path/'partial_matches.jsonl'
        if meta.get('matches_sha256') and meta['matches_sha256']!=sha(matches):return {}
        design=json.loads(views_path.read_text())
        candidates={(e['a'],e['b']):e['delta'] for e in design['candidate_edges']}
        unchanged={i for i,p in enumerate(paths) if p.is_file() and meta['image_sha256'].get(p.name)==sha(p)}
        result={};seen=set();accepted=0
        with matches.open() as f:
            for line in f:
                r=json.loads(line);key=(r['a'],r['b'])
                if key in seen or key not in candidates or r['delta']!=candidates[key]:return {}
                if type(r['passed']) is not bool:return {}
                seen.add(key);accepted+=r['passed']
                if r['a'] in unchanged and r['b'] in unchanged:result[key]=r
        if len(seen)!=meta['checked_edges'] or accepted!=meta['accepted_edges']:return {}
        return result
    except (OSError,ValueError,KeyError,TypeError):return {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tour',required=True);p.add_argument('--kml-root',type=Path,required=True)
    p.add_argument('--image-root',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();tour=a.tour.zfill(4)
    kd=a.kml_root/tour;idir=a.image_root/f'{tour}_redesigned';out=a.audit_root/tour/'visual_audit_redesigned'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'partial.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cache=out/'features';cache.mkdir(exist_ok=True)
    d=json.loads((kd/'views.json').read_text());paths=[idir/(v['id']+'.jpg') for v in d['views']]
    done=verified_images(idir,kd,len(paths));visual.THRESHOLDS.update(THRESHOLDS)
    prior=read_cache(out,kd/'views.json',paths,visual.THRESHOLDS)
    edges=[e for e in d['candidate_edges'] if e['a'] in done and e['b'] in done]
    missing=[e for e in edges if (e['a'],e['b']) not in prior]
    cv2.setNumThreads(1);fs=[None]*len(paths)
    needed=sorted({i for e in missing for i in (e['a'],e['b'])})
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,f in zip(needed,pool.map(lambda i:visual.features(paths[i],cache),needed)):fs[i]=f
    print(f'{tour}: {len(done)} verified images; {len(edges)} known-endpoint edges; {len(missing)} new checks',flush=True)
    temp=out/'partial_matches.jsonl.tmp'
    with temp.open('w') as f,ThreadPoolExecutor(max_workers=a.workers) as pool:
        for r in prior.values():f.write(json.dumps(r,separators=(',',':'))+'\n')
        for n,r in enumerate(pool.map(lambda e:visual.match(e,fs),missing),1):
            prior[(r['a'],r['b'])]=r;f.write(json.dumps(r,separators=(',',':'))+'\n')
            if n%200==0:print(f'{n}/{len(missing)} new checks',flush=True)
    temp.replace(out/'partial_matches.jsonl')
    summary={'schema_version':visual.VERSION,'tour_id':tour,'views_sha256':sha(kd/'views.json'),'thresholds':visual.THRESHOLDS,
             'matches_sha256':sha(out/'partial_matches.jsonl'),
             'image_sha256':{paths[i].name:sha(paths[i]) for i in done},'checked_edges':len(prior),
             'accepted_edges':sum(r['passed'] for r in prior.values()),'verified_images':len(done),
             'limitations':'Unknown image pairs remain optimistic. This is not final visual validation.'}
    save_json(out/'partial_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='image_sha256'},indent=2))

if __name__=='__main__':
    from audit_resources import audit_resources
    with audit_resources():
        main()
