"""Deterministic geometry repair preserving every rendered camera.

Trial designs never touch live KML. Commit only a validated feasible trial.
No image generation, threshold relaxation or changes to rendered cameras.
"""
import copy
import fcntl
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from balance import candidates
from common import Frame, save_json, sha
from design_views import emit

BASE = Path(__file__).resolve().parent


def has_images(root, tour):
    return any((root / f'{tour}_redesigned').rglob('*.jpg'))


def revise(design, attempt, diagnostics):
    d = copy.deepcopy(design)
    digest = hashlib.sha256(f"{d['seed_sha256']}:capacity-repair-v1:{attempt}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    views = d['views']; n = len(views)
    anchor = d['anchor']
    frame = Frame(anchor['longitude'], anchor['latitude'], anchor['terrain_altitude_msl'])
    center = float(d.get('camera_height_center_m',np.mean([v['position_enu_m'][2] for v in views])))
    # Different deterministic schedules improve supply without reducing ranges.
    # Tail-heavy tilts address the large-tilt-bin shortage seen in visual audits.
    edge_fraction = (.8, .9, .7, .85)[(attempt-1) % 4]
    heading_groups = (10, 12, 15, 18)[(attempt-1) % 4]
    xyz = np.empty((n, 3))
    for axis, (lo, hi) in enumerate(((-15,15),(-15,15),(center-16.5,center+16.5))):
        values = lo + (np.arange(n)+rng.random(n))*(hi-lo)/n
        rng.shuffle(values); xyz[:,axis] = values
    order = rng.permutation(n)
    for gi, ids in enumerate(np.array_split(order, heading_groups)):
        for j, idx in enumerate(ids):
            v=views[int(idx)]
            u=j/max(1,len(ids)-1)
            if u < edge_fraction/2: tilt=rng.uniform(0,4)
            elif u > 1-edge_fraction/2: tilt=rng.uniform(41,45)
            else: tilt=rng.uniform(8,37)
            v['heading']=float((gi*360/heading_groups+rng.uniform(-5,5))%360)
            v['tilt']=float(tilt)
    for i,v in enumerate(views):
        lon,lat,alt=frame.to_geo(xyz[i])
        v.update(longitude=float(lon),latitude=float(lat),altitude=float(alt),
                 position_enu_m=xyz[i].tolist(),east=float(xyz[i,0]),north=float(xyz[i,1]),up=float(xyz[i,2]),
                 visible_grid=None,anchor_pixel_native=None,anchor_pixel_crop=None)
    if design.get('image_tilt_constraint'):
        # Retain each existing stratified angle, including unrendered ones.
        # Repair may redistribute headings and locations, never the new quota.
        for v, old in zip(views, design['views']):
            v['tilt'] = old['tilt']
        from image_tilt import validate_image_tilt
        validate_image_tilt(d)
    if d.get('max_camera_range_m') is not None:
        for v in views:
            v['camera_range_m'] = float(np.linalg.norm(v['position_enu_m']))
            if v['camera_range_m'] > d['max_camera_range_m']:
                raise ValueError('Capacity repair exceeds camera range limit')
    d['candidate_edges']=candidates(views)
    d['capacity_repair']={'version':1,'attempt':attempt,'seed_sha256':digest,
                         'reason':diagnostics,'rendered_views_changed':0}
    return d


def recover_commit(kml_root, tours):
    journal=kml_root/f'.repair_{tours[0]}_{tours[-1]}.json'
    if not journal.exists():return
    state=json.loads(journal.read_text())
    # Incomplete commits always roll back to the complete pre-trial state.
    for item in reversed(state['items']):
        live=Path(item['live']);backup=Path(item['backup'])
        if backup.exists():
            if live.exists():
                if live.is_dir():shutil.rmtree(live)
                else:live.unlink()
            backup.rename(live)
    journal.unlink()


def repair(a, report):
    from partial_repair import image_snapshot, preserve_frozen, rebound_summary
    designs={t:json.loads((a.kml_root/t/'views.json').read_text()) for t in a.tours}
    snapshots={t:image_snapshot(a.image_root,t,len(designs[t]['views'])) for t in a.tours}
    mutable=[t for t in a.tours if t not in a.audited_tours
             and len(snapshots[t][0]) < len(designs[t]['views'])
             and (not snapshots[t][1] or t in a.partial_tours)]
    original_hashes={t:sha(a.kml_root/t/'views.json') for t in a.tours}
    cache_hashes={t:sha(a.audit_root/t/'visual_audit_redesigned'/'partial_summary.json')
                  for t in mutable if t in a.partial_tours}
    history=[]
    if not mutable:return None
    # Revisions use the same seeds plus a recorded deterministic attempt ID.
    previous=max((json.loads((a.kml_root/t/'views.json').read_text()).get('capacity_repair',{}).get('attempt',0) for t in mutable),default=0)
    for attempt in range(previous+1,previous+5):
        with tempfile.TemporaryDirectory(prefix='capacity_trial_',dir=a.kml_root) as tmp:
            trial=Path(tmp)
            for t in a.tours:shutil.copytree(a.kml_root/t,trial/t)
            trial_audit=trial/'audit'
            if a.audit_root:
                trial_audit.mkdir()
                for t in a.tours:
                    if t in mutable and t in a.partial_tours:
                        ad=trial_audit/t/'visual_audit_redesigned';ad.mkdir(parents=True)
                        source=a.audit_root/t/'visual_audit_redesigned'
                        (ad/'partial_matches.jsonl').symlink_to((source/'partial_matches.jsonl').resolve())
                    else:
                        (trial_audit/t).symlink_to((a.audit_root/t).resolve())
            rebound={}
            for t in mutable:
                d=revise(json.loads((trial/t/'views.json').read_text()),attempt,
                         report.get('capacity_diagnostics',{}).get('shortages',[]))
                d=preserve_frozen(designs[t],d,snapshots[t][0])
                for i,v in enumerate(d['views']):
                    if i not in snapshots[t][0]:emit(trial/t/(v['id']+'.kml'),v,d['anchor'],d['hfov_deg'])
                save_json(trial/t/'views.json',d)
                if t in a.partial_tours:
                    rebound[t]=rebound_summary(a.audit_root/t/'visual_audit_redesigned',
                        a.kml_root/t,trial/t,a.image_root/f'{t}_redesigned',snapshots[t][0])
                    save_json(trial_audit/t/'visual_audit_redesigned'/'partial_summary.json',rebound[t])
            result=trial/'result.json'
            cmd=[str(BASE/'.venv/bin/python'),str(BASE/'preflight_group.py'),'--tours',*a.tours,
                 '--seed',str(a.seed),'--images',str(a.images),'--per-tour',str(a.per_tour),
                 '--bins',str(a.bins),'--tilt-limit',str(a.tilt_limit),
                 '--max-pairs-per-image',str(a.max_pairs_per_image),'--input-root',str(a.input_root),
                 '--image-root',str(a.image_root),'--kml-root',str(trial),'--output',str(result),'--no-repair']
            if a.audit_root:cmd+=['--audit-root',str(trial_audit)]
            cmd+=['--audited-tours',*a.audited_tours,'--partial-tours',*a.partial_tours]
            subprocess.run(cmd,check=True)
            checked=json.loads(result.read_text())
            history.append({'attempt':attempt,'status':checked['feasibility_status'],'tours':mutable})
            if not checked['geometry_feasible']:continue
            lock=(BASE.parent/'gen_img/export.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                for t in a.tours:
                    if sha(a.kml_root/t/'views.json') != original_hashes[t]:
                        raise RuntimeError('Design changed during capacity trial')
                for t in mutable:
                    if image_snapshot(a.image_root,t,len(designs[t]['views'])) != snapshots[t]:
                        raise RuntimeError('Images changed during capacity trial')
                    if t in cache_hashes and sha(a.audit_root/t/'visual_audit_redesigned'/'partial_summary.json') != cache_hashes[t]:
                        raise RuntimeError('Partial audit changed during capacity trial')
                    if t in rebound:
                        # Recheck matches and images immediately before committing.
                        rebound_summary(a.audit_root/t/'visual_audit_redesigned',a.kml_root/t,
                            trial/t,a.image_root/f'{t}_redesigned',snapshots[t][0])
                backup_root=a.kml_root/'.capacity_revisions'/f'{a.tours[0]}_{a.tours[-1]}'
                backup_root.mkdir(parents=True,exist_ok=True)
                items=[]
                for t in mutable:
                    digest=sha(a.kml_root/t/'views.json')[:16]
                    backup=backup_root/f'{t}_{digest}'
                    if backup.exists():raise FileExistsError(backup)
                    items.append({'live':str(a.kml_root/t),'backup':str(backup),'tour':t,'replacement':str(trial/t)})
                    if t in rebound:
                        live=a.audit_root/t/'visual_audit_redesigned'/'partial_summary.json'
                        meta_backup=backup_root/f'{t}_{digest}_partial_summary.json'
                        if meta_backup.exists():raise FileExistsError(meta_backup)
                        items.append({'live':str(live),'backup':str(meta_backup),'tour':t,
                            'replacement':str(trial_audit/t/'visual_audit_redesigned'/'partial_summary.json')})
                journal=a.kml_root/f'.repair_{a.tours[0]}_{a.tours[-1]}.json'
                save_json(journal,{'items':items})
                try:
                    for item in items:
                        Path(item['live']).rename(item['backup'])
                        Path(item['replacement']).rename(item['live'])
                    journal.unlink()
                except BaseException:
                    recover_commit(a.kml_root,a.tours)
                    raise
            finally:lock.close()
            checked['capacity_repair_history']=history
            checked['capacity_repair_committed_tours']=mutable
            return checked
    report['capacity_repair_history']=history
    return None
