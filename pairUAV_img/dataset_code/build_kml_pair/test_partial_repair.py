"""Disposable-data regression for partial-slot repair, without touching Earth."""
import copy
import json
import tempfile
from pathlib import Path
from common import save_json, sha
from partial_repair import image_snapshot, preserve_frozen, rebound_summary
from partial_visual_audit import read_cache
from repair_capacity import recover_commit
import audit_visual_pairs as visual
from group_batch_pipeline import THRESHOLDS


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);old=root/'old';new=root/'new';ad=root/'audit';im=root/'images'/'0001_redesigned'
        for p in (old,new,ad,im):p.mkdir(parents=True)
        views=[dict(id=f'{i:04d}',position_enu_m=[i*4.,0.,500.],heading=i*30.,tilt=i*10.,visible_grid=None) for i in range(3)]
        from balance import candidates
        d=dict(views=views,candidate_edges=candidates(views))
        save_json(old/'views.json',d)
        revised=copy.deepcopy(d);revised['capacity_repair']={}
        revised['views'][0]['heading']=99
        revised['views'][2]['heading']=123
        revised=preserve_frozen(d,revised,{0,1})
        assert revised['views'][:2]==d['views'][:2] and revised['views'][2]!=d['views'][2]
        assert d['views'][0]['heading']==0
        save_json(new/'views.json',revised)
        for i in range(3):
            (old/f'{i:04d}.kml').write_text(f'camera {i}')
            (new/f'{i:04d}.kml').write_text(f'camera {i}')
        for i in (0,1):(im/f'{i:04d}.jpg').write_bytes(f'image {i}'.encode())
        staging=im/'earth_pro_export'/'native';staging.mkdir(parents=True)
        staged=staging/'tour0001_0002.jpg';staged.write_bytes(b'staged')
        assert image_snapshot(root/'images','0001',3)[0]=={0,1,2}
        staged.unlink()
        unknown=staging/'unknown.jpg';unknown.write_bytes(b'unknown')
        assert image_snapshot(root/'images','0001',3)[0]=={0,1,2}
        unknown.unlink()
        frozen,snapshot=image_snapshot(root/'images','0001',3)
        assert frozen=={0,1}
        thresholds={**visual.THRESHOLDS,**THRESHOLDS}
        row={**d['candidate_edges'][0],'passed':True}
        matches=ad/'partial_matches.jsonl';matches.write_text(json.dumps(row)+'\n')
        meta=dict(schema_version=visual.VERSION,thresholds=thresholds,views_sha256=sha(old/'views.json'),
            matches_sha256=sha(matches),image_sha256={f'{i:04d}.jpg':sha(im/f'{i:04d}.jpg') for i in (0,1)},
            checked_edges=1,accepted_edges=1)
        save_json(ad/'partial_summary.json',meta)
        rebound=rebound_summary(ad,old,new,im,frozen)
        assert json.loads((ad/'partial_summary.json').read_text())==meta
        save_json(ad/'partial_summary.json',rebound)
        assert len(read_cache(ad,new/'views.json',[im/f'{i:04d}.jpg' for i in range(3)],thresholds))==1
        save_json(ad/'partial_summary.json',meta)
        (new/'0000.kml').write_text('modified rendered KML')
        try:rebound_summary(ad,old,new,im,frozen)
        except ValueError:pass
        else:raise AssertionError('changed rendered KML accepted')
        (new/'0000.kml').write_text('camera 0')
        (im/'0000.jpg').write_bytes(b'modified')
        try:rebound_summary(ad,old,new,im,frozen)
        except ValueError:pass
        else:raise AssertionError('changed image accepted')
        # Interrupted transaction rolls back both the directory and summary.
        live=root/'live';live.mkdir();(live/'x').write_text('new')
        backup=root/'backup';backup.mkdir();(backup/'x').write_text('old')
        meta_live=root/'meta.json';meta_live.write_text('new')
        meta_backup=root/'meta_backup.json';meta_backup.write_text('old')
        save_json(root/'.repair_0001_0005.json',{'items':[
            dict(live=str(live),backup=str(backup)),dict(live=str(meta_live),backup=str(meta_backup))]})
        recover_commit(root,['0001','0005'])
        assert (live/'x').read_text()=='old' and meta_live.read_text()=='old'
    print('PASS: frozen poses, staged/unknown images, evidence rebinding, tamper rejection, mixed transaction rollback')

if __name__=='__main__':main()
