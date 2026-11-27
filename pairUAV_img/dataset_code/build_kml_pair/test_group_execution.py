"""Offline orchestration regressions; no Earth GUI or production data."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import group_batch_pipeline as g
from group_execution import execute_group


def scenario(resume=False, fail_capacity=False, render_failure=False):
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        tours=['0001','0002','0003','0004','0005']
        a=SimpleNamespace(kml_root=root/'k', image_root=root/'i', json_root=root/'j',
            input_root=root/'source', images=150, bins=5, per_tour=1000, seed=49,
            jpeg_quality=80, render_chunk=10, render_retries=4)
        a.json_root.mkdir()
        done={t:set() for t in tours}
        full=set(); events=[]; checks=[]; recovered=[]
        if resume:
            full.add('0001');done['0001']=set(range(150))
            done['0002']=set(range(121))
            done['0004']={0,2,4}
        def tid(path):
            return next(p[:4] for p in Path(path).parts if p[:4] in tours)
        def invoke(cmd,**kwargs):
            name=Path(cmd[1]).name
            def arg(key): return cmd[cmd.index(key)+1]
            if name=='export_google_earth_pro.py':
                t=tid(arg('--input'));start=arg('--start');n=arg('--limit')
                assert not (set(range(start,start+n)) & done[t])
                if render_failure and not recovered:
                    done[t].add(start)
                    raise subprocess.CalledProcessError(1,cmd)
                done[t].update(range(start,start+n));events.append(('render',t,len(done[t])))
            elif name=='partial_visual_audit.py':
                events.append(('partial',arg('--tour'),len(done[arg('--tour')])))
            elif name=='audit_visual_pairs.py':
                full.add(arg('--tour'));events.append(('full',arg('--tour'),150))
            elif name=='preflight_group.py':
                i=cmd.index('--audited-tours');j=cmd.index('--partial-tours')
                checks.append((set(cmd[i+1:j]),set(cmd[j+1:]),{t:len(v) for t,v in done.items()}))
                g.atomic_json(arg('--output'),{'geometry_feasible':not fail_capacity,'feasibility_status':'infeasible' if fail_capacity else 'feasible'})
            else: raise AssertionError(name)
        api=SimpleNamespace(BASE=Path('.'),PYTHON='python',EXPORTER=Path('export_google_earth_pro.py'),
            verified_images=lambda im,kd,n:done[tid(im)].copy(),
            audit_valid=lambda ad,im,kd,n:tid(im) in full,
            sync_exports=lambda *args:None,invoke=invoke,load=g.load,
            audit_checkpoints=g.audit_checkpoints,render_count=g.render_count,
            recover_ui=lambda:recovered.append(True))
        if fail_capacity:
            try: execute_group(a,tours,'test',lambda *a,**k:None,lambda:None,{},api)
            except RuntimeError as exc: assert 'Group capacity' in str(exc)
            else: raise AssertionError('infeasible capacity ignored')
            assert len(done['0001'])==50 and all(not done[t] for t in tours[1:])
            return
        execute_group(a,tours,'test',lambda *a,**k:None,lambda:None,{},api)
        assert full==set(tours) and all(len(v)==150 for v in done.values())
        if resume:
            assert checks[0][0]=={'0001'} and checks[0][1]=={'0002','0004'}
            assert not any(e[0]=='render' and e[1]=='0001' for e in events)
        else:
            for t in tours:
                assert ('partial',t,50) in events and ('partial',t,140) in events
            first_over_50=next(i for i,e in enumerate(events) if e[0]=='render' and e[2]>50)
            assert all(('partial',t,50) in events[:first_over_50] for t in tours)
            first_full=next(i for i,e in enumerate(events) if e[0]=='full')
            assert all(('partial',t,140) in events[:first_full] for t in tours)
        if render_failure: assert recovered

for kw in ({},{'resume':True},{'fail_capacity':True},{'render_failure':True}):scenario(**kw)
print('PASS: staged group scheduling, multi-tour cache resume, capacity gate, partial export recovery')
