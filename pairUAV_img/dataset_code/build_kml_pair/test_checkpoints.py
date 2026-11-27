"""Checkpoints must not be skipped by large chunks or noncontiguous resumes."""
from group_batch_pipeline import audit_checkpoints, render_count
for total in (60,100,150):
    checkpoints=audit_checkpoints(total)
    for chunk in (1,10,100):
        completed=0;visited=[]
        while completed<total:
            missing=list(range(completed,total))
            count=render_count(missing,chunk,checkpoints,completed)
            assert 1<=count<=min(chunk,len(missing))
            completed+=count;visited.append(completed)
        assert set(checkpoints)<=set(visited)
assert audit_checkpoints(150)==[50,100,125,140]
assert render_count([20,22,23],10,[50,100,125,140],147)==1
assert render_count(list(range(121,150)),10,[50,100,125,140],121)==4
print('PASS: exact checkpoints, chunk caps and noncontiguous resume')
