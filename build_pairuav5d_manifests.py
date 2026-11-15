#!/usr/bin/env python3
"""Make deterministic tour-disjoint pair manifests, reservoir-sampled within every tour."""
from pathlib import Path
import hashlib, random
ROOT=Path('/root/pairUAV-5D-temp')
OUT=Path('/root/rasr-pairuav/outputs/pairuav5d_60k')
SEED=2026
TOTAL=60_000
files=sorted((ROOT/'pairs').glob('*.jsonl'))
base,extra=divmod(TOTAL,len(files))
# Assign the remainder to a deterministic, independent set of tours.
extra_tours=set(p.stem for p in sorted(files,key=lambda p:hashlib.sha256(f'sample:{SEED}:{p.stem}'.encode()).hexdigest())[:extra])
train=[];test=[];global_idx=0;total=0;test_tours=0
for p in files:
    quota=base+(p.stem in extra_tours)
    rng=random.Random(f'{SEED}:{p.stem}')
    reservoir=[];seen=0
    with p.open('rb') as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            item=(global_idx,offset)
            seen+=1
            if len(reservoir)<quota:reservoir.append(item)
            else:
                j=rng.randrange(seen)
                if j<quota:reservoir[j]=item
            global_idx+=1
    # Fixed 20% test split by tour; split salt differs from sampling salt.
    split_value=int(hashlib.sha256(f'split:{SEED}:{p.stem}'.encode()).hexdigest()[:8],16)/0xffffffff
    destination=test if split_value<.2 else train
    if destination is test:test_tours+=1
    for source_index,offset in sorted(reservoir):
        destination.append(f'{source_index}\tpairs/{p.name}#{offset}\n')
    total+=len(reservoir)
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'train_manifest.txt').write_text(''.join(train))
(OUT/'test_manifest.txt').write_text(''.join(test))
summary=f'tours={len(files)} sampled_pairs={total} train_pairs={len(train)} test_pairs={len(test)} test_tours={test_tours}'
(OUT/'manifest_summary.txt').write_text(summary+'\n')
print(summary)
