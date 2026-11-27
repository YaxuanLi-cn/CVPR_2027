import copy
import numpy as np
from image_tilt import uniform_tilts, validate_image_tilt
from repair_capacity import revise
import json

for count in (100,149,150):
    a=uniform_tilts(np.random.default_rng(75),count,0,45)
    assert np.array_equal(a,uniform_tilts(np.random.default_rng(75),count,0,45))
    assert not np.array_equal(a,uniform_tilts(np.random.default_rng(76),count,0,45))
    d={'views':[{'tilt':x} for x in a], 'image_tilt_constraint':{'bins':10,'range_deg':[0,45]}}
    validate_image_tilt(d)
    try:validate_image_tilt(d,[{'tilt':0.0} for _ in a])
    except ValueError:pass
    else:raise AssertionError('Validator trusted planned counts over actual KML angles')
    d['views'][0]['tilt']=50
    try: validate_image_tilt(d)
    except ValueError: pass
    else: raise AssertionError('Out-of-range tilt accepted')
old=json.load(open(__import__('pathlib').Path(__file__).resolve().parent/'test_fixtures/legacy_0026_views.json'))
old['image_tilt_constraint']={'bins':10,'range_deg':[0,45]}
try:validate_image_tilt(old)
except ValueError:pass
else:raise AssertionError('Legacy end-heavy tilt accepted')
for v,t in zip(old['views'],uniform_tilts(np.random.default_rng(75),150,0,45)):v['tilt']=float(t)
new=revise(old,1,[])
assert [v['tilt'] for v in old['views']]==[v['tilt'] for v in new['views']]
assert max(v['camera_range_m'] for v in new['views'])<=256
validate_image_tilt(new)
print('PASS: deterministic uniform sampling, legacy rejection, range rejection, repair preserves angles/distance')
