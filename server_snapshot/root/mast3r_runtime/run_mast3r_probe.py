"""MASt3R image loader adapter; cache exact 512px model inputs by source file."""
from functools import lru_cache
from dust3r.utils.image import load_images

@lru_cache(maxsize=512)
def _prepared(path, size, patch_size):
    return load_images([path], size=size, square_ok=False, verbose=False, patch_size=patch_size)[0]

def prepare_image(path, idx=0, size=512, patch_size=16):
    view = dict(_prepared(str(path), int(size), int(patch_size)))
    view['idx'] = idx
    view['instance'] = str(idx)
    return view
