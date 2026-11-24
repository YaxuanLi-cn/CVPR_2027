"""Download only the needed parts of Mip-NeRF 360 (360_v2.zip) via HTTP range requests."""
import sys, os, threading
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
SCENES = {"bicycle": 4, "garden": 4, "kitchen": 2, "bonsai": 2}
out = sys.argv[1]
tl = threading.local()
with RemoteZip(URL) as z:
    names = []
    for i in z.infolist():
        p = i.filename.split('/')
        if p[0] in SCENES and not i.is_dir():
            f = SCENES[p[0]]
            if p[1] in (f"images_{f}", "sparse", "poses_bounds.npy"):
                dst = os.path.join(out, i.filename)
                if not os.path.exists(dst) or os.path.getsize(dst) != i.file_size:
                    names.append(i.filename)
print(len(names), "files to fetch", flush=True)
def get(n):
    if not hasattr(tl, "z"): tl.z = RemoteZip(URL)
    tl.z.extract(n, out)
    return n
with ThreadPoolExecutor(12) as ex:
    for k, n in enumerate(ex.map(get, names)):
        if k % 100 == 0: print(k, n, flush=True)
print("DONE", flush=True)
