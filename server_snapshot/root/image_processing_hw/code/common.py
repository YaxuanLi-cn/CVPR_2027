"""Shared data loading: COLMAP binary reader, scene/split definitions, image IO.

Conventions: OpenCV camera frame (x right, y down, z forward); `w2c` are 4x4 world-to-camera
matrices and `K` 3x3 pinhole intrinsics at the resolution of the images actually loaded.
"""
import os
import struct
from dataclasses import dataclass

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (path, image folder, kind)
SCENES = {
    "bicycle": ("data/mip360/bicycle", "images_4", "mip360"),
    "garden": ("data/mip360/garden", "images_4", "mip360"),
    "kitchen": ("data/mip360/kitchen", "images_2", "mip360"),
    "bonsai": ("data/mip360/bonsai", "images_2", "mip360"),
    "truck": ("data/gsqa/Datasets/T&T/truck", "images", "gsqa"),
    "train": ("data/gsqa/Datasets/T&T/train", "images", "gsqa"),
}
# GS-QA: number of training images (the remaining, later-numbered frames are the reference video)
GSQA_NTRAIN = {"truck": 226, "train": 258}
GSQA_FPS = {"truck": 30, "train": 30}

CAM_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
              4: ("OPENCV", 8)}


def _qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap(sparse_dir):
    cams = {}
    with open(os.path.join(sparse_dir, "cameras.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = CAM_MODELS[model]
            params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
            cams[cid] = dict(model=name, w=w, h=h, params=params)
    imgs = []
    with open(os.path.join(sparse_dir, "images.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32))
            t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)
            w2c = np.eye(4)
            w2c[:3, :3] = _qvec2rot(q)
            w2c[:3, 3] = t
            imgs.append(dict(name=name.decode(), cam=cid, w2c=w2c))
    with open(os.path.join(sparse_dir, "points3D.bin"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        xyz = np.zeros((n, 3), np.float32)
        rgb = np.zeros((n, 3), np.uint8)
        for i in range(n):
            d = f.read(43)
            xyz[i] = struct.unpack("<3d", d[8:32])
            rgb[i] = struct.unpack("<3B", d[32:35])
            tl = struct.unpack("<Q", f.read(8))[0]
            f.seek(8 * tl, 1)
    return cams, imgs, xyz, rgb


@dataclass
class Scene:
    name: str
    kind: str
    names: list          # image file names (sorted)
    paths: list          # full paths
    w2c: np.ndarray      # (N,4,4)
    K: np.ndarray        # (3,3)
    W: int
    H: int
    train_idx: np.ndarray
    test_idx: np.ndarray  # held-out views (mip360) or reference-video frames in temporal order (gsqa)
    points: np.ndarray
    points_rgb: np.ndarray
    fps: int = 0

    def load_image(self, i):
        im = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)[..., ::-1]
        assert im.shape[0] == self.H and im.shape[1] == self.W, (im.shape, self.H, self.W)
        return np.ascontiguousarray(im)

    def cam_centers(self):
        return np.linalg.inv(self.w2c)[:, :3, 3]


def load_scene(name):
    rel, img_folder, kind = SCENES[name]
    base = os.path.join(ROOT, rel)
    cams, imgs, xyz, rgb = read_colmap(os.path.join(base, "sparse", "0"))
    imgs = sorted(imgs, key=lambda d: d["name"])
    assert len(cams) == 1, "expected a single shared camera"
    cam = cams[imgs[0]["cam"]]
    assert cam["model"] in ("PINHOLE", "SIMPLE_PINHOLE"), cam["model"]
    if cam["model"] == "PINHOLE":
        fx, fy, cx, cy = cam["params"]
    else:
        fx, cx, cy = cam["params"]
        fy = fx
    names = [d["name"] for d in imgs]
    paths = [os.path.join(base, img_folder, n) for n in names]
    im0 = cv2.imread(paths[0])
    H, W = im0.shape[:2]
    sx, sy = W / cam["w"], H / cam["h"]
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float64)
    w2c = np.stack([d["w2c"] for d in imgs])
    n = len(names)
    if kind == "mip360":
        test = np.arange(0, n, 8)
        train = np.array([i for i in range(n) if i % 8 != 0])
        fps = 0
    else:
        nt = GSQA_NTRAIN[name]
        train = np.arange(0, nt)
        test = np.arange(nt, n)
        fps = GSQA_FPS[name]
    return Scene(name, kind, names, paths, w2c, K, W, H, train, test, xyz, rgb, fps)


def subsample_views(scene, n_views):
    """Pick n_views training cameras spread evenly along the capture order."""
    tr = scene.train_idx
    if n_views is None or n_views <= 0 or n_views >= len(tr):
        return tr
    sel = np.round(np.linspace(0, len(tr) - 1, n_views)).astype(int)
    return tr[sel]


def save_png(path, img_u8):
    cv2.imwrite(path, img_u8[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])


def load_png(path):
    return np.ascontiguousarray(cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1])


def read_video(path):
    """Decode a video to uint8 RGB frames (F,H,W,3) with ffmpeg."""
    import subprocess, json
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", path]))["streams"][0]
    w, h = info["width"], info["height"]
    num, den = info["r_frame_rate"].split("/")
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-sws_flags", "accurate_rnd+full_chroma_int+bicubic", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return frames, float(num) / float(den)
