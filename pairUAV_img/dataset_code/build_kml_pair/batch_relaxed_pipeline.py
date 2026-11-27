#!/usr/bin/env python3
"""Resumable batch runner for the approved relaxed 10-bin / 1050-pair pipeline."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

def size_gb(p: Path) -> float:
    st = shutil.disk_usage(p)
    return st.free / 1e9

def existing_complete(json_root: Path, tour: str) -> bool:
    p = json_root / f"{tour}.json"
    try:
        d = json.loads(p.read_text())
        return d.get("status") == "complete_visual_screen" and d.get("pairs") == 1050 and d.get("bins") == 10
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--seed", type=int, default=49)
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=1050)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--input-root", type=Path, default=Path("/root/pairUAV_img/tours_splited_kml"))
    ap.add_argument("--kml-root", type=Path, default=Path("/root/pairUAV_img/processed_kml_relaxed_batch"))
    ap.add_argument("--image-root", type=Path, default=Path("/root/pairUAV_img/processed_images_relaxed_batch"))
    ap.add_argument("--json-root", type=Path, default=Path("/root/pairUAV_img/processed_paired_json_relaxed_batch"))
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true", default=False)
    a = ap.parse_args()
    if not (0 <= a.start <= a.end <= 100): ap.error("tour range must be within 0000..0100")
    if a.bins != 10 or a.pairs != 1050: ap.error("approved configuration is exactly --bins 10 --pairs 1050")
    py = Path(__file__).resolve().parent / ".venv/bin/python"
    pipeline = Path(__file__).resolve().parent / "redesign_pipeline.py"
    a.kml_root.mkdir(parents=True, exist_ok=True); a.image_root.mkdir(parents=True, exist_ok=True); a.json_root.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(["bash", "-lc", "DISPLAY=:1 wmctrl -l | grep -q 'Google Earth Pro'"], check=False)
    if probe.returncode != 0 and not a.dry_run:
        raise RuntimeError("Google Earth Pro window is not available on DISPLAY=:1; run setup_google_earth_runtime.sh first")
    manifest = a.json_root / "batch_manifest.json"
    rows = []
    if manifest.exists():
        try: rows = json.loads(manifest.read_text()).get("tours", [])
        except Exception: rows = []
    by = {r.get("tour"): r for r in rows}
    for n in range(a.start, a.end + 1):
        tour = f"{n:04d}"
        if existing_complete(a.json_root, tour):
            by[tour] = {"tour": tour, "status": "complete_existing"}; print(f"[{tour}] already complete", flush=True); continue
        free = size_gb(a.json_root)
        if free < a.min_free_gb:
            by[tour] = {"tour": tour, "status": "blocked_low_disk", "free_gb": free}
            manifest.write_text(json.dumps({"configuration": vars(a), "tours": sorted(by.values(), key=lambda x:x.get("tour",""))}, indent=2, default=str) + "\n")
            raise RuntimeError(f"stopped before {tour}: only {free:.2f} GB free; need at least {a.min_free_gb:.2f} GB")
        if a.dry_run:
            print(json.dumps({"tour": tour, "free_gb": free, "would_run": True}), flush=True); continue
        cmd = [str(py), str(pipeline), "--tour", tour, "--seed", str(a.seed + n), "--images", str(a.images), "--pairs", str(a.pairs), "--bins", str(a.bins), "--min-tilt-delta", "45", "--tilt-min", "0", "--tilt-max", "45", "--tilt-group-count", "15", "--edge-fraction", ".8", "--camera-height", "500", "--position-radius", "15", "--hfov", "90", "--min-inliers", "20", "--min-inlier-ratio", ".25", "--min-hull-fraction", ".01", "--min-grid-cells", "4", "--input-root", str(a.input_root), "--kml-root", str(a.kml_root), "--image-root", str(a.image_root), "--json-root", str(a.json_root), "--jpeg-quality", str(a.jpeg_quality)]
        print("+", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True, env={**os.environ, "DISPLAY": ":1", "HOME": "/root/pairUAV_img/earth_runtime", "XDG_CONFIG_HOME": "/root/pairUAV_img/earth_runtime/.config"})
            by[tour] = {"tour": tour, "status": "complete", "free_after_gb": size_gb(a.json_root)}
        except Exception as e:
            by[tour] = {"tour": tour, "status": "failed", "error": repr(e), "free_gb": size_gb(a.json_root)}
            manifest.write_text(json.dumps({"configuration": vars(a), "tours": sorted(by.values(), key=lambda x:x.get("tour",""))}, indent=2, default=str) + "\n")
            if a.stop_on_error: raise
        manifest.write_text(json.dumps({"configuration": vars(a), "tours": sorted(by.values(), key=lambda x:x.get("tour",""))}, indent=2, default=str) + "\n")
    print(json.dumps({"start":a.start,"end":a.end,"completed":sum(v.get("status") in ("complete","complete_existing") for v in by.values()),"manifest":str(manifest)}, indent=2))

if __name__ == "__main__": main()
