#!/usr/bin/env python3
"""Build one reproducible, distance-limited Earth Pro tour and visual pairs."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from group_batch_pipeline import BASE, EXPORTER, PYTHON, audit_valid, invoke, load


def reference_range(path: Path) -> float:
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    node = root.find('.//{http://www.opengis.net/kml/2.2}LookAt/{http://www.opengis.net/kml/2.2}range')
    if node is None or not node.text:
        raise ValueError(f'Reference KML has no LookAt range: {path}')
    return float(node.text)


def design_matches(data, tour, seed, images, camera_range, camera_height, radius):
    if (data.get('tour_id') != tour or data.get('seed') != seed + int(tour)
            or data.get('images') != images
            or abs(data.get('max_camera_range_m', -1) - camera_range) > 1e-8
            or abs(data.get('camera_height_center_m', -1) - camera_height) > 1e-8
            or abs(data.get('position_radius_m', -1) - radius) > 1e-8):
        return False
    return all(v.get('camera_range_m', float('inf')) <= camera_range + 1e-8
               for v in data.get('views', []))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tour', required=True)
    ap.add_argument('--publish-standard', action='store_true', help='Update standard directory symlinks after validation, retaining prior targets')
    ap.add_argument('--uniform-image-tilt', action='store_true', help='Require balanced absolute image tilt in 10 bins over 0..45 degrees')
    ap.add_argument('--seed', type=int, default=49)
    ap.add_argument('--images', type=int, default=150)
    ap.add_argument('--pairs', type=int, default=1000)
    ap.add_argument('--bins', type=int, default=5)
    ap.add_argument('--tilt-limit', type=float, default=45)
    ap.add_argument('--camera-height', type=float, default=220)
    ap.add_argument('--position-radius', type=float, default=15)
    ap.add_argument('--max-camera-range', type=float)
    ap.add_argument('--reference-kml', type=Path, default=Path('/root/pairUAV_img/tours_splited_kml/0000/0000.kml'))
    ap.add_argument('--input-root', type=Path, default=Path('/root/pairUAV_img/tours_splited_kml'))
    ap.add_argument('--kml-root', type=Path, default=Path('/root/pairUAV_img/processed_kml_relaxed_batch'))
    ap.add_argument('--image-root', type=Path, default=Path('/root/pairUAV_img/processed_images_relaxed_batch'))
    ap.add_argument('--json-root', type=Path, default=Path('/root/pairUAV_img/processed_paired_json_relaxed_batch'))
    ap.add_argument('--render-chunk', type=int, default=10)
    ap.add_argument('--render-retries', type=int, default=4)
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    tour = str(a.tour).zfill(4)
    a.tour = tour
    if not 1 <= a.images <= 150 or a.pairs < 1000 or a.bins < 5:
        ap.error('require 1..150 images, at least 1000 pairs, and at least 5 bins')
    if a.pairs % a.bins:
        ap.error('pairs must be divisible by bins')
    source_first = a.input_root / tour / '0000.kml'
    if not source_first.is_file():
        raise FileNotFoundError(source_first)
    if a.max_camera_range is None:
        a.max_camera_range = reference_range(a.reference_kml)
    if a.max_camera_range <= (2 ** .5) * a.position_radius:
        raise ValueError('max-camera-range must exceed the maximum horizontal offset')
    if a.camera_height + 16.5 > (a.max_camera_range**2 - 2*a.position_radius**2)**0.5:
        raise ValueError('camera height/offset design can exceed max-camera-range')

    gid = f'{tour}_{tour}'
    kdir = a.kml_root / tour
    idir = a.image_root / f'{tour}_redesigned'
    staging = idir / 'earth_pro_export'
    audit = a.json_root / tour / 'visual_audit_redesigned'
    report = a.json_root / f'group_{gid}.json'
    kdir.mkdir(parents=True, exist_ok=True)
    idir.mkdir(parents=True, exist_ok=True)
    a.json_root.mkdir(parents=True, exist_ok=True)

    views_path = kdir / 'views.json'
    expected_seed = a.seed + int(tour)
    if views_path.exists():
        design = json.loads(views_path.read_text())
        if not design_matches(design, tour, a.seed, a.images, a.max_camera_range,
                              a.camera_height, a.position_radius):
            raise ValueError(f'Existing design does not match requested deterministic settings: {views_path}')
    else:
        if any(idir.rglob('*.jpg')):
            raise ValueError(f'Images exist without their design metadata; refusing to mix artifacts for {tour}')
        invoke([PYTHON, BASE/'design_views.py', '--tour', tour, '--seed', expected_seed,
            '--images', a.images, '--input-root', a.input_root, '--kml-root', a.kml_root,
            '--camera-height', a.camera_height, '--max-camera-range', a.max_camera_range,
            '--position-radius', a.position_radius, '--tilt-min', 0, '--tilt-max', 45,
            '--paired-tilt-groups', '--tilt-group-count', 15, '--edge-fraction', .8, '--hfov', 90,
            *(['--uniform-image-tilt'] if a.uniform_image_tilt else [])])
        design = json.loads(views_path.read_text())
        if not design_matches(design, tour, a.seed, a.images, a.max_camera_range,
                              a.camera_height, a.position_radius):
            raise RuntimeError('Generated design failed deterministic settings verification')

    if a.uniform_image_tilt:
        from image_tilt import validate_image_tilt
        if not design.get('image_tilt_constraint'):
            raise ValueError('Existing design predates uniform image tilt; use separate output roots to retain old images')
        validate_image_tilt(design)

    # A completed validated tour is idempotent: verify and reuse it as-is.
    existing = load(report, {})
    if (existing.get('status') == 'complete' and existing.get('bins') == a.bins
            and existing.get('per_tour', {}).get(tour, 0) >= a.pairs
            and audit_valid(audit, idir, kdir, a.images)):
        invoke([PYTHON, BASE/'validate_group.py', '--group', gid,
            '--json-root', a.json_root, '--kml-root', a.kml_root, '--image-root', a.image_root])
        invoke([PYTHON, BASE/'compare_candidate_selected.py', '--group', gid, '--json-root', a.json_root])
        invoke([PYTHON, BASE/'group_fine_distributions.py', '--group', gid,
            '--bins', 10, '--json-root', a.json_root])
        if a.uniform_image_tilt:
            invoke([PYTHON, BASE/'plot_image_tilt.py', '--views', views_path,
                '--output', a.json_root/f'{tour}_absolute_image_tilt.png'])
            invoke([PYTHON, BASE/'tilt_contact_sheet.py', '--views', views_path, '--images', idir,
                '--output', a.json_root/f'{tour}_tilt_examples.jpg'])
        if a.publish_standard:
            from publish_group import publish
            publish(a.json_root,a.kml_root,a.image_root,gid,replace=True)
        print(f'{tour}: existing validated artifacts reused')
        return

    preflight = a.json_root / f'group_{gid}_preflight.json'
    invoke([PYTHON, BASE/'preflight_group.py', '--tours', tour, '--seed', a.seed,
        '--images', a.images, '--per-tour', a.pairs, '--bins', a.bins,
        '--input-root', a.input_root, '--kml-root', a.kml_root, '--image-root', a.image_root,
        '--output', preflight])
    pre = load(preflight, {})
    if not pre.get('geometry_feasible'):
        raise RuntimeError(f'Geometry preflight {pre.get("feasibility_status")}; no images rendered')
    if a.preflight_only:
        print(json.dumps(pre, indent=2))
        return

    from types import SimpleNamespace
    from group_execution import execute_group
    run_args = SimpleNamespace(
        kml_root=a.kml_root, image_root=a.image_root, json_root=a.json_root,
        input_root=a.input_root, images=a.images, bins=a.bins, per_tour=a.pairs,
        seed=a.seed, jpeg_quality=80, render_chunk=a.render_chunk,
        render_retries=a.render_retries)

    def status(group, tours, stage, **extra):
        print(group, stage, extra, flush=True)

    def disk_guard():
        free = shutil.disk_usage(a.json_root).free / 1e9
        if free < 5:
            raise RuntimeError(f'Only {free:.2f} GB free; stopped before further rendering')

    execute_group(run_args, [tour], gid, status, disk_guard,
                  {**os.environ, 'DISPLAY': ':1'})
    invoke([PYTHON, BASE/'select_group_visual_pairs.py', '--tours', tour,
        '--per-tour', a.pairs, '--bins', a.bins, '--tilt-limit', a.tilt_limit,
        '--kml-root', a.kml_root, '--audit-root', a.json_root,
        '--image-root', a.image_root, '--output-root', a.json_root, '--time-limit', 600])
    invoke([PYTHON, BASE/'validate_group.py', '--group', gid,
        '--json-root', a.json_root, '--kml-root', a.kml_root, '--image-root', a.image_root])
    invoke([PYTHON, BASE/'compare_candidate_selected.py', '--group', gid, '--json-root', a.json_root])
    invoke([PYTHON, BASE/'group_fine_distributions.py', '--group', gid,
        '--bins', 10, '--json-root', a.json_root])

    if a.uniform_image_tilt:
        invoke([PYTHON, BASE/'plot_image_tilt.py', '--views', views_path,
            '--output', a.json_root/f'{tour}_absolute_image_tilt.png'])
        invoke([PYTHON, BASE/'tilt_contact_sheet.py', '--views', views_path, '--images', idir,
            '--output', a.json_root/f'{tour}_tilt_examples.jpg'])

    if a.publish_standard:
        from publish_group import publish
        publish(a.json_root,a.kml_root,a.image_root,gid,replace=True)


if __name__ == '__main__':
    main()
