"""Render a group in stages so every unfinished tour retains repair capacity."""
import shutil


def execute_group(a, tours, gid, status, disk_guard, env, api=None):
    if api is None:
        import group_batch_pipeline as api
    base, python = api.BASE, api.PYTHON
    full, partial = [], []

    def paths(t):
        kd = a.kml_root / t
        im = a.image_root / f'{t}_redesigned'
        return kd, im, im / 'earth_pro_export', a.json_root / t / 'visual_audit_redesigned'

    def audit(t):
        kd, im, staging, ad = paths(t)
        done = api.verified_images(im, kd, a.images)
        if len(done) == a.images:
            status(gid, tours, 'auditing', tour=t, completed_images=len(done))
            api.invoke([python, base/'audit_visual_pairs.py', '--tour', t,
                '--kml-root', a.kml_root, '--image-root', im, '--image-pattern', '{id}.jpg',
                '--output-dir', ad, '--workers', 8, '--min-inliers', 20,
                '--min-inlier-ratio', .25, '--min-hull-fraction', .01,
                '--min-grid-cells', 4, '--bins', a.bins, '--tilt-limit', 45,
                '--requested-pairs', a.per_tour])
            if not api.audit_valid(ad, im, kd, a.images):
                raise RuntimeError(f'{t}: full audit verification failed')
            full.append(t)
            if t in partial:
                partial.remove(t)
            # Cleanup is optional. Never discard final images or matching evidence.
            for directory in (staging, ad/'features'):
                try:
                    if directory.exists():
                        shutil.rmtree(directory)
                except OSError as exc:
                    print(f'Keeping optional cache {directory}: {exc}', flush=True)
        elif done:
            status(gid, tours, 'partial_visual_audit', tour=t, completed_images=len(done))
            api.invoke([python, base/'partial_visual_audit.py', '--tour', t,
                '--kml-root', a.kml_root, '--image-root', a.image_root,
                '--audit-root', a.json_root])
            if t not in partial:
                partial.append(t)

    def capacity(stage):
        output = a.json_root / f'group_{gid}_partial_capacity.json'
        status(gid, tours, 'checking_group_capacity', stage_target=stage,
               audited_tours=list(full), partial_tours=list(partial))
        api.invoke([python, base/'preflight_group.py', '--tours', *tours,
            '--seed', a.seed, '--images', a.images, '--per-tour', a.per_tour,
            '--bins', a.bins, '--input-root', a.input_root,
            '--kml-root', a.kml_root, '--output', output,
            '--image-root', a.image_root, '--audit-root', a.json_root,
            '--audited-tours', *full, '--partial-tours', *partial])
        result = api.load(output, {})
        if not result.get('geometry_feasible'):
            raise RuntimeError(f'Group capacity {result.get("feasibility_status")}; '
                               f'evidence retained at {output}; no further rendering')

    # Resume every tour before making a new rendering decision. A later tour's
    # existing visual failures must not be treated as untested geometry.
    for t in tours:
        kd, im, staging, ad = paths(t)
        if api.audit_valid(ad, im, kd, a.images):
            full.append(t)
            status(gid, tours, 'audit_reused', tour=t)
            continue
        staging.mkdir(parents=True, exist_ok=True)
        api.sync_exports(staging, im, kd, a.jpeg_quality)
        audit(t)
    if full or partial:
        capacity('resume')

    checkpoints = api.audit_checkpoints(a.images)
    for target in checkpoints + [a.images]:
        for t in tours:
            if t in full:
                continue
            kd, im, staging, ad = paths(t)
            failures = 0
            changed = False
            while True:
                done = api.verified_images(im, kd, a.images)
                if len(done) >= target:
                    break
                missing = sorted(set(range(a.images)) - done)
                disk_guard()
                count = min(target-len(done), api.render_count(
                    missing, a.render_chunk, checkpoints, len(done)))
                status(gid, tours, 'rendering', tour=t, completed_images=len(done),
                       total_images=a.images, stage_target=target,
                       next_index=missing[0], retry=failures)
                import subprocess
                try:
                    api.invoke(['/usr/bin/python3', api.EXPORTER, '--input', kd,
                        '--output', staging, '--start', missing[0], '--limit', count,
                        '--settle', 3, '--timeout', 180, '--hfov', 90],
                        env=env, timeout=count*360+180)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    api.sync_exports(staging, im, kd, a.jpeg_quality)
                    gained = len(api.verified_images(im, kd, a.images)) > len(done)
                    changed = changed or gained
                    failures = 1 if gained else failures+1
                    status(gid, tours, 'retrying_render', tour=t, retry=failures, error=str(exc))
                    if failures >= a.render_retries:
                        raise RuntimeError('Repeated renderer failure; verified outputs retained') from exc
                    api.recover_ui()
                    continue
                api.sync_exports(staging, im, kd, a.jpeg_quality)
                if len(api.verified_images(im, kd, a.images)) <= len(done):
                    raise RuntimeError('Exporter returned success without verified new images')
                changed = True
                failures = 0
            if changed:
                audit(t)
                capacity(target)
    if set(full) != set(tours):
        raise RuntimeError('Group render stages ended without all full audits')
