#!/usr/bin/python3
"""Bounded recovery of the dedicated renderer, under its exclusive GUI lock."""
import os
import sys
import re
import fcntl
import subprocess
import time
from pathlib import Path
os.environ['DISPLAY'] = ':1'
import export_google_earth_pro as e


def close_pickers():
    for pattern in (r'^Open$', r'^Save As$'):
        w = e.visible_window(pattern)
        if w:
            e.focus(w)
            e.key('Escape')
            e.wait_window(pattern, False, timeout=5)


def responsive():
    """Require Qt to actually open and close a dialog; an X window alone is insufficient."""
    try:
        close_pickers()
        if e.visible_window(r'^Google Earth$'):
            e.Earth.dismiss_trust_prompt(timeout=8)
        main = e.visible_window(r'^Google Earth Pro$')
        if not main: return False
        e.focus(main)
        e.key('ctrl+o')
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if e.visible_window(r'^Google Earth$'):
                e.Earth.dismiss_trust_prompt(timeout=5)
            if e.visible_window(r'^Open$'):
                close_pickers()
                return True
            time.sleep(.3)
    except (RuntimeError, subprocess.SubprocessError):
        pass
    return False


def main():
    lock = (e.BASE / 'export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0,str(e.BASE.parent/'build_kml_pair'))
    from audit_resources import recover_affinity
    recover_affinity()
    if responsive():
        print('Earth Pro responded to dialog health check', flush=True)
        return
    print('Earth Pro not responsive; restarting dedicated renderer', flush=True)
    subprocess.run(['supervisorctl', 'restart', 'earth-pro'], check=True, timeout=45)
    end = time.monotonic() + 45
    while time.monotonic() < end:
        crash = e.visible_window(r'^Google Earth Crash Detected$')
        if crash:
            e.focus(crash)
            time.sleep(.5)
            info = e.run('xwininfo', '-id', crash).stdout
            def coord(label): return int(re.search(re.escape(label) + r'\s*:\s*(-?\d+)', info).group(1))
            x,y=coord('Absolute upper-left X'),coord('Absolute upper-left Y')
            width,height=coord('Width'),coord('Height')
            choices = [w for w in e.words((x,y,x+width,y+height),psm=6) if w['text'] == 'Continue']
            if not choices: raise RuntimeError('Crash dialog Continue button missing')
            e.click_word(max(choices, key=lambda w:w['y']))
        if e.visible_window(r'^Google Earth Pro$'):
            diagnostics = e.BASE / 'recovery_diagnostics'
            diagnostics.mkdir(exist_ok=True)
            e.Earth(diagnostics)
            if responsive():
                print('Earth Pro restarted and passed dialog health check', flush=True)
                return
        time.sleep(1)
    raise RuntimeError('Earth Pro failed bounded recovery; saved artifacts retained')


if __name__ == '__main__': main()
