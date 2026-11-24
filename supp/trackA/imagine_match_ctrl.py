#!/usr/bin/env python3
"""Control variants of the imagination re-matching step. Thin wrapper around stage2_imagination/imagine_match.py
(the original module is imported and run unchanged; only its canvas size can be overridden here).

  --rot-only --max-grid 3   compute-matched control: the same tile-wise native-resolution matching, but the target is only
                            rotated into the source orientation (no translation, no ground plane, no camera height)
  --max-grid 1              plane-induced warp with a single 1280x720 tile (the budget of the coarse stage's zoom pass)
  --canvas 512 288          plane-induced warp rendered at the resolution of a latent-diffusion image (single tile)
  --guide <pkl>             alternative guide poses (e.g. the ground-truth pose: oracle imagination)
Usage: imagine_match_ctrl.py [--canvas W H] <arguments of imagine_match.py>
"""
import sys
sys.path.insert(0, '/root/rasr-pairuav/stage2_imagination')
import imagine_match as im
if '--canvas' in sys.argv:
    i = sys.argv.index('--canvas'); im.CW, im.CH = int(sys.argv[i + 1]), int(sys.argv[i + 2]); del sys.argv[i:i + 3]
print('canvas', im.CW, im.CH, flush=True)
im.main()
