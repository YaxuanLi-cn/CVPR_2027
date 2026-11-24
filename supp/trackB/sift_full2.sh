#!/bin/bash
# after the SIFT chain on the sampled sets has finished: SIFT on the full 6,364-pair test set (CPU only, low priority), in a fixed random order
source /venv/main/bin/activate; cd /root/rasr-pairuav
until [ -d outputs/supp/trackB/matches/sift/real_out ] && [ $(ls outputs/supp/trackB/matches/sift/real_out | wc -l) -ge 300 ]; do sleep 30; done
nice -n 19 python supp/trackB/run_sift.py test_full 4; echo "SIFT_FULL_DONE $(date)"
