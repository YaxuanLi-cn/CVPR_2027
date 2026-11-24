#!/bin/bash
# MASt3R on the 1000 test pairs and on 300 train pairs (scale calibration), then DUSt3R on both
cd /root/rasr-pairuav && source /venv/main/bin/activate
L=outputs/reviewer_baselines/logs
python reviewer_baselines/rb_recon.py mast3r 1000 test  > $L/recon_mast3r_test.log 2>&1
python reviewer_baselines/rb_recon.py mast3r 300 train  > $L/recon_mast3r_train.log 2>&1
python reviewer_baselines/rb_recon.py dust3r 1000 test  > $L/recon_dust3r_test.log 2>&1
python reviewer_baselines/rb_recon.py dust3r 300 train  > $L/recon_dust3r_train.log 2>&1
