#!/usr/bin/env python3
"""Second worker pool for sift_pairs.py: the training pairs of the control subset in reverse order (pairs already matched are skipped)."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, '/root/rasr-pairuav/supp/trackA'); import sift_pairs as S
if __name__ == '__main__':
    ids = np.load(f'{S.O}/ids_ctrl_train.npy'); P = pd.read_pickle(f'{S.SG}/pairs.pkl').set_index('manifest_index').loc[ids].sample(frac=1.0, random_state=0)
    jobs = [(int(i), f'{S.DATA}/{r.image_a}', f'{S.DATA}/{r.image_b}') for i, r in P.iterrows()][::-1]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p: print('matched', sum(p.imap_unordered(S.match, jobs, chunksize=2)))
