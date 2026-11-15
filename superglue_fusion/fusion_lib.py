"""Shared helpers: data loading, grouped OOF LightGBM, metrics."""
import os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
ROOT = '/root/rasr-pairuav/outputs/pairuav5d_60k'; SG = f'{ROOT}/superglue'
META = ['manifest_index', 'scene_id', 'pair_id', 'json_rel_path', 'image_a', 'image_b', 'heading_num', 'range_num', 'tilt_num', 'is_valid', 'failure_reason',
        'pair_wall_seconds', 'x_num', 'y_num', 'z_num']

def load_mast3r():
    """MASt3R feature tables (same rows/features as the baseline heads), cached as pickle."""
    c = f'{SG}/mast3r_feats.pkl'
    if os.path.exists(c): return pd.read_pickle(c)
    out = []
    for split in ('train', 'test'):
        d = pd.read_csv(f'{ROOT}/features/{split}_xyz.csv'); d['split'] = split; out.append(d)
    d = pd.concat(out, ignore_index=True)
    feats = [c for c in d.columns if c not in META + ['split']]
    d[feats] = d[feats].astype(np.float32)
    d = d[['manifest_index', 'split'] + feats]; d.to_pickle(c); return d

def wrap(a): return (np.asarray(a) + 180.0) % 360.0 - 180.0

def add_targets(df):
    """Derived targets. lx/ly = horizontal displacement in cam-a's levelled frame (right, forward)."""
    df = df.copy(); ha = np.deg2rad(df.heading_a)
    df['dist'] = np.sqrt(df.x**2 + df.y**2 + df.z**2); df['logd'] = np.log(df.dist)
    df['lx'] = np.cos(ha) * df.x - np.sin(ha) * df.y; df['ly'] = np.sin(ha) * df.x + np.cos(ha) * df.y
    df['ux'] = df.lx / df.dist; df['uy'] = df.ly / df.dist; df['uz'] = df.z / df.dist
    df['hsin'] = np.sin(np.deg2rad(df.heading)); df['hcos'] = np.cos(np.deg2rad(df.heading))
    return df

LGB = dict(n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=30, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
           reg_lambda=1.0, max_bin=127, verbose=-1, n_jobs=8)

def _fit_predict(Xtr, ytr, Xs, params, wtr=None):
    m = lgb.LGBMRegressor(**params); m.fit(Xtr, ytr, sample_weight=wtr)
    return [m.predict(X) for X in Xs]

def oof_fit(X, y, groups, Xtest, params=None, n_splits=5, weight=None, mask=None):
    """Tour-grouped OOF predictions on train + full-train prediction on test. mask: rows usable for training."""
    params = {**LGB, **(params or {})}; X = np.asarray(X, np.float32); Xtest = np.asarray(Xtest, np.float32); y = np.asarray(y, np.float64)
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    folds = list(GroupKFold(n_splits).split(X, y, groups))
    def job(tr, va):
        tr = tr[mask[tr]]
        return _fit_predict(X[tr], y[tr], [X[va]], params, None if weight is None else weight[tr])[0]
    jobs = [delayed(job)(tr, va) for tr, va in folds] + [delayed(_fit_predict)(X[mask], y[mask], [Xtest], params, None if weight is None else weight[mask])]
    res = Parallel(n_jobs=len(jobs))(jobs)
    oof = np.zeros(len(y))
    for (tr, va), p in zip(folds, res[:-1]): oof[va] = p
    return oof, res[-1][0]

def report(df, px, py, pz, ptilt, phead, name=''):
    r = dict(MAE_x_m=np.mean(np.abs(px - df.x)), MAE_y_m=np.mean(np.abs(py - df.y)), MAE_z_m=np.mean(np.abs(pz - df.z)),
             MAE_tilt_deg=np.mean(np.abs(ptilt - df.tilt)), MAE_heading_deg_circular=np.mean(np.abs(wrap(phead - df.heading))),
             MAE_distance_from_xyz_m=np.mean(np.abs(np.sqrt(px**2 + py**2 + pz**2) - np.sqrt(df.x**2 + df.y**2 + df.z**2))))
    r = {k: float(v) for k, v in r.items()}
    if name: print(name, ' '.join(f'{k.split("_")[1]}={v:.4f}' for k, v in r.items()), flush=True)
    return r
