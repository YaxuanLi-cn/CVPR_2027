"""Part 2 analysis: agreement between metrics and the GS-QA subjective scores (DMOS)."""
import glob, json, os
import numpy as np
import pandas as pd
import openpyxl
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT
from metrics import SIGN, LABEL

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
METS = ["psnr", "ssim", "lpips_vgg", "lpips_alex", "flip", "cvvdp_img", "cvvdp_vid"]
rng = np.random.default_rng(0)
NBOOT = 2000


def read_xlsx(path):
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    hdr, out, scene = rows[0], [], None
    for r in rows[1:]:
        scene = r[0] or scene
        out.append(dict(zip(hdr, (scene,) + r[1:])))
    return pd.DataFrame(out)


def load():
    dmos = read_xlsx(f"{ROOT}/data/gsqa/dmos.xlsx").rename(columns={"Scene": "scene", "Method": "method", "DMOS": "dmos"})
    rows, frames = [], {}
    for f in sorted(glob.glob(f"{RES}/gsqa/*.json")):
        d = json.load(open(f))
        r = {"scene": d["scene"], "method": d["method"], "traj": "360" if d["group"] == "T&T" else "FF",
             "fps": d["fps"], "cvvdp_vid": d["cvvdp_vid"], "tpsnr": d["tpsnr"]}
        for k, v in d["frames"].items():
            r[k] = float(np.mean(v))
        frames[(d["scene"], d["method"])] = {k: np.asarray(v) for k, v in d["frames"].items()}
        rows.append(r)
    df = pd.DataFrame(rows).merge(dmos, on=["scene", "method"], validate="one_to_one")
    return df, frames


def srocc(x, y):
    return stats.spearmanr(x, y).statistic


def logistic(x, a, b, c, d):
    return (a - b) / (1 + np.exp(-(x - c) / (abs(d) + 1e-9))) + b


def plcc_fitted(x, y):
    """Pearson correlation after the 4-parameter monotonic logistic mapping (ITU-T P.1401 style)."""
    try:
        p0 = [y.max(), y.min(), np.mean(x), np.std(x) + 1e-6]
        p, _ = curve_fit(logistic, x, y, p0=p0, maxfev=20000)
        return stats.pearsonr(logistic(x, *p), y).statistic
    except Exception:
        return stats.pearsonr(x, y).statistic


def within_scene(df, m, fn=srocc):
    return float(np.mean([fn(SIGN.get(m, 1) * g[m].values, g["dmos"].values) for _, g in df.groupby("scene")]))


def boot_ci(df, mets, stat, nboot=NBOOT, stratified=False):
    """Bootstrap over stimuli (stratified: resample within each scene). Returns dict m -> array of replicates."""
    reps = {m: [] for m in mets}
    groups = [g.index.values for _, g in df.groupby("scene")]
    for _ in range(nboot):
        if stratified:
            idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        else:
            idx = rng.choice(df.index.values, len(df))
        b = df.loc[idx]
        for m in mets:
            reps[m].append(stat(b, m))
    return {m: np.asarray(v) for m, v in reps.items()}


def s_all(b, m):
    return srocc(SIGN.get(m, 1) * b[m].values, b["dmos"].values)


def s_within(b, m):
    out = []
    for _, g in b.groupby("scene"):
        if g[m].nunique() > 2:
            out.append(srocc(SIGN.get(m, 1) * g[m].values, g["dmos"].values))
    return float(np.nanmean(out))


def table(df, mets, name):
    rows = []
    for subset, sub in [("all", df), ("360", df[df.traj == "360"]), ("FF", df[df.traj == "FF"])]:
        sub = sub.reset_index(drop=True)
        ra = boot_ci(sub, mets, s_all)
        rw = boot_ci(sub, mets, s_within, stratified=True)
        for m in mets:
            x, y = SIGN.get(m, 1) * sub[m].values, sub["dmos"].values
            rows.append({"subset": subset, "metric": m, "n": len(sub),
                         "srocc": srocc(x, y), "srocc_lo": np.nanpercentile(ra[m], 2.5), "srocc_hi": np.nanpercentile(ra[m], 97.5),
                         "krocc": stats.kendalltau(x, y).statistic, "plcc": plcc_fitted(x, y),
                         "srocc_within": within_scene(sub, m), "within_lo": np.nanpercentile(rw[m], 2.5),
                         "within_hi": np.nanpercentile(rw[m], 97.5)})
        # paired bootstrap: difference to the best metric on the within-scene statistic
        best = max(mets, key=lambda m: within_scene(sub, m))
        for r in rows:
            if r["subset"] == subset:
                d = rw[best] - rw[r["metric"]]
                r["best_within"] = best
                r["p_worse_than_best"] = float(np.mean(d <= 0)) if r["metric"] != best else np.nan
    t = pd.DataFrame(rows)
    t.to_csv(f"{RES}/{name}.csv", index=False)
    return t


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    df, frames = load()
    print("stimuli:", len(df))
    df.to_csv(f"{RES}/gsqa_scores.csv", index=False)

    # ---- sanity check against the numbers published with the dataset
    pub = read_xlsx(f"{ROOT}/data/gsqa/metrics.xlsx").rename(columns={"Scene": "scene", "Method": "method"})
    chk = df.merge(pub, on=["scene", "method"])
    print("sanity (ours vs published): PSNR-YUV r=%.3f, SSIM r=%.3f, LPIPS r=%.3f" % (
        stats.pearsonr(chk.psnr, chk["PSNR-YUV"]).statistic, stats.pearsonr(chk.ssim, chk["SSIM"]).statistic,
        stats.pearsonr(chk.lpips_vgg, chk["LPIPS"]).statistic))

    # ---- temporal pooling variants of the per-frame metrics
    pool_rows = []
    for (scene, method), fr in frames.items():
        r = {"scene": scene, "method": method}
        for k, v in fr.items():
            q = SIGN[k] * v                      # quality-oriented: higher is better
            r[f"{k}|mean"] = q.mean()
            r[f"{k}|p10"] = np.percentile(q, 10)  # mean of the worst frames matters?
            r[f"{k}|min"] = q.min()
            r[f"{k}|std"] = -q.std()             # temporal fluctuation of per-frame quality
        pool_rows.append(r)
    pool = pd.DataFrame(pool_rows).merge(df[["scene", "method", "dmos", "traj"]], on=["scene", "method"])
    prow = []
    for c in [c for c in pool.columns if "|" in c]:
        k, pl = c.split("|")
        prow.append({"metric": k, "pooling": pl, "srocc": srocc(pool[c], pool.dmos),
                     "srocc_within": float(np.mean([srocc(g[c], g.dmos) for _, g in pool.groupby("scene")]))})
    pd.DataFrame(prow).to_csv(f"{RES}/gsqa_pooling.csv", index=False)

    t = table(df, METS + ["tpsnr"], "gsqa_correlations")
    pd.set_option("display.width", 250)
    print(t.round(3).to_string())
    print(pd.DataFrame(prow).pivot(index="metric", columns="pooling", values="srocc_within").round(3))

    # per-scene SROCC matrix
    ps = pd.DataFrame({m: {s: srocc(SIGN.get(m, 1) * g[m], g.dmos) for s, g in df.groupby("scene")} for m in METS})
    ps.to_csv(f"{RES}/gsqa_per_scene_srocc.csv")
    print(ps.round(2))

    # leave-one-method-out is not meaningful with 8 methods; instead: per-method SROCC across scenes
    pm = pd.DataFrame({m: {s: srocc(SIGN.get(m, 1) * g[m], g.dmos) for s, g in df.groupby("method")} for m in METS})
    pm.to_csv(f"{RES}/gsqa_per_method_srocc.csv")

    # ---------------- figures
    cmap = plt.get_cmap("tab10")
    scenes = list(dict.fromkeys(df.scene))
    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    for ax, m in zip(axs.flat, METS + ["tpsnr"]):
        for i, s in enumerate(scenes):
            g = df[df.scene == s]
            ax.scatter(g[m], g.dmos, color=cmap(i), s=28, marker="o" if g.traj.iloc[0] == "360" else "^", label=s)
        r = t[(t.subset == "all") & (t.metric == m)].iloc[0]
        ax.set_title(f"{LABEL[m]}\nSROCC {r.srocc:.2f}  (within-scene {r.srocc_within:.2f})", fontsize=10)
        ax.set_xlabel(LABEL[m] + (" (lower = better)" if SIGN[m] < 0 else ""))
        ax.set_ylabel("DMOS (higher = better)")
        ax.grid(alpha=.3)
    axs.flat[0].legend(fontsize=7, ncol=2, title="circles: 360°, triangles: forward-facing", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/gsqa_scatter.png", dpi=130)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, (col, lo, hi, title) in zip(axs, [("srocc", "srocc_lo", "srocc_hi", "SROCC over all stimuli of the subset"),
                                               ("srocc_within", "within_lo", "within_hi", "Mean within-scene SROCC")]):
        wd = 0.26
        for j, (subset, c) in enumerate([("all", "#333333"), ("360", "#1f77b4"), ("FF", "#ff7f0e")]):
            tt = t[t.subset == subset].set_index("metric").loc[METS]
            x = np.arange(len(METS)) + (j - 1) * wd
            ax.bar(x, tt[col], wd, color=c, label=f"{subset} (n={tt.n.iloc[0]})",
                   yerr=[tt[col] - tt[lo], tt[hi] - tt[col]], capsize=2, error_kw={"lw": .8})
        ax.set_xticks(np.arange(len(METS)))
        ax.set_xticklabels([LABEL[m] for m in METS], rotation=25, ha="right")
        ax.set_title(title)
        ax.axhline(0, color="k", lw=.5)
        ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("Spearman correlation with DMOS")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(f"{FIG}/gsqa_srocc.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(ps.loc[scenes, METS].values, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(METS))); ax.set_xticklabels([LABEL[m] for m in METS], rotation=25, ha="right")
    ax.set_yticks(range(len(scenes))); ax.set_yticklabels([f"{s} ({df[df.scene == s].traj.iloc[0]})" for s in scenes])
    for i in range(len(scenes)):
        for j in range(len(METS)):
            ax.text(j, i, f"{ps.loc[scenes[i], METS[j]]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="SROCC with DMOS (8 methods per scene)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/gsqa_per_scene.png", dpi=130)
