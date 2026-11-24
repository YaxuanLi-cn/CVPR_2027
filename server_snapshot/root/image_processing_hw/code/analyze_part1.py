"""Part 1 analysis: controlled degradations (3DGS on the GS-QA 'truck' reference camera path)."""
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT
from metrics import SIGN, LABEL

RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
METS = ["psnr", "ssim", "lpips_vgg", "lpips_alex", "flip", "cvvdp_img", "cvvdp_vid", "tpsnr"]
# (axis, level label, tag, step)
CONDS = [("iterations", "7000", "q_full", 7000), ("iterations", "3000", "q_full", 3000), ("iterations", "1000", "q_full", 1000),
         ("views", "226 (all)", "q_full", 7000), ("views", "32", "q_views32", 7000), ("views", "16", "q_views16", 7000),
         ("views", "8", "q_views8", 7000),
         ("gaussians", "1M", "q_cap1000000", 7000), ("gaussians", "100k", "q_cap100000", 7000),
         ("gaussians", "10k", "q_cap10000", 7000)]
rows = []
for axis, lab, tag, step in CONDS:
    f = f"{RES}/part1/truck__{tag}__step_{step:06d}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    r = {"axis": axis, "level": lab, "tag": tag, "step": step, "n_gs": d["log"]["num_gs"].get(str(step)),
         "cvvdp_vid": d["cvvdp_vid"], "tpsnr": d["tpsnr"]}
    r.update({k: float(np.mean(v)) for k, v in d["frames"].items()})
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{RES}/part1_truck_conditions.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(4).to_string())

u = df.drop_duplicates(["tag", "step"]).reset_index(drop=True)
Q = pd.DataFrame({m: SIGN[m] * u[m] for m in METS})           # quality-oriented
rho = Q.corr(method="spearman")
tau = Q.corr(method="kendall")
rho.to_csv(f"{RES}/part1_truck_metric_agreement_spearman.csv")
print("\nSpearman rank agreement between metrics over", len(u), "conditions\n", rho.round(2))
print("\nKendall tau\n", tau.round(2))

# pairs of conditions on which two metrics disagree about which is better
print("\nRank reversals between metrics (condition A vs B):")
names = [f"{a}={l}" for a, l in zip(u.axis, u.level)]
rev = []
for i in range(len(u)):
    for j in range(i + 1, len(u)):
        s = {m: np.sign(Q[m][i] - Q[m][j]) for m in METS}
        if len(set(s.values())) > 1:
            rev.append((names[i], names[j], [m for m in METS if s[m] > 0], [m for m in METS if s[m] < 0]))
            print(f"  {names[i]:22s} vs {names[j]:22s}: A better per {rev[-1][2]}; B better per {rev[-1][3]}")
json.dump(rev, open(f"{RES}/part1_truck_reversals.json", "w"), indent=1)

fig, axs = plt.subplots(2, 4, figsize=(16, 7))
col = {"iterations": "#1f77b4", "views": "#d62728", "gaussians": "#2ca02c"}
for ax, m in zip(axs.flat, METS):
    for axis, g in df.groupby("axis", sort=False):
        ax.plot(range(len(g)), g[m], "o-", color=col[axis], label=axis)
        for x, (l, v) in enumerate(zip(g.level, g[m])):
            ax.annotate(l, (x, v), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(LABEL[m] + (" (lower = better)" if SIGN[m] < 0 else ""))
    ax.set_xlabel("degradation level (0 = best)")
    ax.set_xticks(range(4)); ax.grid(alpha=.3)
axs.flat[0].legend(title="degraded factor")
fig.suptitle("3DGS on 'truck' (300-frame reference camera path): metric response to controlled degradations")
fig.tight_layout()
fig.savefig(f"{FIG}/part1_truck_curves.png", dpi=130)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(rho.values, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(METS))); ax.set_xticklabels([LABEL[m] for m in METS], rotation=35, ha="right")
ax.set_yticks(range(len(METS))); ax.set_yticklabels([LABEL[m] for m in METS])
for i in range(len(METS)):
    for j in range(len(METS)):
        ax.text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", color="w" if rho.values[i, j] < .7 else "k", fontsize=8)
fig.colorbar(im, label="Spearman rank correlation")
ax.set_title(f"Agreement between metrics ({len(u)} degraded 3DGS conditions)")
fig.tight_layout()
fig.savefig(f"{FIG}/part1_truck_agreement.png", dpi=130)
