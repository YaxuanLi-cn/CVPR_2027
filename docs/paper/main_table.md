# 主表（`tab:main_result`）：加入重训后的 π0.5 和 π0-FAST

所有行都在同一批 6,364 对测试样本上统计，数字来自 `figures/tables.json`（由 `figures/calc_tables.py` 生成）。

## LaTeX

```latex
\begin{table*}[!t]
    \centering
    \caption{\textbf{Image-goal pose estimation on PairUAV.} We report the MAE of each pose parameter, the success rate (SR) and the AUC of the pose error, the latter two in percents \%. DreamNav outperforms all policies, world models and representations when using either one or both stages. $\ddagger$ indicates distances predicted by the official distance head of the model.}
    \label{tab:main_result}
    \small
    \setlength{\tabcolsep}{4.2pt}
    \begin{tabular}{l|cccccc|c|ccc}
        \toprule
        & \multicolumn{6}{c|}{MAE $\downarrow$} & SR $\uparrow$ & \multicolumn{3}{c}{Pose AUC $\uparrow$} \\
        Methods & $\Delta x$ (m) & $\Delta y$ (m) & $\Delta z$ (m) & $d$ (m) & $\Delta\theta$ ($^\circ$) & $\Delta\phi$ ($^\circ$) & @1\,m & @5$^\circ$ & @10$^\circ$ & @20$^\circ$ \\
        \midrule
        AI2-THOR~\citep{7989381} & 7.46 & 7.44 & 15.41 & 19.32 & 85.80 & 18.22 & 0.1 & 0.3 & 0.8 & 1.8 \\
        $\pi_{0.5}$~\citep{intelligence2025pi_} & 9.36 & 9.57 & 19.45 & 7.64 & 80.72 & 4.16 & 8.4 & 0.9 & 2.8 & 6.7 \\
        $\pi_0$-FAST~\citep{pertsch2025fast} & 10.38 & 10.59 & 15.16 & 6.12 & 76.13 & 2.67 & 11.3 & 4.0 & 7.7 & 10.2 \\
        ViNT~\citep{shah2023vint} & 7.46 & 7.46 & 15.44 & 5.92$^\ddagger$ & 51.83 & 11.27 & 10.2$^\ddagger$ & 1.5 & 4.4 & 11.4 \\
        NoMaD~\citep{sridhar2024nomad} & 9.75 & 9.49 & 19.53 & 6.78$^\ddagger$ & 86.19 & 15.93 & 9.2$^\ddagger$ & 0.4 & 1.2 & 3.7 \\
        DINO-WM~\citep{zhou2025dinowm} & 8.92 & 9.06 & 19.29 & 10.23 & 84.59 & 11.09 & 6.2 & 0.4 & 1.4 & 4.3 \\
        Visual Foresight~\citep{ebert2018visual} & 8.80 & 8.67 & 18.64 & 10.30 & 83.09 & 7.97 & 5.7 & 0.5 & 1.9 & 5.6 \\
        DINOv3 ViT-7B~\citep{simeoni2025dinov3} & 7.70 & 7.70 & 14.27 & 13.39 & 86.23 & 4.44 & 2.7 & 0.7 & 2.3 & 5.5 \\
        Sample4Geo~\citep{deuser2023sample4geo} & 8.60 & 8.29 & 15.88 & 10.59 & 85.90 & 11.22 & 5.6 & 0.3 & 1.1 & 3.7 \\
        \midrule
        Ours (Stage I)  & 1.37 & 1.35 & 0.81 & 0.92 & 0.12 & 0.44 & 73.1 & 91.7 & 95.5 & 97.7 \\
        Ours (Stage II) & \textbf{0.99} & \textbf{0.98} & \textbf{0.54} & \textbf{0.68} & \textbf{0.07} & \textbf{0.20} & \textbf{83.3} & \textbf{96.3} & \textbf{97.9} & \textbf{98.9} \\
        \bottomrule
    \end{tabular}
\end{table*}
```

`\citep{pertsch2025fast}` 的 bib 条目在 `experiment_new_refs.bib` 里。

## 同一张表（便于核对）

| Methods | Δx (m) | Δy (m) | Δz (m) | d (m) | Δθ (°) | Δφ (°) | SR@1 m | AUC@5° | AUC@10° | AUC@20° |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AI2-THOR | 7.46 | 7.44 | 15.41 | 19.32 | 85.80 | 18.22 | 0.1 | 0.3 | 0.8 | 1.8 |
| π0.5 | 9.36 | 9.57 | 19.45 | 7.64 | 80.72 | 4.16 | 8.4 | 0.9 | 2.8 | 6.7 |
| π0-FAST | 10.38 | 10.59 | 15.16 | 6.12 | 76.13 | 2.67 | 11.3 | 4.0 | 7.7 | 10.2 |
| ViNT | 7.46 | 7.46 | 15.44 | 5.92‡ | 51.83 | 11.27 | 10.2‡ | 1.5 | 4.4 | 11.4 |
| NoMaD | 9.75 | 9.49 | 19.53 | 6.78‡ | 86.19 | 15.93 | 9.2‡ | 0.4 | 1.2 | 3.7 |
| DINO-WM | 8.92 | 9.06 | 19.29 | 10.23 | 84.59 | 11.09 | 6.2 | 0.4 | 1.4 | 4.3 |
| Visual Foresight | 8.80 | 8.67 | 18.64 | 10.30 | 83.09 | 7.97 | 5.7 | 0.5 | 1.9 | 5.6 |
| DINOv3 ViT-7B | 7.70 | 7.70 | 14.27 | 13.39 | 86.23 | 4.44 | 2.7 | 0.7 | 2.3 | 5.5 |
| Sample4Geo | 8.60 | 8.29 | 15.88 | 10.59 | 85.90 | 11.22 | 5.6 | 0.3 | 1.1 | 3.7 |
| Ours (Stage I) | 1.37 | 1.35 | 0.81 | 0.92 | 0.12 | 0.44 | 73.1 | 91.7 | 95.5 | 97.7 |
| Ours (Stage II) | **0.99** | **0.98** | **0.54** | **0.68** | **0.07** | **0.20** | **83.3** | **96.3** | **97.9** | **98.9** |

## 相对你贴的版本改了什么

1. **π0.5 一行整行换成重训后的结果**（官方 `pi05_base`，LoRA，1 轮，关掉了 openpi 默认的随机裁剪和旋转增强）。你贴的那行（8.93 / 9.14 / 19.10 / 8.28 / 83.99 / 7.50 / 7.5 / 0.5 / 1.7 / 5.2）是旧权重的结果。
2. **新增 π0-FAST 一行**（官方 `pi0_fast_base`，LoRA，1 轮，贪心解码），放在 π0.5 下面。
3. **ViNT、NoMaD 的 $d$ 上补回了 ‡**。你贴的版本只在 SR 上留了 ‡，但 5.92 和 6.78 这两个距离误差同样来自模型自带的距离头，而标题里 ‡ 的说明写的正是 "distances predicted by the official distance head"。不标的话，这两个数会被当成和其他行同口径。如果坚持不标，$d$ 应换成由预测位移模长算出的 18.91（ViNT）和 7.14（NoMaD）。

其余行没有改动。

## 正文里需要同步的地方

- **Baselines 一段的基线列表**里要加上 π0-FAST：`…AI2-THOR, $\pi_{0.5}$, $\pi_0$-FAST~\citep{pertsch2025fast}, ViNT, and NoMaD…`。仓库里的 `experiment.tex` 已经加了。
- **新的 π0.5 和 π0-FAST 数字不影响正文里的任何结论句**："Regression-based methods like $\pi_{0.5}$ and DINOv3 cannot predict a substantially more accurate horizontal displacement than the constant predictor" 仍然成立（两者的 Δx、Δy 都在 9–10.6 m，常数预测是 7.45 m）。
