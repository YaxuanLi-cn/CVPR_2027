# 真实照片表（`tab:real`）更新版

## 结论

- **你贴的版本里所有数字都是最新的。** 我用 `/root/realdata/mezieres/pred.pkl` 逐格复算了一遍，60 个单元格全部一致；SR 用的也已经是和主表相同的口径（预测位移模长与真值模长之差 ≤ 1 m / 2 m）。
- **唯一和你最近的决定不一致的是 `Dir.` 这一列**（平移方向误差的中位数）。你已经决定 Pose AUC 不再考虑平移方向、只看旋转，所以下面的更新版把 `Dir.` 换成了航向变化的 MAE（`Δθ`），和主表的列一致。其余五列数字不变。

## LaTeX（更新版）

```latex
\begin{table*}[!t]
    \centering
    \caption{\textbf{Zero-shot pose estimation on real drone photographs.} With data from a DJI Phantom 4 RTK, DreamNav outperforms the regression baseline and the constant predictor in both coarse and refined settings, in the range seen in training. Rescaling by the flight height does not close the gap of the regression baseline but strongly benefits our approach. The median of the position error and the MAEs of both the heading change and the tilt change are reported, with the SR at 1\,m and 2\,m.}
    \label{tab:real}
    \small
    \setlength{\tabcolsep}{3.5pt}
    \begin{tabular}{@{}l|ccccc|ccccc@{}}
    \toprule
    & \multicolumn{5}{c|}{Within training range (300 pairs)} & \multicolumn{5}{c}{Beyond training range (300 pairs)} \\
    Methods & Pos. (m) & SR@1\,m & SR@2\,m & $\Delta\theta$ & $\Delta\phi$ & Pos. (m) & SR@1\,m & SR@2\,m & $\Delta\theta$ & $\Delta\phi$ \\
    \midrule
    Constant (zero motion)       & 7.84  & 1.3  & 3.7  & 83.71 & \textbf{0.45} & 21.71 & 0.0 & 0.0 & 101.73 & \textbf{0.55} \\
    RASR + sin/cos               & 11.42 & 5.7  & 11.0 & 3.72 & 3.40 & 23.54 & 3.0 & 5.0 & 19.95 & 7.27 \\
    RASR + sin/cos, rescaled     & 5.10  & 18.3 & 34.3 & 3.72 & 3.40 & 18.88 & 0.0 & 2.0 & 19.95 & 7.27 \\
    \midrule
    Ours (Stage II), no flight height & 16.42 & 2.0 & 3.7 & \textbf{2.25} & 1.62 & 24.13 & 1.0 & 2.3 & \textbf{17.11} & 4.94 \\
    Ours (Stage I)               & 1.43  & 45.0 & 69.3 & 2.31 & 1.90 & 18.27 & 0.0 & 7.3 & 17.37 & 5.12 \\
    Ours (Stage II)              & \textbf{0.84} & \textbf{63.3} & \textbf{80.3} & \textbf{2.25} & 1.62 & \textbf{18.03} & \textbf{5.0} & \textbf{11.0} & \textbf{17.11} & 4.94 \\
    \bottomrule
    \end{tabular}
\end{table*}
```

## 同一张表（便于核对）

训练范围内（300 对）：

| Methods | Pos. (m) | SR@1 m | SR@2 m | Δθ (°) | Δφ (°) |
|---|---:|---:|---:|---:|---:|
| Constant (zero motion) | 7.84 | 1.3 | 3.7 | 83.71 | **0.45** |
| RASR + sin/cos | 11.42 | 5.7 | 11.0 | 3.72 | 3.40 |
| RASR + sin/cos, rescaled | 5.10 | 18.3 | 34.3 | 3.72 | 3.40 |
| Ours (Stage II), no flight height | 16.42 | 2.0 | 3.7 | **2.25** | 1.62 |
| Ours (Stage I) | 1.43 | 45.0 | 69.3 | 2.31 | 1.90 |
| Ours (Stage II) | **0.84** | **63.3** | **80.3** | **2.25** | 1.62 |

训练范围外（300 对）：

| Methods | Pos. (m) | SR@1 m | SR@2 m | Δθ (°) | Δφ (°) |
|---|---:|---:|---:|---:|---:|
| Constant (zero motion) | 21.71 | 0.0 | 0.0 | 101.73 | **0.55** |
| RASR + sin/cos | 23.54 | 3.0 | 5.0 | 19.95 | 7.27 |
| RASR + sin/cos, rescaled | 18.88 | 0.0 | 2.0 | 19.95 | 7.27 |
| Ours (Stage II), no flight height | 24.13 | 1.0 | 2.3 | **17.11** | 4.94 |
| Ours (Stage I) | 18.27 | 0.0 | 7.3 | 17.37 | 5.12 |
| Ours (Stage II) | **18.03** | **5.0** | **11.0** | **17.11** | 4.94 |

## 相对你贴的版本改了什么

1. **`Dir.` 列换成 `Δθ`（航向变化的 MAE，单位度）**，并挪到 `Δφ` 前面，列顺序和主表一致。被换掉的数值（平移方向误差中位数）：范围内 90.0 / 41.0 / 41.0 / 1.9 / 4.4 / 1.9，范围外 90.0 / 55.5 / 55.5 / 45.5 / 48.1 / 45.5。
2. **标题最后一句相应改写**：原句是 "The medians of both the position error and the translation-direction error and the MAE of the tilt change are reported, …"，现在是 "The median of the position error and the MAEs of both the heading change and the tilt change are reported, …"。句式仍然沿用原来的出处（LoFTR："The evaluation results on both the … and the … are reported."）。
3. **加粗**：`Δθ` 列里 Stage II 两行并列最好（带不带飞行高度只影响尺度，不影响旋转），和原来 `Dir.` 列的加粗方式相同。

## 需要你知道的几点

- **如果你想保留 `Dir.` 列，那你贴的版本不用改**，它已经是最新的。
- **换成 `Δθ` 之后，正文不需要改数字**：Results 一段（T10–T16）没有引用 `Dir.` 列的数值；T10 "in all metrics" 对新的一列仍然成立（范围内 2.25° 对 3.72°，范围外 17.11° 对 19.95°）。
- **`Δθ` 在训练范围外的数值很大（17°–20°）**，这是均值被少数匹配失败的样本拉高的结果，中位数是 3.3°（Stage II）和 6.5°（RASR）。如果想让这一列和 `Pos.` 一样报中位数，范围内是 89.34 / 1.93 / 1.93 / 0.67 / 0.72 / 0.67，范围外是 91.45 / 6.52 / 6.52 / 3.32 / 3.88 / 3.32。
- **如果想放只看旋转的 Pose AUC（和主表同口径）**，@5° / 10° / 20° 的数值是：范围内 Constant 1.5 / 2.6 / 4.5，RASR + sin/cos 32.2 / 59.9 / 77.8，Stage I 63.8 / 78.2 / 87.4，Stage II 68.5 / 80.8 / 88.6；范围外 Constant 16.3 / 17.5 / 18.3，RASR + sin/cos 8.4 / 24.0 / 43.4，Stage I 27.4 / 37.8 / 52.2，Stage II 30.6 / 39.9 / 53.5。
- **仓库里的 `experiment.tex` 还没有同步这一改动**，这份文件只是给你的更新版；确认要用之后我再把表和标题句同步进去。
