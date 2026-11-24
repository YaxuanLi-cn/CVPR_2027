# 真实照片表（`tab:real`）：只保留训练范围内，加入三个基线

训练范围内 300 对，三个基线都是零样本、输出不做飞行高度缩放。数字来自 `/root/realdata/mezieres/baselines/real_baseline_metrics.json`（由 `online/real_baselines.py` 生成）。

## LaTeX

```latex
\begin{table}[!t]
    \centering
    \caption{\textbf{Zero-shot pose estimation on real drone photographs.} With data from a DJI Phantom 4 RTK, DreamNav outperforms the policy baselines in both coarse and refined settings, in the range seen in training. The medians of both the position error and the translation-direction error and the MAE of the tilt change are reported, with the SR at 1\,m and 2\,m.}
    \label{tab:real}
    \resizebox{\columnwidth}{!}{%
    \begin{tabular}{@{}l|ccccc@{}}
    \toprule
    Methods & Pos. (m) & SR@1\,m & SR@2\,m & $\Delta\phi$ & Dir. \\
    \midrule
    ViNT~\citep{shah2023vint} & 7.75 & 3.7 & 8.0 & 14.53 & 90.6 \\
    NoMaD~\citep{sridhar2024nomad} & 21.21 & 0.3 & 2.3 & 17.86 & 89.5 \\
    $\pi_0$-FAST~\citep{pertsch2025fast} & 25.93 & 0.0 & 0.0 & 4.43 & 91.8 \\
    \midrule
    Ours (Stage I) & 1.43 & 45.0 & 69.3 & 1.90 & 4.4 \\
    Ours (Stage II) & \textbf{0.84} & \textbf{63.3} & \textbf{80.3} & \textbf{1.62} & \textbf{1.9} \\
    \bottomrule
    \end{tabular}%
    }
\end{table}
```

## 同一张表（便于核对）

| Methods | Pos. (m) | SR@1 m | SR@2 m | Δφ (°) | Dir. (°) |
|---|---:|---:|---:|---:|---:|
| ViNT | 7.75 | 3.7 | 8.0 | 14.53 | 90.6 |
| NoMaD | 21.21 | 0.3 | 2.3 | 17.86 | 89.5 |
| π0-FAST | 25.93 | 0.0 | 0.0 | 4.43 | 91.8 |
| Ours (Stage I) | 1.43 | 45.0 | 69.3 | 1.90 | 4.4 |
| Ours (Stage II) | **0.84** | **63.3** | **80.3** | **1.62** | **1.9** |

## 相对你贴的版本改了什么

1. **加了三行基线**（ViNT、NoMaD、π0-FAST），放在我们的方法上面，用 `\midrule` 隔开，引用键和主表相同。
2. **标题第一句换了比较对象**："the regression baseline and the constant predictor" 改成 "the policy baselines"，因为那两行已经不在表里了。句式没有动。
3. **标题删掉了第二句**（"Rescaling by the flight height does not close the gap of the regression baseline but strongly benefits our approach."）：表里已经没有缩放前后对比的行，这句话没有数字支撑。想保留的话原样加回即可。
4. **Δφ 列的 1.62 加粗了**：原来这一列最好的是常数预测器（0.45），那一行去掉之后 Stage II 是最好的。
5. **两行 Ours 的数字没有变。**

## 需要你知道的几点

- **两行 Ours 用了飞行高度（60 m），三个基线没有用。** 这是按"只用不 rescaled 的结果"的决定来的；不用飞行高度的 Stage II 是 16.42 m / 2.0 / 3.7。
- **ViNT 和 NoMaD 的 SR 用的是预测位移的模长**，和表里其他行同一口径，所以没有加主表里的 ‡。如果想和主表一样改用模型自带距离头的输出，ViNT 是 0.0 / 0.0，NoMaD 是 0.0 / 0.3。
- **基线的位置误差和 Dir. 是在世界坐标系（东、北、上）里算的**，因为它们学的就是这个标签；Ours 是在第一张图的相机坐标系里算的。SR 是模长之差，和坐标系无关。把基线的输出当成相机坐标系来算，Pos. 是 ViNT 7.89 m、NoMaD 20.78 m、π0-FAST 25.61 m，结论不变。
- **三个基线都没有迁移到真实照片上**（真值距离平均 7.2 m，"原地不动"的 Pos. 是 7.84 m）：
  - ViNT 的位移输出接近 0（模长中位数 1.1 m），所以 Pos. 和"原地不动"基本相同。
  - NoMaD 输出的是训练尺度下的随机采样（模长中位数 19.4 m），远大于真值距离。
  - π0-FAST 的输出模长中位数是 25.1 m，300 对的 x 全部是同一个值（−9.69 m），所以 Pos. 最大、SR 为 0。
  - 这批照片的俯仰几乎不变（真值 Δφ 平均 0.45°）。π0-FAST 的俯仰最接近（4.43°），ViNT 和 NoMaD 仍然预测出十几度的变化。
- **π0-FAST 的数字在不同次运行之间有很小的波动**：贪心解码在这块 GPU 上不是逐位可复现的，换一个进程重跑，约三分之一的样本会解码出不同的结果。4 次独立运行的范围是 Pos. 25.88–25.93 m，SR 都是 0.0 / 0.0，Δφ 4.40–4.46°，Dir. 91.1–91.8°；表里填的是保存下来的那一次。这个现象在 PairUAV 测试集上同样存在（原评测脚本重跑两次，64 对里只有 45 对输出相同），所以主表里 π0-FAST 一行重跑时末位也可能变动，具体幅度我没有在全部测试集上测过。
- **仓库里的 `experiment.tex` 没有改**，这份文件只是更新后的表格。

## 怎么复现

- ViNT / NoMaD：`/root/baseline/visualnav-transformer/pairuav5d_v2/eval_real.py`。用同一份代码重跑 256 对 PairUAV 测试样本，和已保存的测试预测逐位一致。
- π0-FAST：`/root/baseline/openpi/pairuav_v2/eval_real.py`。图像预处理和已有的 224 缓存逐像素一致。
- 指标：`online/real_baselines.py`。用它重算的 Constant、RASR 和三行 Ours，和 `real_table.md` 里的数字完全一致。
- 逐对的预测保存在 `/root/realdata/mezieres/baselines/`。
