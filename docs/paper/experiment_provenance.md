# Experiment 逐句结构来源对照表

对应文件：`experiment.tex`（同目录）。共 **131 句**（正文 104 句，图表标题 27 句），每一句的句式骨架都取自下表 11 篇论文中的一句原文；内容（数字、设置、结论）来自仓库里的实验记录和本次重新统计的结果，出处见“数字来源”。

## 核验方式

- **原句逐字核验**：每条“原”都由脚本在该论文的 arXiv LaTeX 源码（Nature 论文为 nature.com 正文）中做了逐字匹配，全部命中；`experiment.tex` 里除表格和小标题外没有未登记的句子。下面显示时只做了排版清理：引用写成 `[cite]`、交叉引用写成 `[ref]`、去掉 `\emph` 等格式命令。
- **会议正式版交叉核对**：又把每条原句的文字部分与会议官方 PDF（CVF Open Access / NeurIPS Proceedings）做了自动比对，129/131 句完全一致。两处例外都已人工确认：**I1**（Reloc3r）在正式版里文字相同，只是 PDF 中这句话被一张表格隔成两段，自动比对没接上；**T8**（DUSt3R）正式版的括号内措辞与 arXiv 版略有不同（“serve … in Sec. F of the suppl.”，句末多了 “(zero-shot settings)”），表中 T8 已直接采用正式版措辞，句式骨架不受影响。
- 闭环一节的句子（S7–S18、C20、C21）是仿真跑完后补写的：来自 Nature 论文的句子比对的就是 nature.com 正式版正文，其余来自 PixLoc、OrienterNet、LoFTR、NWM 的句子已与 CVF 正式版 PDF 比对一致。
- **只用正文**：所有原句都出自论文正文或正文图表的标题，没有用补充材料里的句子。
- **句子拆分口径**：以句号为界；带编号的列举（E3、B1）整体算一句；图表标题里加粗的小标题与紧随其后的第一句算一句。段落小标题、表格内容不计。

## 来源论文

| 简称 | 论文 | 发表 | 取用句数 |
|---|---|---|---:|
| Kaufmann et al. (Swift) | Kaufmann et al., *Champion-level drone racing using deep reinforcement learning* | Nature 620, 2023 | 21 |
| PixLoc | Sarlin et al., *Back to the Feature: Learning Robust Camera Localization from Pixels to Pose* | CVPR 2021 | 18 |
| SuperGlue | Sarlin et al., *SuperGlue: Learning Feature Matching with Graph Neural Networks* | CVPR 2020 (Oral) | 17 |
| OrienterNet | Sarlin et al., *OrienterNet: Visual Localization in 2D Public Maps with Neural Matching* | CVPR 2023 | 16 |
| Pixel-Perfect SfM | Lindenberger et al., *Pixel-Perfect Structure-from-Motion with Featuremetric Refinement* | ICCV 2021 (Oral) | 16 |
| LoFTR | Sun et al., *LoFTR: Detector-Free Local Feature Matching with Transformers* | CVPR 2021 | 12 |
| Navigation World Models | Bar et al., *Navigation World Models* | CVPR 2025 | 10 |
| LightGlue | Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed* | ICCV 2023 | 8 |
| Kendall & Gal | Kendall and Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* | NIPS 2017 | 8 |
| DUSt3R | Wang et al., *DUSt3R: Geometric 3D Vision Made Easy* | CVPR 2024 | 3 |
| Reloc3r | Dong et al., *Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization* | CVPR 2025 | 2 |

没有使用列表之外的来源；ECCV、CoRL、ICRA、TPAMI、arXiv-only 等不在你指定范围内的论文一律未用作句式来源（ViNT、NoMaD、MASt3R 等只作为被引用的方法出现）。

## 数字来源

| 位置 | 内容 | 出处 |
|---|---|---|
| 表 `tab:main_result` | 所有方法在**同一批 6,364 对**测试样本上的 MAE、成功率、位姿 AUC | `figures/calc_tables.py` 重新统计，结果存在 `figures/tables.json`。我们的两行、RASR 两行与实验日志里的数字一致 |
| 表 `tab:ablation` | 逐项累加的消融，同一批 6,364 对 | 同上；各行对应的预测文件见脚本 |
| 表 `tab:geom` | 未融合的两视图几何精度，全部 31,273 对 | `SUPERGLUE_EXPERIMENTS.md` §19 |
| 表 `tab:imag` | (a) 400 对上的想象配置对比；(b) 3,000 对上的融合对照 | `STAGE2_IMAGINATION.md` §2、§5.1 |
| A4 的 47% / 99.3% | 原始匹配器 / 旋转搜索后能解出位姿的比例 | `SUPERGLUE_EXPERIMENTS.md` §3（2,000 对抽样）、§4（600 对抽样），两个抽样不同 |
| G8、G9 | 想象图与真实图的位移 7.6 → 5.0 像素；平移方向中位误差 1.16° → 0.41° → 0.38° | `STAGE2_IMAGINATION.md` §2、§5.2（400 对） |
| 图 `fig:cdf` | 误差累积分布 | `figures/errs.npz`（由 `calc_tables.py` 生成） |
| 图 `fig:breakdown`、`fig:reliability`，N5–N6、V4、V7 | 按难度分组、置信区间、误差预测的可靠性 | `outputs/analysis/extended_metrics.json` |
| L2 的 2,422 / 44% / 17% / 39% | 1 m 阈值下失败样本的归因 | `figures/calc_tables.py` 最后一段（本次新增统计） |
| 表 `tab:real`、图 `fig:real`、T9 的数字 | 600 对真实无人机照片 | `/root/realdata/mezieres/real_metrics.json`、`pred.pkl` |
| I1 的耗时 | 粗估计约 1 s、精修 0.2–0.5 s | `/root/realdata/mezieres/run.log` 里的分阶段计时 |
| 图 `fig:qualitative` | 单栏 3×3 九宫格（方形小图）：前两列是 PairUAV 测试集的 2 对，最后一列是 Mézières 真实照片 1 对；三行依次为 source（取整幅图中包含黄框的等高方形窗口）、想象的 target、真实 target（后两行裁到两台相机都能看到的最大正方形） | `figures/make_qualitative.py`。样本是人工挑的成功例：先筛出“公共可见区域 ≥ 40%、想象精修成功且残差位移中位数 ≤ 8 px（真实照片 ≤ 40 px）、视角变化明显”的样本，再按场景多样性手选；变换用的始终是粗估计位姿，不是真值 |
| 表 `tab:nav`、图 `fig:nav`、S10b–S18 | 闭环仿真（8 个地点、160 个回合、最多 5 步），自停规则下的 Stage I 与带近距离模式的 Stage II（仿真日志里的策略 `s1` 与 `s2p`） | `outputs/sim/nav_log.pkl`、`outputs/sim/nav_report.md`；阈值扫描由 `figures/make_figures.py nav` 计算（$\tau$ = 0.5、1、2 m 与 `nav_report.md` 一致） |

## 待补和需要你确认的地方

1. **闭环一节的 “Stage II” 是带近距离模式（地平面伺服）的 Stage II。** 按你的决定，Method 末尾补了 “Short-Range Mode” 一段（`method.tex`，标签 `sec:servo`），闭环设置里用 S7b 说明：闭环评测启用该模式，前面的离线实验没有启用。各阈值下的结果都如实报告：$\tau$ ≤ 1.25 m 时 Stage II 明显更好（1 m 时 81.9% 对 56.9%，按回合重采样的 95% 区间为 +16 到 +35 个百分点）；$\tau$ = 1.5 m 时 73.1% 对 70.0%，差异不显著；$\tau$ = 2 m 时 Stage I 略高（70.6% 对 67.5%，同样不显著）。正文 S14、S14b 写了这一点。不带近距离模式的 Stage II、世界系变体、RASR 的闭环结果和固定飞满 5 步的成功率没有写进论文，原始数字在 `outputs/sim/nav_report.md`。离线表格（主表、消融、真实数据）里的 Stage II 都不含近距离模式。
2. **ViNT、NoMaD 两行已填**，来自 `visualnav-transformer/pairuav5d_v2/runs/` 的新结果，并限制在同一批 6,364 对上。它们的 $d$ 和 SR 用的是模型自带的距离头（表中标 ‡，与 `benchmark_results.md` 的做法一致）；若像其他行一样由 xyz 的模长计算，ViNT 是 18.91 m / 0.3%，NoMaD 是 7.14 m / 8.6%。
3. **成功率的口径。** 表里的 SR 按你的要求用回原口径：预测位移的模长与真值模长相差不超过 1 m 即算成功（与 `benchmark_results.md` 一致）。这个口径不看方向，所以绝对航向那两步（消融表的最后一行、“真实绝对航向”一行）不改变 SR。图 `fig:cdf`、`fig:breakdown`、`fig:real` 和失败归因（L2）用的仍是三维位置误差，正文 Metrics 一段对两者都做了定义。
4. **基线的数字与 `benchmark_results.md` 略有出入**，因为这里把基线也限制在与我们相同的 6,364 对上（原来是 12,313 对）。$\pi_{0.5}$ 一行现在是重训后的结果（`baseline/openpi/pairuav_v2/results/pi05/`：官方 `pi05_base`，LoRA，1 轮，关掉了 openpi 默认的随机裁剪和旋转增强），和其他行一样在这 6,364 对上统计，原来的 † 已去掉；旧权重的 5 维预测保存在 `results/pi05_old/`，同一批样本上是 8.93 / 9.14 / 19.10 / 8.28 / 83.99 / 7.50 / SR 7.5 / AUC 0.5 / 1.7 / 5.2。新增的 $\pi_0$-FAST 一行来自 `baseline/openpi/pairuav_v2/results/pi0_fast/`（官方 `pi0_fast_base`，LoRA，1 轮，贪心解码），B1 的基线列表里相应加了它，bib 条目 `pertsch2025fast` 在 `experiment_new_refs.bib`。
5. **数据集描述按仓库里实际使用的数据写**（1,412 个地点、每个地点 150 张图、训练 24,909 对、测试 6,364 对），与原稿的“72 scenes / 1,402 buildings / 1:1:4 划分”不同，需要和数据集一节统一；换成全量结果时这几个数字要一起改。
6. **B2 的表述**：“所有基线都在同一划分的训练地点上训练”是准确的，但基线用的是 47,687 对训练样本，我们的融合只用了 24,909 对（第二阶段 12,000 对）。
7. **A5 指向附录 I**：需要把“按 $|\Delta\theta|$ 分组的匹配数”那张表（`SUPERGLUE_EXPERIMENTS.md` §3）放进附录。
8. **粗估计约 1 s、精修 0.2–0.5 s 的耗时是在 GPU 被其他任务占用时测的**，偏保守；这句话已从正文移到附录（见第 14 条），放进附录前最好在空闲的 GPU 上重测。
9. **近距离模式的相机高度**：仿真里它在第一步由粗估计阶段给出，之后按已执行的爬升量递推（`online/sim_nav.py`）；Method 里只写了“使用式 (homography) 的相机高度 $h$”，如需写得更细可以补一句。
10. **删掉了原稿的两块内容**：Drone-X（YouTube 视频）一节和扩散想象的 $3\times3$ 图。新方法不使用扩散模型，仓库里也没有新方法在 Drone-X 上的结果；真实数据实验换成了 M\'ezi\`eres。
11. **M\'ezi\`eres 数据集的引用条目**（`experiment_new_refs.bib` 里的 `heigvd_mezieres`）是按数据集 README 写的，投稿前请核对数据集作者希望的引用格式。
12. **表 `tab:imag`c（扩散检索对比）与 G10–G12、C10b 是新加的。** “Diffusion retrieval” 一行的五个数字直接取自 `docs/paper/diffusion_results.jpg`（X 1.958、Y 1.872、Z 1.429、tilt 0.655、heading 0.433）。那张图的“初始预测”一列（1.372 / 1.349 / 0.812 / 0.440 / 0.121）与我们 Stage I 在 6,364 对上的数字（1.373 / 1.346 / 0.809 / 0.442 / 0.121）保留两位小数后完全相同，但不是逐位相同，说明扩散实验用的测试样本与这 6,364 对略有出入；图里没有记录样本数，所以表头没有写样本数，“Geometric imagination (ours)” 一行用的是 6,364 对上的 Stage II。需要你确认三点：（1）扩散实验的测试样本数；（2）G10 对做法的描述（在粗估计附近取候选位姿、用扩散模型合成视图、与目标视图比较后选最接近的）是按旧版 `dreamNav/step2` 的流程写的，5 自由度版本若有不同请改；（3）G12 的原因解释是推断，没有单独做实验验证。
13. **Pose AUC 的口径改成只看旋转。** 按你的决定，位姿误差取航向误差和俯仰误差中的较大值，不再包含平移方向误差（M4 和表 `tab:main_result` 的三列都已更新，计算在 `analysis/extended_metrics.py` 的 `pose_ang`）。这样基线之间能区分开（ViNT 11.4、RASR 71.0，@20°），但这一列不再反映平移方向，所以绝对航向那两步（规范视图航向、真实绝对航向）对它没有影响。改之前的数值（含平移方向）：Stage I 46.2 / 62.5 / 75.5，Stage II 61.9 / 74.5 / 83.7，RASR 0.2 / 1.0 / 4.8，其余基线 @20° 不超过 0.3。
14. **Implementation Details 一段按你的要求压缩成三句**（I0 基线初始化、I1 GPU、I3 指向附录）。从正文删掉、需要放进附录 I 的内容：DreamNav 用 PyTorch 实现；粗估计一对图像约 1 s、精修 0.2–0.5 s（取决于图块数）；SuperPoint、SuperGlue、MASt3R 使用原始预训练模型。各基线的具体做法也应写进附录：π0.5 用官方 `pi05_base` 做 LoRA 微调；ViNT、NoMaD 用官方权重全量微调；DINOv3、Sample4Geo 冻结官方主干、只训练位姿头；DINO-WM 冻结官方 DINOv2 编码器、预测器从头训练；RASR 冻结 MASt3R、回归头从头训练；Visual Foresight 没有官方权重，在缓存的 DINOv3 特征上从头训练；AI2-THOR 目前没有用任何预训练权重（输入是 16×16 缩略图），I0 里的 “when available” 对它是勉强成立的，建议按原论文改用冻结的 ImageNet ResNet-50 特征重跑。

## 逐句对照

每条三行：**新** = `experiment.tex` 中的句子；**原** = 来源原句；**构** = 两句共用的句式骨架。

### Opening

**E1** · OrienterNet（CVPR 2023）
```text
新: We evaluate our single model for image-goal navigation in the context of both offline pose estimation and closed-loop flight.
原: We evaluate our single model for localization in the context of both driving and AR.
构: We evaluate our single model for X in the context of both A and B.
```
**E2** · OrienterNet（CVPR 2023）
```text
新: Figure~\ref{fig:qualitative} shows qualitative examples, while Figure~\ref{fig:breakdown} illustrates the effectiveness of the imagination-refined stage.
原: [ref] shows qualitative examples, while [ref] illustrates the effectiveness of multi-frame fusion.
构: Fig. A shows qualitative examples, while Fig. B illustrates the effectiveness of X.
```
**E3** · OrienterNet（CVPR 2023）
```text
新: Our experiments show that: 1) DreamNav is more accurate than existing navigation policies, world models and visual representations for pose estimation between aerial views; 2) Geometric imagination helps estimate every pose parameter more accurately than the coarse stage alone; 3) DreamNav is significantly more accurate than a regression baseline on real drone photographs when considering baselines within the training range.
原: Our experiments show that: 1) OrienterNet is more effective than existing deep networks for localization with 2D maps; 2) Planimetric maps help localize more accurately than overhead satellite imagery; 3) OrienterNet is significantly more accurate than an embedded consumer-grade GPS sensor when considering multiple views.
构: Our experiments show that: 1) X is more [adj] than existing A for B; 2) C help [verb] more accurately than D; 3) X is significantly more accurate than E when considering F.
```

### Experimental setup: dataset

**D1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: We use PairUAV (Section~\ref{PairUAV}), a large-scale aerial dataset composed of 1,412 sites with ground truth poses for every image, and well-defined training and test splits corresponding to different sites.
原: We use ScanNet [cite], a large-scale indoor dataset composed of monocular sequences with ground truth poses and depth images, and well-defined training, validation, and test splits corresponding to different scenes.
构: We use X, a large-scale [adj] dataset composed of A with ground truth B, and well-defined training ... and test splits corresponding to different scenes.
```
**D2** · LoFTR（CVPR 2021）
```text
新: Each site contains 150 images with headings over the full circle and tilts of up to $45^\circ$ from nadir.
原: ScanNet contains 1613 monocular sequences with ground truth poses and depth maps.
构: X contains N [items] with A and B.
```
**D3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Holding out 290 sites for testing, we select 24,909 training and 6,364 test pairs.
原: Discarding pairs with too small or too large overlap, we select 230M training and 1500 test pairs.
构: [Gerund] A, we select N training and M test pairs.
```
**D4** · LoFTR（CVPR 2021）
```text
新: This dataset contains image pairs with baselines of up to 38\,m and arbitrary heading changes.
原: This dataset contains image pairs with wide baselines and extensive texture-less regions.
构: This dataset contains image pairs with A and B.
```

### Experimental setup: metrics

**M1** · PixLoc（CVPR 2021）
```text
新: We report for each pose parameter the mean absolute error (MAE), in metres or degrees, as well as the success rate (SR) at 1\,m for the distance.
原: We report for each scene the median translation (cm) and rotation (°) errors [cite], as well as the average localization recall at (5cm, 5°) for 7Scenes [cite].
构: We report for each A the [statistic] B errors, as well as the C at [thresholds] for D.
```
**M2** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The position error measures the distance between the predicted and the true goal position when executing the estimated displacement from the source pose.
原: The gate margin metric measures the distance between the drone and the closest point on the gate when crossing the gate plane.
构: The X measures the distance between A and B when [gerund] C.
```
**M3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Distance errors are obtained from the norm of the estimated displacement.
原: Relative poses are obtained from essential matrix estimation with RANSAC.
构: A are obtained from B.
```
**M3b** · SuperGlue（CVPR 2020 (Oral)）
```text
新: We also report the 3-D position error and the SR, where a pair is deemed successful based on its distance error.
原: We also report the match precision and the matching score [cite], where a match is deemed correct based on its epipolar distance.
构: We also report the A and the B, where a C is deemed correct based on its D.
```
**M4** · SuperGlue（CVPR 2020 (Oral)）
```text
新: As in previous work~\citep{sarlin2020superglue,sun2021loftr}, we report the AUC of the pose error at the thresholds ($5^\circ$, $10^\circ$, $20^\circ$), where the pose error is the maximum of the angular errors in heading and tilt.
原: As in previous work [cite], we report the AUC of the pose error at the thresholds (5°, 10°, 20°), where the pose error is the maximum of the angular errors in rotation and translation.
构: As in previous work, we report the AUC of the pose error at the thresholds (...), where the pose error is the maximum of the angular errors in A and B.
```

### Experimental setup: baselines

**B1** · LoFTR（CVPR 2021）
```text
新: We compare DreamNav with four categories of methods: 1) visuomotor policies including AI2-THOR~\citep{7989381}, $\pi_{0.5}$~\citep{intelligence2025pi_}, $\pi_0$-FAST~\citep{pertsch2025fast}, ViNT~\citep{shah2023vint}, and NoMaD~\citep{sridhar2024nomad}, 2) world models that plan by imagination, i.e., DINO-WM~\citep{zhou2025dinowm} and Visual Foresight~\citep{ebert2018visual}, 3) visual representations including DINOv3~\citep{simeoni2025dinov3} and Sample4Geo~\citep{deuser2023sample4geo}, and 4) a regression on pooled MASt3R statistics, i.e., RASR~\citep{liang2026rasr}.
原: We compare LoFTR with three categories of methods: 1) detector-based local features including R2D2 [cite], D2Net [cite], and DISK [cite], 2) a detector-based local feature matcher, i.e., SuperGlue [cite] on top of SuperPoint [cite] features, and 3) detector-free matchers including Sparse-NCNet [cite] and DRC-Net [cite].
构: We compare X with N categories of methods: 1) A including a1, a2, and a3, 2) a B, i.e., b1 ..., and 3) C including c1 and c2.
```
**B2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: All baselines, including RASR, are trained on the training sites of the same split, supervised by the five pose parameters of each pair.
原: All learned methods, including SuperGlue, are trained on ground truth correspondences, found by projecting keypoints from one image to the other.
构: All A, including X, are trained on B, [participle clause].
```
**B3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Other predictors are evaluated as reference: a constant output of zero motion, and RASR with its heading regressed as a sine--cosine pair using the same features.
原: Other local features are evaluated as reference: ORB [cite] with GMS, D2-Net [cite], and ContextDesc [cite] using the publicly available trained models.
构: Other A are evaluated as reference: a1 ..., and a2 using the B.
```

### Experimental setup: implementation

**I0** · Reloc3r（CVPR 2025）
```text
新: We initialize the baselines with their official pre-trained weights, when available, before training on PairUAV.
原: We initialize Reloc3r with DUSt3R's pre-trained 512-DPT weights.
构: We initialize X with Y's pre-trained weights.
```
**I1** · Reloc3r（CVPR 2025）
```text
新: All experiments are conducted on a single NVIDIA RTX 5090 GPU.
原: All evaluations are conducted on a 24GB NVIDIA GeForce RTX 4090 GPU and we use mixed precision fp16 / fp32 to improve speed and memory efficiency without sacrificing accuracy.
构: All [evaluations] are conducted on a [N] NVIDIA X GPU (first clause of the source).
```
**I3** · PixLoc（CVPR 2021）
```text
新: More details are in Appendix~I.
原: More details are in the supplemental.
构: More details are in the X.
```

### Comparison with baselines

**R1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: As shown in Table~\ref{tab:main_result}, DreamNav outperforms all baselines, on all pose parameters, when using either the coarse stage alone or both stages.
原: As shown in Table [ref], SuperGlue outperforms all baselines, at all relative pose thresholds, when applied to both SuperPoint and SIFT.
构: As shown in Table, X outperforms all baselines, at all A, when [participle] both B and C.
```
**R2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Most notably, the success rate of the resulting navigation is very high (83.3\%), reinforcing the view that the last meter is within reach of a monocular camera.
原: Most notably, the precision of the resulting matching is very high (84.9\%), reinforcing the analogy that SuperGlue “glues” together local features.
构: Most notably, the A of the resulting B is (very) high (N%), reinforcing the C that D.
```
**R3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Regression-based methods like $\pi_{0.5}$ and DINOv3 cannot predict a substantially more accurate horizontal displacement than the constant predictor itself, lacking any cue on the absolute heading (see Table~\ref{tab:main_result} and Figure~\ref{fig:cdf}).
原: Outlier rejection methods like PointCN and OANet cannot predict more correct matches than the NN matcher itself, overly relying on the initial descriptors (see Figure [ref] and supplemental [ref]).
构: A methods like a1 and a2 cannot predict more [adj] B than the C itself, [participle clause] (see Fig. and ...).
```
**R4** · OrienterNet（CVPR 2023）
```text
新: Direct regression disregards any knowledge of projective geometry and performs mere interpolation between training pairs without any geometric constraint.
原: a) Map retrieval disregards any knowledge of projective geometry and performs mere recognition without any geometric constraint.
构: X disregards any knowledge of projective geometry and performs mere Y without any geometric constraint.
```
**R5** · LoFTR（CVPR 2021）
```text
新: DreamNav outperforms RASR by a large margin (83.3\% against 11.7\% at SR@1\,m), which demonstrates the effectiveness of the correspondence-based design.
原: LoFTR outperforms SuperGlue by a large margin (13\% at AUC@10°), which demonstrates the effectiveness of the detector-free design.
构: X outperforms Y by a large margin (N% at METRIC), which demonstrates the effectiveness of the Z design.
```
**R6** · PixLoc（CVPR 2021）
```text
新: The imagination-refined stage brings consistent improvement on PairUAV, especially in the tilt change, with up to $-$54.6\% error.
原: PixLoc brings consistent improvement on CMU, especially in the fine threshold, with up to +2.4\% recall.
构: X brings consistent improvement on D, especially in the A, with up to N% B.
```
**R7** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: At all error thresholds up to the 99th percentile, the refined estimate is consistently more accurate than its coarse counterpart.
原: At all error thresholds, featuremetric optimization is consistently more accurate than its geometric counterparts.
构: At all error thresholds, A is consistently more accurate than its B counterpart(s).
```

### Ablation study

**A1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: To evaluate our design decisions, we repeat the experiments on the same test pairs, but this time focusing on different DreamNav variants.
原: To evaluate our design decisions, we repeat the indoor experiments with SuperPoint features, but this time focusing on different SuperGlue variants.
构: To evaluate our design decisions, we repeat the A experiments with/on B, but this time focusing on different X variants.
```
**A2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: This ablation study, presented in Table~\ref{tab:ablation}, shows that all DreamNav components are useful and bring performance gains.
原: This ablation study, presented in Table [ref], shows that all SuperGlue blocks are useful and bring substantial performance gains.
构: This ablation study, presented in Table, shows that all X blocks are useful and bring (substantial) performance gains.
```
**A3** · SuperGlue（CVPR 2020 (Oral)）
```text
新: When we regress the sine and cosine of the heading change with an MLP while keeping the features of RASR, we observe an improvement in $\mathrm{MAE}_\theta$ from 5.70$^\circ$ to 1.82$^\circ$.
原: When we additionally backpropagate through the SuperPoint descriptor network while training SuperGlue, we observe an improvement in AUC@20° from 51.84 to 53.38.
构: When we [verb] A while [gerund] B, we observe an improvement in METRIC from N1 to N2.
```
**A4** · Navigation World Models（CVPR 2025）
```text
新: We find that running the original matcher leads to a valid relative pose for only 47\% of the pairs, while searching over four rotations leads to a valid pose for 99.3\% of them.
原: We find that running the model with time only leads to poor performance, while not conditioning on time leads to small drop in performance as well.
构: We find that running A leads to B, while [gerund] C leads to D.
```
**A5** · LightGlue（ICCV 2023）
```text
新: Without the rotation search, the matcher loses its ability to match pairs whose heading change exceeds $60^\circ$, as shown in Appendix~I.
原: Without the matchability classifier, the network loses its ability to discriminate between good and bad matches, as shown in Figure [ref].
构: Without the A, the B loses its ability to [verb] C, as shown in Fig.
```
**A6** · OrienterNet（CVPR 2023）
```text
新: World-frame displacements stay at the level of the constant predictor when the absolute heading is unknown.
原: b) Featuremetric refinement converges to incorrect locations when the initial pose is inaccurate.
构: A [verb] to/at B when the C is [adj].
```
**A7** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Zoom re-matching is particularly effective in weakly matched pairs, as full-image matching cannot fully exploit the co-visible region due to the fixed budget of keypoints.
原: Featuremetric optimization is particularly effective in this situation, as geometric optimization cannot fully suppress the detection noise due to the small number of observations.
构: A is particularly effective in B, as C cannot fully [verb] D due to the E.
```
**A8** · PixLoc（CVPR 2021）
```text
新: Verifying the heading in the canonical view only marginally improves the performance, confirming that the remaining heading error can be sufficiently systematic within a site for a more exact input representation to leave it unchanged.
原: Localizing with the oracle prior only marginally improves the performance, confirming that image retrieval can be sufficiently accurate for the pose optimization to converge to the correct minimum.
构: [Gerund] A only marginally improves the performance, confirming that B can be sufficiently [adj] for C to [verb] D.
```
**A9** · PixLoc（CVPR 2021）
```text
新: Using the true absolute heading halves the horizontal errors, and makes DreamNav accurate to half a metre in both $\Delta x$ and $\Delta y$.
原: Using the oracle prior partially bridges the gap, and makes PixLoc competitive on driving datasets like CMU and RobotCar.
构: Using the oracle A [verb] the B, and makes X [predicate].
```

### Ablation of the imagination-refined stage

**G1** · Navigation World Models（CVPR 2025）
```text
新: We render imagined views while varying the number of tiles per pair from $1$ to $3{\times}3$ (see Table~\ref{tab:imag}a).
原: We train models while varying the number of conditioning frames from $1$ to $4$ (see Table [ref]).
构: We [verb] A while varying the number of B from N1 to N2 (see Table).
```
**G2** · Navigation World Models（CVPR 2025）
```text
新: Unsurprisingly, more tiles help, and with a single tile the matcher often ``runs out'' of keypoints, leading to five times fewer inliers.
原: Unsurprisingly, more context helps, and with short context the model often “lose track”, leading to poor predictions.
构: Unsurprisingly, more A helps, and with [little] A the B often ``...'', leading to C.
```
**G3** · PixLoc（CVPR 2021）
```text
新: Re-aligning with the coarse pose, replacing the rotation-only warp by the plane-induced homography, and matching tiles at the native resolution all contribute to the performance of the refinement over the zoom pass of the coarse stage.
原: Unrolling the optimizer and learning features, damping factor, and confidences all contribute to the performance of PixLoc over classical photometric alignment.
构: [Gerund] A, B, and C all contribute to the performance of X over Y.
```
**G4** · PixLoc（CVPR 2021）
```text
新: This improvement is not brought by the additional layer of regressors.
原: This improvement is not brought by the significantly less accurate image retrieval.
构: This improvement is not brought by the A.
```
**G5** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The findings of our control reveal that the same regressors without the imagined matches result in no gain over the coarse stage and a slight increase in error (Table~\ref{tab:imag}b).
原: The findings of our analysis reveal that subsequent fine-tuning operations result in negligible enhancements in performance and alterations in behaviour (Extended Data Fig. 2 ).
构: The findings of our A reveal that B result in [negligible] C and D (Fig./Table).
```
**G6** · Navigation World Models（CVPR 2025）
```text
新: Results (Table~\ref{tab:imag}b) show DreamNav refines effectively without the verification scores, with only minor performance drops (0.455\,m to 0.468\,m in $\ell_x$).
原: Results (Table [ref]) show NWM plans effectively under constraints, with only minor performance drops (see examples in Figure [ref]).
构: Results (Table) show X [verb] effectively under/without A, with only minor performance drops (...).
```
**G7** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: We investigate the extent of variations in accuracy across iterations.
原: We investigate the extent of variations in behaviour across iterations.
构: We investigate the extent of variations in A across iterations.
```
**G8** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: We observe that the displacement between imagined and real views, which is a label-free measure of the accuracy of the guiding pose, is 7.6 pixels after the coarse stage and 5.0 pixels after one refinement.
原: We observe that the difference in distance from gate centres, which is a metric for the safety of the policy, is 0.09 ± 0.08 m.
构: We observe that the A, which is a metric for the B of the C, is N.
```
**G9** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Note that the gain of a second round in the median translation-direction error (0.03$^\circ$) is substantially smaller than the difference between the errors of the zoom pass and the first round (0.75$^\circ$).
原: Note that this lap-time difference is substantially smaller than the difference between the single-lap completion times of Swift and the human pilots (0.16 s).
构: Note that this A is substantially smaller than the difference between the B of C and D (N).
```
**G10** · OrienterNet（CVPR 2023）
```text
新: Diffusion retrieval replaces the plane-induced warp and matching by a comparison of the target view and of the views that a diffusion model synthesizes for candidate poses around the coarse estimate.
原: a) Map retrieval [cite] replaces the BEV inference and matching by a correlation of the neural map and with a global image embedding.
构: X retrieval replaces the A and matching by a [comparison] of the B and (with) C.
```
**G11** · LoFTR（CVPR 2021）
```text
新: Replacing the geometric imagination by diffusion retrieval on top of the same coarse stage results in a significant increase in error for every pose parameter, up to 0.43$^\circ$ from 0.12$^\circ$ in $\mathrm{MAE}_\theta$ (Table~\ref{tab:imag}c).
原: 1) Replacing the LoFTR module by convolution with a comparable number of parameters results in a significant drop in AUC as expected.
构: Replacing the A by B with C results in a significant [drop/increase] in D.
```
**G12** · LoFTR（CVPR 2021）
```text
新: We attribute the degradation to the use of synthesized views and the retrieval among candidate poses, giving up pixel-level geometry and tying the estimate to the sampled candidates.
原: We attribute the prominence to the use of the Transformer and the optimal transport layer, taking advantage of global information and jointly bringing global consensus into the final matches.
构: We attribute the A to the use of the B and the C, [gerund] D and [gerund] E.
```

### Analysis: where the refinement helps

**N1** · LightGlue（ICCV 2023）
```text
新: Figure~\ref{fig:breakdown} studies the effectiveness of the two stages -- coarse and refined -- on the test pairs for different ranges of inlier count, rotation and distance.
原: Table [ref] studies the effectiveness of the two pruning mechanisms -- adaptive depth and width -- on MegaDepth image pairs for different ranges of visual overlap.
构: Table/Fig. studies the effectiveness of the two A -- a1 and a2 -- on B pairs for different ranges of C.
```
**N2** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: The improvements are consistent across all bins, both in rotation and in distance.
原: The improvements are consistent across all local features, both indoors and outdoors.
构: The improvements are consistent across all A, both B and C.
```
**N3** · LightGlue（ICCV 2023）
```text
新: For moderately difficult samples, such as the pairs with 100 to 200 coarse inliers, the refinement recovers many more correspondences and nearly halves the position error, resulting in a drop from 3.98\,m to 2.21\,m.
原: For easy samples, such as the successive frames of a video, the network quickly converges and exits after a few layers, resulting in a 1.86$\times$ speedup.
构: For [adj] samples, such as the A, the B [verb] and [verb] ..., resulting in a N [change].
```
**N4** · LightGlue（ICCV 2023）
```text
新: In cases of low visual overlap, e.g. fewer than 100 inliers, the refinement requires more correspondences to succeed.
原: In cases of low visual overlap, e.g. loop closure, the network requires more layers to converge.
构: In cases of low visual overlap, e.g. A, the B requires more C to [verb].
```
**N5** · Navigation World Models（CVPR 2025）
```text
新: After evaluation, we resample the 290 test sites 1,000 times to report 95\% confidence intervals.
原: After training, we sample $5$ times from each model to report mean and std results.
构: After A, we sample N times from B to report C.
```
**N6** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The refined stage more consistently reaches the goal, exhibiting a higher success rate by 10.2 points (interval of 9.1 to 11.3) and a lower mean position error by 0.68\,m (0.61 to 0.74).
原: The autonomous drone more consistently pushes for fast lap times, exhibiting lower mean and variance.
构: The A more consistently [verb] B, exhibiting higher/lower C and D.
```

### Analysis: reliability of the predicted error

**V0** · Kendall & Gal（NIPS 2017）
```text
新: In this section we wish to study the effectiveness of predicting the error of every candidate estimate.
原: In this section we wish to study the effectiveness of modeling aleatoric and epistemic uncertainty.
构: In this section we wish to study the effectiveness of [gerund] A.
```
**V1** · Kendall & Gal（NIPS 2017）
```text
新: In particular, we wish to quantify the performance of the error predictions that weight our fusion and analyze what they capture.
原: In particular, we wish to quantify the performance of these uncertainty measurements and analyze what they capture.
构: In particular, we wish to quantify the performance of these A and analyze what they capture.
```
**V2** · Kendall & Gal（NIPS 2017）
```text
新: Firstly, in Figure~\ref{fig:reliability} (left) we show sparsification curves for four pose parameters.
原: Firstly, in Figure [ref] we show precision-recall curves for regression and classification models.
构: Firstly, in Figure we show A curves for B.
```
**V3** · Kendall & Gal（NIPS 2017）
```text
新: They show how our model performance improves by removing pairs with predicted error larger than various percentile thresholds.
原: They show how our model performance improves by removing pixels with uncertainty larger than various percentile thresholds.
构: They show how our model performance improves by removing A with B larger than various percentile thresholds.
```
**V4** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Keeping the most confident half reduces the mean error of the tilt change from 0.20$^\circ$ to 0.02$^\circ$ for the test pairs.
原: Precomputing distance maps reduces the peak memory requirement of the bundle adjustment from 80 GB to less than 10GB for 1000 images.
构: [Gerund] A reduces the B of the C from N1 to N2 for D.
```
**V5** · Kendall & Gal（NIPS 2017）
```text
新: Secondly, in Figure~\ref{fig:reliability} (right) we analyze the quality of our error prediction using calibration plots from our model on the test set.
原: Secondly, in Figure [ref] we analyze the quality of our uncertainty measurement using calibration plots from our model on the test set.
构: Secondly, in Figure we analyze the quality of our A using calibration plots from our model on the test set.
```
**V6** · Kendall & Gal（NIPS 2017）
```text
新: Better performing error predictions should correlate more accurately with the line $y=x$ in the calibration plots.
原: Better performing uncertainty estimates should correlate more accurately with the line $y=x$ in the calibration plots.
构: Better performing A should correlate more accurately with the line y=x in the calibration plots.
```
**V7** · PixLoc（CVPR 2021）
```text
新: The predicted error is however lower than the actual one by a nearly constant factor of 1.5 to 2, which is mostly due to the square-root regression target, as our regressors cannot recover the mean of a skewed distribution if the fit is performed after such a concave transform.
原: PixLoc is however less robust than the feature matching pipelines, which is mostly due to the naive pose prior, as our algorithm cannot converge if the retrieval returns the incorrect location.
构: X is however less/lower [adj] than Y, which is mostly due to the A, as our B cannot [verb] if C.
```

### Analysis: limitations

**L1** · OrienterNet（CVPR 2023）
```text
新: Estimating the pose of a pair is challenging when the two views barely overlap or when the scene lacks cues on the absolute heading.
原: Localizing an image or a sequence is challenging when the environments lacks distinctive elements or when they are not registered in the map.
构: [Gerund] A is challenging when B or when C.
```
**L2** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Out of the 2,422 position errors above 1\,m recorded for DreamNav, 44\% were because of an error in the absolute heading alone, 17\% because of fewer than 200 coarse inliers and 39\% because of the remaining errors in scale and translation direction.
原: Out of the 10 losses recorded for Swift, 40% were because of a collision with the opponent, 40% because of collision with a gate and 20% because of the drone being slower than the human pilot.
构: Out of the N losses recorded for X, a% were because of A, b% because of B and c% because of C.
```
**L3** · PixLoc（CVPR 2021）
```text
新: DreamNav can also fail for sites in the southern hemisphere due to their reversed shadows and is more sensitive to changes of the flight altitude.
原: PixLoc can also fail for large outliers ratios due prominent occluders and is more sensitive to camera miscalibration.
构: X can also fail for A due (to) B and is more sensitive to C.
```

### Zero-shot transfer to real drone photographs

**T1** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: While the previous experiments precisely quantify the accuracy of DreamNav, they do not contain any variations of sensor, altitude or image formation.
原: While the previous experiments precisely quantify the accuracy of the refinement, they do not contain any variations of appearance or camera models.
构: While the previous experiments precisely quantify the accuracy of X, they do not contain any variations of A or B.
```
**T2** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: We thus turn to real drone photographs and evaluate the benefits of our geometric imagination in a zero-shot setting.
原: We thus turn to crowd-sourced imagery and evaluate the benefits of our featuremetric optimization in an end-to-end reconstruction pipeline.
构: We thus turn to A and evaluate the benefits of our B in a(n) C.
```
**T3** · OrienterNet（CVPR 2023）
```text
新: We consider the pose estimation in real flights with the M\'ezi\`eres dataset~\citep{heigvd_mezieres}, following the metrics of Section~\ref{sec:setup}.
原: We consider the localization in driving scenarios with the KITTI dataset [cite], following the closest existing setup [cite].
构: We consider the A in B scenarios with the X dataset, following the C.
```
**T4** · OrienterNet（CVPR 2023）
```text
新: Images are captured by the camera of a DJI Phantom 4 RTK flying 60\,m above a village and have GT poses from RTK and bundle adjustment.
原: Images are captured by cameras mounted on a car driving in urban and residential areas and have GT poses from RTK.
构: Images are captured by cameras [of/on] a A [gerund] in/above B and have GT poses from RTK.
```
**T5** · LoFTR（CVPR 2021）
```text
新: We only use the pairs of photographs whose cameras are at most 30\,m apart, from which we sample 600 pairs for evaluation.
原: We follow DISK [cite] to only use the scenes of “Sacre Coeur” and “St. Peter's Square” for validation, from which we sample 1500 pairs for a fair comparison.
构: We ... only use the A of B ..., from which we sample N pairs for C.
```
**T6** · LightGlue（ICCV 2023）
```text
新: The pairs were sampled in a way that the difficulty is balanced based on the baseline-to-height ratio.
原: The data was collected in a way that the difficulty is balanced based on visual overlap.
构: The A was collected/sampled in a way that the difficulty is balanced based on B.
```
**T7** · OrienterNet（CVPR 2023）
```text
新: We undistort the photographs and resample them such that their intrinsics match the calibration $\mathbf{K}$ of PairUAV.
原: We render $64m×64m$ map tiles and resize the images such that their focal length matches the median of MGL.
构: We [verb] A and resize/resample the images such that their B matches the C of D.
```
**T8** · DUSt3R（CVPR 2024）
```text
新: We emphasize that all results are obtained with the \emph{same} DreamNav model (the only site-specific input is the flight height reported by the aircraft, by which the estimated displacement is rescaled), \ie we never finetune our model on real photographs.
原: We emphasize that all results are obtained with the same DUSt3R model (our default model is denoted as ‘DUSt3R 512’, other DUSt3R models serve for the ablations in Sec. F of the suppl.), i.e. we never finetune our model on a particular downstream task (zero-shot settings).
构: We emphasize that all results are obtained with the same X model (...), i.e. we never finetune our model on A.
```
**T9** · OrienterNet（CVPR 2023）
```text
新: We compute the position error in the levelled frame of the source camera since the absolute heading is not transferable from rendered to real shadows.
原: We compute the position error along directions perpendicular (lateral) and parallel (longitudinal) to the viewing axis [cite] since the pose is generally less constrained along the road.
构: We compute the position error along/in A since the B is [adj] C.
```
**T10** · OrienterNet（CVPR 2023）
```text
新: Table~\ref{tab:real} shows that DreamNav outperforms all regression baselines in both ranges of baseline-to-height ratio, in all metrics.
原: [ref] (a-b) shows that OrienterNet outperforms all existing approaches based on both satellite and OSM maps, in all metrics.
构: Table shows that X outperforms all A ... both B and C, in all metrics.
```
**T10b** · PixLoc（CVPR 2021）
```text
新: However, no improvement can be observed on the tilt change, while the constant predictor is more accurate there since the gimbal barely moves between these photographs.
原: However, no improvement can be observed on RobotCar Day, while the refinement is detrimental on Aachen at 0.25m.
构: However, no improvement can be observed on A, while the B is [adj] on/there C.
```
**T11** · PixLoc（CVPR 2021）
```text
新: It is accurate within the training range, despite being trained on rendered images only.
原: It is competitive for indoor scenes, despite being trained on outdoor Internet data only.
构: It is [adj] for/within A, despite being trained on B only.
```
**T12** · PixLoc（CVPR 2021）
```text
新: The refinement also increases the success rate at all thresholds on real photographs, which exhibit stronger parallax, a difficult condition for a plane-induced warp.
原: It also increases the pose accuracy at all thresholds on RobotCar Night, which exhibits significant motion blur, a difficult condition for sparse keypoint detection.
构: It also increases the A at all thresholds on B, which exhibits C, a difficult condition for D.
```
**T13** · PixLoc（CVPR 2021）
```text
新: This confirms that a plane-induced warp is all we need for the imagination to pay off and that it generalizes well despite a fusion fitted on rendered pairs.
原: This confirms that deep features are all we need for accurate localization and that they generalize well despite end-to-end training.
构: This confirms that A are all we need for B and that they generalize well despite C.
```
**T14** · OrienterNet（CVPR 2023）
```text
新: The performance is however significantly lower beyond the training range (5.0\% SR), which highlights the difficulty of wide baselines at low altitude and the need for further research.
原: The performance is however significantly lower than with driving data ( [ref]), which highlights the difficulty of AR-like conditions and the need for further research.
构: The performance is however significantly lower than/beyond A (...), which highlights the difficulty of B and the need for further research.
```
**T15** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: By contrast with its relative geometry, DreamNav's scale recovery assumes that the flight altitude is consistent with what was observed during training.
原: By contrast, Swift’s perception system assumes that the appearance of the environment is consistent with what was observed during training.
构: By contrast, X’s A assumes that the B is consistent with what was observed during training.
```
**T16** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: If this assumption fails, the system can fail: without the reported flight height, the SR drops from 63.3\% to 2.0\%.
原: If this assumption fails, the system can fail.
构: If this assumption fails, the system can fail.
```

### Closed-loop navigation in simulation

**S1** · Navigation World Models（CVPR 2025）
```text
新: Next, we turn to describe experiments that measure how well we can navigate using DreamNav in closed loop.
原: Next, we turn to describe experiments that measure how well can we navigate using a NWM.
构: Next, we turn to describe experiments that measure how well can we navigate using X.
```
**S2** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: To this end, we evaluate performance in an image-goal navigation task when deployed in eight held-out test sites.
原: To this end, we evaluate performance in a racing task when deployed in four different settings.
构: To this end, we evaluate performance in a(n) A task when deployed in N different B.
```
**S3** · LightGlue（ICCV 2023）
```text
新: We first reconstruct a 3D Gaussian splatting model~\citep{kerbl20233d} from 135 of the 150 images of each site, with known poses and calibration, using gsplat~\citep{ye2025gsplat}.
原: We first triangulate a sparse 3D point cloud from the 4328 daytime reference images, with known poses and calibration, using COLMAP [cite].
构: We first [verb] a 3D A from the N images, with known poses and calibration, using TOOL.
```
**S4** · LightGlue（ICCV 2023）
```text
新: For each of 10 held-out goal images per site, we sample 2 start poses among the other cameras, render the view of the drone, and execute the estimated pose change with actuation noise.
原: For each of the 824 daytime and 98 nighttime queries, we retrieve 50 images with NetVLAD [cite], match each of them, and estimate a camera pose with RANSAC and a Perspective-n-Point solver.
构: For each of the N A, we [verb] M B ..., [verb] C, and [verb] a D with E.
```
**S5** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: These observations are rendered by sampling the splat model at the native resolution and are independent of the real image that serves as the goal.
原: These noisy observations are generated by sampling one realization from the residual observation model and are independent of the perception awareness of the deployed controller.
构: These A observations are generated by sampling B from the C model and are independent of the D.
```
**S6** · Navigation World Models（CVPR 2025）
```text
新: Each command is randomly perturbed by 3\% of its length and 5\,cm per axis, $0.2^\circ$ in heading and $0.1^\circ$ in tilt.
原: Each goal is randomly chosen between $\pm16$ seconds window around the current state.
构: Each A is randomly [participle] [by/between] B ...
```
**S7** · PixLoc（CVPR 2021）
```text
新: We assess the role of imagination by comparing two variants of DreamNav, with and without the imagination-refined stage.
原: We justify our design decisions by comparing different variants of PixLoc.
构: We [verb] our A by comparing different variants of X.
```
**S7b** · PixLoc（CVPR 2021）
```text
新: The refined stage is run with its short-range mode (Section~\ref{sec:servo}) when evaluated in closed loop (and without it in the experiments above).
原: PixLoc is trained on MegaDepth (CMU) when evaluated on Aachen (RobotCar and CMU).
构: X is [participle] on/with A (...) when evaluated on/in B (...).
```
**S8** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Policies are allowed to continue flying for up to 5 steps, provided their estimated distance to the goal is still above a threshold $\tau$.
原: Pilots are allowed to continue racing even after a crash, provided their vehicle is still able to fly.
构: A are allowed to continue [gerund] ..., provided their B is still C.
```
**S8b** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Choosing a smaller threshold can bring the drone closer to the goal but can also increase the risk of acting on baselines shorter than those of training.
原: Leaving a smaller gate margin can increase speed but can also increase the risk of collision or missing the gate.
构: [Gerund] a smaller A can [verb] B but can also increase the risk of C.
```
**S9** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Performance is assessed by evaluating the position error at the stop, the number of steps flown and the percentage of episodes ending within 1\,m of the goal.
原: Performance is assessed by evaluating the fastest lap time, the average and minimum observed gate margin of successfully passed gates and the percentage of track successfully completed.
构: Performance is assessed by evaluating the A, the B and the percentage of C.
```
**S10** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The results are summarized in Table~\ref{tab:nav}.
原: The results are summarized in Extended Data Table 1c .
构: The results are summarized in Table.
```
**S10b** · PixLoc（CVPR 2021）
```text
新: DreamNav however lags behind on rendered views, where the textures are significantly blurrier and the estimates are therefore coarser (0.48\,m against 0.36\,m of median $\ell_x$ error on the same pairs).
原: It however lags behind on Aachen, where the reference images are significantly sparser and the initial priors are therefore much coarser.
构: X however lags behind on A, where the B are significantly [adj] and the C are therefore (much) coarser.
```
**S11** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Both variants manage to cover most of the initial distance when deployed for a single step, with the refined estimate yielding the lowest median error (1.32\,m from a start at 18.9\,m).
原: All approaches manage to successfully complete the task when deployed in idealized dynamics and ground-truth observations, with the time-optimal baseline yielding the lowest lap time.
构: All A manage to [verb] B when deployed in/for C, with the D yielding the lowest E.
```
**S12** · OrienterNet（CVPR 2023）
```text
新: The SR of the full model is raised from 38.8\% to over 81\% by closing the loop for only 2.2 steps on average.
原: The recall at 3m/3° is saturated to over 95\% by fusing the predictions from sequences of only 20 seconds.
构: The A at/of B is saturated/raised to over N% by [gerund] C ... of only D.
```
**S13** · LoFTR（CVPR 2021）
```text
新: The full model outperforms the coarse stage alone by 25 points of SR at $\tau = 1$\,m, demonstrating the effectiveness of the short-range mode.
原: LoFTR outperforms the detector-free method DRC-Net by 61\% at AUC@10°, demonstrating the effectiveness of the Transformer.
构: X outperforms the A by N% at METRIC, demonstrating the effectiveness of the B.
```
**S14** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: Although the full model is globally more successful than the coarse stage, it is not more successful at all individual thresholds of the sweep (Figure~\ref{fig:nav}).
原: Although Swift is globally faster than all human pilots, it is not faster on all individual segments of the track (Extended Data Table 1 ).
构: Although X is globally more A than B, it is not more A on all individual C of the D (Table/Fig.).
```
**S14b** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: The coarse stage is as successful for larger thresholds as the full model then stops after its first step in more than half of the episodes, before the short-range mode can act, while smaller thresholds leave it the steps to do so.
原: Patch Flow is more complete for larger thresholds as it partly solves a different problem by increasing the keypoint repeatability with its large receptive field, while we focus on their localization.
构: A is more/as [adj] for larger thresholds as it/B [verb] ..., while C ...
```
**S15** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: One hypothesis is that the fusion extrapolates poorly to baselines shorter than 5\,m, which make up fewer than 1\% of its training pairs.
原: One hypothesis is that Swift optimizes trajectories on a longer timescale than human pilots.
构: One hypothesis is that X [verb] A ... than B.
```
**S16** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: This is apparent, for example between 0.5\,m and 1\,m from the goal, for which the remaining distance is overestimated to 1.34\,m in median by the coarse stage, but estimated at 0.78\,m by the short-range mode.
原: This is apparent, for example in the split S (Fig. 4b,d ), for which human pilots are faster in the beginning and at the end of the manoeuvre, but slower overall (Extended Data Table 1d , segment 3).
构: This is apparent, for example in A, for which B are/is C ..., but D ...
```
**S17** · Navigation World Models（CVPR 2025）
```text
新: We report four thresholds (Table~\ref{tab:nav}) and find that the short-range mode improves navigation performance, with thresholds of at most 1.25\,m yielding better results.
原: We report ATE and RPE on all in-domain datasets (Table [ref]) and find that NWM-based trajectory ranking improves navigation performance, with more samples yielding better results.
构: We report A (Table) and find that B improves navigation performance, with C yielding better results.
```
**S18** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: The full model is consistently faster at meeting the stopping condition, with 2.2 steps against 3.8 at $\tau = 1$\,m.
原: Swift is consistently faster at the start and in tight turns such as the split S.
构: X is consistently faster at A ...
```

### Qualitative results

**Q1** · DUSt3R（CVPR 2024）
```text
新: The plane-induced warp yields realistic imagined views even in challenging situations.
原: DUSt3R yields high-quality dense 3D reconstructions even in challenging situations.
构: X yields [adj] A even in challenging situations.
```
**Q2** · LoFTR（CVPR 2021）
```text
新: The imagined views recover both the layout and the fine details of the real target, successfully coping with elevated structures and large heading and altitude changes.
原: LoFTR obtains more correct matches and fewer mismatches, successfully coping with low-texture regions and large viewpoint and illumination changes.
构: X obtains/recovers A and B, successfully coping with C and large D and E changes.
```

### Captions

**C1** · SuperGlue（CVPR 2020 (Oral)）
```text
新: \textbf{Image-goal pose estimation on PairUAV.} We report the MAE of each pose parameter, the success rate (SR) and the AUC of the pose error, the latter two in percents \%.
原: Wide-baseline indoor pose estimation. We report the AUC of the pose error, the matching score (MS) and precision (P), all in percents \%.
构: Title. We report the A, the B (b) and C (c), all in percents %.
```
**C2** · SuperGlue（CVPR 2020 (Oral)）
```text
新: DreamNav outperforms all policies, world models and representations when using either one or both stages.
原: SuperGlue outperforms all handcrafted and learned matchers when applied to both SIFT and SuperPoint.
构: X outperforms all A and B [baselines] when [participle] both C and D.
```
**C3** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: \textbf{Cumulative errors.} We plot the cumulative position, heading and tilt errors on the test pairs of PairUAV.
原: Camera pose estimation. We plot the cumulative translation error and report its AUC.
构: Title. We plot the cumulative A error(s) and ...
```
**C4** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Our refinement improves the accuracy of the coarse estimates for all pose parameters, even for the heading change, whose coarse estimates are already accurate to a fraction of a degree.
原: Our refinement improves the accuracy of the query camera poses for all local features, even when for SIFT, whose detections are already well-localized.
构: Our refinement improves the accuracy of the A for all B, even for C, whose D are already [adj].
```
**C5** · SuperGlue（CVPR 2020 (Oral)）
```text
新: \textbf{Ablation of DreamNav.} While the sine--cosine regression alone improves over the baseline RASR regressor, the rotation-aware two-view geometry explains the majority of the gains brought by DreamNav.
原: Ablation of SuperGlue. While the optimal matching layer alone improves over the baseline Nearest Neighbor matcher, the Graph Neural Network explains the majority of the gains brought by SuperGlue.
构: Ablation of X. While the A alone improves over the baseline B, the C explains the majority of the gains brought by X.
```
**C6** · SuperGlue（CVPR 2020 (Oral)）
```text
新: Both the absolute heading and the zoom re-matching are critical for accurate displacements, and the imagination-refined stage further improves the accuracy.
原: Both cross-attention and positional encoding are critical for strong gluing, and a deeper network further improves the precision.
构: Both A and B are critical for C, and D further improves the E.
```
**C7** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: \textbf{Raw two-view geometry.} The learned priors improve the accuracy of poses estimated by epipolar geometry (E) or by the MAP estimator of Eq.~\eqref{eq:map} (MAP) with rotation-aware matches.
原: End-to-end SfM. The proposed refinement improves the accuracy of poses estimated by epipolar geometry (stereo) or a complete SfM pipeline (multiview) with crowd-sourced imagery.
构: Title. The A improve(s) the accuracy of poses estimated by epipolar geometry (a) or B (b) with C.
```
**C8** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: Improvements are substantial for both first-pass (pass 1) and pose-guided (zoom) matching configurations, especially when the rotation and the translation direction are weakly constrained.
原: Improvements are substantial for both standard (SIFT) and recent (SuperGlue) matching configurations, especially when few images $N$ observe the scene.
构: Improvements are substantial for both A (a) and B (b) matching configurations, especially when C.
```
**C9** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: \textbf{Ablation of the imagination-refined stage.} a)~The plane-induced warp and the tile-wise matching both largely improve the number of inliers and the accuracy of the relative pose.
原: Ablation study on ETH3D. i) Featuremetric keypoint and bundle adjustments (KA and BA) both largely improve the triangulation and localization accuracy.
构: Ablation ... i) A and B both largely improve the C and D accuracy.
```
**C10** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: b)~Different refined fusions are better than the coarse stage when given the imagined matches but the full model (our default) works best.
原: iii) Different image representations are better than the unrefined detections but S2DNet (our default) works best.
构: iii) Different A are better than the B but X (our default) works best.
```
**C10b** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: c)~Diffusion retrieval produces a full imagined view because of its generative prior but is less accurate than the coarse stage.
原: Patch Flow produces a longer track length because of its larger receptive field but is less accurate.
构: X produces a [adj] A because of its B but is less accurate.
```
**C11** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: \textbf{Where the refinement helps.} We show the position error of the two stages, in linear scale, for varying ranges of difficulty.
原: Run-times. We show the duration, in logarithmic scale, of the refinement for varying numbers of images.
构: Title. We show the A, in [linear/logarithmic] scale, of the B for varying C.
```
**C12** · LightGlue（ICCV 2023）
```text
新: Refinement helps most on moderately matched pairs, where the error drops by almost half.
原: Early stopping helps most on smaller scenes, where the network stops after just half the layers.
构: A helps most on B, where the C [verb] ... half ...
```
**C13** · Kendall & Gal（NIPS 2017）
```text
新: \textbf{Reliability of the predicted error.} Left: sparsification plots demonstrating that the predicted error of each parameter can effectively capture accuracy, as the error decreases with decreasing fraction of retained pairs.
原: Precision Recall plots demonstrating both measures of uncertainty can effectively capture accuracy, as precision decreases with increasing uncertainty.
构: A plots demonstrating (that) B can effectively capture accuracy, as C decreases with increasing/decreasing D.
```
**C14** · Kendall & Gal（NIPS 2017）
```text
新: Right: calibration plots. This plot shows how well the predicted error is calibrated, where perfect calibration corresponds to the line $y=x$, shown dashed.
原: Uncertainty calibration plots. This plot shows how well uncertainty is calibrated, where perfect calibration corresponds to the line $y=x$, shown in black.
构: A calibration plots. This plot shows how well B is calibrated, where perfect calibration corresponds to the line y=x, shown in [style].
```
**C15** · OrienterNet（CVPR 2023）
```text
新: \textbf{Zero-shot pose estimation on real drone photographs.} With data from a DJI Phantom 4 RTK, DreamNav outperforms the regression baseline and the constant predictor in both coarse and refined settings, in the range seen in training.
原: Localization of head-mounted devices for AR. With data from Aria glasses, OrienterNet outperforms the map retrieval baseline and the embedded GPS sensor in both single- and multi-frame settings, in both cities.
构: Title. With data from A, X outperforms the B baseline and the C in both a- and b- settings, in [both] D.
```
**C16** · OrienterNet（CVPR 2023）
```text
新: Rescaling by the flight height does not close the gap of the regression baseline but strongly benefits our approach.
原: Multi-frame fusion does not filter out the high noise of the GPS but strongly benefits our approach.
构: A does not [verb] the B of the C but strongly benefits our approach.
```
**C17** · Navigation World Models（CVPR 2025）
```text
新: \textbf{Comparing the cumulative position error} of DreamNav and RASR within and beyond the training range of baseline-to-height ratio, up to 60\,m of error on the 600 real pairs of the M\'ezi\`eres dataset.
原: Comparing generation accuracy and quality of NWM and DIAMOND at $1$ and $4$ FPS as function of time, up to $16$ seconds of generated video on the RECON dataset.
构: Comparing A of X and Y at/within B ..., up to N of C on the D dataset.
```
**C18** · Pixel-Perfect SfM（ICCV 2021 (Oral)）
```text
新: \textbf{Geometric imagination.} We show the target views $\widehat{G}$ imagined from the source $S$ for 2 test pairs of PairUAV and 1 pair of real photographs of the M\'ezi\`eres dataset.
原: Refined SfM tracks. We show patches centered around reprojections of 3x 3D points observed in 4 images of the St. Peter's Square scene.
构: Title. We show A ... of N B ... in M C of the D.
```
**C18c** · DUSt3R（CVPR 2024）
```text
新: From top to bottom: source, imagined target, real target over the same region.
原: From left to right: RGB, depth map, confidence map, reconstruction.
构: From X to Y: A, B, C.
```
**C19** · PixLoc（CVPR 2021）
```text
新: Although only based on a planar warp of the source pixels, the imagination works well thanks to the flight altitude, despite the coarse pose estimate and large viewpoint changes.
原: Although only based on local gradients, direct alignment works well thanks to deep features, despite the coarse initial pose estimate and strong appearance changes.
构: Although only based on A, X works well thanks to B, despite the coarse (initial) pose estimate and strong/large C changes.
```
**C20** · Navigation World Models（CVPR 2025）
```text
新: \textbf{Closed-loop image-goal navigation.} We present results for navigating with DreamNav under four stopping thresholds $\tau$, reporting the mean number of steps, the median position error at the stop (Err., in m) and the SR at 1\,m (\%) next to the one-step result.
原: Planning with Navigation Constraints. We present results for planning with NWM under three action constraints, reporting the differences in final position ($\delta u$) and yaw ($\delta \phi$) relative to the no-constraints baseline.
构: Title. We present results for [gerund] with X under N A, reporting the B ... and C ...
```
**C1b** · LoFTR（CVPR 2021）
```text
新: $\ddagger$ indicates distances predicted by the official distance head of the model.
原: † indicates models trained on MegaDepth.
构: † indicates A [participle] on B.
```
**C7b** · LoFTR（CVPR 2021）
```text
新: The mean (median) of the angular errors in degrees is reported.
原: The AUC of the corner error in percentage is reported.
构: The A of the B error in C is reported.
```
**C16b** · LoFTR（CVPR 2021）
```text
新: The medians of both the position error and the translation-direction error and the MAE of the tilt change are reported, with the SR at 1\,m and 2\,m.
原: The evaluation results on both the local feature evaluation track and the full visual localization track are reported.
构: The A on/of both the B and the C are reported.
```
**C18b** · PixLoc（CVPR 2021）
```text
新: For a yellow region in the source (top), we show in the target camera (middle) the view rendered by the plane-induced warp of the coarse pose.
原: For a red point in the reference image (left), we highlight in the query (right) the multilevel basin of attraction colored by the 2D gradient angle $\nicefrac{\partialF_q}{\partialp_q^i}^\top r_k^i$.
构: For a [colour] A in the B (position), we highlight/show in the C (position) the D [participle] by E.
```
**C21** · Kaufmann et al. (Swift)（Nature 620, 2023）
```text
新: \textbf{Effect of the stopping threshold.} Comparison of the closed-loop behaviour of each variant, illustrated by the SR at 1\,m (left) and the mean number of steps flown (right) for varying thresholds $\tau$.
原: Comparison of the fastest race of each pilot, illustrated by the time behind Swift.
构: Comparison of the A of each B, illustrated by the C.
```
