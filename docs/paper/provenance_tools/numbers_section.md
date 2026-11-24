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
