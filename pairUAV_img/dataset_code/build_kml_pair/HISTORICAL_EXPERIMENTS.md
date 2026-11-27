# KML / pair / Google Earth Pro 数据生成

已完成 `tour0000` 的实际端到端数据：150 张调整后 KML 图片、1200 个 pair，输出在：

- `/root/pairUAV_img/processed_kml/0000/`
- `/root/pairUAV_img/processed_images/0000/`
- `/root/pairUAV_img/processed_paired_json/0000.json`

## 对任意 tour 运行

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/pipeline.py \
  --tour 0000 --seed 0 --images 150 --pairs 1200 \
  --settle 20 --timeout 180
```

只需替换 `--tour`。`--images` 最大 150；`--pairs` 必须大于 1000 且能被 `--bins` 整除。相同 tour、相同 seed 和相同代码版本结果可复现；改 tour 或 seed 会生成不同视角。

分阶段运行：

```bash
.../build_pairs.py --tour 0000 --seed 0 --images 150 --pairs 1200
.../render_pairs.py --tour 0000 --settle 20 --timeout 180
.../validate_pairs.py --tour 0000 --require-images
```

`build_pairs.py` 只修改 `/root/pairUAV_img/processed_kml/`，不会修改原始 `/root/pairUAV_img/tours_splited_kml/`。`render_pairs.py` 使用 `/root/pairUAV_img/dataset_code/gen_img/export_google_earth_pro.py` 的 Google Earth Pro 原生 3D Save Image，输出 3840×2160 原生图并交付约 3686×2073 的去署名 JPEG。

## 几何和 pair 定义

- x：东向位移；y：北向位移；z：上向位移，单位都是米。
- 位移是两个相机实际位置的 `B-A`，在固定的 first-key WGS84 ENU 坐标系中计算。
- heading 和 tilt 是 `B-A`；heading 使用 `[-180,180)` 环绕差。
- 约束：`|x|≤15m`、`|y|≤15m`、`|z|≤33m`、`|heading|≤180°`、`|tilt|≤60°`。
- 12 个等宽分箱分别对五个变量强制均匀；1200 pairs 时每个变量每箱精确 100 个。
- A/B 是无序图片组合中固定的一种方向；反向不重复计数。
- 每个候选还要通过 first-key 投影和共同可见网格筛选；距离和姿态相同但没有共同地标区域的 pair 不进入最终集合。

KML 使用 first KML 的 LookAt 目标作为 first-key 区域。当前源数据没有建筑物语义 ID，所以 JSON 明确记录 `feature_key=first-key` 及 source hash；这表示目标建筑区域，而不是自动识别出的建筑名称。

## 输出和校验

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/validate_pairs.py \
  --tour 0000 --require-images
```

校验会拒绝少于 1001 个 pair、超出五个范围、分箱不均、缺图或尺寸错误。每张图片应为 3686×2073 JPEG；原生导出保存在 `processed_images/<tour>/earth_pro_export/native/`。

`build_pairs.py` 会写 `views.json`，其中包括 seed、源文件哈希、相机位置、姿态、投影锚点、候选 pair 和分箱结果。`processed_paired_json/<tour>.json` 只有在图片完成后由 `finalize_pairs.py` 或 `render_pairs.py` 写成 `status=complete`。

## 环境

安装和启动 Google Earth Pro：

```bash
bash /root/pairUAV_img/dataset_code/gen_img/install_google_earth.sh
bash /root/pairUAV_img/dataset_code/gen_img/setup_google_earth_runtime.sh
```

Pair 生成依赖脚本目录的 `.venv`，已安装 numpy、scipy、opencv、pyproj 和 Pillow。Earth Pro 必须在 GPU Xorg `:1` 和 Openbox 下运行；脚本会保留安装、启动、渲染和失败日志。

## 论文级限制

图片几何使用 WGS84 地心坐标转换到局部 ENU，KML 高度使用相对 first-key 的局部高度基准；这保证 pair 内部位移可解释，但不等同于严格大地水准面 MSL 高程。first-key 共同可见性是针孔投影/网格启发式，最终发布前应人工抽查 montage 和随机 pair。当前 `0000` 已完成 150 张图片的 Earth Pro 实渲染和 1200 pair 校验；没有穷举所有 tour 的视觉质量。

## 实际图像可预测性审计（新增）

不能只用相机投影网格来断定 pair 可预测。`audit_visual_pairs.py` 对已由 Earth Pro 导出的图像做 SIFT 双向 ratio 匹配、Fundamental Matrix 几何一致性检查，并要求足够的内点比例、图像覆盖范围和网格分散度。它输出逐候选匹配记录、通过/拒绝示例拼图、各变量分箱可用量图及 JSON/PDF 摘要：

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/audit_visual_pairs.py \
  --tour 0000 --workers 8
```

`0000` 当前审计位于 `/root/pairUAV_img/processed_paired_json/0000/visual_audit/`。对 150 张当前图像检查了 3,663 个几何候选，1,514 个达到严格特征匹配门槛。可查看 `visual_distribution.png`、`accepted_examples.jpg`、`rejected_examples.jpg` 和 `summary.json`。审计显示 tilt 差接近 ±60° 的分箱没有一个候选通过视觉筛选；因此当前图像集不可能同时提供全 `[-60,60]` 倾角覆盖和视觉筛选后的均匀 pair。当前 1,200-pair JSON 仍是旧的“数值/几何筛选”结果，不能描述成已经通过图像可预测性验证。

此审计是可复现的必要筛查证据，不是对真实建筑物身份的语义识别，也不能单凭两张图片保证米制平移的尺度和地理 ENU 方向唯一可观测。正式发布前，需重新设计视角和 anchor 采样，再通过真实影像筛选、联合分布整数求解及人工审阅；如果某个约束下没有超过 1,000 个通过筛查的均匀 pair，应明确报告不可行，不能发布旧的几何候选来凑数。

Earth Pro 导出器的 Qt Open 对话框已修复为检查路径存在、尝试显式激活 Open 按钮并确认对话框关闭；复现命令仍由 `render_pairs.py` 调用。


## 0000 当前视觉筛选版本

`audit_visual_pairs.py` 完成后，运行以下命令对 pair 做精确联合约束求解：

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/select_visual_pairs.py \
  --tour 0000 --count 1200 --bins 2 --step 0.001
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/validate_pairs.py \
  --tour 0000 --require-images
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/plot_distributions.py --tour 0000
```

当前 `0000` 结果为 1,200 对，每个变量以 2 个等宽区间各 600 对精确平衡；实际图像候选支持的最大对称 tilt 范围为约 `[-39.62°, +39.62°]`，0.001° 网格搜索找到 `[-39.619°, +39.619°]` 的可行边界（原候选最大绝对差约 39.6197°）。注意已选 1,200 对的实际最小/最大 tilt 差约为 `[-34.80°, +39.61°]`；两箱只保证正负半区各 600 对，不保证连续范围端点覆盖。注意这是**很粗的两箱边际均匀**，不能等同于细粒度连续均匀，也不是五维联合分布均匀。五变量 3 箱联合试验在超过 1,000 对时不可行。因此 JSON 将分箱数、tilt 范围、特征匹配阈值和限制写明。此前未做视觉筛选的 JSON 备份为 `/root/pairUAV_img/processed_paired_json/0000_unfiltered_legacy.json`。


## 高倾角差重新设计（必须 tilt Δ 至少 ±45°）

`--bins 12` 是默认正式配置；旧 `--bins 2` 仅为前一轮可行性探测，不能用于交付。新方案用 `design_views.py --paired-tilt-groups` 将相近航向/区域中的相机 tilt 覆盖 0°–60°，并提高相机高度和水平视场角以增加大 tilt 差下的共同场景。`redesign_pipeline.py` 在隔离输出目录执行原生 4K 导出、实际图像特征审计、精确联合 MILP，再按可行结果选最大 tilt 差区间；`--min-tilt-delta 45` 是硬约束，找不到 1,200 对时会失败，不会回退到 ±20° 或 2 箱。

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/redesign_pipeline.py \
  --tour 0000 --seed 47 --images 150 --pairs 1200 --bins 12 \
  --min-tilt-delta 45 --camera-height 500 --hfov 90
```

这会写入 `processed_kml_redesigned/0000`、`processed_images/0000_redesigned` 和 `processed_paired_json/0000.json`（旧 0000 视觉审计版本先备份）。当前 16 张高/低 tilt 试点中，约 59° tilt 差的 2 对通过严格特征筛查；其余多对失败，说明必须靠 150 张全图和 MILP 验证总体配额，不能仅根据少数成功例宣称 ±45° 已可行。

## 2026-10-21 约束可行性结论（tour0000）

`redesign_pipeline.py` 现在会在真实 Google Earth Pro 图片完成后运行 SIFT mutual-ratio + USAC-MAGSAC 审计，再运行整数约束求解；审计目录通过 `--audit-dir` 显式传递，求解失败会直接退出，不会发布未验证 pair。`audit_visual_pairs.py` 支持通过命令行调整视觉门槛，默认仍为严格门槛：40 个内点、内点比例 0.45、覆盖 4%、至少 6 个网格。

对 `tour0000` 的三批 150 张原生 3840×2160、交付 3686×2073 的 Earth Pro 图像进行了实际审计。严格门槛下有 2266 个通过候选，但在 tilt 半范围至少 45°时，12 个等宽分箱的乐观上界只有 48 对。为了检验“少量共同内容”是否足够，使用记录在案的放宽门槛（20 内点、比例 0.25、覆盖 1%、4 网格）重新审计；seed 49 有 3757 个候选，seed 50 有 2511 个候选。对 3、4、5、6、8、10、12 分箱进行 exact marginal MILP 求解，均无法取得 1200 对；只有 2 分箱的诊断可取得 1200 对。2 分箱结果不作为交付数据。

完整证据保存在 `/root/pairUAV_img/processed_paired_json/0000_redesign_capacity_report.json`，seed 50 的 KML、图片和审计副本在 `processed_kml_redesigned/0000_seed50`、`processed_images/0000_seed50`、`processed_paired_json/0000_seed50`。因此当前 tour0000 没有被错误地写入一个声称满足用户底线的最终 pair JSON；要得到可发布结果，需要增加视图数量、扩大可见共同内容，或放宽“每个等宽分箱精确等量”的要求。

## 放宽视觉门槛的已完成结果

用户确认采用放宽视觉门槛后，`tour0000` 已完成一组可复现结果：

- 150 张 Earth Pro 3D 图片，3686×2073 JPEG；
- 10 个等宽 marginal bins；
- 1050 个 pair，每个变量每箱精确 105 个；
- tilt 差范围 `[-45°, 45°]`；
- 视觉筛选：20 内点、内点比例至少 0.25、凸包覆盖至少 1%、至少 4 个 4×4 网格；
- 1050 个 unordered pairs 唯一，单张图最多参与 46 个 pair。

结果文件：

- `/root/pairUAV_img/processed_kml_relaxed/0000/`
- `/root/pairUAV_img/processed_images_relaxed/0000/`
- `/root/pairUAV_img/processed_paired_json/0000_relaxed_10bins.json`
- `/root/pairUAV_img/processed_paired_json/0000_relaxed_10bins.jsonl`
- `/root/pairUAV_img/processed_paired_json/0000_relaxed_10bins_validation.json`
- `/root/pairUAV_img/processed_paired_json/0000_relaxed_10bins/pair_distributions.png`
- `/root/pairUAV_img/processed_paired_json/0000_relaxed_10bins/pair_scatter.png`

可复现命令（使用隔离输出目录，避免覆盖旧数据）：

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/redesign_pipeline.py \
  --tour 0000 --seed 49 --images 150 --pairs 1050 --bins 10 \
  --min-tilt-delta 45 --tilt-min 0 --tilt-max 45 \
  --tilt-group-count 15 --edge-fraction 0.8 \
  --camera-height 500 --position-radius 15 --hfov 90 \
  --min-inliers 20 --min-inlier-ratio 0.25 \
  --min-hull-fraction 0.01 --min-grid-cells 4 \
  --input-root /root/pairUAV_img/tours_splited_kml \
  --kml-root /root/pairUAV_img/processed_kml_relaxed \
  --image-root /root/pairUAV_img/processed_images_relaxed \
  --json-root /root/pairUAV_img/processed_paired_json
```

## 0000–0100 批处理

`batch_relaxed_pipeline.py` 支持 `0000` 到 `0100`（共 101 个 tour），会逐 tour 运行已确认的配置，跳过已经完成的 tour，并在每个 tour 开始前检查可用磁盘。每个 tour 完成视觉审计后会删除 SIFT 特征缓存，只保留匹配记录、摘要、分布图和 pair JSON；导出 native 图片也会在压缩后删除。

当前容器剩余约 11.85GB。使用 `--jpeg-quality 80` 时，按已完成的 0000 样本估算，101 个 tour 的交付图片约需 32GB，加上 JSON/KML/审计文件建议至少准备 40GB 可用空间。不要使用 quality 30 来强行塞入当前磁盘，因为那会明显降低 4K 图像的 JPEG 细节质量。

挂载足够大的 volume 后执行：

```bash
/root/pairUAV_img/dataset_code/build_kml_pair/.venv/bin/python \
  /root/pairUAV_img/dataset_code/build_kml_pair/batch_relaxed_pipeline.py \
  --start 0 --end 100 --seed 49 --images 150 --pairs 1050 --bins 10 \
  --jpeg-quality 80 --min-free-gb 5 \
  --input-root /root/pairUAV_img/tours_splited_kml \
  --kml-root /root/pairUAV_img/processed_kml_relaxed_batch \
  --image-root /root/pairUAV_img/processed_images_relaxed_batch \
  --json-root /root/pairUAV_img/processed_paired_json_relaxed_batch
```

脚本会生成 `/root/pairUAV_img/processed_paired_json_relaxed_batch/batch_manifest.json`，中断后重新执行同一命令即可从未完成 tour 继续。

批处理期间 Google Earth Pro 的 supervisor 服务已设置 `ulimit -c 0`，防止崩溃生成几十 GB 的 core dump；Google Earth 的 tile cache 仍可能增长，但不会留下 core 文件吞掉输出空间。

