绝对图片 tilt 均匀版本（0026）：

```bash
cd /root/pairUAV_img/dataset_code/build_kml_pair
.venv/bin/python build_single_tour.py --tour 0026 --uniform-image-tilt --publish-standard \
  --kml-root /root/pairUAV_img/processed_kml_uniform_tilt \
  --image-root /root/pairUAV_img/processed_images_uniform_tilt \
  --json-root /root/pairUAV_img/processed_paired_json_uniform_tilt
```

150 个 Camera tilt 在 0–45° 分层随机采样，10 个等宽箱各 15 张；角度从垂直向下算起。角度采样由 tour/seed 确定，每图单独计一次，与 pair 的 tilt 差值五箱约束分别验证。距离仍不超过 256m。容量修复保持每个视角的绝对 tilt，不得回退到旧两端采样。旧版图片保留在 relaxed_batch 目录，新版独立存放；通过验证后 --publish-standard 只更新标准入口符号链接，旧链接目标记入 publication_history；不会因旧版 pair 已完成而跳过新约束。脚本包含断点恢复，完整验证之后生成 absolute_image_tilt 图。`test_image_tilt.py` 检查复现性、旧采样拒绝、范围越界及修复保持约束。

# KML / pair / Google Earth Pro 数据生成

当前使用 `group_batch_pipeline.py`，任务范围为 0001–0050。下文是当前复现流程；此前逐 tour 均衡、两分箱等探索记录归档到 [HISTORICAL_EXPERIMENTS.md](HISTORICAL_EXPERIMENTS.md)，不能作为当前交付配置。

单 tour 可用 `build_single_tour.py` 完整生成/恢复 KML、Google Earth Pro 图片及视觉 pair；默认从 `/root/pairUAV_img/tours_splited_kml/0000/0000.kml` 读取参考 LookAt range，限制相机到 first-key ENU 核心点的三维距离。0026 使用 256 m 最大距离、220 m 中心高度、15 m 水平偏移，最多 150 张；不会启动其他 tour：

```bash
cd /root/pairUAV_img/dataset_code/build_kml_pair
.venv/bin/python build_single_tour.py --tour 0026
```

设计随机种子由基础 seed 与 tour id 确定。脚本会保留并验证已有结果，按 50/100/125/140/150 张断点续跑；只有全量视觉匹配与五轴 5 箱 MILP 通过后才选择、哈希验证并发布。Earth Pro 需要在 `DISPLAY=:1` 上运行；图片导出器使用系统 Python，单 tour 流程会调用它。

## 五个 tour 联合均衡与断点恢复（当前流程）

当前任务范围 `0001–0050`，每五个连续 tour 为一组。每 tour 最多 150 张图片、恰好 1000 个不重复无序 pair；允许选择 pair 方向，但不能同时选择正反方向。组内五个变量分别使用五个等宽分箱，每箱 1000 对，范围为 ENU x/y ±15 m、z ±33 m、tilt ±45°、heading ±180°。这是边缘分箱均衡，不代表联合分布或箱内连续均匀。

流程：几何 MILP 预检 → 复用通过校验的已有图片 → 分块补齐缺失图片 → 真实图片视觉审计 → 联合选择 → 独立验证和分布图。几何预检与最终选择均约束单图最多参与 150 对。几何通过是必要条件，不能在没有图片时保证真实视觉匹配通过。视觉门槛为 20 内点、内点比例 0.25、凸包覆盖 0.01、4 个网格；它不证明同一语义建筑物或米制位姿唯一可解。

运行与复现：

```bash
cd /root/pairUAV_img/dataset_code/build_kml_pair
# 前台运行（与 supervisor 二选一，文件锁防止两个批处理同时运行）
.venv/bin/python group_batch_pipeline.py --start 1 --end 50
# 只做五个 tour 的几何可行性预检，不导出图片
.venv/bin/python group_batch_pipeline.py --start 11 --end 15 --preflight-only
# 安装持久运行配置；配置文件、启动命令均在本目录保存
cp service/kml-group-batch.conf /etc/supervisor/conf.d/
supervisorctl reread
supervisorctl update
supervisorctl start kml-group-batch
# 查看进度
supervisorctl status kml-group-batch
tail -n 40 /var/log/kml-group-batch.log
cat /root/pairUAV_img/processed_paired_json_relaxed_batch/group_batch_manifest.json
```

输出：`/root/pairUAV_img/processed_kml_relaxed_batch/<tour>/`、`/root/pairUAV_img/processed_images_relaxed_batch/<tour>_redesigned/`、`/root/pairUAV_img/processed_paired_json_relaxed_batch/`。源 KML 不修改。每组生成 `group_<first>_<last>.json`、`group_<first>_<last>_validation.json`、`group_<first>_<last>_distributions.png`，分布图用不同颜色显示每个 tour 的补偿贡献。

渲染原生尺寸 3840×2160，裁剪后 JPEG 为 3686×2073。`render_receipt.json` 记录 KML、原始导出、裁剪导出及最终 JPEG 哈希；断点只认可有完整清单且哈希匹配的裁剪图。每次补齐最多十张，失败保留已成功项并尝试恢复 GUI，不会跳过失败组继续盲目渲染后续组。反复失败会在状态文件记录原因并停止服务，便于修复后以同一命令恢复。

新设计通过 `calibrate_earth.py` 在 first-key 坐标读取 Earth Pro 地形遥测，标定缓存带源 KML 哈希。历史设计只要已有任何导出 JPEG 就保留其坐标，以复用图片。尚未渲染且缺少当地高程标定的设计，会在临时目录完整生成校准版本后替换；原设计保留在 KML 根目录的 `.legacy_terrain/<tour>/`，随后重新求解组容量。单 tour 随机种子由 tour id 与种子共同确定。已有设计与指定图片数/种子不符时会报错，避免错误复用。

### 运行与验收补充

`bash setup_batch.sh` 安装 Python 依赖和 supervisor 配置；`requirements-batch.txt` 固定当前使用的主要库版本。现有机器已配置，无需在渲染过程中重新安装。当前 service 配置从 0011 继续到 0050，0001–0010 已独立验收；更改 `service/kml-group-batch.conf` 的起止环境变量后重新安装配置即可调整范围。不要同时运行单张 GUI 调试和批处理。

```bash
.venv/bin/python batch_status.py
.venv/bin/python validate_group.py --group 0001_0005
```

状态查询分别显示已经写入最终图片收据的数量和导出清单已提交的图片数量，后者可能领先一个分块。只有 `group_*_validation.json` 通过才算组级验收通过。

每次开始尚未审计的 tour 前，用已审计 tour 的真实通过边和剩余 tour 的几何候选重新求解容量；如这一乐观上界仍不满足，立即停止后续渲染并保留证据。此检查在断点恢复时同样执行。

2026-10-23 新导出包含 `observed_camera_telemetry`，要求新鲜遥测与 KML 的经纬度、高度、heading、tilt 对应后才保存图片。目前实测交互视口的 `horizFov` 是 60°，KML 请求值为 90°；二者分别记录，不能将请求值作为已经标定的导出相机内参。历史图片没有逐张遥测证据。裁剪、原生导出视口与内参关系需要另行标定后才能用于依赖相机内参的定量实验。

地形标定和相机采样可通过保留 `calibration/` 缓存及源 KML 哈希复现；Google Earth 在线影像、地形更新会影响重新渲染，不能保证跨版本像素一致。


2026-10-23 恢复与中途检查补充：

- 恢复时通过打开并关闭 Qt 文件对话框确认响应，无法响应才重启专用 Earth 服务；崩溃提示按实际窗口坐标识别 Continue。
- 子进程使用独立进程组，超时和服务终止会清理导出器及其 OCR 子进程。Earth supervisor 启动命令永久设置 core 上限为 0。
- 每累计 50 张已校验图片执行部分视觉审计；剔除已确认不通过的边，未生成图片的边仍按乐观几何候选计算组容量。中途通过不等于最终可行，仍需全量审计与最终联合求解。
- 部分审计缓存绑定图像、views.json、算法版本和视觉门槛；最终审计复用匹配记录，避免重复计算。
- 验收检查五个连续 tour、JSON/JSONL 一致性、图像及 KML 对应关系，并把 pair 与审计文件哈希写入验证报告；发布时拒绝已修改的文件。

2026-10-23 GUI 修复：启动调查只在实际小型 Earth 弹窗内识别，避免对地图做 OCR；重新加载本流程生成文件的确认框支持识别后确认。0013 的实际高程校准已通过，0012–0015 的未渲染设计已按各自当地高程更新。

2026-10-23 资源与缓存恢复补充：

- `audit_resources.py` 在持有导出锁期间临时限制空闲 Earth 的 CPU affinity，给图像匹配留出 CPU；退出恢复原线程配置。恢复日志绑定进程/线程启动时间，避免 PID 重用；若审计被 SIGKILL，下一次导出、高程标定或 GUI 恢复会先恢复日志中的配置。
- JPEG 解码校验按文件内容 SHA256 缓存，仍逐次读取并哈希，不信任时间戳。
- 部分审计新增匹配文件 SHA256、记录数、唯一性及候选几何一致性检查；新旧快照错配、截断或重复记录不能复用。完整审计先写临时匹配文件再替换。
- `capacity_diagnostics.py` 报告每 tour 的边数、各分箱及分箱集合的乐观供给缺口，正反方向不能重复计数；通过这些必要条件仍需联合整数求解。超时 `unknown` 不等于证明不可行。
- 后续单 tour 审计图默认使用 5 分箱、tilt ±45°，仅显示候选供给；每 tour 不要求单独均衡，最终以 `group_*_distributions.png` 与独立验证报告为准。

离线恢复回归检查：`.venv/bin/python test_resume.py`、`.venv/bin/python test_partial_cache.py`、`.venv/bin/python test_audit_resources.py`。这些测试不操作真实 Earth GUI 或数据。

容量修复：`preflight_group.py` 对已证明不可行的结果自动调用 `repair_capacity.py`，在临时目录尝试最多四个确定性方案；允许调整完全未渲染的 tour，以及具有完整部分审计证据的 tour 中尚未渲染的视角。所有已有 JPEG（包括暂存导出）对应的视角冻结；完整审计 tour 全部冻结。联合求解可行后，持导出锁再次核对图片、设计和审核摘要未变化才提交；旧 KML 保留在 `.capacity_revisions/`，提交日志支持中断回滚。记录修订编号、随机摘要和触发缺口。`--no-repair` 可关闭修复以单独诊断。求解超时不会触发盲目改图。全部 tour 已出图时仍不能靠几何重设计解决视觉容量不足；程序保留证据，不放宽门槛或静默跳过。

2026-10-23 GPU 渲染：该服务器的 NVIDIA EGL 图形库可用，已安装 VirtualGL 3.1.5，`vglrun -d egl glxinfo -B` 确认 RTX 3090。`../gen_img/install_virtualgl.sh` 固定版本并校验安装包 SHA256；不安装或修改宿主 NVIDIA 驱动。GPU supervisor 配置保存于 `../gen_img/earth-pro-gpu.conf`，软件配置保存于 `../gen_img/earth-pro-software.conf`。在导出分块结束后停止批处理，复制所选 Earth 配置到 `/etc/supervisor/conf.d/earth-pro.conf`，执行 `supervisorctl reread`、`supervisorctl update`、`/usr/bin/python3 ../gen_img/recover_google_earth.py`，验证新导出后恢复 `supervisorctl start kml-group-batch`。0012/0010 已使用 GPU 成功原生导出并通过相机遥测检查；不同渲染后端可能产生像素差异，保留逐图导出记录。

完整审计完成后，`cleanup_partial_audit.py` 逐条确认部分审计记录包含于完整证据，且几何、图像、门槛和文件哈希一致，才清理重复部分缓存。完整匹配证据、图片与 KML 保留。

缓存清理容错：部分缓存元数据缺失、损坏或锁被占用时，保留文件并记录原因，不中断已经完成的视觉审核；完整审核仍须通过独立校验。回归命令：`.venv/bin/python test_cleanup_partial.py`。

部分视角容量修复：`partial_repair.py` 保留已有 KML 的字节及相机信息；逐条校验视觉缓存端点不变后重绑定设计哈希。设计与审核摘要共同写入可回滚事务。离线测试 `.venv/bin/python test_partial_repair.py`；真实数据副本集成测试 `test_partial_repair_integration.py` 依赖 0012 部分审核缓存，结果保存在 `diagnostics/partial_repair_integration_result.json`。该测试保留 51 个已拍摄 KML 和 746 条匹配记录，五个 tour 联合求解通过，未修改真实产物。容量修复仍只证明剩余几何候选有可行解，不能保证未拍摄图片必定通过视觉审核。

拍摄检查点更新：150 张上限时，在 50、100、125、140 张执行部分视觉审核和剩余容量检查，分块大小自动截断到检查点；保留后期未拍摄名额用于修复。匹配缓存继续复用。回归命令：`.venv/bin/python test_checkpoints.py`。

五个 tour 分阶段拍摄：`group_execution.py` 按 50、100、125、140、150 张阶段轮流推进组内 tour。每次增加图像后审计，并联合使用所有已完成和部分审计 tour 的视觉结果检查容量；不会把其他已拍摄 tour 的失败边重新当作未知候选。断点恢复先同步所有 tour 的导出并补齐审计。已有完整审核直接复用；已有图片与 KML 冻结。这样保留多个 tour 的未拍摄名额用于容量修复，但仍不能保证未知图像必定通过视觉门槛。离线回归：`.venv/bin/python test_group_execution.py`。

0027–0030 新生成任务与已有 0026 组成五-tour 组。运行 `.venv/bin/python build_uniform_group.py --tours 0026 0027 0028 0029 0030`，先复用已验证的 0026 图像和 KML，再分阶段生成其余四个 tour；每 tour 1000 对、组内五轴各五箱1000对。每 tour 自身150张图片的绝对 tilt 十箱各15张，并要求所有图片进入最终 pair。全组验证完成后更新标准目录链接；原始版本不删除。状态命令：`.venv/bin/python uniform_progress.py`。Supervisor 配置保存于 `service/kml-0026-0030-uniform.conf`，日志 `/var/log/kml-0026-0030-uniform.log`。

2026-11-02 当前授权范围为 0160–0239（80 个 tour），按 0160–0164、0165–0169……0235–0239 分为16组。`service/run_0160_0239.sh` 是可重复命令入口；`service/kml-0160-0239.conf` 为 supervisor 配置。主程序 `build_uniform_range.py --start 160 --end 239` 遇到未解决失败即保留证据并停止，不跳过失败组；重跑会复用图片及审核缓存。每组先联合预检，不再强求每个 tour 在出图前单独配平；预检也包含所有图片至少进入一个 pair 的约束。

```bash
cd /root/pairUAV_img/dataset_code/build_kml_pair
.venv/bin/python range_progress.py --start 160 --end 239
supervisorctl status kml-0160-0239
tail -n 30 /var/log/kml-0160-0239.log
```

图片/KML/匹配证据的实际目录分别为 `/root/pairUAV_img/processed_images_uniform_tilt/`、`/root/pairUAV_img/processed_kml_uniform_tilt/`、`/root/pairUAV_img/processed_paired_json_uniform_tilt/`。通过独立验证之后才更新 `processed_images/<tour>`、`processed_kml/<tour>`、`processed_paired_json/<tour>.json[ l]` 标准入口。范围状态文件为 `range_0160_0239_progress.json`；完成代表审核和配对通过，而不是仅出图结束。
