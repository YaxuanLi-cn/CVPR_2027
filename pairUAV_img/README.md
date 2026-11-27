# pairUAV_img — 从 KML 到 pair（图 + KML + JSON）的全流程

本目录是 pairUAV-5D 数据集生成所需的全部代码与原始数据。输入是 University-1652
的航线 KML，输出是每个 tour 的 150 张 Google Earth Pro 原生导出图、150 个相机 KML，
以及 1000 个经过真实图片视觉匹配筛选的 pair（JSON / JSONL）。

目录固定放在 `/root/pairUAV_img`（代码和服务脚本里写的是这个绝对路径，见文末“移动目录”）。
在 CVPR_2027 仓库里它位于 `pairUAV_img/` 子目录，使用时把该子目录复制或链接到 `/root/pairUAV_img`。

**原始 KML 不在本目录里，也不在 git 仓库里。** 使用前把 University-1652 的航线 KML 拆分版放到
`/root/pairUAV_img/tours_splited_kml/`（每个 tour 一个目录，`<tour>/0000.kml` …；合并版可用
`dataset_code/step1/split_tours_kml.py` 拆分），或者在命令里用 `--input-root <路径>` 指定它的位置。
本目录的其余内容全部入库，包括 `.venv`、运行时状态和缓存；`.venv` 里的解释器链接指向
`/venv/main/bin/python3`，换机器后如不可用，用 `dataset_code/build_kml_pair/requirements-lock.txt` 重建。

## 目录结构

```
/root/pairUAV_img/
├── README.md                    本文件
├── tours_splited_kml/           原始 University-1652 航线 KML（拆分版，1652 个 tour × 271 帧；需自行放入，不在仓库里）
├── dataset_code/
│   ├── step1/                   合并版 KML → 拆分版 的脚本（拆分版即由它生成，留作记录）
│   ├── gen_img/                 Google Earth Pro 出图：导出器、环境安装、GUI 恢复、输出校验
│   ├── build_kml_pair/          设计视角/KML、分阶段出图、视觉审计、选 pair、验证、发布
│   │   ├── .venv/               流水线用的 Python 环境（numpy/scipy/opencv/pyproj/matplotlib）
│   │   ├── calibration/         各 tour 的地形高程标定缓存（绑定源 KML 哈希，保证设计可复现）
│   │   ├── service/             supervisor 配置与启动脚本模板
│   │   └── README.md            流水线的详细设计说明与历史记录
│   └── hf_upload/               把一个已验证的组整理成 images/ kml/ pairs/ 交付格式并上传 HF
└── earth_runtime/               Earth Pro 的专用 HOME（配置、My Places 里的相机遥测链接）
```

运行后会在本目录下生成（现在不存在，属于产物）：

| 目录 | 内容 |
| --- | --- |
| `processed_kml_uniform_tilt/<tour>/` | `0000.kml`…`0149.kml`、`views.json`（设计与候选边） |
| `processed_images_uniform_tilt/<tour>_redesigned/` | `0000.jpg`…`0149.jpg`、`render_receipt.json` |
| `processed_paired_json_uniform_tilt/` | `<tour>_group_<组>.json/.jsonl`、`group_<组>_validation.json`、分布图、审计证据 |
| `processed_kml/` `processed_images/` `processed_paired_json/` | 验证通过后的“标准入口”符号链接 |

## 全流程：一条命令

每五个连续 tour 为一组（组内联合配平）。前提是 Earth 相关服务在运行（见“环境”）。

```bash
cd /root/pairUAV_img/dataset_code/build_kml_pair

# 一组（五个连续 tour）
.venv/bin/python build_uniform_group.py --tours 0640 0641 0642 0643 0644

# 一段范围（必须是整组，遇到失败即停，不跳组）
.venv/bin/python build_uniform_range.py --start 640 --end 659
```

两条命令都可以随时中断后原样重跑：已通过哈希校验的图片、审计缓存、设计都会复用。
一组（750 张图）在本机约 6–7 小时。长时间运行建议用 supervisor，模板见
`build_kml_pair/service/kml-0560-0639.conf` 与 `run_0560_0639.sh`（改起止编号即可）。

各阶段对应的脚本（`build_uniform_group.py` 按此顺序调用）：

| 阶段 | 脚本 | 产物 |
| --- | --- | --- |
| 1 地形标定 | `calibrate_earth.py` | `calibration/<tour>.json` |
| 2 设计 150 个视角并写 KML | `design_views.py`（`image_tilt.py`、`balance.py`） | `<tour>/NNNN.kml`、`views.json` |
| 3 组内几何可行性预检 | `preflight_group.py`（必要时 `repair_capacity.py`） | `group_<组>_preflight.json` |
| 4 分阶段出图 50/100/125/140/150 | `group_execution.py` → `gen_img/export_google_earth_pro.py` | `NNNN.jpg`、`render_receipt.json` |
| 5 视觉匹配审计 | `partial_visual_audit.py`、`audit_visual_pairs.py` | `<tour>/visual_audit_redesigned/matches.jsonl` |
| 6 联合选 pair | `select_group_visual_pairs.py` | `<tour>_group_<组>.json/.jsonl`、`group_<组>.json` |
| 7 独立验证 | `validate_group.py` | `group_<组>_validation.json`、分布图 |
| 8 发布标准入口 | `publish_group.py` | `processed_kml/ images/ paired_json/` 下的符号链接 |

查看进度：

```bash
.venv/bin/python range_progress.py --start 640 --end 659
cat /root/pairUAV_img/processed_paired_json_uniform_tilt/group_0640_0644_progress.json
```

## 输出标准

- 图片：Earth Pro 原生 3840×2160 导出后裁掉署名条，成品 **3686×2073 JPEG**（质量 80）。
  每张图导出时用相机遥测核对经纬度、高度、heading、tilt 与 KML 一致，哈希写入
  `render_receipt.json`；只有哈希匹配的图才算完成。
- 每个 tour：150 张图；绝对 tilt 在 0–45° 的 10 个等宽箱中各 15 张；相机到核心点距离 ≤ 256 m。
- 每个 tour：恰好 1000 个不重复无序 pair，每张图至少进入一个 pair，单图最多 150 对。
- 每组（5000 对）：x/y（±15 m）、z（±33 m）、tilt 差（±45°）、heading 差（±180°）五个量各分 5 个
  等宽箱，每箱 1000 对。
- 视觉门槛：内点 ≥ 20、内点比例 ≥ 0.25、凸包覆盖 ≥ 0.01、覆盖网格 ≥ 4。
- 只有 `group_<组>_validation.json` 的 `status` 为 `validated` 才算该组通过。

一条 pair 记录（JSONL 的一行）包含：`pair_id`、`tour_id`、两张图的编号 `a`/`b`、
`delta`（`x_m`、`y_m`、`z_m`、`tilt_deg`、`heading_deg`）、匹配统计（`inliers`、`matches`、
`inlier_ratio`、`hull_fraction`、`grid_cells`、`model`）、`kml_a`/`kml_b`、`image_a`/`image_b`
以及 `anchor`（核心点坐标与地形标定）。

## 整理成交付格式并上传（可选）

```bash
cd /root/pairUAV_img/dataset_code/hf_upload
# 把已验证的组 0640–0644 整理到 /root/pairUAV_img/hf_upload_staging/payload/
../build_kml_pair/.venv/bin/python stage_group.py 640
# 上传（需要 huggingface_hub，可用 /venv/main/bin/python3；token 放在一个 0600 文件里）
HF_UPLOAD_GROUP=0640_0644 HF_UPLOAD_TOKEN_FILE=/path/to/token /venv/main/bin/python3 upload_group.py
```

交付布局：`images/<tour>/NNNN.jpg`、`kml/<tour>/NNNN.kml`、`pairs/<tour>.jsonl`
（JSONL 里的路径改写为上述相对路径）。这两个脚本取自最后一次实际上传（0635–0639）所用的版本，
迁移时只改了路径并把组号改成参数，迁移后没有重新实际上传过。

## 只出图（不配对）

```bash
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0003 --limit 10
/root/pairUAV_img/dataset_code/gen_img/verify_outputs.py /root/pairUAV_img/images/0003
```

直接按原始航线 KML 出图，输出在 `/root/pairUAV_img/images/<tour>/`。详见 `gen_img/README.md`。

## 环境

目录外的系统级依赖（本机已装好）：Google Earth Pro 7.3.7、VirtualGL 3.1.5、Xvfb、Openbox、
xdotool、wmctrl、tesseract，以及四个 supervisor 服务
`earth-xvfb`、`earth-window-manager`、`earth-pro`、`earth-camera-telemetry`。

```bash
supervisorctl status | grep earth          # 四个都应为 RUNNING
/usr/bin/python3 /root/pairUAV_img/dataset_code/gen_img/recover_google_earth.py   # Earth 窗口无响应时
```

在新机器（Ubuntu 24.04 amd64、有 NVIDIA GPU、有 supervisor）上重装：

```bash
bash /root/pairUAV_img/dataset_code/build_kml_pair/setup_batch.sh
```

它会安装 Earth Pro 和依赖、建 `.venv`、写入并启动上述服务。

注意：

- Earth Pro 会自己在 `/root/.googleearth/` 建影像缓存（与 HOME 设置无关）。那是可丢弃的缓存，
  删掉后下次启动自动重建。
- 容器 stop/start 后 `/tmp/.X1-lock` 会残留导致 Xvfb 起不来；`earth-xvfb` 的启动脚本已加入
  自动清理残留锁的处理。
- 同一桌面一次只能跑一个导出任务（`gen_img/export.lock`）。
- Google Earth 在线影像和地形会更新，重新渲染不保证与旧图逐像素一致。

## 离线自检（不动 Earth、不动数据）

```bash
cd /root/pairUAV_img/dataset_code/build_kml_pair
for t in test_resume test_partial_cache test_audit_resources test_checkpoints test_cleanup_partial \
         test_full_audit_cache test_group_execution test_partial_repair test_uniform_range test_image_tilt; do
  .venv/bin/python $t.py
done
```

`test_partial_repair_integration.py` 依赖 0011/0012 的历史中间数据，这些数据已删除，该测试无法再运行。

## 整理后的验证记录（2026-11-26）

代码从 `/root/dataset_code` 迁到本目录、旧产物删除前后，逐段实际跑过：

| 环节 | 做法 | 结果 |
| --- | --- | --- |
| 出图 | 用迁移后的导出器重渲已交付的 0639 的 3 个视角，与原交付图比较 | 3686×2073，相机遥测核对通过；与原图相关系数 1.0000，平均灰度差 0.33 |
| 全视觉审计 | 对 0639 的 150 张原图重跑 `audit_visual_pairs.py` | 6633 条候选边、4710 条通过，逐条判定与原审计完全一致 |
| 选 pair / 验证 / 出图表 / 发布 | 对 0635–0639 的原始数据重跑阶段 6–8 | `validated`，5000 对；五个 tour 的 pair JSONL 与原交付逐字节相同 |
| 新组全流程前半 | 旧数据删除后，对全新的组 0640–0644 运行 `build_uniform_group.py` | 5 个 tour 地形标定、设计、组预检通过；0640 出图 50 张（含一次中断续跑）、部分视觉审计与容量检查通过后人工停止 |
| 只出图入口 | 最终状态下 `generate.py --tour 0003 --limit 2` | 2 张 3686×2073，`verify_outputs.py` PASS |
| 离线自检 | 上面列出的 10 个测试 | 全部 PASS |

没有做的：没有在新位置把一个全新的组从头跑到 `validated`（约 6–7 小时）；`hf_upload` 没有重新实际上传。
上述测试产物均已删除；`calibration/` 里保留了 0640–0644 的标定缓存。

## 移动目录

路径 `/root/pairUAV_img` 写在代码默认参数、`earth_runtime/.config` 和
`/opt/supervisor-scripts/earth-*.sh` 里。要换位置：整体移动后对 `dataset_code` 和
`earth_runtime/.config` 做一次字符串替换，再重跑 `gen_img/setup_google_earth_runtime.sh`
并重装遥测服务脚本（`setup_batch.sh` 的最后几行）。
