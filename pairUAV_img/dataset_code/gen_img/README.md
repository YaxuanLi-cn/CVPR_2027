# Google Earth Pro KML 批量导图

## 日常使用：只改 tour ID 和张数

```bash
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0003 --limit 10
```

首次运行自动安装缺少的软件、启动 Xvfb `:1` + Openbox + Google Earth Pro。VirtualGL 使用 NVIDIA EGL 进行 GPU 渲染，桌面分辨率为 3840×2160，无需宿主 Xorg 显示驱动。
使用系统 `/usr/bin/python3`，无需激活虚拟环境。
**只需要 KML，不需要任何参考截图。** 同一桌面一次运行一个导出任务。

输出目录：`/root/pairUAV_img/images/0003/`（随 tour 自动变化）

- `tour0003_0000.jpg` 等：裁边后的 JPEG，3686×2073。
- `native/`：Earth Pro 原生 3840×2160 JPEG，含来源署名。
- `preview_montage.jpg`：前 20 张的缩略预览，预览下方带文件名。
- `manifest.json`：当前运行状态、相机 XML、源文件和图片 SHA256、尺寸、裁剪框。
- `export.log`、`runs/`：完整日志和诊断图。

```bash
# 换 tour，前 25 张
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0004 --limit 25
# 所有 KML
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0003 --all
# 从第 11 个文件开始做 10 张
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0003 --start 10 --limit 10
# 只检查输入和输出路径，不启动环境
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0003 --limit 10 --check
# 自定义数据位置和输出根目录
/root/pairUAV_img/dataset_code/gen_img/generate.py --tour 0003 --limit 10 \
  --input-root /path/to/tours --output-root /root/pairUAV_img/images
```

默认识别 `/root/pairUAV_img/tours_split_kml/` 和 `/root/pairUAV_img/tours_splited_kml/`。
按文件名排序（数据集为四位补零文件名）；X 大于剩余文件数时处理全部剩余文件。
再次运行会覆盖本次选中的同名输出；未选中的旧图片保留，当前批次以 manifest 为准。
慢网络可使用 `--settle 60 --timeout 300`，每张至少等待 60 秒再检查场景稳定。

## 环境安装命令

支持 Ubuntu 24.04 amd64，需要网络、root 和正在运行的 Supervisor。
本实例已有 Supervisor，安装保留其现有配置，不重启服务器管理服务。

```bash
bash /root/pairUAV_img/dataset_code/gen_img/install_google_earth.sh
bash /root/pairUAV_img/dataset_code/gen_img/setup_google_earth_runtime.sh
```

安装日志在脚本目录的 `install.log`；服务由 Supervisor 管理，运行日志在 `/var/log/portal/earth-*.log`。
停止 Earth 服务时使用 `supervisorctl stop earth-pro`，不用停止 Caddy 等平台服务。

## 工作原理与画质

`export_google_earth_pro.py` 从 KML 复制原始 LookAt/Camera 全部参数到不可见
Point Placemark，导入后由 Earth Pro 执行相机定位。开启 3D Buildings 和 Terrain，
关闭地图标签、标题、图例、指南针等叠加层，调用原生 Save Image。
本工具没有生成式补图、瓦片透视合成或图片放大步骤。

最终 JPEG 是原生 4K 的裁剪，不是严格 3840×2160：裁掉两侧各 2%，
并把垂直裁剪量集中到底部以去除整个署名条，结果为 3686×2073。
裁剪会轻微改变取景范围，完整原始取景保存在 native 中。
原生导出分辨率不等于无限细节：源影像、当地 3D 数据和加载情况决定可见细节，
不能承诺任意地区、任意放大倍数都保持锐利。没有 3D 覆盖的地区无法凭空生成建筑模型。

脚本拒绝尺寸错误、中心近乎纯色的导出；失败时返回非零并写入 failed 状态。
场景稳定是加载启发式判断，不能代替对新地区源数据质量的检查。
只检查 manifest.status=complete 不能证明所有相机参数都已在渲染器中准确执行；
可通过保留的相机 XML、原始 JPEG 和预览核对。

## 校验本次输出

```bash
/root/pairUAV_img/dataset_code/gen_img/verify_outputs.py /root/pairUAV_img/images/0003
```

每次生成完成会自动执行校验，也可以单独运行上述命令。
校验源 KML、临时 KML 和 manifest 的相机 XML 是否一致，检查原生与裁剪 JPEG
尺寸、文件完整性及 SHA256。脚本不把相机 XML 校验等同于画面视觉验证。

本次新服务器上测试了 `0003` 前十张，以及切换到 `0002`、从第 3 张开始导出两张。
没有穷举所有 1652 个 tour；入口对 tour ID 和选中数量没有硬编码。
