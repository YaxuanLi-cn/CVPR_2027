# PairUAV-5D Baseline Archive

This directory preserves the baseline-specific code and small result artifacts
from the experiment server. `server_snapshot/` keeps their source-relative paths
where practical. The summary tables are in
[`server_snapshot/PAIRUAV5D_MAST3R_RESULTS.md`](server_snapshot/PAIRUAV5D_MAST3R_RESULTS.md)
and [`server_snapshot/benchmark_results.md`](server_snapshot/benchmark_results.md).
The real-data reports are under [`server_snapshot/realdata/mezieres/`](server_snapshot/realdata/mezieres/).

The snapshot includes PairUAV adaptations, metric JSON, logs, and per-row
prediction arrays. It excludes checkpoints, pretrained weights, datasets,
training caches, and virtual environments. Prediction NPZ files retain model
predictions but have ground-truth arrays removed. Recompute metrics only with
the authorized dataset and the split described in the result reports.

Some historical shell launchers retain the server's original absolute paths.
Use the included Python entry points and provide the dataset/checkpoint paths
for a new machine; the result reports document the exact split and run setup.

## Upstream Source Revisions

The baseline adapters depend on these upstream source revisions:

| Source | Repository | Revision |
| --- | --- | --- |
| DreamNav baseline collection | `YaxuanLi-cn/dreamNav` | `ab46b47697098681b4561881d2fbad91630edbee` |
| DINO-WM | `world-model-2026/dino-wm` | `359aae94edd9399ff4428e33671713ffe36e3f61` |
| DINO-WM PairUAV fork | `gaoyuezhou/dino_wm` | `0a9492fa12044b852ae9e001cc74604b79c8bb0c` |
| ViNT / NoMaD | `robodhruv/visualnav-transformer` | `dca79815b704e5aa9c6bdc3082351f9e3b2848c2` |
| PairUAV source | `YaxuanLi-cn/PairUAV` | `ad8f13e9d7be40bab7d25069f7eca14abbcb226e` |

The revisions identify the upstream source trees used on the server. The
server-specific PairUAV adaptation scripts and result artifacts are included in
this archive; external model weights and datasets must be obtained separately
from their original sources. See the root
[`docs/REPRODUCTION.md`](../../docs/REPRODUCTION.md) for the released RASR
pipeline and required external artifacts.
