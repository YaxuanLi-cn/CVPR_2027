# Root Snapshot

This snapshot covers the non-hidden, project-related files under `/root` that
were not already stored in their own GitHub repository. Paths below `root/`
mirror their original paths relative to `/root`.

| Server path | Snapshot location or source |
| --- | --- |
| `/root/rasr-pairuav` | This repository; its code, reports, and results are committed directly. |
| `/root/baseline` | [`baselines/pairuav5d/`](../baselines/pairuav5d/), including local adapters, reports, metrics, and prediction-only arrays. |
| `/root/PairUAV-5D-stage2-repro-bundle` | `baselines/pairuav5d/server_snapshot/PairUAV-5D-stage2-repro-bundle/` |
| `/root/image_processing_hw` | `server_snapshot/root/image_processing_hw/` with source, reports, figures, logs, and result tables. |
| `/root/mast3r_runtime` | The local probe script is in `server_snapshot/root/mast3r_runtime/`; its upstream source revisions are listed below. Model weights are excluded. |
| `/root/realdata` | Metrics and reports are in `server_snapshot/root/realdata/`; raw images and pair data are excluded. |
| `/root/supp` | `server_snapshot/root/supp/`, including the server's supplementary source and rendered PDF. |
| `/root/PairUAV` | Existing GitHub repository at `ad8f13e9d7be40bab7d25069f7eca14abbcb226e`; the server-only PairUAV-5D runner is archived under `baselines/pairuav5d/server_snapshot/PairUAV/`. |
| `/root/dreamNav` | Existing GitHub repository at `ab46b47697098681b4561881d2fbad91630edbee`. |
| `/root/pairUAV-5D-temp` | Dataset and match cache; excluded. |

## Pinned Upstream Repositories

These checkouts are pinned to the exact revisions used on the server. MASt3R,
its `dust3r/croco` submodules, and tiny-cuda-nn source snapshots are included
under `server_snapshot/root/`; the local tiny-cuda-nn `setup.py` is overlaid
there. PairUAV and DreamNav are already available at their listed GitHub commits
and are not duplicated.

| Server checkout | Upstream | Revision | Local state captured here |
| --- | --- | --- | --- |
| `/root/PairUAV` | `YaxuanLi-cn/PairUAV` | `ad8f13e9d7be40bab7d25069f7eca14abbcb226e` | PairUAV-5D runner under the baseline snapshot. |
| `/root/dreamNav` | `YaxuanLi-cn/dreamNav` | `ab46b47697098681b4561881d2fbad91630edbee` | Checkout was clean. |
| `/root/mast3r_runtime/mast3r` | `naver/mast3r` | `f5209afc300cec36239a7ac992263f36847bbba0` | Probe script copied to the root snapshot. |
| MASt3R submodule `dust3r` | `naver/dust3r` | `3cc8c88c413bb9e34c41db0e0eef99c2ee010b12` | No source changes; nested `croco` revision follows. |
| MASt3R submodule `dust3r/croco` | `naver/croco` | `d7de0705845239092414480bd829228723bf20de` | No source changes. |
| `/root/image_processing_hw/third_party/tiny-cuda-nn` | `NVlabs/tiny-cuda-nn` | `3daa6e5b8b8e0fa17971a1a20acfbb3754494b38` | Modified `bindings/torch/setup.py` copied to the root snapshot. |
| tiny-cuda-nn submodule `dependencies/cmrc` | `vector-of-bool/cmrc` | `952ffddba731fc110bd50409e8d2b8a06abbd237` | Pin from tiny-cuda-nn tree. |
| tiny-cuda-nn submodule `dependencies/cutlass` | `NVIDIA/cutlass` | `82f5075946e2569589439d500733b700a3141374` | Pin from tiny-cuda-nn tree. |
| tiny-cuda-nn submodule `dependencies/fmt` | `fmtlib/fmt` | `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa` | Pin from tiny-cuda-nn tree. |

## Exclusions

The snapshot leaves out hidden configuration and credentials, `.git` metadata,
virtual environments, Python caches, raw datasets and video collections,
training runs, model checkpoints, pretrained weights, and large feature caches.
PairUAV prediction archives contain predictions only; target arrays are
omitted. Reproduction still requires the documented datasets and weights from
their original sources.
