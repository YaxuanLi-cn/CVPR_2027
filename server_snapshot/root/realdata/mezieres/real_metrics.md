
### All sampled pairs (600 pairs, mean true distance 14.0 m)

| Method | pos. err mean (m) | median | rel. pos. err | <1 m | <2 m | <10% of dist. | heading MAE (deg) | median | tilt MAE (deg) | transl. dir. median (deg) | dist. rel. err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DreamNav stage 2 (refined) | 9.03 | 4.98 | 52.4% | 29.7% | 40.7% | 22.8% | 9.685 | 1.009 | 3.279 | 8.85 | 41.3% |
| DreamNav stage 1 (coarse) | 9.67 | 5.80 | 58.1% | 17.0% | 31.7% | 11.2% | 9.838 | 1.070 | 3.512 | 15.48 | 46.0% |
| DreamNav stage 2, no altitude input | 20.49 | 20.42 | 185.8% | 0.0% | 0.0% | 0.0% | 9.685 | 1.009 | 3.279 | 8.85 | 160.7% |
| MASt3R-stat MLP (RASR + sin/cos) | 18.05 | 17.59 | 157.7% | 0.0% | 0.0% | 0.0% | 11.833 | 2.887 | 5.335 | 46.82 | 110.6% |
| MASt3R-stat MLP, altitude-scaled | 11.55 | 8.02 | 80.8% | 0.8% | 4.2% | 0.0% | 11.833 | 2.887 | 5.335 | 46.82 | 55.8% |
| Constant (zero motion) | 13.99 | 10.35 | 100.0% | 0.7% | 1.8% | 0.0% | 92.721 | 90.058 | 0.502 | 90.00 | 100.0% |

### Baseline/height inside the training range (300 pairs, mean true distance 7.2 m)

| Method | pos. err mean (m) | median | rel. pos. err | <1 m | <2 m | <10% of dist. | heading MAE (deg) | median | tilt MAE (deg) | transl. dir. median (deg) | dist. rel. err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DreamNav stage 2 (refined) | 2.13 | 0.84 | 32.6% | 57.7% | 72.3% | 43.3% | 2.255 | 0.666 | 1.616 | 1.89 | 19.7% |
| DreamNav stage 1 (coarse) | 2.74 | 1.43 | 40.6% | 34.0% | 59.0% | 22.3% | 2.310 | 0.720 | 1.901 | 4.36 | 25.1% |
| DreamNav stage 2, no altitude input | 16.39 | 16.42 | 244.2% | 0.0% | 0.0% | 0.0% | 2.255 | 0.666 | 1.616 | 1.89 | 225.8% |
| MASt3R-stat MLP (RASR + sin/cos) | 12.34 | 11.42 | 196.1% | 0.0% | 0.0% | 0.0% | 3.720 | 1.929 | 3.403 | 40.98 | 143.4% |
| MASt3R-stat MLP, altitude-scaled | 5.28 | 5.10 | 78.1% | 1.7% | 8.3% | 0.0% | 3.720 | 1.929 | 3.403 | 40.98 | 42.9% |
| Constant (zero motion) | 7.18 | 7.84 | 100.0% | 1.3% | 3.7% | 0.0% | 83.710 | 89.339 | 0.452 | 90.00 | 100.0% |

### Baseline/height beyond the training range (300 pairs, mean true distance 20.8 m)

| Method | pos. err mean (m) | median | rel. pos. err | <1 m | <2 m | <10% of dist. | heading MAE (deg) | median | tilt MAE (deg) | transl. dir. median (deg) | dist. rel. err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DreamNav stage 2 (refined) | 15.92 | 18.03 | 72.2% | 1.7% | 9.0% | 2.3% | 17.114 | 3.317 | 4.942 | 45.51 | 62.9% |
| DreamNav stage 1 (coarse) | 16.59 | 18.27 | 75.6% | 0.0% | 4.3% | 0.0% | 17.366 | 3.876 | 5.123 | 48.06 | 66.9% |
| DreamNav stage 2, no altitude input | 24.59 | 24.13 | 127.4% | 0.0% | 0.0% | 0.0% | 17.114 | 3.317 | 4.942 | 45.51 | 95.7% |
| MASt3R-stat MLP (RASR + sin/cos) | 23.76 | 23.54 | 119.3% | 0.0% | 0.0% | 0.0% | 19.946 | 6.524 | 7.267 | 55.54 | 77.7% |
| MASt3R-stat MLP, altitude-scaled | 17.81 | 18.88 | 83.4% | 0.0% | 0.0% | 0.0% | 19.946 | 6.524 | 7.267 | 55.54 | 68.7% |
| Constant (zero motion) | 20.81 | 21.71 | 100.0% | 0.0% | 0.0% | 0.0% | 101.731 | 91.451 | 0.551 | 90.00 | 100.0% |

### Large heading change (> 60 deg) (506 pairs, mean true distance 14.0 m)

| Method | pos. err mean (m) | median | rel. pos. err | <1 m | <2 m | <10% of dist. | heading MAE (deg) | median | tilt MAE (deg) | transl. dir. median (deg) | dist. rel. err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DreamNav stage 2 (refined) | 9.84 | 5.92 | 57.3% | 31.4% | 38.3% | 23.7% | 11.409 | 1.387 | 3.797 | 15.03 | 45.0% |
| DreamNav stage 1 (coarse) | 10.47 | 6.96 | 63.3% | 17.6% | 30.0% | 10.9% | 11.584 | 1.464 | 4.013 | 31.27 | 49.5% |
| DreamNav stage 2, no altitude input | 19.94 | 19.91 | 184.2% | 0.0% | 0.0% | 0.0% | 11.409 | 1.387 | 3.797 | 15.03 | 155.1% |
| MASt3R-stat MLP (RASR + sin/cos) | 16.94 | 16.19 | 153.1% | 0.0% | 0.0% | 0.0% | 13.673 | 3.367 | 5.641 | 54.67 | 102.3% |
| MASt3R-stat MLP, altitude-scaled | 12.16 | 9.11 | 85.1% | 1.0% | 4.3% | 0.0% | 13.673 | 3.367 | 5.641 | 54.67 | 60.4% |
| Constant (zero motion) | 13.99 | 10.03 | 100.0% | 0.8% | 2.2% | 0.0% | 108.085 | 91.158 | 0.519 | 90.00 | 100.0% |

### Small heading change (<= 15 deg) (76 pairs, mean true distance 14.7 m)

| Method | pos. err mean (m) | median | rel. pos. err | <1 m | <2 m | <10% of dist. | heading MAE (deg) | median | tilt MAE (deg) | transl. dir. median (deg) | dist. rel. err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DreamNav stage 2 (refined) | 5.11 | 2.13 | 28.8% | 11.8% | 48.7% | 11.8% | 0.304 | 0.180 | 0.506 | 5.47 | 23.2% |
| DreamNav stage 1 (coarse) | 5.85 | 2.76 | 33.3% | 5.3% | 34.2% | 5.3% | 0.347 | 0.188 | 0.902 | 4.61 | 29.2% |
| DreamNav stage 2, no altitude input | 24.27 | 24.52 | 189.2% | 0.0% | 0.0% | 0.0% | 0.304 | 0.180 | 0.506 | 5.47 | 184.8% |
| MASt3R-stat MLP (RASR + sin/cos) | 24.85 | 23.06 | 176.8% | 0.0% | 0.0% | 0.0% | 1.982 | 1.791 | 3.703 | 31.22 | 151.0% |
| MASt3R-stat MLP, altitude-scaled | 8.55 | 6.38 | 56.6% | 0.0% | 0.0% | 0.0% | 1.982 | 1.791 | 3.703 | 31.22 | 31.6% |
| Constant (zero motion) | 14.70 | 11.62 | 100.0% | 0.0% | 0.0% | 0.0% | 2.165 | 0.613 | 0.217 | 90.00 | 100.0% |

World-frame x / y with the learned absolute-heading network (all pairs): MAE 7.49 / 7.59 m; median absolute-heading error 32.6 deg.
Diagnostics: pass-1 inliers median 115, zoom 134, imagination re-matching 546; imagined-vs-real residual median 73.3 px.
