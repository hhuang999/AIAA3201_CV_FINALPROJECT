### Stereo: equal-weight mean over 15 scenes

| configuration   |   mae_px |   valid_fraction |   cpu_seconds |
|:----------------|---------:|-----------------:|--------------:|
| reflect_3way    |   1.3409 |           0.7409 |        0.0631 |
| reflect_hh4     |   1.2899 |           0.7457 |        0.1766 |
| unpad_3way      |   1.4116 |           0.6207 |        0.0767 |

### NYU: 654 held-out images, RGB only

| method         |   abs_rel |   rmse |   delta1 |
|:---------------|----------:|-------:|---------:|
| raw            |    0.1145 | 0.3988 |   0.8923 |
| global_scale   |    0.1020 | 0.3893 |   0.8995 |
| monotone_curve |    0.1008 | 0.3870 |   0.9032 |

### Scene-grouped training out-of-fold scores

| method             |   abs_rel |   rmse |   delta1 |
|:-------------------|----------:|-------:|---------:|
| raw                |    0.1179 | 0.4304 |   0.8842 |
| global_scale       |    0.1050 | 0.4150 |   0.8978 |
| monotone_curve     |    0.1031 | 0.4037 |   0.9013 |
| residual_l0.1_s0.5 |    0.1019 | 0.3997 |   0.9051 |
| residual_l0.1_s1   |    0.1023 | 0.4001 |   0.9062 |
| residual_l1_s0.5   |    0.1020 | 0.4003 |   0.9042 |
| residual_l1_s1     |    0.1018 | 0.3992 |   0.9057 |
| residual_l10_s0.5  |    0.1027 | 0.4025 |   0.9021 |
| residual_l10_s1    |    0.1024 | 0.4015 |   0.9029 |

### Training-only camera diagnosis: 795 images

| method           |   abs_rel |   rmse |   delta1 |   training_median_scale |
|:-----------------|----------:|-------:|---------:|------------------------:|
| known_intrinsics |    0.0971 | 0.3962 |   0.9205 |                  0.9889 |
| rgb_only         |    0.1179 | 0.4304 |   0.8842 |                  0.9561 |

### Efficiency

| model                             | gpu        | precision                       | network_hw   |   seconds_per_image_or_frame |   peak_allocated_mib |   peak_reserved_mib |
|:----------------------------------|:-----------|:--------------------------------|:-------------|-----------------------------:|---------------------:|--------------------:|
| UniDepthV2 RGB-only               | NVIDIA A40 | official internal FP16 autocast | 490x644      |                       0.0250 |             498.0771 |            604.0000 |
| UniDepthV2 known camera, training | NVIDIA A40 | official internal FP16 autocast | 490x644      |                       0.0263 |             497.0161 |            624.0000 |
| Depth Anything V2 Small           | NVIDIA A40 | FP32                            | 518x924      |                       0.0581 |             375.3647 |            448.0000 |
| Video Depth Anything Small        | NVIDIA A40 | official FP16 autocast          | 518x924      |                       0.0618 |            2689.2197 |           3138.0000 |

Times use each model's reported inference scope; offline video time is amortized over the 70 decoded frames. Peak memory is allocated GPU memory; reserved memory is also retained in the CSV and runtime JSON files.

The final method is the monotone curve. No residual candidate passed all metrics in every fold. The residual stage is inactive; its recorded incremental runtime is zero.

### Video: mean adjacent normalized-depth change

| method    |   mean_adjacent_change |
|:----------|-----------------------:|
| framewise |               0.103695 |
| video     |               0.103065 |
