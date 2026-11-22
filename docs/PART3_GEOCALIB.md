# Part 3: `GeoCalib-3D` — Zero-GT-Intrinsics Joint Geometric & Semantic Monocular Metric Depth Calibration

This technical report documents the full design, training-only residual diagnosis, dual-track 26-candidate scene-disjoint cross-validation (`5` folds $\times$ `4` seeds), parameter freeze, held-out `654`-image evaluation, mechanistic case audit, uncontended latency benchmarks, and 300-DPI figures for **`GeoCalib-3D`** on **NYU Depth V2**.

> **Zero-Leakage & Experimental Integrity Protocol**:
> - **Zero Ground-Truth Camera Intrinsics ($K_{\text{true}}$)**: Uses strictly frozen **UniDepthV2 ViT-S** RGB-only predictions (`lpiccinelli/unidepth-v2-vits14`), never ground-truth intrinsics. Cache parity on training split: `median |Delta Z| = 0.000228 m`, `MAE = 0.001001 m`, `fx` relative difference `0.000523`.
> - **Strict Training/Test Isolation**: All oracle diagnosis (`diag_*.csv`), feature PCA/centering, spline knot selection, hyperparameter regularization, 4-seed (`20261031, 42, 100, 20261121`) cross-validation, and full-train parameter fitting ran **exclusively on the `795` official NYU training images (`249` scenes)** with `splits["test"]` blocked by `VerifiedCache(allowed_split="train")` / `VerifiedGeoCalibCache(allowed_split="train")`.
> - **Pre-Test Parameter Freeze & One-Shot Test Evaluation**: Locked in [`frozen_geocalib_model.json`](../results/part3_geocalib/frozen_geocalib_model.json) (`schema: "GeoCalib_3D"`, `complete: true`, `test_read: false`, SHA-256 `816f799a82784d86d8985ee255607fa6a7d2806c09c3df423be038654fc14ebd`) prior to held-out evaluation on `654` test images (`215` scenes) via [`depthlab/geocalib_evaluate.py`](../depthlab/geocalib_evaluate.py).
> - **Explicit Dual-Track Accounting**: **Single-Pass Main Track (`1x` ViT-S forward)** and **Multi-Pass 2× Flip-TTA Track (`2x` ViT-S forward)** are segregated across all parameter counts, timings, tables, and plots.

---

## 1. Training-Split Mechanistic & Oracle Diagnosis (`N=795` Train Only)

Using 5-fold OOF predictions on the `795` training images (`raw` `0.117945 / 0.430373 / 0.884173`, `monotone_spline` `0.103380 / 0.404847 / 0.901091`, geometry-only `geo_focal_3d` `0.092502 / 0.379230 / 0.926687`), [`depthlab/geocalib_diagnose.py`](../depthlab/geocalib_diagnose.py) establishes four structural findings:

1. **Remaining Post-Geometry Oracle Headroom ([`diag_oracles.csv`](../results/part3_geocalib/diag_oracles.csv))**:
   - Even after `geo_focal_3d` (`0.092502`), a per-image scalar minimizes AbsRel down to `0.060208` (`geo_oracle_absrel_scalar`, `+0.032294` headroom; `geo_oracle_med_scalar`: `0.061137`; `geo_oracle_rmse_scalar`: `0.068150` / `0.295606 m`).
   - Adding affine depth or spatial terms yields `0.056390` (`geo_oracle_log_affine`), `0.058401` (`geo_oracle_inv_affine`), `0.054323` (`geo_oracle_spatial_linear`), and `0.049761` (`geo_oracle_spatial_quadric`). Thus ~`0.032` AbsRel sits in image-level metric scale error and ~`0.006–0.010` sits in residual 2D spatial/radial distortion.
2. **Low-Dim Geometry Linearly Saturated vs Rich Internal ViT-S Semantics ([`diag_signal_r2.csv`](../results/part3_geocalib/diag_signal_r2.csv))**:
   - Standard 5-fold scene-CV Ridge regression confirms handcrafted geometry (`geo_sp3d_control`: `CV R^2(e_geo_med) = -0.0094`), confidence summary statistics (`-0.0005`), and second-order geometric interactions (`-0.0016`) are linearly exhausted with respect to median scale residual.
   - Conversely, frozen internal **UniDepthV2 ViT-S** representations predict significant out-of-fold variance: `cls_token` (`384-d` $\rightarrow$ `PCA-32`, `lam=10`) reaches `CV R^2(e_geo_med) = +0.0226` (full `384-d`, `lam=100` reaches `+0.0406` on `e_geo_radial` and `+0.0313` on `e_geo_slope_v`), `depth_feat_stats` (`512-d` $\rightarrow$ `PCA-32`, `lam=10`) reaches `+0.0223` (`+0.0220` on `e_geo_absrel`), and `patch_mean_std` (`768-d`, `lam=100`) reaches `+0.0554` on radial bias (`+0.0383` on vertical slope).
3. **Flip-TTA `-17 sigma` Principal-Point (`cx_off`) Out-of-Distribution Bug Discovered**:
   - Applying geometry calibration directly to horizontally mirrored `cx` (`639 - cx_flip`) shifts `cx_off = (cx - 320)/640` by `-17 sigma` relative to Pass-1 training standardization, blowing up Pass-2 predictions to `0.112093` and degrading `geo_flip_log_avg` from `0.092502` to `0.097793`. Dropping `cx_off` (using flip-symmetric geometry) immediately recovers `flip_geo_only` (`0.091986 / 0.377239 / 0.926907`).
4. **Left-Tail Focal Bimodality (`f_tilde < 0.75`, `n=43`: `26` True Collapse, `7` False Alarm, `10` Mild; [`diag_left_tail_discriminators.csv`](../results/part3_geocalib/diag_left_tail_discriminators.csv))**:
   - When UniDepthV2 predicts very low focal length (`f_tilde < 0.75`), two opposite regimes occur: wide-FoV rooms whose depth truly shrank (`true_collapse`), and close counter/wall shots whose raw depth was already accurate (`false_alarm`).
   - False alarms exhibit much larger metric vertical depth span (`log_span_y`: mean `2.000` vs `1.599`, `AUC = 0.8022`) and more far pixels (`frac_far_4m`: mean `42.0%` vs `19.8%`, `AUC = 0.7912`), while multi-pass flip focal agreement (`f_flip_tilde` `AUC = 0.8297`, `f_tilde_min` `AUC = 0.8187`) and internal encoder predictions (`oof_df_pred_e_geo_med` `AUC = 0.7857`, `Cohen's d = +1.12`) cleanly distinguish regimes.

---

## 2. `GeoCalib-3D` Architecture & 26-Candidate Scene-CV Ablation (`N=795` Train)

[`depthlab/geocalib_model.py`](../depthlab/geocalib_model.py) systematically evaluates `26` structured candidates across 4 design axes (joint focal/depth/semantic optimization under balanced multi-metric log/AbsRel/RMSE loss):

- **Pure-Geometry & Alternative Parametrizations (`geo_free_left_focal .. geo_conf_ctrl`)**:
  - Quantile knots + Huber loss (`geo_qknots_robust`: `0.092372`) and inverse residual (`geo_inv_residual`: `0.092567`) match or marginally edge `geo_focal_3d` (`0.092502`); rational depth (`0.093340`) and confidence stats (`0.093351`) degrade OOF.
- **Internal Semantic Scale Prior (`geosem_cls .. geosem_pca128`)**:
  - Concatenating fold-centered L2-normalized single-pass encoder blocks (`[l2n(cls_384), l2n(patch_mean_384), l2n(patch_std_384), l2n(depth_feat_512)]`, `1688-d` with Ridge L2 `sem_reg=0.05`) cuts Train OOF AbsRel from `0.092502` down to **`0.087323`** (`geosem_scale`, `CV DoF = 113.1`); `PCA-128` (`geosem_pca128`) retains `0.087443` in `152` raw parameters (`DoF = 83.7`).
- **Joint Winner — Track A Primary `geocalib_3d`**:
  - Unifying (1) monotone **inverse-depth spline** (`inv_spline(1/Z_hat)` across knots `[1/12, 1/6, 1/3, 1/1.5, 1/0.7, 1/0.3] m^-1`, naturally linearizing far-field S-curve disparity), (2) flip-symmetric `clean` 3D geometry (`cx_off` excluded, `z_clip=4.0`), (3) single-pass L2-normalized Ridge semantic scale prior (`sem_reg=0.05`), and (4) Ridge-regularized **semantic-conditioned 2D spatial field** (`v_tilde, r^2`, `sem_sp_reg=0.08`) achieves Canonical Train OOF **`0.086894 / 0.366619 / 0.932284`** (`5018` raw params / `137.9` CV Eff.DoF, `146.9` on full-795 fit).
  - It beats `geo_focal_3d` on **5/5** canonical folds (`+0.005608` AbsRel, `95% CI [+0.003786, +0.007455]`) and **4/4** seeds (`4-seed mean = 0.087383 +/- 0.000463` vs `0.092209 +/- 0.000245`). An automated fold-internal stacking blend (`geocalib_3d_blend`) puts weight `alpha = 0.0` on alternative branches, independently selecting pure `geocalib_3d`.
- **Sample Efficiency ([`oof_learning_curve.csv`](../results/part3_geocalib/oof_learning_curve.csv))**:
  - Trained on just `25%` of fitting scenes per fold (~`50` scenes, `154` images), `geocalib_3d` reaches **`0.092241`** OOF AbsRel — already outperforming `geo_focal_3d` trained on `100%` of scenes (`0.092502`).
- **Track B Primary `geocalib_3d_flip` (`2x` Forward)**:
  - Adding a second horizontal-flip ViT-S forward pass with un-mirrored `K2`, learnable log-depth fusion weight `w_flip`, flip-consistency signals, symmetrized `cls`, and `inv_spline` yields Canonical OOF **`0.086697 / 0.366778 / 0.932848`** (`4-seed mean = 0.087196 +/- 0.000451`, `5024` raw / `136.8` CV DoF, `150.9` full-795 DoF).

---

## 3. Main Results: 5-Fold Train OOF (`N=795`) vs Held-Out Test (`N=654`)

### 3.1 Complete 11-Method Locked Evaluation Ladder ([`oof_ablation_summary.csv`](../results/part3_geocalib/oof_ablation_summary.csv) & [`test_summary.csv`](../results/part3_geocalib/test_summary.csv))

| Method | Track (`Passes`) | Params (`Raw / Eff.DoF`) | OOF `AbsRel` | OOF `RMSE` (m) | OOF `delta1` | Test `AbsRel` $\downarrow$ | Test `RMSE` (m) $\downarrow$ | Test `delta1` $\uparrow$ | Test Gain vs `monotone_spline` | Test Gain vs `geo_focal_3d` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `raw` | 1-Pass (`1x`) | `0 / 0.0` | 0.117945 | 0.430373 | 0.884173 | 0.114540 | 0.398839 | 0.892293 | `-0.01371 / -0.01186 / -1.10 pp` | `-0.02800 / -0.04959 / -4.27 pp` |
| `global_scale` | 1-Pass (`1x`) | `1 / 1.0` | 0.105587 | 0.416395 | 0.897421 | 0.102033 | 0.389248 | 0.899513 | `-0.00121 / -0.00227 / -0.37 pp` | `-0.01550 / -0.04000 / -3.55 pp` |
| `monotone_spline` | 1-Pass (`1x`) | `6 / 6.0` | 0.103380 | 0.404847 | 0.901091 | 0.100827 | 0.386977 | 0.903253 | `0.00000 / 0.00000 / 0.00 pp` | `-0.01429 / -0.03773 / -3.18 pp` |
| `geo_focal_3d` | 1-Pass (`1x`) | `25 / 25.0` | 0.092502 | 0.379230 | 0.926687 | 0.086535 | 0.349250 | 0.935017 | `+0.01429 / +0.03773 / +3.18 pp` | `0.00000 / 0.00000 / 0.00 pp` |
| `geo_qknots_robust` | 1-Pass (`1x`) | `27 / 27.0` | 0.092372 | 0.379342 | 0.927501 | 0.086202 | 0.347125 | 0.935889 | `+0.01462 / +0.03985 / +3.26 pp` | `+0.00033 / +0.00213 / +0.09 pp` |
| `geo_inv_residual` | 1-Pass (`1x`) | `26 / 26.0` | 0.092567 | 0.378949 | 0.926980 | 0.086451 | 0.348720 | 0.935412 | `+0.01438 / +0.03826 / +3.22 pp` | `+0.00008 / +0.00053 / +0.04 pp` |
| `geosem_scale` | 1-Pass (`1x`) | `1688 / 134.3` | 0.087323 | 0.369819 | 0.931481 | **0.082977** | 0.341153 | **0.938765** | `+0.01785 / +0.04582 / +3.55 pp` | `+0.00356 / +0.00810 / +0.37 pp` |
| **`geocalib_3d` (Primary)** | **1-Pass (`1x`)** | `5018 / 146.9` | **0.086894** | **0.366619** | **0.932284** | **0.083024** | **0.340792** | **0.938413** | **`+0.01780 / +0.04619 / +3.52 pp`** | **`+0.00351 / +0.00846 / +0.34 pp`** |
| `flip_geo_only` | 2-Pass (`2x`) | `25 / 25.0` | 0.091986 | 0.377239 | 0.926907 | 0.086813 | 0.348347 | 0.935119 | `+0.01401 / +0.03863 / +3.19 pp` | `-0.00028 / +0.00090 / +0.01 pp` |
| `flip_geo_gated` | 2-Pass (`2x`) | `41 / 41.0` | 0.092294 | 0.378924 | 0.927448 | 0.087495 | 0.351140 | 0.935521 | `+0.01333 / +0.03584 / +3.23 pp` | `-0.00096 / -0.00189 / +0.05 pp` |
| **`geocalib_3d_flip` (Track B)** | **2-Pass (`2x`)** | `5024 / 150.9` | **0.086697** | **0.366778** | **0.932848** | **0.083189** | **0.339846** | **0.938350** | **`+0.01764 / +0.04713 / +3.51 pp`** | **`+0.00335 / +0.00940 / +0.33 pp`** |

*Note*: `test_summary.csv` reports full-795 frozen `effective_dof` (`146.9` for `geocalib_3d`, `150.9` for `geocalib_3d_flip`), whereas `oof_ablation_summary.csv` reports the 5-fold CV mean effective DoF (`137.9` and `136.8`).

### 3.2 Held-Out Scene-Cluster Bootstrap (`10,000` Replicates, `215` Scenes; [`test_paired_bootstrap.csv`](../results/part3_geocalib/test_paired_bootstrap.csv) & [`test_change_counts.csv`](../results/part3_geocalib/test_change_counts.csv))

- **`geocalib_3d` vs `monotone_spline`**:
  - `AbsRel` gain: **`+0.017803`**, 95% CI **`[+0.013784, +0.022111]`** (`p = 0.0`)
  - `RMSE` gain: **`+0.046185 m`**, 95% CI **`[+0.034878, +0.058758 m]`** (`p = 0.0`)
  - `delta1` gain: **`+0.035160`** (`+3.52 pp`), 95% CI **`[+0.023056, +0.049139]`** (`p = 0.0`)
  - Image win/loss (`N=654`): **`300` all-better vs `150` all-worse** (`204` mixed); per-metric improvements on `411 / 434 / 380` images.
- **`geocalib_3d` vs `geo_focal_3d`**:
  - `AbsRel` gain: **`+0.003512`**, 95% CI **`[+0.001204, +0.005810]`** (`p = 0.0008`)
  - `RMSE` gain: **`+0.008458 m`**, 95% CI **`[+0.003151, +0.013915 m]`** (`p = 0.0011`)
  - `delta1` gain: **`+0.003396`** (`+0.34 pp`), 95% CI `[-0.000169, +0.007285]` (`p = 0.0299`; meanwhile `geosem_scale` achieves `+0.003748` with CI `[+0.000340, +0.007501]`, `p = 0.0145`).
  - Image win/loss (`N=654`): **`226` all-better vs `201` all-worse** (`227` mixed); per-metric improvements on `348 / 337 / 324` images.
- **Negative Finding — `geocalib_3d_flip` (`2x`) vs `geocalib_3d` (`1x`)**:
  - `AbsRel` diff: `-0.000166`, CI `[-0.000737, +0.000423]` (`p = 0.7086`); `RMSE` diff: `+0.000946 m`, CI `[-0.000892, +0.002810 m]` (`p = 0.1569`); `delta1` diff: `-0.000063`, CI `[-0.001671, +0.001662]` (`p = 0.5210`).
  - **Takeaway**: Once single-pass encoder semantics and semantic-conditioned 2D spatial modulation are active, running a second horizontal-flip ViT-S forward pass provides no statistically reliable test benefit while doubling inference latency.

### 3.3 Held-Out Test Depth-Bin Comparison ([`test_depth_bins.csv`](../results/part3_geocalib/test_depth_bins.csv))

| Depth Bin (`Images / Pixels`) | `raw` (`AR / d1`) | `monotone_spline` (`AR / d1`) | `geo_focal_3d` (`AR / d1`) | **`geocalib_3d (1x)` (`AR / d1`)** | `geocalib_3d_flip (2x)` (`AR / d1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `[0.001, 1.0) m` (`254 / 4.74M`) | `0.3273 / 0.5500` | `0.2592 / 0.6509` | `0.2240 / 0.7230` | **`0.2208 / 0.7304`** | `0.2233 / 0.7261` |
| `[1.0, 2.0) m` (`641 / 45.09M`) | `0.1566 / 0.8268` | `0.1290 / 0.8600` | `0.1099 / 0.9108` | **`0.1052 / 0.9149`** | **`0.1050 / 0.9143`** |
| `[2.0, 4.0) m` (`645 / 84.15M`) | `0.1027 / 0.9150` | `0.0984 / 0.9018` | `0.0855 / 0.9302` | **`0.0822 / 0.9346`** | `0.0824 / 0.9334` |
| `[4.0, 6.0) m` (`415 / 17.24M`) | `0.1311 / 0.8196` | `0.1342 / 0.8099` | `0.1189 / 0.8397` | **`0.1175 / 0.8432`** | **`0.1169 / 0.8466`** |
| `[6.0, 10.0) m` (`220 / 4.80M`) | `0.1892 / 0.6637` | `0.2013 / 0.6121` | `0.1786 / 0.6962` | **`0.1766 / 0.6855`** | **`0.1759 / 0.6941`** |

---

## 4. Mechanism Audit: Left-Tail Cases, Spatial Bias & Latency

### 4.1 Explicit Left-Tail Case Audit ([`test_left_tail_case_audit.csv`](../results/part3_geocalib/test_left_tail_case_audit.csv))

| Test Index (`Scene`) | Left-Tail Regime | `f_tilde` | `geocalib_3d` Shift (`focal / geo / sem / total`) | `geo_focal_3d` (`AR / d1`) | **`geocalib_3d (1x)` (`AR / d1`)** | `geocalib_3d_flip (2x)` (`AR / d1`) | Net `geocalib_3d - geo_focal_3d` |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`#839` (`kitchen_0039`)** | False Focal Alarm | `0.350` | `+0.234 / +0.021 / -0.098 / +0.153` | `0.2683 / 0.5125` | **`0.1699 / 0.9034`** | `0.1762 / 0.9035` | **`-0.0983` AR, `+39.09 pp` d1** |
| **`#832` (`kitchen_0038`)** | False Focal Alarm | `0.350` | `+0.234 / +0.015 / -0.091 / +0.152` | `0.2460 / 0.5404` | **`0.1906 / 0.6513`** | `0.1988 / 0.6329` | **`-0.0554` AR, `+11.10 pp` d1** |
| **`#326` (`classroom_0023`)** | False Focal Alarm | `0.364` | `+0.230 / -0.074 / -0.058 / +0.097` | `0.2168 / 0.7455` | **`0.1481 / 0.8060`** | `0.1603 / 0.7921` | **`-0.0687` AR, `+6.05 pp` d1** |
| **`#1150` (`bedroom_0110`)** | Wide-FoV Collapse | `0.676` | `+0.145 / +0.018 / +0.121 / +0.257` | `0.1369 / 0.8265` | **`0.1249 / 0.9168`** | `0.1310 / 0.8214` | **`-0.0121` AR, `+9.03 pp` d1** |
| **`#1151` (`bedroom_0110`)** | Wide-FoV Collapse | `0.486` | `+0.197 / +0.001 / +0.097 / +0.265` | `0.0929 / 0.9732` | **`0.0576 / 0.9957`** | `0.0811 / 0.9945` | **`-0.0353` AR, `+2.25 pp` d1** |
| **`#1097` (`bedroom_0091`)** | Wide-FoV Collapse | `0.705` | `+0.137 / -0.011 / +0.091 / +0.205` | **`0.0398 / 0.9994`** | `0.0513 / 0.9993` | `0.0500 / 0.9990` | `+0.0115` AR, `-0.01 pp` d1 |

- Without hard-coded heuristics, the single-pass semantic branch outputs **negative priors (`-0.098 / -0.091`)** on close kitchen counters (`#839 / #832`) while simultaneously outputting **positive priors (`+0.121 / +0.097`)** on genuinely compressed wide-FoV bedrooms (`#1150 / #1151`). Across all `f_tilde < 0.55` held-out frames (`n=34`), AbsRel improves from `0.102483` (`geo_focal_3d`) to **`0.091200`** (`geocalib_3d`).
- **Focal & Spatial Residual Alignment ([`test_mechanism_diagnostics.csv`](../results/part3_geocalib/test_mechanism_diagnostics.csv), [`test_spatial_residuals.json`](../results/part3_geocalib/test_spatial_residuals.json))**:
  - Focal correlation `r(f_tilde, median_log_err)` drops from `+0.5639` (`raw`) and `-0.0854` (`geo_focal_3d`) to **`+0.0094`** (`geocalib_3d`), with median-log-error `SD` tightening from `0.09655 -> 0.07384 -> 0.06770` (`MAE 0.08201 -> 0.05577 -> 0.05178`).
  - `12 x 16` crop-grid RMS spatial log bias falls monotonically: `0.04186` (`raw`, max `0.07936`) $\rightarrow$ `0.01730` (`monotone_spline`) $\rightarrow$ `0.01222` (`geo_focal_3d`) $\rightarrow$ **`0.01101`** (`geocalib_3d`, max `0.03195`) $\rightarrow$ `0.01068` (`geocalib_3d_flip`).
- **Notable Regressions ([`test_representative_cases.csv`](../results/part3_geocalib/test_representative_cases.csv))**:
  - Top `geocalib_3d` gains vs `geo_focal_3d`: `#329 classroom_0024` (`0.2084 / 0.580 -> 0.1150 / 0.941`), `#839 kitchen_0039` (`0.2683 -> 0.1699`), `#328 classroom_0024` (`0.2082 -> 0.1130`).
  - Worst regressions vs `geo_focal_3d`: `#167 living_room_0013` (`0.0229 -> 0.1146`, `delta1 0.994`), `#1234 living_room_0036` (`0.0370 -> 0.1035`, `delta1 0.991`), `#1148 bedroom_0109` (`0.0713 -> 0.1440`, `delta1 0.925 -> 0.845`) — near-wall high-focal scenes (`f_tilde > 1.20`) where `geo_focal_3d` was already near-oracle and semantics slightly over-shifted scale.

### 4.2 Uncontended Latency & Numerical Parity ([`runtime_overhead.json`](../results/part3_geocalib/runtime_overhead.json), `N=200` Test Images, RTX 4060 Laptop GPU)

| Stage / Variant | Backbone Passes | Backbone (`ms/img`) | CPU Torch Post (`ms/img`) | GPU CUDA Post (`ms/img`) | End-to-End Total (`ms/img`) | Torch vs NumPy `max |Diff|` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw` (UniDepthV2 ViT-S) | `1x` | `43.04` | `—` | `—` | `43.04` (GPU) | `—` |
| `monotone_spline` | `1x` | `43.04` | `21.98` | `—` | `65.02` | `—` |
| `geo_focal_3d` | `1x` | `43.04` | `4.71` | `—` | `47.75` (CPU-post) | `—` |
| **`geocalib_3d` (Primary)** | **`1x`** | **`43.04`** | **`5.77`** (NumPy `37.35`) | **`3.89`** | **`46.93` (GPU) / `48.81` (CPU)** | **`7.11e-15 m`** (CPU & GPU) |
| `geocalib_3d_flip` (Track B) | `2x` | `87.17` | `15.29` | `5.61` | `92.78` (GPU) / `102.46` (CPU) | **`7.11e-15 m`** (CPU & GPU) |

---

## 5. Publication Figures & Verification

- [fig1_architecture_and_oracles.png](../results/figures/part3_geocalib/fig1_architecture_and_oracles.png) — Dual-track architecture, training-only residual oracle ceiling waterfall, and zero-leakage 5-fold CV $R^2$ signal discovery.
- [fig2_ablation_and_ci.png](../results/figures/part3_geocalib/fig2_ablation_and_ci.png) — Self-contained OOF vs held-out test progression and 10,000-replicate scene-bootstrap 95% CIs vs `geo_focal_3d`.
- [fig3_mechanism_and_spatial_bias.png](../results/figures/part3_geocalib/fig3_mechanism_and_spatial_bias.png) — Left-tail focal bimodality resolution (`#839/#832/#326` vs `#1150/#1151`) and `12 x 16` spatial log-bias heatmaps.
- [fig4_depth_bins_learning_curve_pareto.png](../results/figures/part3_geocalib/fig4_depth_bins_learning_curve_pareto.png) — 5-depth-bin test bars, training-only sample-efficiency curve (`10% -> 100%`), and accuracy–latency Pareto frontier.
- [fig5_qualitative_and_failure_cases.png](../results/figures/part3_geocalib/fig5_qualitative_and_failure_cases.png) — Qualitative comparison across left-tail false alarms (`#839/#832/#326`), wide-FoV collapse (`#1151`), and worst regression (`#167`).

```powershell
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/manifest.json
D:\anaconda\envs\vb\python.exe -m pytest -q --ignore=tests/test_video_recovery.py -k "not shell_defaults and not writer_crash_resume"
```
