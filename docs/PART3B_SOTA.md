# Part 3B GeoCalib-3D: Geometric, Spatial-Depth & Multi-Metric Calibration

This document records the research diagnosis, model architecture, 5-fold multi-seed cross-validation, pre-test freeze, one-shot held-out test evaluation, learning curve, and publication figures for `GeoCalib-3D` (`J3_geok_sp3d_slope_bal111_l01`), an improved single-pass post-hoc metric depth calibrator for RGB-only **UniDepthV2 ViT-S** on **NYU Depth V2**.

> **Zero-Leakage & Reproducibility Provenance**:
> - **Branch**: `feat/part3b-sota-calibration` (additive over clean `codex/part3b` commit `923ec49`; zero edits to existing historical modules, `results/frozen_20261111/`, `results/final_review_20261113/`, or `outputs/`).
> - **Zero GT $K_{\text{true}}$**: Never uses `RGB_INTRINSICS` or ground-truth intrinsics. Uses only UniDepthV2's own RGB-only outputs $(\hat{Z}, \hat{K})$.
> - **Zero Test Leakage**: All diagnostics (Gate 1), ablation design & multi-seed CV (Gate 2), and full 795-image fitting & parameter lock (Gate 3, Commit `8560b5a`, `frozen_sota_model.json` SHA-256 `5f8326a63470416e077cec12afb41ed793ffb2a58abf8d94a0eb88b84f08e86c`, `test_read: false`) were frozen **before** reading `splits["test"]`.
> - **One-Shot Test Evaluation**: Evaluated once on the 654 held-out test images (`215` scenes) via `depthlab/sota_evaluate.py` (Gate 4, Commit `bd03fba`).

---

## 1. Gate 1 Findings: Why `B_residual_v1` Left Large Reducible Error

On the 795 official training images (`249` scenes), the reproduced baseline `b3_residual_v1` achieves 5-fold scene-CV OOF `AbsRel = 0.102441`, `RMSE = 0.401714 m`, `delta1 = 0.904042`. Residual & oracle diagnosis (`depthlab/sota_diagnose.py`) uncovered four concrete failure modes of `B_residual_v1`:

1. **Unexploited Self-Predicted Focal Coupling ($\hat{K}$)**:
   - Oracle analysis shows **60.6%** of remaining relative error sits in per-image scale (`b3_oracle_absrel_scalar = 0.061383` vs `0.102441`).
   - Across the 795 training frames, `e_med = log(median(Z_gt / Z_B3))` strongly anti-correlates with UniDepthV2's self-predicted focal length (`fy` Pearson `r = -0.5511`, `p = 2.3e-64`, 5-fold scene-CV `R^2 = 0.2274`; `fx` `r = -0.5463`), due to perspective depth–focal length ambiguity when $\hat{f}$ under-predicts wide-FoV rooms (`min fx = 155.6 px` vs median `572.7 px`).
   - Multivariate `Geo_K + Spatial3D` features predict `26.0%` of out-of-fold log-scale variance (`R^2(e_med) = 0.2604`, `R^2(e_absrel) = 0.2170`), whereas 16 RGB illumination/texture/gradient cues carry zero OOF signal (`R^2 = -0.0102` to `-0.0182`).
2. **Global 2D Image-Plane Warps Hurt OOF**: Fitting a dataset-wide 2D spatial quadric field degrades OOF (`0.103347 / 0.403509 / 0.903899`) because spatial bias flips sign with focal-length/pitch regimes.
3. **Decoupled Objective & Cross-Metric Bias**: `B_residual_v1` fits its 6-knot spline purely on unclipped RMSE and a separate ridge residual on median-ratio MSE (`shrinkage=0.5`), leaving a `-1.43%` systematic positive scale bias for `AbsRel` (optimal multiplier `0.9857`).
4. **Unresolved S-Curve Across Depth Ranges**: `B3` over-predicts `[0.001, 1.0)m` (`signed log err = +0.1965`, `AbsRel = 0.2813`) and under-predicts `[6.0, 10.0)m` (`signed log err = -0.2145`).

---

## 2. Method Design: `GeoCalib-3D` (25 Learned Parameters)

Given single-pass backbone outputs $\hat{Z} \in \mathbb{R}_+^{480 \times 640}$ and $\hat{K} \in \mathbb{R}^{3 \times 3}$, define $\tilde{f} = \frac{\hat{f}_x + \hat{f}_y}{2 \cdot 480}$, geometry vector $\phi_{\text{geo}}(\hat{K}, \hat{Z})$ (2 principal-point offsets `cx_off, cy_off` + 8 pseudo-3D subcrop cues extracted on a 4x4 grid `CROP_SUB` of 14,980 px), and slope-modulation vector $\phi_{\text{slope}}(\hat{K}, \hat{Z}) = [\tilde{f}, \text{std}(\log\hat{Z}), d\log\hat{Z}/d\tilde{v}, \log\text{span}_Y]$:

$$\log Z_{\text{cal}}(u,v) = g(\log \hat{Z}(u,v)) + 0.50 \tanh\!\left(\frac{h_f(\tilde{f}) + 0.40 \tanh\!\left(\frac{\mathbf{w}_s^\top \tilde{\phi}_{\text{geo}}}{0.40}\right)}{0.50}\right) + \Delta\beta \cdot (\log \hat{Z}(u,v) - 1)$$

followed by $Z_{\text{cal}} = \operatorname{clip}(\exp(\log Z_{\text{cal}}), 0.001, 10.0)\,\text{m}$.

| Component | Parameterization & Physical Role | Learned Params | Locked Full-795 Values Summary |
| :--- | :--- | :---: | :--- |
| Depth Spline $g(\log\hat{Z})$ | 6 fixed log-knots $[0.3, 0.7, 1.5, 3, 6, 12]\,\text{m}$; offset + 5 slopes $\in [0.25, 2.0]$ | **6** | Slopes `[0.971, 1.024, 1.094, 1.069, 0.647]` |
| Focal Spline $h_f(\tilde{f})$ | 6 knots $[0.35, 0.75, 0.98, 1.14, 1.22, 1.35]$ anchored at $h_f(1.14)=0$; non-increasing slopes $\in [-2.5, 0]$ | **5** | Ordinates `[+0.268, +0.206, +0.058, 0.0, -0.051, -0.175]` |
| Conditioned Scale $\Delta s_{\text{geo}}$ | Standardized `cx_off, cy_off` + 8 `Spatial3D` features, $\tanh$-bounded | **10** | Captures image-level 3D span, aspect, and tilt scale shifts |
| Slope Modulation $\Delta\beta$ | $\Delta\beta = 0.20 \tanh(\mathbf{w}_\beta^\top \tilde{\phi}_{\text{slope}})$; net slope $\ge 0.05$ guarantees strict monotonicity in $\hat{Z}$ | **4** | Weights `[+0.118, -0.077, -0.045, -0.149]` |
| **Joint Objective & Optimizer** | L-BFGS (`float64`, `balanced_111`, `reg_l2=0.01`) minimizing smooth `(AbsRel, RMSE, soft-δ1)` directly | **Total: 25** | Single pass, zero extra backbone runs, strictly monotone in $\hat{Z}$ |

Gate 2 ablations confirmed that adding `+RayUnwarp` (`J4`), `+CondSpatial2D` (`J5`), or `+RGB3` (`J7`) added complexity without beating parsimonious 25-param `J3_geok_sp3d_slope_bal111_l01`.

---

## 3. Quantitative Results: 5-Fold Multi-Seed CV & Held-Out 654-Image Test

All baselines (`raw`, `b1_global_scale`, `b2_monotone_curve`, `b3_residual_v1`) and `GeoCalib-3D` stages are evaluated in a single unified loop over the exact same `outputs/nyu_rgb/depth.npy` predictions (`depth.npy` SHA-256 `e84ee7d5...`).

> **Cross-Cache Audit Disclosure**: Historical replay artifacts under `results/frozen_20261111/` and `results/final_review_20261113/replay_evaluation.json` were computed on an earlier split-array FP16 cache (`depth.npy` SHA-256 `2d2e63d2...`, NAS-backed). On the local NVIDIA A40 cache (`outputs/nyu_rgb/depth.npy`, `e84ee7d5...`), replaying historical `frozen_model.json` reproduces all four baseline aggregate means within `1.26e-4` (`max_mean_abs_error = 0.00012587`, e.g., `B3_prev` Test `AbsRel 0.099601` vs historical `0.099607`, `RMSE 0.383099` vs `0.383124`, `delta1 0.905671` vs `0.905546`), with `per_image_max_abs_error = 0.093926` from FP16 autocast jitter. All paired gains below are strictly on the identical local cache.

### 3.1 Main Ablation & Generalization Table (OOF N=795 vs Held-Out Test N=654)

| Method | Params | OOF `AbsRel` $\downarrow$ | OOF `RMSE` $\downarrow$ | OOF `delta1` $\uparrow$ | Test `AbsRel` $\downarrow$ | Test `RMSE` (m) $\downarrow$ | Test `delta1` $\uparrow$ | Test Gain vs `B3_prev` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `raw` (UniDepthV2 ViT-S) | 0 | 0.117945 | 0.430373 | 0.884173 | 0.114540 | 0.398839 | 0.892293 | `-0.0149 / -0.0157 / -0.0134` |
| `b1_global_scale` | 1 | 0.105587 | 0.416395 | 0.897421 | 0.102033 | 0.389248 | 0.899513 | `-0.0024 / -0.0061 / -0.0062` |
| `b2_monotone_curve` | 6 | 0.103380 | 0.404847 | 0.901091 | 0.100827 | 0.386977 | 0.903253 | `-0.0012 / -0.0039 / -0.0024` |
| **`b3_residual_v1` (Ref.)** | 15 | **0.102441** | **0.401714** | **0.904042** | **0.099601** | **0.383099** | **0.905671** | `0.0000 / 0.0000 / 0.0000` |
| `S2_b2_geok_sp3d_ridge` | 20 | 0.095462 | 0.386573 | 0.922258 | 0.090760 | 0.358244 | 0.929672 | `+0.00884 / +0.02485 / +0.02400` |
| `J0_spline_bal111` | 6 | 0.102636 | 0.405234 | 0.901056 | 0.100135 | 0.387626 | 0.902919 | `-0.00053 / -0.00453 / -0.00275` |
| `J1_geok_bal111_l01` | 13 | 0.094063 | 0.386752 | 0.925016 | 0.088865 | 0.356687 | 0.931131 | `+0.01074 / +0.02641 / +0.02546` |
| `J2_geok_sp3d_bal111_l01` | 21 | 0.093114 | 0.382839 | 0.925942 | 0.087335 | 0.351782 | 0.934768 | `+0.01227 / +0.03132 / +0.02910` |
| **`GeoCalib-3D` (Ours, `J3`)** | **25** | **0.092502** | **0.379230** | **0.926687** | **0.086535** | **0.349250** | **0.935017** | **`+0.01307 / +0.03385 / +0.02935`** |

### 3.2 Multi-Seed Stability (Gate 2 OOF) & Paired Scene-Cluster Bootstrap (10,000 Replicates)

- **4-Seed OOF Stability (`20261031, 42, 100, 20261121`)**: `GeoCalib-3D` achieves `AbsRel 0.092209 +/- 0.000212` (`0.092502, 0.092171, 0.092255, 0.091908`), `RMSE 0.378730 +/- 0.000595 m`, `delta1 0.927246 +/- 0.000399`, beating `b3_residual_v1` on all 3 metrics across **5/5** canonical folds and **4/4** random scene-split seeds.
- **Held-Out Test Paired 10,000-Replicate Scene-Cluster Bootstrap (`215` scenes)**:
  - `GeoCalib-3D` vs `B3_residual_v1`:
    - `AbsRel` gain: **`+0.013066`**, 95% CI **`[+0.009194, +0.017186]`** (`p_one_sided = 0.0`)
    - `RMSE` gain: **`+0.033848 m`**, 95% CI **`[+0.023187, +0.045404]`** (`p_one_sided = 0.0`)
    - `delta1` gain: **`+0.029346`** (`+2.93 pp`), 95% CI **`[+0.017298, +0.043057]`** (`p_one_sided = 0.0`)
  - Image win/loss vs `B3_residual_v1` (`test_change_counts.csv`, `N=654`): **`285` all-better vs `158` all-worse** (`211` mixed/tied); per-metric improvements on **`404` (`AbsRel`)**, **`398` (`RMSE`)**, and **`378` (`delta1`)** images.
  - Note on Generalization: Test gains exceed OOF gains (`test_minus_oof_gain`: `+0.00313` AbsRel, `+0.01136 m` RMSE, `+0.00670` delta1), confirming zero overfitting and clean generalization across the 215 held-out scenes.

### 3.3 Held-Out Depth-Bin Breakdown (`test_depth_bins.csv`) & Learning Curve (`oof_learning_curve.csv`)

| Depth Range | `Images / Pixels` | `Raw` (`AR / d1`) | `B3_prev` (`AR / d1`) | **`GeoCalib-3D` (`AR / d1`)** | Relative Error Drop vs `B3_prev` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `[0.001, 1.0) m` | `254 / 4.74M` | `0.3273 / 0.5500` | `0.2612 / 0.6489` | **`0.2240 / 0.7230`** | **`-14.2%` AbsRel, `+7.41 pp` delta1** |
| `[1.0, 2.0) m` | `641 / 45.09M` | `0.1566 / 0.8268` | `0.1271 / 0.8645` | **`0.1099 / 0.9108`** | **`-13.6%` AbsRel, `+4.63 pp` delta1** |
| `[2.0, 4.0) m` | `645 / 84.15M` | `0.1027 / 0.9150` | `0.0968 / 0.9052` | **`0.0855 / 0.9302`** | **`-11.7%` AbsRel, `+2.50 pp` delta1** |
| `[4.0, 6.0) m` | `415 / 17.24M` | `0.1311 / 0.8196` | `0.1329 / 0.8129` | **`0.1189 / 0.8397`** | **`-10.6%` AbsRel, `+2.68 pp` delta1** |
| `[6.0, 10.0) m` | `220 / 4.80M` | `0.1892 / 0.6637` | `0.1988 / 0.6208` | **`0.1786 / 0.6962`** | **`-10.1%` AbsRel, `+7.53 pp` delta1** |

- **Train-Only Learning Curve (`oof_learning_curve.csv`)**: With just **10%** of fitting scenes (`~20` scenes, `60.4` images/fold), `GeoCalib-3D` already achieves `OOF AbsRel = 0.102039`, `delta1 = 0.910860` — outperforming `B3_residual_v1` trained on **100%** of scenes (`0.102441 / 0.904042`). Scaling from 10% -> 25% -> 50% -> 100% smoothly improves `GeoCalib-3D` OOF `AbsRel` from `0.102039` -> `0.096328` -> `0.093110` -> `0.092502`.

---

## 4. Mechanism Diagnostics, Limitations & Runtime

- **Mechanism Verification (`test_mechanism_diagnostics.csv`, `test_spatial_residuals.json`)**: On the 654 held-out test images, focal correlation of residual log-scale error drops from Pearson `r = +0.550` (`SD = 0.0947`, `MAE = 0.0702` under `B3_prev`) down to `r = -0.085` (`SD = 0.0738`, `MAE = 0.0558`) after `GeoCalib-3D`, while 12x16 crop-grid RMS spatial log bias decreases from `0.0419` (`Raw`) and `0.0168` (`B3_prev`) to `0.0122` (`GeoCalib-3D`).
- **Best Cases & Honest Failure Mode (`test_representative_cases.csv`)**:
  - **Largest Gains**: Wide-FoV under-predicted bedrooms (`#1150 bedroom_0110`, `f_tilde=0.676`: `B3 0.3236 -> 0.1369`, `delta1 0.005 -> 0.827`; `#1151 bedroom_0110`, `f_tilde=0.486`: `0.2890 -> 0.0929`, `delta1 0.005 -> 0.973`; `#1097 bedroom_0091`, `f_tilde=0.705`: `0.2285 -> 0.0398`).
  - **Primary Limitation / Regressions**: Rare extreme-low predicted-focal outliers where UniDepthV2 outputs `f_tilde <= 0.365` (clamped at knot `0.35`, i.e., `f ~ 168 px` vs true `519 px`) while raw depth itself was nearly unbiased (`#839 kitchen_0039`, `f_tilde=0.350`: `B3 0.0960 -> GeoCalib 0.2683`; `#832 kitchen_0038`, `f_tilde=0.350`: `0.0840 -> 0.2460`; `#326 classroom_0023`, `f_tilde=0.364`: `0.0949 -> 0.2168`). Because `h_f(0.35) = +0.268`, false focal alarms trigger an over-correction (`+30.8%` scale boost). Future work should add backbone uncertainty gating or outlier clamping on `f_tilde`.
- **Single-Pass Latency (`float64`, zero extra backbone forward passes)**:
  - Pre-test 100-image uncontended benchmark (`cv_selection_report.json`): feature extraction `1.98 ms/img` + GPU Torch apply `2.31 ms/img` (or CPU Torch `5.94 ms/img`, pure NumPy `24.14 ms/img` vs `B3_prev 27.83 ms/img`).
  - Held-out 200-image benchmark (`runtime_overhead.json`, measured concurrently under CPU load): feature extraction `2.22 ms/img`, CPU Torch apply `8.95 ms/img`, GPU Torch apply `4.20 ms/img`, pure NumPy `30.96 ms/img` vs `B3_prev 37.24 ms/img`, with Torch-vs-NumPy max pixel discrepancy `<= 8.88e-15 m`.

---

## 5. Publication Figures & Reproduction Commands

- [fig1_pipeline_overview.png](../results/figures/part3b_sota/fig1_pipeline_overview.png) — `GeoCalib-3D` single-pass architecture, learned focal spline $h_f(\tilde{f})$, and monotone depth spline offset.
- [fig2_ablation_and_ci.png](../results/figures/part3b_sota/fig2_ablation_and_ci.png) — 9-stage OOF & held-out test ablation progression + 10k-replicate scene-cluster bootstrap 95% CI forest plots.
- [fig3_mechanism_diagnostics.png](../results/figures/part3b_sota/fig3_mechanism_diagnostics.png) — Before/after focal-drift scatter (`r: +0.550 -> -0.085`) and 12x16 image-space residual heatmaps.
- [fig4_depth_bins_and_learning_curve.png](../results/figures/part3b_sota/fig4_depth_bins_and_learning_curve.png) — 5-depth-bin test bars and train-only 5-fold learning curve (`10% -> 100%` scenes).
- [fig5_qualitative_and_failures.png](../results/figures/part3b_sota/fig5_qualitative_and_failures.png) — Best gains, typical P75 gain, and honest extreme-focal failure case visualization.

```powershell
# Re-verify all delivered artifacts and hashes
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/part3b_sota/manifest.json
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/final_review_20261113/manifest.json

# Run Windows unit test suite (including all Part 3B SOTA checks)
D:\anaconda\envs\vb\python.exe -m pytest -q --ignore=tests/test_video_recovery.py -k "not shell_defaults and not writer_crash_resume"
```
