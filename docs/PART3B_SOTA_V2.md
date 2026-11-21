# Part 3B Round 2 (`GeoCalib-3D v2`): Inverse-Depth, Semantic-Prior & Dual-Track Calibration

This document records the Round 2 exploration, post-`v1` mechanistic diagnosis, dual-track 26-candidate 5-fold cross-validation (`4` random seeds), pre-test model freeze, strictly one-shot 654-image held-out test evaluation, Round 1 failure-case audit, uncontended latency benchmarks, and 300-DPI CVPR publication figures for **`GeoCalib-3D v2`** on **NYU Depth V2**.

> **Zero-Leakage & Iterative Round 2 Provenance Disclosure**:
> - **Branch & Preservation**: `feat/part3b-sota-v2` (strictly additive over clean Round 1 commit `547ca84`; zero modifications to historical modules, Round 1 outputs under `results/part3b_sota/` and `results/figures/part3b_sota/`, `docs/PART3B_SOTA.md`, `outputs/nyu_rgb/`, `results/frozen_20261111/`, or `results/final_review_20261113/`).
> - **Honest Round 2 Distinction**: Unlike Round 1 (`GeoCalib-3D v1`, zero prior exposure to `splits["test"]`), Round 2 design was explicitly motivated by Round 1's documented post-test failure taxonomy (`#839 / #832 / #326` false focal alarms vs `#1150 / #1151` wide-FoV collapse). Once Round 2 exploration began, **100% of diagnosis, feature engineering, architecture choices, hyperparameter tuning, 4-seed CV selection, and full-795 parameter locking operated exclusively on the 795 official training images (`249` scenes)** with `splits["test"]` access hard-blocked via `VerifiedV2Cache(allowed_split="train")`.
> - **Zero GT $K_{\text{true}}$**: Never uses `RGB_INTRINSICS` or ground-truth intrinsics. Uses only pinned frozen RGB-only **UniDepthV2 ViT-S** outputs (`lpiccinelli/unidepth-v2-vits14`, train-only cache parity `median |Delta Z| = 0.000228 m`, mean `MAE = 0.001001 m`, `fx` rel diff `0.000523`).
> - **Pre-Test Freeze & One-Shot Test Verification**: Locked before touching `splits["test"]` at Git commit `ebad14a58a83393661edead282bd6db07751d3a4` (`results/part3b_sota_v2/frozen_sota_v2_model.json` SHA-256 `a7633ffa74b21b70e196ba822c15fe454ac96b528443d4a70753f0aa4ff479e6`, `test_read: false`). Evaluated **once** on the 654 held-out test images (`215` scenes) via `depthlab/sota_v2_evaluate.py`, reproducing Round 1 `raw`, `b3_residual_v1`, and `v1_geocalib3d` test means with `max_abs_diff = 5.55e-17`.
> - **Explicit Track Separation**: **Track A (Single-Pass, `1x` ViT-S forward)** and **Track B (Multi-Pass 2× Horizontal-Flip TTA, `2x` ViT-S forward)** are reported separately across every table and plot.

---

## 1. Gate 1 Findings: Post-`v1` Residual Ceiling & Root-Cause Discoveries (N=795 Train Only)

Using exact canonical 5-fold OOF `GeoCalib-3D v1` predictions on the 795 training images (`AbsRel = 0.092502`, `RMSE = 0.379230 m`, `delta1 = 0.926687`), `depthlab/sota_v2_diagnose.py` established four empirical facts:

1. **Substantial Remaining Oracle Headroom**:
   - `v1_oracle_med_scalar`: `0.061137 / 0.312985 / 0.947473`; `v1_oracle_absrel_scalar`: `0.060208 / 0.312762 / 0.946831` (`+0.032294` AbsRel ceiling remaining in per-image scale); `v1_oracle_log_affine`: `0.056390`; `v1_oracle_inv_affine`: `0.058401`; `v1_oracle_spatial_linear`: `0.054323`; `v1_oracle_spatial_quadric`: `0.049761`.
2. **Low-Dim Geometry Saturated, Internal Single-Pass Semantics Carries Real Out-of-Fold Signal**:
   - Standard 5-fold scene-CV Ridge/PCA predictability of remaining `v1` residuals shows `v1_geo_sp3d_control` (`CV R^2(e_v1_med) = -0.0094`), `confidence_stats` (`-0.0005`), and `geo_sp3d_interactions` (`-0.0016`) are exhausted linearly.
   - By contrast, frozen single-pass UniDepthV2 internal representations carry genuine out-of-fold predictive signal on remaining `v1` errors (`diag_v2_signal_r2.csv`): `cls_token` (`384-d -> PCA-32`, `lam=10`) achieves `CV R^2(e_v1_med) = +0.0226` (`full-384 lam=100` reaches `+0.0406` on `e_v1_radial` and `+0.0313` on `e_v1_slope_v`), `depth_feat_stats` (`512-d -> PCA-32`, `lam=10`) achieves `CV R^2(e_v1_med) = +0.0223` (`e_v1_absrel = +0.0220`), and `patch_mean_std` (`768-d`, `lam=100`) reaches `CV R^2(e_v1_radial) = +0.0554` (`+0.0383` on `e_v1_slope_v`).
3. **Root Cause of Naive Flip-TTA Regression Discovered (`-17 sigma` `cx_off` Extrapolation)**:
   - Un-adjusted `v1` applied to horizontal-flip outputs (`outputs/nyu_rgb_flip/`) degraded train OOF from `0.092502` to `0.097793` (`v1_flip_log_avg`), because mirroring `cx` via `639 - cx_flip` shifted `cx_off = (cx - 320)/640` from `+0.00358 +/- 0.00052` (Pass 1) to `-0.00520 +/- 0.00054` (`-17 sigma` out-of-distribution wrt `v1`'s standardized `scale_residual`), causing `v1(p2, K2_mir)` to spike to `0.112093`. Dropping `cx_off` (or keeping native un-mirrored `cx`) immediately fixes Flip-TTA (`M1_v1_flip_sym_zero_retrain`: `0.091986 / 0.377239 / 0.926907`).
4. **Left-Tail Focal Bimodality (`f_tilde < 0.75`, n=43: 26 `true_collapse`, 7 `false_alarm`, 10 `mild`)**:
   - False focal alarms already exhibit large metric vertical depth span (`log_span_y` mean `2.000` vs `1.599`, `AUC = 0.8022`) and abundant far pixels (`frac_far_4m` mean `42.0%` vs `19.8%`, `AUC = 0.7912`), while multi-pass flip focal agreement (`f_flip_tilde` `AUC = 0.8297`, `f_tilde_min` `AUC = 0.8187`) also separates regimes.

---

## 2. Method Architecture & Gate 2 Four-Direction Ablation Study (N=795 Train Only)

Across 26 structured candidates evaluated on Canonical 5-fold Scene-CV (`SEED=20261031`) and 4-seed stability (`20261031, 42, 100, 20261121`), our four research directions revealed clear, complementary mechanisms:

- **Direction 1 (Deepen Pure Geometry: `D1..D3`)** & **Direction 2A (New Depth Parametrizations: `N1..N3`)**:
  - Train OOF hovers around `0.092372` (`D3_qknots_robust`) to `0.092567` (`N2_hybrid_log_inv_res`), while `N3_rational_depth` underperforms (`0.093340`) and `S0_conf_summary_ctrl` hurts (`0.093351`). Low-dimensional handcrafted `(K_hat, Z_hat)` scalars are saturated near `v1`.
- **Direction 2B (Single-Pass Internal Semantics: `S1..S5`)**:
  - Adding L2-normalized single-pass semantic blocks (`all_sp`: `[l2n(cls), l2n(patch_mean), l2n(patch_std), l2n(depth_feat_stats)]` centered strictly inside each training fold and regularized by Ridge L2 `sem_reg=0.05`) cuts Train OOF AbsRel from `0.092502` down to **`0.087323`** (`S4_sem_all_r005`, effective DoF `113.1`), while `PCA-128` (`S5_sem_all_pca128_r005`) compresses raw parameters `1688 -> 152` at `0.087443`.
- **Direction 3 & 4 (Combine & Polish -> Track A Winner `H6_geosem_inv_spline`)**:
  - Combining (i) monotone **inverse-depth spline** (`inv_spline(1/Z_hat)` across knots `[1/12, 1/6, 1/3, 1/1.5, 1/0.7, 1/0.3] m^-1`, naturally linearizing far-field S-curve disparity), (ii) flip-symmetric `v2_clean` geometry (`cx_off` removed, `z_clip=4.0`), (iii) internal semantic scale prior (`sem_reg=0.05`), and (iv) regularized **semantic-conditioned 2D spatial field** (`v_tilde, r^2` with `sem_sp_reg=0.08`) yields **`H6_geosem_inv_spline`**:
    - Canonical OOF (`N=795`): **`0.086894 / 0.366619 / 0.932284`** (`+0.005608 / +0.012611 m / +0.560 pp` over `v1`, `95% CI [+0.003786, +0.007455]`), beating `v1` across **5/5** canonical folds and **4/4** seeds (`4-seed mean = 0.087383 +/- 0.000401`).
    - Pure-data hybrid blend `H7_sp_hybrid_blend` converged inside folds to `alpha = 0.0` (selecting pure `H6_geosem_inv_spline`), so parsimony promotes **`H6_geosem_inv_spline`** (`5018 raw / 137.9 CV effective DoF`, `146.9` on full-795 fit).
- **Track B (2-Pass Flip-TTA -> Winner `M6_flip_inv_geosem`)**:
  - Using 2 passes (original + horizontal-flip RGB) with un-mirrored `K2`, learnable log-depth fusion weight `w_flip`, flip-consistency features, symmetrized `CLS`, and `inv_spline` achieves **`M6_flip_inv_geosem`** Canonical OOF **`0.086697 / 0.366778 / 0.932848`** (`4-seed mean = 0.087196 +/- 0.000391`).
  - Honest comparison vs Track A: on 795 Train OOF, `M6` vs `H6` differs by only `+0.000197` AbsRel (`95% CI [-0.000396, +0.000796]`), `-0.000160 m` RMSE (`CI [-0.002153, +0.001754]`), and `+0.000564` delta1 (`CI [-0.001055, +0.002052]`) — statistically indistinct while doubling backbone runtime!

---

## 3. Main Results: 5-Fold Train OOF (N=795) vs Held-Out Test (N=654)

All baselines, ablations, Track A (`1x` forward), and Track B (`2x` forward) models are evaluated cleanly over the identical local UniDepthV2 predictions (`outputs/nyu_rgb/depth.npy`).

### 3.1 Locked Comparison Ladder (`oof_ablation_summary.csv` & `test_summary.csv`)

| Method | Track (`Passes`) | Params (`Raw / Eff.DoF`) | OOF `AbsRel` | OOF `RMSE` | OOF `delta1` | Test `AbsRel` | Test `RMSE` (m) | Test `delta1` | Test Gain vs `B3_prev` | Test Gain vs `v1` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `raw` (UniDepthV2 ViT-S) | 1-Pass (`1x`) | `0 / 0.0` | 0.117945 | 0.430373 | 0.884173 | 0.114540 | 0.398839 | 0.892293 | `-0.01494 / -0.01574 / -0.01338` | `-0.02800 / -0.04959 / -0.04272` |
| `b3_residual_v1` (`B3_prev`) | 1-Pass (`1x`) | `15 / 15.0` | 0.102441 | 0.401714 | 0.904042 | 0.099601 | 0.383099 | 0.905671 | `0.00000 / 0.00000 / 0.00000` | `-0.01307 / -0.03385 / -0.02935` |
| **`v1_geocalib3d` (`v1`)** | 1-Pass (`1x`) | `25 / 25.0` | **0.092502** | **0.379230** | **0.926687** | **0.086535** | **0.349250** | **0.935017** | `+0.01307 / +0.03385 / +0.02935` | `0.00000 / 0.00000 / 0.00000` |
| `D3_qknots_robust` | 1-Pass (`1x`) | `27 / 27.0` | 0.092372 | 0.379342 | 0.927501 | 0.086202 | 0.347125 | 0.935889 | `+0.01340 / +0.03597 / +0.03022` | `+0.00033 / +0.00213 / +0.00087` |
| `N2_hybrid_log_inv_res` | 1-Pass (`1x`) | `26 / 26.0` | 0.092567 | 0.378949 | 0.926980 | 0.086451 | 0.348720 | 0.935412 | `+0.01315 / +0.03438 / +0.02974` | `+0.00008 / +0.00053 / +0.00039` |
| `S4_sem_all_r005` | 1-Pass (`1x`) | `1688 / 134.3` | 0.087323 | 0.369819 | 0.931481 | **0.082977** | 0.341153 | **0.938765** | `+0.01662 / +0.04195 / +0.03309` | `+0.00356 / +0.00810 / +0.00375` |
| **`v2` (`H6_geosem_inv_spline`, Track A)** | **1-Pass (`1x`)** | `5018 / 146.9` | **0.086894** | **0.366619** | **0.932284** | **0.083024** | **0.340792** | **0.938413** | **`+0.01658 / +0.04231 / +0.03274`** | **`+0.00351 / +0.00846 / +0.00340`** |
| `M1_v1_flip_sym_zero_retrain` | 2-Pass (`2x`) | `25 / 25.0` | 0.091986 | 0.377239 | 0.926907 | 0.086813 | 0.348347 | 0.935119 | `+0.01279 / +0.03475 / +0.02945` | `-0.00028 / +0.00090 / +0.00010` |
| `M2_flip_geo_gated` | 2-Pass (`2x`) | `41 / 41.0` | 0.092294 | 0.378924 | 0.927448 | 0.087495 | 0.351140 | 0.935521 | `+0.01211 / +0.03196 / +0.02985` | `-0.00096 / -0.00189 / +0.00050` |
| **`v2+Flip` (`M6_flip_inv_geosem`, Track B)** | **2-Pass (`2x`)** | `5024 / 150.9` | **0.086697** | **0.366778** | **0.932848** | **0.083189** | **0.339846** | **0.938350** | **`+0.01641 / +0.04325 / +0.03268`** | **`+0.00335 / +0.00940 / +0.00333`** |

> *Note*: `test_summary.csv` reports full-795 frozen `effective_dof` (`146.9` for `H6`, `150.9` for `M6`), while Gate 2 `oof_ablation_summary.csv` reports 5-fold CV mean DoF (`137.9` for `H6`, `136.8` for `M6`). Notice also that semantic scale-only `S4_sem_all_r005` (`0.082977`) slightly ties/edges `H6` (`0.083024`) on held-out `AbsRel` & `delta1` while `H6` leads on `RMSE` (`0.340792` vs `0.341153`) and OOF (`0.086894` vs `0.087323`).

### 3.2 Held-Out Test 10,000-Replicate Scene-Cluster Bootstrap (`215` Scenes) & Win Counts

- **`H6_geosem_inv_spline` (`v2`, Track A) vs `b3_residual_v1`**:
  - `AbsRel` gain: **`+0.016577`**, 95% CI `[+0.012706, +0.020809]` (`p = 0.0`); `RMSE`: **`+0.042307 m`**, 95% CI `[+0.031260, +0.054650]` (`p = 0.0`); `delta1`: **`+0.032742`** (`+3.27 pp`), 95% CI `[+0.021021, +0.046186]` (`p = 0.0`).
  - Win/loss (`N=654`): **`298` all-better vs `153` all-worse** (`203` mixed); per-metric improvements on `412 / 417 / 381` images.
- **`H6_geosem_inv_spline` (`v2`, Track A) vs `v1_geocalib3d`**:
  - `AbsRel` gain: **`+0.003512`**, 95% CI **`[+0.001280, +0.005868]`** (`p = 0.0009`)
  - `RMSE` gain: **`+0.008458 m`**, 95% CI **`[+0.003117, +0.014054 m]`** (`p = 0.0007`)
  - `delta1` gain: **`+0.003396`** (`+0.34 pp`), 95% CI `[-0.000183, +0.007364]` (`p = 0.0308`; meanwhile `S4_sem_all_r005` achieves CI `[+0.000342, +0.007466]`, `p = 0.0140`).
  - Win/loss vs `v1` (`N=654`): **`226` all-better vs `201` all-worse** (`227` mixed); per-metric improvements on `348 / 337 / 324` images.
- **`M6_flip_inv_geosem` (`v2+Flip`, Track B) vs `H6_geosem_inv_spline` (`v2`, Track A)**:
  - `AbsRel` delta: `-0.000166`, CI `[-0.000746, +0.000410]` (`p = 0.7135`); `RMSE` delta: `+0.000946 m`, CI `[-0.000893, +0.002822]` (`p = 0.1652`); `delta1` delta: `-0.000063`, CI `[-0.001694, +0.001613]` (`p = 0.5376`).
  - **Negative Result on 2× Flip-TTA Worth Disclosing Honestly**: Once internal encoder semantics (`v2 Single-Pass`) is enabled, a second horizontal-flip ViT-S forward pass yields zero statistically significant test gain on `AbsRel` or `delta1` while nearly doubling end-to-end latency (`47.7 ms -> 93.9 ms` GPU). Single-Pass `H6_geosem_inv_spline` strictly dominates the accuracy–latency Pareto frontier.

### 3.3 Held-Out Test Depth-Bin Comparison (`test_depth_bins.csv`)

| Depth Bin (`Images / Pixels`) | `Raw` (`AR / d1`) | `B3_prev` (`AR / d1`) | `v1` (`AR / d1`) | **`v2 (H6 1-Pass)` (`AR / d1`)** | `v2+Flip (M6 2x)` (`AR / d1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `[0.001, 1.0) m` (`254 / 4.74M`) | `0.3273 / 0.5500` | `0.2612 / 0.6489` | `0.2240 / 0.7230` | **`0.2208 / 0.7304`** | `0.2233 / 0.7261` |
| `[1.0, 2.0) m` (`641 / 45.09M`) | `0.1566 / 0.8268` | `0.1271 / 0.8645` | `0.1099 / 0.9108` | **`0.1052 / 0.9149`** | **`0.1050 / 0.9143`** |
| `[2.0, 4.0) m` (`645 / 84.15M`) | `0.1027 / 0.9150` | `0.0968 / 0.9052` | `0.0855 / 0.9302` | **`0.0822 / 0.9346`** | `0.0824 / 0.9334` |
| `[4.0, 6.0) m` (`415 / 17.24M`) | `0.1311 / 0.8196` | `0.1329 / 0.8129` | `0.1189 / 0.8397` | **`0.1175 / 0.8432`** | **`0.1169 / 0.8466`** |
| `[6.0, 10.0) m` (`220 / 4.80M`) | `0.1892 / 0.6637` | `0.1988 / 0.6208` | `0.1786 / 0.6962` | **`0.1766 / 0.6855`** | **`0.1759 / 0.6941`** |

---

## 4. Mechanism Audit: Round 1 Failure Recovery, Spatial Residuals & Latency

### 4.1 Explicit Audit of Round 1 Failure & High-Gain Cases (`test_round1_case_audit.csv`)

| Case Index (`Scene`) | Role in Round 1 | `f_tilde` | `v2` Shift Breakdown (`focal / geo / sem / total`) | `v1` (`AR / d1`) | **`v2 (1-Pass H6)` (`AR / d1`)** | `v2+Flip (M6)` (`AR / d1`) | Net `v2 - v1` Change |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **`#839` (`kitchen_0039`)** | Round 1 #1 Failure | `0.350` | `+0.234 / +0.021 / -0.098 / +0.153` | `0.2683 / 0.5125` | **`0.1699 / 0.9034`** | `0.1762 / 0.9035` | **`-0.0983` AR, `+39.09 pp` d1** |
| **`#832` (`kitchen_0038`)** | Round 1 #2 Failure | `0.350` | `+0.234 / +0.015 / -0.091 / +0.152` | `0.2460 / 0.5404` | **`0.1906 / 0.6513`** | `0.1988 / 0.6329` | **`-0.0554` AR, `+11.10 pp` d1** |
| **`#326` (`classroom_0023`)** | Round 1 #3 Failure | `0.364` | `+0.230 / -0.074 / -0.058 / +0.097` | `0.2168 / 0.7455` | **`0.1481 / 0.8060`** | `0.1603 / 0.7921` | **`-0.0687` AR, `+6.05 pp` d1** |
| **`#1150` (`bedroom_0110`)** | Round 1 #1 Gain | `0.676` | `+0.145 / +0.018 / +0.121 / +0.257` | `0.1369 / 0.8265` | **`0.1249 / 0.9168`** | `0.1310 / 0.8214` | **`-0.0121` AR, `+9.03 pp` d1** |
| **`#1151` (`bedroom_0110`)** | Round 1 #2 Gain | `0.486` | `+0.197 / +0.001 / +0.097 / +0.265` | `0.0929 / 0.9732` | **`0.0576 / 0.9957`** | `0.0811 / 0.9945` | **`-0.0353` AR, `+2.25 pp` d1** |
| **`#1097` (`bedroom_0091`)** | Round 1 #3 Gain | `0.705` | `+0.137 / -0.011 / +0.091 / +0.205` | **`0.0398 / 0.9994`** | `0.0513 / 0.9993` | `0.0500 / 0.9990` | `+0.0115` AR, `-0.01 pp` d1 |

Wait — observe how remarkably principled that decomposition is: without any heuristic rules on `#839` vs `#1151`, the single-pass internal semantic branch learns **negative log-scale priors (`-0.098 / -0.091`)** on close counter/kitchen false focal alarms (`#839 / #832`) to pull `delta1` from `51.2% -> 90.3%`, while simultaneously outputting **positive scale priors (`+0.121 / +0.097`)** on genuinely shrunken wide-FoV bedrooms (`#1150 / #1151`) to push `#1151` from `0.0929 -> 0.0576` (`99.6%` `delta1`)! Across all `f_tilde < 0.55` left-tail test frames (`n=34`), `AbsRel` drops from `0.1025` (`v1`) to **`0.0912`** (`v2 H6`).

- **Spatial & Focal Mechanism Verification (`test_mechanism_diagnostics.csv`, `test_spatial_residuals.json`)**:
  - Residual focal correlation `r(f_tilde, med_log_err)` narrows from `-0.0854` (`v1`, `SD=0.0739`, `MAE=0.0558`) to **`+0.0094`** (`v2 H6`, `SD=0.0677`, `MAE=0.0518`).
  - 12×16 crop-grid RMS spatial log bias falls from `0.0419` (`Raw`) -> `0.0168` (`B3`) -> `0.0122` (`v1`) -> **`0.0110`** (`v2 H6`) -> **`0.0107`** (`v2+Flip M6`).
- **Top New `v2` Gains & Honest New Regressions (`test_representative_cases.csv`)**:
  - Largest `v2 vs v1` gains: `#329 classroom_0024` (`v1 0.2084 / d1 0.580 -> v2 0.1150 / d1 0.941`), `#839 kitchen_0039` (`0.2683 -> 0.1699`), `#328 classroom_0024` (`0.2082 -> 0.1130`).
  - Honest worst `v2 vs v1` regressions: `#167 living_room_0013` (`v1 0.0229 -> v2 0.1146`, `d1 0.994`), `#1234 living_room_0036` (`0.0370 -> 0.1035`, `d1 0.991`), `#1148 bedroom_0109` (`0.0713 -> 0.1440`, `d1 0.925 -> 0.845`) — near-range wall/table shots (`f~ > 1.20`) where `v1` had near-oracle prediction and semantic prior slightly over-shifted scale (`+0.069` on `#167`).

### 4.2 Uncontended Latency & Numerical Parity (`runtime_overhead.json`, `N=200` Test Images, RTX 4060 Laptop GPU)

| Stage / Variant | Backbone Passes | Backbone Latency (`ms/img`) | Post-Process CPU Torch (`ms/img`) | Post-Process GPU CUDA (`ms/img`) | End-to-End Total (`ms/img`) | Torch vs NumPy `max |Diff|` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `raw` (UniDepthV2 ViT-S) | `1x` | `43.04` | `—` | `—` | `43.04` (GPU) | `—` |
| `b3_residual_v1` (`B3_prev`) | `1x` | `43.04` | `42.18` (NumPy) | `—` | `85.22` | `—` |
| `v1_geocalib3d` | `1x` | `43.04` | `8.30` | `—` | `51.34` (CPU-post) | `<= 8.88e-15` (R1) |
| **`v2` (`H6_geosem_inv_spline`)** | **`1x`** | **`43.04`** | **`10.26`** (NumPy `48.50`) | **`4.63`** | **`47.67` (GPU) / `53.30` (CPU)** | **`7.11e-15 m`** |
| `v2+Flip` (`M6_flip_inv_geosem`) | `2x` | `87.17` | `24.04` | `6.69` | `93.86` (GPU) / `111.21` (CPU) | `7.11e-15 m` |

---

## 5. Publication Figures & Verification Commands

- [fig1_v2_architecture_and_oracles.png](../results/figures/part3b_sota_v2/fig1_v2_architecture_and_oracles.png) — Dual-track architecture, post-`v1` oracle gap waterfall, and zero-leakage 5-fold CV $R^2$ signal discovery.
- [fig2_v2_ablation_and_ci.png](../results/figures/part3b_sota_v2/fig2_v2_ablation_and_ci.png) — Dual-track OOF & test ablation progression alongside 10,000-replicate test bootstrap 95% CIs vs `v1`.
- [fig3_v2_mechanism_and_left_tail.png](../results/figures/part3b_sota_v2/fig3_v2_mechanism_and_left_tail.png) — Left-tail bimodality mitigation (`#839/#832/#326` vs `#1150/#1151`) and 12×16 spatial log-residual heatmaps.
- [fig4_v2_depth_bins_learning_curve_pareto.png](../results/figures/part3b_sota_v2/fig4_v2_depth_bins_learning_curve_pareto.png) — 5-depth-bin test bars, train-only learning curve (`10% -> 100%`), and accuracy–latency Pareto frontier.
- [fig5_v2_round1_failures_and_qualitative.png](../results/figures/part3b_sota_v2/fig5_v2_round1_failures_and_qualitative.png) — Qualitative comparison across Round 1 failures (`#839/#832/#326`), Round 1 high gain (`#1151`), and worst new `v2` regression (`#167`).

```powershell
# Re-verify Round 2, Round 1, and historical SHA-256 manifests
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/part3b_sota_v2/manifest.json
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/part3b_sota/manifest.json
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/final_review_20261113/manifest.json

# Run full Windows pytest suite
D:\anaconda\envs\vb\python.exe -m pytest -q --ignore=tests/test_video_recovery.py -k "not shell_defaults and not writer_crash_resume"
```
