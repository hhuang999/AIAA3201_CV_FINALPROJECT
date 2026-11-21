# Part 3B Overview: Baseline Restoration, Round 1 (`GeoCalib-3D v1`) & Round 2 (`GeoCalib-3D v2`)

This index summarizes the additive Part 3B research progression built on top of the restored course-project codebase for RGB-only **UniDepthV2 ViT-S** post-hoc calibration on **NYU Depth V2**.

> **Why `README.md` Remains Unchanged**: Root [README.md](../README.md) and the restored pipeline sources are SHA-256 locked inside `results/final_review_20261113/manifest.json` (`336` files), `results/acceptance_20261113/manifest.json` (`344` files), and `results/restoration_20261113/manifest.json` (`318` files). Keeping `README.md` byte-identical to `codex/part3b` guarantees `python -m depthlab.verify` passes unconditionally alongside the Round 1 and Round 2 manifests.

---

## 1. Branch & Provenance Map

| Branch | Anchor Commit | Scope & Scientific Discipline | Primary Report & Manifest |
| :--- | :---: | :--- | :--- |
| `codex/part3b` | `923ec49` | Restored Part 1/2/3B baseline (`b3_residual_v1` + Part 1 stereo + Part 2 video) | [FINAL_CODE_REVIEW_20261113.md](FINAL_CODE_REVIEW_20261113.md)<br>[`final_review_20261113/manifest.json`](../results/final_review_20261113/manifest.json) (`336` files) |
| `feat/part3b-sota-calibration` | `547ca84` | **Round 1 (`GeoCalib-3D v1`)**: 25-param single-pass geometric + Spatial3D + slope calibrator; zero prior test exposure, zero GT $K_{\text{true}}$, pre-test freeze (`8560b5a`), one-shot 654-test eval | [PART3B_SOTA.md](PART3B_SOTA.md)<br>[`part3b_sota/manifest.json`](../results/part3b_sota/manifest.json) (`31` files) |
| `feat/part3b-sota-v2` | `HEAD` | **Round 2 (`GeoCalib-3D v2`)**: Round 1 failure-informed iteration (`#839/#832/#326` vs `#1150/#1151`), strict train-only 795-image 5-fold/4-seed CV (`test` blocked), pre-test freeze (`ebad14a`), one-shot 654-test eval, zero GT $K_{\text{true}}$, explicit 1-Pass vs 2-Pass Flip-TTA separation | [PART3B_SOTA_V2.md](PART3B_SOTA_V2.md)<br>[`part3b_sota_v2/manifest.json`](../results/part3b_sota_v2/manifest.json) (`32` files) |

---

## 2. Verified Summary Table (NYU Depth V2, Identical Local UniDepthV2 ViT-S Cache)

All methods below are evaluated over the identical local `outputs/nyu_rgb/depth.npy` predictions (`795` official train images / `249` scenes via 5-fold scene-grouped OOF; `654` held-out test images / `215` scenes evaluated once after parameter lock).

| Method | Track (`Passes`) | Params (`Raw / Eff.DoF`) | OOF `AbsRel` $\downarrow$ | OOF `RMSE` $\downarrow$ | OOF `delta1` $\uparrow$ | Test `AbsRel` $\downarrow$ | Test `RMSE` (m) $\downarrow$ | Test `delta1` $\uparrow$ | Test Gain vs `B3_prev` / `v1` | End-to-End GPU (`ms/img`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `raw` (UniDepthV2 ViT-S) | 1-Pass (`1x`) | `0 / 0.0` | 0.117945 | 0.430373 | 0.884173 | 0.114540 | 0.398839 | 0.892293 | `—` | `43.04` |
| `b3_residual_v1` (`B3_prev`) | 1-Pass (`1x`) | `15 / 15.0` | 0.102441 | 0.401714 | 0.904042 | 0.099601 | 0.383099 | 0.905671 | Baseline (`—`) | `85.22` (NumPy post) |
| `v1_geocalib3d` (`GeoCalib-3D v1`) | 1-Pass (`1x`) | `25 / 25.0` | 0.092502 | 0.379230 | 0.926687 | 0.086535 | 0.349250 | 0.935017 | `+0.01307 / +0.03385 m / +2.93 pp` vs `B3_prev` | `51.34` (CPU post) |
| **`v2` (`H6_geosem_inv_spline`, Track A Primary)** | **1-Pass (`1x`)** | `5018 / 146.9` | **0.086894** | **0.366619** | **0.932284** | **0.083024** | **0.340792** | **0.938413** | **`+0.01658 / +0.04231 m / +3.27 pp` vs `B3_prev`**<br>**`+0.00351 / +0.00846 m / +0.34 pp` vs `v1`** | **`47.67`** |
| `v2+Flip` (`M6_flip_inv_geosem`, Track B Extension) | 2-Pass (`2x`) | `5024 / 150.9` | 0.086697 | 0.366778 | 0.932848 | 0.083189 | 0.339846 | 0.938350 | Statistically tied with 1-Pass `v2` on `AbsRel`/`delta1` (`p > 0.5`) at ~2x runtime | `93.86` |

- **10,000-Replicate Scene-Cluster Bootstrap (`v2` vs `v1`, `215` test scenes)**: `AbsRel` `+0.003512` (`95% CI [+0.001280, +0.005868]`), `RMSE` `+0.008458 m` (`95% CI [+0.003117, +0.014054 m]`), `delta1` `+0.003396` (`95% CI [-0.000183, +0.007364]`).
- **Round 1 Failure Recovery (`v2` vs `v1`)**: `#839` (`kitchen_0039`: `AbsRel 0.2683 -> 0.1699`, `delta1 51.3% -> 90.3%`), `#832` (`kitchen_0038`: `0.2460 -> 0.1906`), `#326` (`classroom_0023`: `0.2168 -> 0.1481`), while further improving wide-FoV `#1151` (`0.0929 -> 0.0576`, `delta1 99.6%`).

---

## 3. Code, Results, Figures & Tests Layout

```text
depthlab/
  sota_io.py                     # Train-only verified I/O & cache checksum guard (Round 1)
  sota_diagnose.py               # Pre-v1 residual, oracle & focal-drift diagnosis (Round 1)
  sota_calibration.py            # GeoCalib-3D v1 model, L-BFGS fitting & 5-fold/4-seed CV
  sota_evaluate.py               # One-shot 654-test evaluation & 10k scene bootstrap (Round 1)
  sota_figures.py                # 300-DPI CVPR figures (Round 1)
  sota_v2_cache.py               # Train/test split-isolated aux & horizontal-flip cache builder
  sota_v2_diagnose.py            # Post-v1 oracle ceiling, CV R^2 & left-tail bimodality audit
  sota_v2_calibration.py         # Dual-track v2 models (inverse-depth spline + semantic field)
  sota_v2_evaluate.py            # One-shot 654-test evaluation, Round 1 case audit & parity check
  sota_v2_figures.py             # 300-DPI CVPR figures (Round 2)

docs/
  PART3B_OVERVIEW.md             # This index (Round 1 + Round 2 overview)
  PART3B_SOTA.md                 # Full Round 1 (GeoCalib-3D v1) technical report
  PART3B_SOTA_V2.md              # Full Round 2 (GeoCalib-3D v2) technical report

results/
  part3b_sota/                   # Round 1 frozen model, OOF/test tables & 31-file manifest
  part3b_sota_v2/                # Round 2 frozen model, OOF/test tables & 32-file manifest
  figures/part3b_sota/           # Round 1 fig1..fig5 (300 DPI)
  figures/part3b_sota_v2/        # Round 2 fig1..fig5 (300 DPI)

tests/
  test_part3b_sota.py            # Round 1 invariant, monotonicity, split-guard & parity tests
  test_part3b_sota_v2.py         # Round 2 inverse-spline, flip-symmetry & parity tests
```

---

## 4. Verification Commands

Run inside `AIAA3102_CV_FINALPROJECT`:

```powershell
# Verify all 4 SHA-256 manifests (baseline review, acceptance, Round 1 SOTA, Round 2 SOTA)
D:\anaconda\envs\vb\python.exe -m depthlab.verify
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/acceptance_20261113/manifest.json
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/part3b_sota/manifest.json
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/part3b_sota_v2/manifest.json

# Run Windows pytest suite (excludes Linux shell-default / POSIX crash-resume checks)
D:\anaconda\envs\vb\python.exe -m pytest -q --ignore=tests/test_video_recovery.py -k "not shell_defaults and not writer_crash_resume"
```
