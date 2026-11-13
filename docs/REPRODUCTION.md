# Reproducing the frozen Part 1/2/3B implementation

The rebuilt `depthlab` package is self-contained. Historical code is used only in independent migration checks, never imported by production modules. The frozen artifact remains byte-identical to commit `91091a1b1d27fc2e19e56c2910dd0b3020921f59`.

## Environment and inputs

Use Linux, Python 3.10 and `requirements-reproduction.txt` for the numerical parity route. That file includes the previously validated PyTorch 2.8 / CUDA 12.8 lock. The original `requirements-metric.txt` and `requirements-video.txt` describe the earlier rewrite environment; they are retained for historical results and are not the numerical parity environment.

The restored code fixes the upstream PyTorch attention branches even when xFormers is installed. UniDepth uses RGB only, FP32 weights with official internal CUDA autocast, bilinear returned-depth interpolation and seed zero. DAV2 remains FP32, VDA uses its official FP16 offline path. Both consume the same OpenCV-decoded RGB frames.

Source repositories must be clean Git checkouts at their pinned revisions. Checkpoint file hashes are verified, so equivalent-looking model names or unverified source archives are not substitutes.

Use these pinned sources and input locations:

| Source repository | Revision | Local directory |
|---|---|---|
| [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` | `third_party/UniDepth` |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` | `third_party/Depth-Anything-V2` |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | `4f5ae23172ba60fd7bc11ef671cca678842c7072` | `third_party/Video-Depth-Anything` |

| Input | Location |
|---|---|
| [NYU labeled data](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) | `data/nyu_depth_v2_labeled.mat` |
| [Official NYU splits](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) | `data/splits.mat` |
| [Middlebury Q input](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip) and [GT](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip) | `data/MiddEval3-data-Q.zip`, `data/MiddEval3-GT0-Q.zip` |
| VDA `assets/example_videos/davis_rollercoaster.mp4` | `data/davis_rollercoaster.mp4` |
| [UniDepthV2 ViT-S config and safetensors](https://huggingface.co/lpiccinelli/unidepth-v2-vits14/tree/main) | `data/weights/unidepth_v2_vits14/` |
| [DAV2 Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/tree/main) | `data/weights/depth_anything_v2_small/depth_anything_v2_vits.pth` |
| [VDA Small checkpoint](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/tree/main) | `data/weights/video_depth_anything_small/video_depth_anything_vits.pth` |

The decoded clip contains 70 frames at 24 FPS. NYU uses the official 795/654 train/test split, filled depth in meters, crop `[45:471, 41:601]`, valid targets strictly between 0.001 and 10 meters, prediction clipping to `[0.001, 10]`, and equal per-image averaging. No test-time target-based scale alignment is applied.

Place the official NYU `.mat` and `splits.mat`, Middlebury Q input/ground-truth ZIPs, and `davis_rollercoaster.mp4` in `data/`. The current server reuses verified existing files through symlinks; `depthlab.prepare` avoids rewriting already present extraction members. Large validation artifacts are under `outputs/restored_20261113/`, which resolves to NAS storage.

## Baselines

Run from the repository root using the reproduction interpreter:

```bash
export METRIC_PYTHON=/path/to/reproduction/python
export VIDEO_PYTHON="$METRIC_PYTHON"
"$METRIC_PYTHON" -m pytest -q
bash scripts/run_baselines.sh data third_party outputs/reproduction results/reproduction
```

This generates raw predictions and numerical artifacts, without model selection. Use a new run directory for independent experiments. NYU inference can resume the same interrupted cache only when identity and runtime checks match; complete caches are checksum-verified without reloading a model.

For stereo-only validation without creating figures:

```bash
"$METRIC_PYTHON" -m depthlab.stereo --data-root data \
  --output outputs/stereo_check --no-images --benchmark-passes 3
```

The primary is reflected-padding 3-way SGBM with LR tolerance one. The unpadded and HH4 alternatives remain ablations. Raw disparity, validity and depth arrays, region diagnostics and configuration metadata are retained.

## Frozen residual deployment and reproduction

The primary artifact is:

`results/frozen_20261111/part3b_residual_20261111/final/frozen_model.json`

Its SHA-256 is `176a0b027af87d11fb14a44a843ed50b8e068e4284ee8096e4daf34009c55ea5`. It contains the original selected `residual_median_ratio_l1_a0.5`, original base knots, all residual coefficients, strict scalar control, train IDs and provenance.

```bash
bash scripts/run_evaluation.sh data outputs/reproduction results/reproduction
```

The evaluation script performs fixed-parameter numerical replay only. It checks the original dataset/split, a complete RGB-only prediction cache, the frozen model digest and all saved per-image metrics. It never fits on test labels or chooses a candidate. The output must be new or empty. Reports, figures and final videos are not generated by this script.

For an additional pixel-level comparison, invoke `python -m depthlab.evaluate --help` and supply `--reference-depth` with the original external `primary_depth.npy`. The expected external depth SHA is `94d351f467c4ae16fcca37222413dc9cf67556d99fbd2e9d7985de936ec245a4`.

For direct application without ground truth:

```python
from depthlab.frozen import load_frozen, apply_frozen
model = load_frozen("results/frozen_20261111/part3b_residual_20261111/final/frozen_model.json",
    expected_sha256="176a0b027af87d11fb14a44a843ed50b8e068e4284ee8096e4daf34009c55ea5")
calibrated_depth = apply_frozen(raw_depth_480x640, model)
```

The canonical mapping preserves the original order of spline interpolation, base clipping, raw-depth features, bounded residual scaling, and final clipping.

## Optional training-only studies

These are explicit new studies in fresh directories. They do not replace the frozen deployment automatically.

```bash
"$METRIC_PYTHON" -m depthlab.cache_subset --predictions outputs/reproduction/nyu_rgb \
  --data-root data --split train --output outputs/reproduction/nyu_train
"$METRIC_PYTHON" -m depthlab.calibration --data-root data \
  --predictions outputs/reproduction/nyu_train --output outputs/new_residual_study
"$METRIC_PYTHON" -m depthlab.calibration_study --data-root data \
  --predictions outputs/reproduction/nyu_train --output outputs/new_discovery
```

The residual study restores all 14 candidates, scene seed `20261031`, per-image pixel seed `20261105 + original_id`, original LBFGS settings, and all-fold three-metric eligibility. The broader discovery restores the 18 mapping/loss candidates plus raw, median and Huber controls. Saved fold models and uncertainty are retained. Historical GPU fitting and a new CPU fit need not be bitwise identical; exact deployment uses the verified frozen parameters.

Learning curves can reuse a completed new discovery, or the migrated original discovery with its frozen hash chain, through `python -m depthlab.learning_curve --help`. This restores fractions 0.1/0.25/0.5/1.0 and fixed validation scenes. Saved-score distributions, training bias and scene bootstrap are available through `python -m depthlab.calibration_analysis --help`; plots are opt-in.

## Artifact identity and scope

`results/frozen_20261111/IMPORT_MANIFEST.json` identifies every copied historical CSV/JSON/Markdown artifact and its original SHA-256. Root-level old result tables describe the pre-restoration seed-42 rewrite and are retained as historical evidence, not silently relabelled as restored results. Fresh preservation checks are under `results/restoration_20261113/` when delivered.

Compact prediction caches always carry original dataset IDs; consumers never assume row positions equal IDs. Hash binding and finite-value checks prevent mismatched datasets, unfinished caches and invalid depth outputs from entering a completed result. These checks do not claim scientific novelty or new independent test evidence: the replay establishes preservation of the existing frozen result.

## Delivered verification

Run `python -m depthlab.verify` to check the source, tests, dependency specifications, historical evidence and new parity results against `results/restoration_20261113/manifest.json`. The verifier uses only Python's standard library. Large external prediction arrays remain in ignored runtime storage and carry their own complete cache manifests.

The restored source uses LF checkouts on Windows and Linux. `results/**` is preserved byte-for-byte so CSV newline conversion cannot invalidate frozen scientific hashes.
