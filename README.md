# Stereo and monocular depth

AIAA 3201 course project: stereo geometry (Part 1), RGB-only metric depth and video depth consistency (Part 2), and training-only calibration (Part 3B).

This branch restores the previously frozen implementation within the new `depthlab` architecture. See [reproduction instructions](docs/REPRODUCTION.md) for the environment, data, fixed-parameter evaluation and optional training studies. The production package does not import the old project.

## Verified restoration

Fresh executions in the pinned Linux reproduction environment were compared with the original frozen results:

| Component | Verification |
|---|---|
| Part 1 | All 15 scenes, three stereo configurations and 135 saved arrays match the original byte-for-byte. |
| Part 2: NYU | All 1,449 newly inferred raw depth maps match the original element-for-element. |
| Part 2: video depth | Both methods' 70-frame raw depth streams match the original element-for-element. |
| Part 3B | All 654 final residual-corrected depth maps match; 2,616 per-image metric rows differ by at most 4.45e-16 after CSV round trips. |
| Code and environment | 107 tests, 15 CLI entrypoints, shell syntax and 87 installed package dependencies passed. |

[Validation record](results/restoration_20261113/validation.json), [new replay evidence](results/restoration_20261113/), and [unchanged historical artifacts](results/frozen_20261111/IMPORT_MANIFEST.json) retain the details. These checks establish preservation of an existing frozen result, not a new independent holdout experiment.

## Part 3B primary and controls

The primary is the unchanged frozen **monotone curve plus bounded prediction-conditioned residual**, `residual_median_ratio_l1_a0.5`. The teacher's strict global scalar is retained as a control; the curve and residual are explicitly extensions of that baseline.

| Method | AbsRel | RMSE (m) | delta1 |
|---|---:|---:|---:|
| Raw RGB-only UniDepthV2 | 0.114544 | 0.398866 | 0.892170 |
| Global scalar | 0.102042 | 0.389288 | 0.899400 |
| Monotone curve | 0.100831 | 0.387002 | 0.903200 |
| Frozen residual primary | **0.099607** | **0.383124** | **0.905546** |

The [frozen model](results/frozen_20261111/part3b_residual_20261111/final/frozen_model.json) retains the original parameters and provenance. Deployment applies the original spline, clipping, raw-depth features, residual scaling and final clipping order. Selection used scene-grouped training folds; the replay never refits or selects on test data. The 14-candidate residual study, 18-candidate broader discovery, learning curves and saved-score uncertainty analysis are available as separate workflows.

## Layout

```text
depthlab/                      Core modules and python -m entrypoints
scripts/                       Baseline and frozen-replay shell entrypoints
tests/                         Unit and numerical regression tests
docs/REPRODUCTION.md            Current reproduction instructions
results/frozen_20261111/        Original frozen parameters and historical evidence
results/restoration_20261113/   Fresh restoration checks and delivery manifest
data/                          External datasets and weights (ignored)
third_party/                   Pinned upstream Git checkouts (ignored)
outputs/                       Raw arrays and runtime artifacts (ignored)
requirements-reproduction.txt Current reproduction environment entrypoint
```

Root-level legacy CSV/JSON files and existing figures under `results/` describe the earlier seed-42 rewrite and remain historical evidence. Its original README, setup and conclusions are available at [the original repository commit](https://github.com/hhuang999/AIAA3201_CV_FINALPROJECT/blob/411687c4cce2a0c8be021f188c7ae9ef6b85de56/README.md). Follow this README and the reproduction guide for the restored implementation.

## Running

Use Linux and Python 3.10 with [requirements-reproduction.txt](requirements-reproduction.txt), which includes the validated PyTorch 2.8 / CUDA 12.8 lock. The older metric/video requirements files remain historical environment records. Data, weights and upstream sources must be prepared as described in the reproduction guide.

```bash
git clone --branch codex/part3b https://github.com/hhuang999/AIAA3201_CV_FINALPROJECT.git
cd AIAA3201_CV_FINALPROJECT
export METRIC_PYTHON=/path/to/reproduction/python
export VIDEO_PYTHON="$METRIC_PYTHON"
"$METRIC_PYTHON" -m depthlab.verify
"$METRIC_PYTHON" -m pytest -q
bash scripts/run_baselines.sh data third_party outputs/reproduction results/reproduction
bash scripts/run_evaluation.sh data outputs/reproduction results/reproduction
```

The baseline script generates raw predictions and stereo diagnostics. The evaluation script performs numerical frozen replay and checks the saved per-image reference. It does not create a report or render final videos. Use new output directories for independent experiments; incomplete prediction caches can resume only with matching identity and runtime settings.

`python -m depthlab.verify` requires only Python's standard library and checks delivered source and results against the manifest. Large raw arrays are kept on the server, with complete checksum-bound cache manifests; they are not stored in Git.

## Sources and licenses

- [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/submit3/), and Hirschmueller, *Stereo Processing by Semiglobal Matching and Mutual Information*, TPAMI 2008.
- [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html).
- [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), Silberman et al., ECCV 2012.
- [UniDepthV2](https://arxiv.org/abs/2502.20110) and [official source](https://github.com/lpiccinelli-eth/UniDepth); source is CC BY-NC 4.0.
- [Depth Anything V2](https://arxiv.org/abs/2406.09414) and [official source](https://github.com/DepthAnything/Depth-Anything-V2); Small source/weights are Apache 2.0.
- [Video Depth Anything](https://arxiv.org/abs/2501.12375) and [official source](https://github.com/DepthAnything/Video-Depth-Anything); Small source/weights are Apache 2.0.

Downloaded repositories retain their original license files. Data and model licenses continue to apply to the external inputs.
