# End-to-End Reproduction Guide

All modules in [`depthlab/`](../depthlab/) are self-contained and verify against [`results/manifest.json`](../results/manifest.json). Large datasets (`data/`), pinned upstream repositories (`third_party/`), and cached array outputs (`outputs/`) stay outside Git with explicit SHA-256 checksums.

---

## 1. Environment & Pinned Inputs

Use Linux + Python 3.10 with [`requirements-reproduction.txt`](../requirements-reproduction.txt) (referencing the pinned PyTorch 2.8 / CUDA 12.8 environment [`requirements-linux-cu128.lock.txt`](../requirements-linux-cu128.lock.txt)) for full backbone GPU reproduction, or Windows Conda environment `vb` for verification, held-out replay, figure regeneration, and unit tests.

Upstream Git checkouts and local checkpoints are verified by commit revision and SHA-256 digest:

| Upstream Repository | Pinned Commit | Local Directory |
| :--- | :--- | :--- |
| [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` | `third_party/UniDepth` |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` | `third_party/Depth-Anything-V2` |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | `4f5ae23172ba60fd7bc11ef671cca678842c7072` | `third_party/Video-Depth-Anything` |

| Dataset / Weight Input | Expected Local Path |
| :--- | :--- |
| [NYU Depth V2 labeled `.mat`](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) | `data/nyu_depth_v2_labeled.mat` |
| [Official NYU `splits.mat`](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) | `data/splits.mat` |
| [Middlebury TrainingQ data](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip) & [GT](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip) | `data/MiddEval3-data-Q.zip`, `data/MiddEval3-GT0-Q.zip` |
| DAVIS clip (`davis_rollercoaster.mp4`, `70` frames @ `24 FPS`) | `data/davis_rollercoaster.mp4` |
| [UniDepthV2 ViT-S config + safetensors](https://huggingface.co/lpiccinelli/unidepth-v2-vits14/tree/main) | `data/weights/unidepth_v2_vits14/` |
| [Depth Anything V2 Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/tree/main) | `data/weights/depth_anything_v2_small/depth_anything_v2_vits.pth` |
| [Video Depth Anything Small checkpoint](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/tree/main) | `data/weights/video_depth_anything_small/video_depth_anything_vits.pth` |

Evaluation protocol invariants ([`depthlab/metrics.py`](../depthlab/metrics.py)): official `795 / 654` train/test split (`249` train / `215` test disjoint scenes), NYU Eigen crop `[45:471, 41:601]`, ground-truth validity `(0.001, 10.0) m`, output depth clipping `[0.001, 10.0] m`, and equal per-image metric averaging (`AbsRel`, `RMSE`, `delta1`).

---

## 2. Running Parts 1 & 2 Pipelines

```bash
export METRIC_PYTHON=/path/to/reproduction/python
export VIDEO_PYTHON="$METRIC_PYTHON"

# Extract datasets & verify official NYU splits
"$METRIC_PYTHON" -m depthlab.prepare --data-root data --output results/part2_depth_video/data_summary.json

# Part 1: Middlebury 15-scene StereoSGBM ablations + region diagnostics + timing
"$METRIC_PYTHON" -m depthlab.stereo --data-root data --output results/part1_stereo --figures results/figures/part1_stereo

# Part 2: RGB-only UniDepthV2 cache, known-K training diagnostic, and DAVIS video models + bokeh render
"$METRIC_PYTHON" -m depthlab.infer_nyu --data-root data --source third_party/UniDepth --weights data/weights/unidepth_v2_vits14 --output outputs/nyu_rgb
"$METRIC_PYTHON" -m depthlab.diagnose --data-root data --predictions outputs/nyu_rgb --known-predictions outputs/nyu_known --output results/part2_depth_video
"$VIDEO_PYTHON" -m depthlab.infer_video --method decode --video data/davis_rollercoaster.mp4 --output outputs/frames
"$VIDEO_PYTHON" -m depthlab.infer_video --method image --source third_party/Depth-Anything-V2 --weights data/weights/depth_anything_v2_small/depth_anything_v2_vits.pth --frames outputs/frames --output outputs/video_image
"$VIDEO_PYTHON" -m depthlab.infer_video --method video --source third_party/Video-Depth-Anything --weights data/weights/video_depth_anything_small/video_depth_anything_vits.pth --frames outputs/frames --output outputs/video_temporal
"$VIDEO_PYTHON" -m depthlab.render_video --frames outputs/frames --image outputs/video_image --video outputs/video_temporal --results results/figures/part2_video --delivery outputs/delivery
```

Or run the bundled shell pipeline directly:

```bash
bash scripts/run_baselines.sh data third_party outputs results
```

---

## 3. Running & Replaying Part 3 (`GeoCalib-3D`)

The frozen parameter artifact is [`results/part3_geocalib/frozen_geocalib_model.json`](../results/part3_geocalib/frozen_geocalib_model.json) (`SHA-256 = 816f799a82784d86d8985ee255607fa6a7d2806c09c3df423be038654fc14ebd`). It binds dataset/split/cache digests, verified source digests, all `795` training IDs (`test_read: false`), and exact parameters for all `11` locked evaluation methods.

### 3.1 Replay Held-Out 654-Image Evaluation (Zero Retraining)

```bash
# Exact SHA-256-verified held-out replay using cached arrays under outputs/
bash scripts/run_evaluation.sh data outputs results/part3_geocalib
```

Equivalent direct Python CLI:

```bash
"$METRIC_PYTHON" -m depthlab.geocalib_evaluate \
  --data-root data \
  --orig-preds outputs/nyu_rgb \
  --aux-dir outputs/nyu_rgb_aux \
  --flip-dir outputs/nyu_rgb_flip \
  --frozen-model results/part3_geocalib/frozen_geocalib_model.json \
  --expected-model-sha256 816f799a82784d86d8985ee255607fa6a7d2806c09c3df423be038654fc14ebd \
  --oof-summary results/part3_geocalib/oof_ablation_summary.csv \
  --output results/part3_geocalib --force
```

### 3.2 Optional Auxiliary Cache, Training-Only Diagnosis & Refitting CLIs

Inspect `--help` on any module for all flags:

```bash
# 1. Extract internal ViT-S semantic features (outputs/nyu_rgb_aux/) & 2-pass horizontal-flip cache (outputs/nyu_rgb_flip/)
"$METRIC_PYTHON" -m depthlab.geocalib_cache build --help

# 2. Run training-only (795 images) oracle ceiling, CV R^2 & left-tail diagnosis
"$METRIC_PYTHON" -m depthlab.geocalib_diagnose --help

# 3. Run 5-fold / 4-seed scene-disjoint CV ablation study and/or full-train parameter freeze (795 train only)
"$METRIC_PYTHON" -m depthlab.geocalib_model --help

# 4. Regenerate pipeline.png, nyu_raw_examples.png, all 5 Part 3 figures, and results/manifest.json
"$METRIC_PYTHON" -m depthlab.geocalib_figures --help
```

### 3.3 Direct Programmatic Inference Without Ground Truth

```python
from depthlab.geocalib_model import (
    assemble_semantic_vector,
    apply_geocalib_model,
    load_frozen_geocalib,
)

frozen = load_frozen_geocalib(
    "results/part3_geocalib/frozen_geocalib_model.json",
    expected_sha256="816f799a82784d86d8985ee255607fa6a7d2806c09c3df423be038654fc14ebd",
)
model = frozen["models"][frozen["primary_single_pass_method"]]  # "geocalib_3d"
x_sem = assemble_semantic_vector("all_sp", cls_orig, patch_mean, patch_std, depth_feat_stats)
calibrated_480x640 = apply_geocalib_model(
    raw_depth_480x640,
    pred_k_3x3,
    model,
    conf_summary=conf_summary_17,
    cached_sem=x_sem,
    use_torch_fast=True,
)
```

---

## 4. Integrity Verification & Test Suite

[`depthlab/verify.py`](../depthlab/verify.py) checks every code file, test, script, doc, CSV/JSON table, and PNG figure listed in [`results/manifest.json`](../results/manifest.json) using Python's standard library only:

```bash
# Linux
python -m depthlab.verify --manifest results/manifest.json
python -m pytest -q
```

```powershell
# Windows (Conda env vb)
D:\anaconda\envs\vb\python.exe -m depthlab.verify --manifest results/manifest.json
D:\anaconda\envs\vb\python.exe -m pytest -q --ignore=tests/test_video_recovery.py -k "not shell_defaults and not writer_crash_resume"
```
