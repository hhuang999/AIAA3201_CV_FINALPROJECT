# Stereo and monocular depth

AIAA 3201 course project on stereo geometry, RGB-only metric depth, video depth consistency, and training-only depth calibration.

## Installation and data

Use Python 3.10 and a CUDA-capable GPU. Two environments preserve the upstream models' different PyTorch and NumPy versions:

```bash
conda create -n depth-metric python=3.10 -y
conda create -n depth-video python=3.10 -y
conda run -n depth-metric pip install -r requirements-metric.txt
conda run -n depth-video pip install -r requirements-video.txt
```

Obtain the official code at these revisions:

| Repository | Revision |
|---|---|
| [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) | `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | `4f5ae23172ba60fd7bc11ef671cca678842c7072` |

Clone each repository into its named directory under `third_party`, then check out the revision above. Official GitHub source archives at these revisions are equivalent inputs. The inference commands import the source directly. UniDepth's optional KNN and patch-extraction training/3D-evaluation extensions are not used by this project's 2D depth evaluation. Video frames are decoded with OpenCV, shared by both video pipelines.

Place these downloads in `data/`:

- [Middlebury quarter-resolution input](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip) and [left ground truth](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip), retaining their original ZIP filenames.
- [NYU Depth V2 labeled data](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and [official splits.mat](https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat).
- `davis_rollercoaster.mp4` from Video Depth Anything's `assets/example_videos/` directory. The actual clip contains **70 frames at 24 FPS**; requesting the first 100 uses all 70, without repeating frames.

Download weights into the following directories:

| Directory under data/weights | Official files |
|---|---|
| `unidepth_v2_vits14/` | `config.json` and `model.safetensors` from [UniDepthV2 ViT-S](https://huggingface.co/lpiccinelli/unidepth-v2-vits14/tree/main) |
| `depth_anything_v2_small/` | `depth_anything_v2_vits.pth` from [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/tree/main) |
| `video_depth_anything_small/` | `video_depth_anything_vits.pth` from [Video Depth Anything Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/tree/main) |

The local UniDepth folder avoids requiring an online Hub connection during inference. Data, weights, third-party source, raw arrays, and videos are excluded from Git.

## Metric inference

```bash
conda run -n depth-metric python -m depthlab.prepare --data-root data
conda run -n depth-metric python -m depthlab.infer_nyu \
  --data-root data --source third_party/UniDepth \
  --weights data/weights/unidepth_v2_vits14 --output outputs/nyu_rgb
```

UniDepthV2 ViT-S receives RGB only and retains its official internal FP16 autocast and default resolution bounds. RGB input and returned depth are 480x640; the network input is recorded separately. Floating-point depth and estimated camera matrices are saved for all 1,449 images. This command does not calculate test metrics.

The NYU protocol uses filled `depths` in meters, MATLAB split indices minus one, and the Python crop `[45:471, 41:601]`. Targets satisfy `0.001 < depth < 10`; predictions are clipped to `[0.001, 10]`. AbsRel, RMSE and delta1 (`ratio < 1.25`) use float64 and equal per-image averaging, with no test-time scale alignment. Any necessary output resizing is bilinear with `align_corners=False`.

## Stereo geometry

Use the 15 Middlebury quarter-resolution training scenes, including `im0.png`, `im1.png`, `calib.txt`, and `disp0GT.pfm` under `MiddEval3/trainingQ/`.

```bash
python -m depthlab.stereo --data-root data --output results
python -m pytest
```

Three settings compare unpadded 3-way SGBM, reflected-padding 3-way SGBM, and reflected-padding HH4. Matching uses grayscale images, a five-pixel block, and a disparity range rounded up to a multiple of 16. Both directions use the same matcher and padding. Predicted disparities must be finite, positive, inside the original corresponding image, and agree within one pixel with the reverse prediction sampled at the nearest column. Padding supplies context without making out-of-image matches valid.

Disparity MAE is computed on the intersection of valid predictions and finite positive ground truth. Coverage divides that intersection by the ground-truth valid count. Scene scores are averaged equally. Depth uses quarter-resolution calibration, `focal * baseline_m / (disparity + doffs)`. Invalid depths are NaN.

## Sources

- [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/submit3/).
- Hirschmüller, *Stereo Processing by Semiglobal Matching and Mutual Information*, TPAMI 2008.
- [OpenCV StereoSGBM](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html).
