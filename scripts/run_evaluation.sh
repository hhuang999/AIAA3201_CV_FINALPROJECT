#!/usr/bin/env bash
set -euo pipefail
data=${1:-data}
sources=${2:-third_party}
outputs=${3:-outputs}
results=${4:-results}
metric=${METRIC_PYTHON:-python}
"$metric" -m depthlab.evaluate --data-root "$data" --predictions "$outputs/nyu_rgb" --parameters "$results/calibration_parameters.json" --output "$results"
"$metric" -m depthlab.infer_nyu --data-root "$data" --source "$sources/UniDepth" --weights "$data/weights/unidepth_v2_vits14" --output "$outputs/nyu_known" --split train --known-intrinsics
"$metric" -m depthlab.diagnose --data-root "$data" --predictions "$outputs/nyu_rgb" --known-predictions "$outputs/nyu_known" --output "$results"
"$metric" -m depthlab.figures --data-root "$data" --predictions "$outputs/nyu_rgb" --results "$results"
"$metric" -m depthlab.render_video --frames "$outputs/frames" --image-predictions "$outputs/video_image" --video-predictions "$outputs/video_temporal" --output "$outputs/delivery" --results "$results/figures/video"
"$metric" -m depthlab.summarize --outputs "$outputs" --results "$results" --readme README.md
