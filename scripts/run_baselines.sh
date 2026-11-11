#!/usr/bin/env bash
set -euo pipefail
data=${1:-data}
sources=${2:-third_party}
outputs=${3:-outputs}
results=${4:-results}
metric=${METRIC_PYTHON:-python}
video=${VIDEO_PYTHON:-python}
"$metric" -m depthlab.prepare --data-root "$data" --output "$results/data_summary.json"
"$metric" -m depthlab.stereo --data-root "$data" --output "$results"
"$metric" -m depthlab.infer_nyu --data-root "$data" --source "$sources/UniDepth" --weights "$data/weights/unidepth_v2_vits14" --output "$outputs/nyu_rgb"
"$video" -m depthlab.infer_video --method decode --video "$data/davis_rollercoaster.mp4" --output "$outputs/frames"
"$video" -m depthlab.infer_video --method image --source "$sources/Depth-Anything-V2" --weights "$data/weights/depth_anything_v2_small/depth_anything_v2_vits.pth" --frames "$outputs/frames" --output "$outputs/video_image"
"$video" -m depthlab.infer_video --method video --source "$sources/Video-Depth-Anything" --weights "$data/weights/video_depth_anything_small/video_depth_anything_vits.pth" --frames "$outputs/frames" --output "$outputs/video_temporal"
