#!/usr/bin/env bash
set -euo pipefail
# Numerical held-out replay using the frozen training-only GeoCalib-3D parameters.
data=${1:-data}
outputs=${2:-outputs}
results=${3:-results/part3_geocalib}
metric=${METRIC_PYTHON:-python}
frozen=${4:-results/part3_geocalib/frozen_geocalib_model.json}
"$metric" -m depthlab.geocalib_evaluate \
  --data-root "$data" \
  --orig-preds "$outputs/nyu_rgb" \
  --aux-dir "$outputs/nyu_rgb_aux" \
  --flip-dir "$outputs/nyu_rgb_flip" \
  --frozen-model "$frozen" \
  --expected-model-sha256 816f799a82784d86d8985ee255607fa6a7d2806c09c3df423be038654fc14ebd \
  --oof-summary results/part3_geocalib/oof_ablation_summary.csv \
  --output "$results" --force
