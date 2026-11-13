#!/usr/bin/env bash
set -euo pipefail
# Numerical frozen replay only. Presentation generation is a separate task.
data=${1:-data}
outputs=${2:-outputs/reproduction}
results=${3:-results/reproduction}
metric=${METRIC_PYTHON:-python}
frozen=results/frozen_20261111/part3b_residual_20261111/final
"$metric" -m depthlab.evaluate \
  --data-root "$data" --predictions "$outputs/nyu_rgb" \
  --parameters "$frozen/frozen_model.json" \
  --expected-model-sha256 176a0b027af87d11fb14a44a843ed50b8e068e4284ee8096e4daf34009c55ea5 \
  --reference "$frozen/test/per_image.csv" --output "$results/frozen_replay"
