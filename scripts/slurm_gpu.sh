#!/usr/bin/env bash
#SBATCH --job-name=depth-inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
set -euo pipefail
mapfile -t visible_uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
if [[ ${#visible_uuids[@]} -eq 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${visible_uuids[0]}"
fi
exec "$@"
