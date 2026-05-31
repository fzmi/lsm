#!/usr/bin/env bash
set -euo pipefail

# Evaluate STF-CNN under the four-fold LSM protocol.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-lsm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

DATASET_PATH="${DATASET_PATH:-data/lsm}"
LAMBDAS="${LAMBDAS:-0.0483 1 5 10}"

python main.py \
    --dataset-path "${DATASET_PATH}" \
    --methods stf \
    --stf-arch cnn \
    --cv-eval \
    --stf-lambdas ${LAMBDAS} \
    --stf-checkpoint-template outputs/stf/cnn/lambda{lambda_tag}_fold{fold}_best.pth
