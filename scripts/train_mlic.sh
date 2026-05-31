#!/usr/bin/env bash
set -euo pipefail

# Train MLIC++ under the four-fold LSM protocol.
#
# Examples:
#   bash scripts/train_mlic.sh
#   LAMBDAS="1 2 5 10" EPOCHS=40 bash scripts/train_mlic.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-lsm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

DATASET_PATH="${DATASET_PATH:-data/lsm}"
LAMBDAS="${LAMBDAS:-1 2 5 10 25 50 100}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.2}"
MAX_PATCHES="${MAX_PATCHES:-50000}"

mkdir -p outputs/mlic outputs/logs

python train_mlic.py \
    --dataset-path "${DATASET_PATH}" \
    --lambdas ${LAMBDAS} \
    --save-path "outputs/mlic/lambda{lambda_tag}_fold{fold}.pth" \
    --cross-validate \
    --train-sample-ratio "${SAMPLE_RATIO}" \
    --train-max-patches "${MAX_PATCHES}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 8
