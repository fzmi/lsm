#!/usr/bin/env bash
set -euo pipefail

# Evaluate MLIC++ checkpoints under the four-fold LSM protocol.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-lsm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

DATASET_PATH="${DATASET_PATH:-data/lsm}"
LAMBDAS="${LAMBDAS:-1 2 5 10 25 50 100}"

echo "=== Evaluating MLIC++ (cv-eval) lambdas=${LAMBDAS} on $(hostname) ==="
python main.py \
    --dataset-path "${DATASET_PATH}" \
    --methods mlic \
    --cv-eval \
    --mlic-lambdas ${LAMBDAS} \
    --mlic-checkpoint-template outputs/mlic/lambda{lambda_tag}_fold{fold}_best.pth
