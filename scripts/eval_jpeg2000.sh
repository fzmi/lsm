#!/usr/bin/env bash
set -euo pipefail

# Evaluate full-image, band-wise JPEG2000 on one or more LSM regions.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-lsm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

DATASET_PATH="${DATASET_PATH:-data/lsm}"
REGIONS="${REGIONS:-brisbane camarillo cambridge hawick kagoshima lagos lamington logan-village manaus munich muscat vancouver}"
QUALITIES="${QUALITIES:-5 10 15 20 25 30 35 40 50 60}"

python main.py \
    --dataset-path "${DATASET_PATH}" \
    --regions ${REGIONS} \
    --methods jpeg2000 \
    --jpeg2000-qualities ${QUALITIES}
