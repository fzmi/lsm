#!/usr/bin/env bash
set -euo pipefail

# Train MIDNet for one region across JPEG2000 intermediate qualities.
#
# Examples:
#   REGION=brisbane bash scripts/train_midnet.sh
#   REGION=hawick QUALITIES="10 30 50" EPOCHS=200 bash scripts/train_midnet.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-lsm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

REGION="${REGION:?Set REGION=<region_name> when submitting}"
QUALITIES="${QUALITIES:-10 30 50}"
RESBLOCKS="${RESBLOCKS:-16}"
TOTAL_PATHS="${TOTAL_PATHS:-1}"
N_FEATS="${N_FEATS:-64}"
EPOCHS="${EPOCHS:-200}"
DATASET_PATH="${DATASET_PATH:-data/lsm}"

if [[ "$RESBLOCKS" -eq 16 && "$TOTAL_PATHS" -eq 1 && "$N_FEATS" -eq 64 ]]; then
    OUT_DIR="outputs/midnet"
else
    OUT_DIR="outputs/midnet-r${RESBLOCKS}-p${TOTAL_PATHS}-f${N_FEATS}"
fi
mkdir -p "$OUT_DIR" outputs/logs

for q in $QUALITIES; do
    save_path="${OUT_DIR}/${REGION}_q${q}.pth"
    if [[ -f "${save_path%.pth}_best.pth" ]]; then
        echo "skip ${REGION} q=${q}: ${save_path%.pth}_best.pth already exists"
        continue
    fi
    echo "=== Training MIDNet ${REGION} q=${q} (resblocks=${RESBLOCKS}, paths=${TOTAL_PATHS}, n_feats=${N_FEATS}, epochs=${EPOCHS}, no-amp) ==="
    python train_midnet.py \
        --dataset-path "${DATASET_PATH}" \
        --region "${REGION}" \
        --intermediate-method jpeg2000 \
        --intermediate-quality "${q}" \
        --total-paths "${TOTAL_PATHS}" \
        --resblocks "${RESBLOCKS}" \
        --n-feats "${N_FEATS}" \
        --epochs "${EPOCHS}" \
        --no-amp \
        --clip-max-norm 1.0 \
        --save-path "${save_path}" \
        --jpeg2000-root outputs/jpeg2000
done
