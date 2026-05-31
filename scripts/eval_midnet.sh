#!/usr/bin/env bash
set -euo pipefail

# Evaluate MIDNet checkpoints across regions and JPEG2000 qualities.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-lsm}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi

QUALITIES="${QUALITIES:-10 30 50}"
REGIONS="${REGIONS:-brisbane camarillo cambridge hawick kagoshima lagos lamington logan-village manaus munich muscat vancouver}"
TOTAL_PATHS="${TOTAL_PATHS:-1}"
RESBLOCKS="${RESBLOCKS:-16}"
N_FEATS="${N_FEATS:-64}"
# Default delivery dtype matches the `_dfp16` suffix used in current
# results.csv rows. Override with DELIVERY_DTYPE=fp32 for the heavier variant.
DELIVERY_DTYPE="${DELIVERY_DTYPE:-fp16}"
DATASET_PATH="${DATASET_PATH:-data/lsm}"

for q in $QUALITIES; do
    for region in $REGIONS; do
        ckpt="outputs/midnet/${region}_q${q}_best.pth"
        if [[ ! -f "$ckpt" ]]; then
            echo "skip ${region} q=${q}: checkpoint $ckpt missing"
            continue
        fi
        echo "=== Eval MIDNet ${region} q=${q} (delivery=${DELIVERY_DTYPE}) ==="
        python main.py \
            --dataset-path "${DATASET_PATH}" \
            --regions "${region}" \
            --methods midnet \
            --midnet-checkpoint "${ckpt}" \
            --midnet-intermediate-method jpeg2000 \
            --midnet-intermediate-quality "${q}" \
            --midnet-total-paths "${TOTAL_PATHS}" \
            --midnet-resblocks "${RESBLOCKS}" \
            --midnet-n-feats "${N_FEATS}" \
            --midnet-delivery-dtype "${DELIVERY_DTYPE}"
    done
done
