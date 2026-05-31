# LSM: Multispectral Satellite Image Delivery Benchmark

<img src="docs/LSM%20Title.svg" alt="LSM Dataset" width="500" height="172" style="border: 1px solid #ddd; margin: 0 0 10px 0;" />

LSM is a benchmark for multispectral satellite image delivery. It evaluates how well compression and neural delivery methods preserve 16-bit multispectral imagery under a transmission-size budget, using reconstruction quality, delivery size, and encoding/decoding time.

## Dataset

The dataset will be published on Hugging Face. After downloading it, place or symlink the extracted dataset at:

```text
data/lsm
```

Expected layout:

```text
data/lsm/
├── brisbane
│   └── image.tiff
├── camarillo
│   └── image.tiff
...
```

Each `image.tiff` is expected to be a `uint16` multispectral image with shape `H x W x C`.

## Installation

Create an environment and install the required packages:

```bash
conda create -n lsm python=3.11 -y
conda activate lsm
pip install -r requirements.txt
```

The neural baselines require PyTorch with CUDA for practical training and evaluation. Install the PyTorch build that matches your CUDA driver before installing the rest of the requirements if your platform needs a specific wheel.

## Quick Start

Run JPEG2000 on one region:

```bash
python main.py \
  --dataset-path data/lsm \
  --regions brisbane \
  --methods jpeg2000 \
  --jpeg2000-qualities 5 10 15 20 25 30 35 40 50 60
```

Results are appended to:

```text
outputs/results.csv
```

BD-PSNR and BD-SSIM summaries, when computable against the JPEG2000 anchor, are written to:

```text
outputs/bd_metrics.csv
```

## Benchmark Methods

### JPEG2000

JPEG2000 is evaluated band-wise on the full image. The benchmark records total transmitted bits, bpp, bpp per band, patch-averaged MSE/PSNR/SSIM, full-image PSNR, and encode/decode time.

```bash
bash scripts/eval_jpeg2000.sh
```

You can override the regions and qualities:

```bash
REGIONS="brisbane hawick" QUALITIES="10 20 30 40 50" bash scripts/eval_jpeg2000.sh
```

JPEG2000 evaluation also writes cached `.jp2` files under `outputs/jpeg2000/`. MIDNet uses these cached JPEG2000 reconstructions as intermediate inputs.

### MIDNet

MIDNet is a parameter-delivering method. It uses JPEG2000 intermediate reconstructions plus a compact per-region neural model. The benchmark counts both intermediate JPEG2000 bits and delivered model-weight bits.

Generate JPEG2000 intermediates first:

```bash
REGIONS="brisbane" QUALITIES="10 30 50" bash scripts/eval_jpeg2000.sh
```

Train MIDNet for one region:

```bash
REGION=brisbane QUALITIES="10 30 50" bash scripts/train_midnet.sh
```

Evaluate MIDNet:

```bash
REGIONS="brisbane" QUALITIES="10 30 50" bash scripts/eval_midnet.sh
```

Expected MIDNet checkpoint paths:

```text
outputs/midnet/<region>_q<quality>_best.pth
```

### MLIC++

MLIC++ is trained and evaluated under the four-fold LSM cross-validation protocol defined in `utils/regions.py`.

Train:

```bash
LAMBDAS="1 2 5 10 25 50 100" bash scripts/train_mlic.sh
```

Evaluate:

```bash
LAMBDAS="1 2 5 10 25 50 100" bash scripts/eval_mlic.sh
```

Expected MLIC++ checkpoint paths:

```text
outputs/mlic/lambda<lambda_tag>_fold<fold>_best.pth
```

### STF-CNN

STF-CNN is trained and evaluated under the same four-fold LSM cross-validation protocol.

Train:

```bash
LAMBDAS="0.0483 1 5 10 25 50 100 200" bash scripts/train_stf_cnn.sh
```

Evaluate lower-rate checkpoints:

```bash
bash scripts/eval_stf_cnn_A.sh
```

Evaluate higher-rate checkpoints:

```bash
bash scripts/eval_stf_cnn_B.sh
```

Expected STF-CNN checkpoint paths:

```text
outputs/stf/cnn/lambda<lambda_tag>_fold<fold>_best.pth
```

## Command-Line Interface

The main benchmark entry point is:

```bash
python main.py --help
```

Common options:

```text
--dataset-path              Path to the dataset root.
--regions                   One or more region names for single-region evaluation.
--methods                   One or more of jpeg2000, midnet, mlic, stf.
--cv-eval                   Use the predefined four-fold cross-validation protocol.
--patch-size-h              Patch height for metric aggregation. Default: 64.
--patch-size-w              Patch width for metric aggregation. Default: 64.
--max-value                 Maximum sample value. Default: 65535.
```

Example cross-validation evaluation for MLIC++:

```bash
python main.py \
  --dataset-path data/lsm \
  --methods mlic \
  --cv-eval \
  --mlic-lambdas 1 2 5 10 25 50 100 \
  --mlic-checkpoint-template outputs/mlic/lambda{lambda_tag}_fold{fold}_best.pth
```

Example cross-validation evaluation for STF-CNN:

```bash
python main.py \
  --dataset-path data/lsm \
  --methods stf \
  --stf-arch cnn \
  --cv-eval \
  --stf-lambdas 0.0483 1 5 10 25 50 100 200 \
  --stf-checkpoint-template outputs/stf/cnn/lambda{lambda_tag}_fold{fold}_best.pth
```

## Outputs

The benchmark writes generated artefacts under `outputs/`, which is ignored by git:

```text
outputs/
├── results.csv
├── bd_metrics.csv
├── jpeg2000/
├── midnet/
├── mlic/
├── stf/
└── logs/
```

`results.csv` columns:

```text
region,method,config,bits_total,mse_patch_avg,psnr_patch_avg,psnr_img,ssim_patch_avg,bpp,bpppb,encode_time,decode_time
```
