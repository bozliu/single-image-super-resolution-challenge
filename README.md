# Single Image Super-Resolution (x4) Challenge

This repository is a compact PyTorch pipeline for **single image super-resolution (SISR)**: given a low-resolution (LR) image, predict a high-resolution (HR) image at **4x** scale.

The project is designed around the Mini-DIV2K setting (500 train pairs + 80 validation pairs) and a strict **parameter budget**. It aims to be:

- reproducible and scriptable (train/eval/infer are simple CLI entry points),
- practical on **Apple Silicon** via `mps`,
- competitive under a small-model constraint using carefully chosen augmentations.

## Related Article

- [Single Image Super-Resolution Challenge (Medium)](https://bozliu.medium.com/single-image-super-resolution-challenge-6f4835e5a156)
![Single Image Super-Resolution Challenge](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*rmQjKd4uETJtwYMRTDzmrg.png)

## 1. Challenge Description

Super-resolution is useful when the acquisition process is resolution-limited: mobile zoom, old photos, thumbnails, bandwidth-constrained streaming, or upscaling in computer vision pipelines.

This project targets the common SISR evaluation protocol:

- input: LR image (downsampled by 4x),
- output: reconstructed HR image,
- metric: **PSNR** between prediction and ground-truth HR (computed on RGB with a small border crop).

## 2. Dataset and Constraints

The code assumes the Mini-DIV2K dataset layout and follows typical SISR challenge constraints:

- train from scratch on the provided training pairs only,
- no external data or pretrained weights,
- no model ensembling,
- **parameter limit**: `< 1,821,085` trainable parameters.

Dataset setup instructions live in `data/README.md`.

## 3. What’s Unique Here

- **Apple Silicon first**: device selection defaults to `mps` when available, otherwise CPU.
- **RAM-cached decoding**: the long training config can cache decoded PNGs (uint8) in memory to avoid repeatedly decoding 2K images, which is a major bottleneck on macOS.
- **Paired augmentation for SISR**: CutBlur-style patch replacement plus flips/rotations and RGB permutation to improve robustness without breaking LR/HR alignment.
- **Strict parameter check**: training refuses to run if the model exceeds the budget.

## 4. Method

Model:

- MSRResNet-style x4 network (EDSR/SRResNet family),
- 20 residual blocks, 64 channels,
- PixelShuffle upsampling,
- residual learning with a bilinear-upsampled base.

Training:

- loss: Charbonnier (smooth L1-like),
- optimizer: Adam + cosine annealing,
- gradient clipping,
- EMA tracking for checkpoint selection.

Augmentations (configurable):

- random horizontal/vertical flip,
- random 90-degree rotations,
- RGB channel permutation,
- CutBlur-style LR/HR patch mixing.

## 5. Results Snapshot

Current reference run (example):

- validation PSNR: **~28.36 dB** at **10k** iterations (Mini-DIV2K val, x4, RGB, crop border 4)
- parameters: **1,812,995** (under the `< 1,821,085` limit)

Exact PSNR may vary with hardware, PyTorch version, seeds, and training length.

The latest Markdown report lives at `report/project2_report.md` (anonymized).

## 6. How To Run

### Install

1. Install PyTorch for your platform by following the official instructions:

- [PyTorch install instructions](https://pytorch.org/get-started/locally/)

2. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### Prepare Data

Follow `data/README.md` to place the dataset under `data/`.

### Dataset Download Links

- challenge package (train/val/test): [Dropbox dataset package](https://www.dropbox.com/scl/fo/f88w72e55xuy5ofjup77b/AGVFqg57pq_AmZwzbg5_RnQ?rlkey=lyph59zpbdkohlqx0i0zg9suc&dl=0)
- official reference: [DIV2K official site](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- local placement details: see `data/README.md` (train: `data/Mini-DIV2K/Train/{HR,LR_x4}`, val: `data/Mini-DIV2K/Val/{HR,LR_x4}`, test LR: `data/test/LR`)

### Train / Eval / Inference

Quick smoke run:

```bash
python scripts/check_env.py
python scripts/train.py --config configs/smoke.yaml --device auto
```

Score-focused training:

```bash
python scripts/train.py --config configs/long.yaml --device auto
python scripts/eval.py --config configs/long.yaml --checkpoint checkpoints/best.pth --device auto
```

Generate x4 outputs for a test LR folder (expects `0001.png` ... `0080.png`):

```bash
python scripts/infer_testset.py --config configs/long.yaml --checkpoint checkpoints/best.pth --device auto
```

Generate a Markdown report (no PDF toolchain required):

```bash
python scripts/generate_report.py --config configs/long.yaml --md-only
```

### Quick Quality Checks

```bash
python scripts/check_env.py
python -m pytest -q
```

## 7. Repository Layout

- `src/project2/`: datasets, model, training loop, metrics, inference
- `scripts/`: CLI entry points
- `configs/`: run presets (smoke, quick, long)
- `data/`: dataset instructions (dataset files are git-ignored)
- `report/`: anonymized Markdown report template/output

## 8. References

- DIV2K dataset: E. Agustsson and R. Timofte, NTIRE 2017 SISR challenge.
- EDSR: Lim et al., Enhanced Deep Residual Networks for Single Image Super-Resolution.
- SRResNet/SRGAN: Ledig et al., Photo-Realistic Single Image Super-Resolution Using a GAN.
- CutBlur: Yoo et al., Rethinking Data Augmentation for Image Super-Resolution.
