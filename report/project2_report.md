# Single Image Super-Resolution (x4)

This report summarizes the dataset, model, training setup, and validation PSNR for an x4 single-image super-resolution (SISR) solution.

## 1. Data Pre-processing

This solution trains on the provided Mini-DIV2K dataset only: 500 training pairs and 80 validation pairs.

To keep training practical on a local Apple M3 Pro, the pipeline avoids repeatedly decoding full 2K PNGs by caching the decoded images in RAM (uint8) and sampling paired random crops from the cache.

- Scale factor: x4
- Train patch: 128x128 (HR), 32x32 (LR)
- Train batch size: 8
- Cache images: True

## 2. Data Augmentation

The training pipeline applies lightweight paired augmentations:

- random horizontal / vertical flips (p=0.5/0.5),
- random 90-degree rotation (p=0.5),
- random RGB channel permutation (p=0.35),
- CutBlur-style patch replacement between LR and HR domains (p=0.35).

## 3. Model Architecture

The model is an MSRResNet-style x4 super-resolution network with:

- 20 residual blocks without batch normalization,
- 64 feature channels,
- PixelShuffle-based x4 upsampling.

The total trainable parameter count is **1,812,995** (limit: < 1,821,085).

## 4. Loss Function and Optimization

The default objective is Charbonnier loss, chosen for robust convergence and good runtime on Apple M-series MPS backend.  
Optimization uses Adam (`lr=2e-4`, `betas=(0.9,0.99)`), cosine annealing schedule, gradient clipping, and EMA checkpoint tracking.

## 5. Training Configuration

- training iterations: 80000
- validation interval: 10000 (full 80-image PSNR, RGB, crop border=4)
- training from scratch only (no external pretrained weights)
- no external data, no model ensemble

## 6. Results

This repository does not version large binaries (datasets, checkpoints, generated images). The numbers below are from a reference run.

- checkpoint (generated): `checkpoints/best.pth`
- checkpoint step: **10000**
- validation PSNR (best recorded during training): **28.3595**
- validation PSNR (final evaluation script): **28.3595**

## 7. Conclusion

The refreshed pipeline runs locally on Apple M-series GPUs with MPS acceleration and satisfies the strict parameter constraint. Once the dataset is placed under `data/`, it can train from scratch and generate x4 outputs for all 80 test LR images.
