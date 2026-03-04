from __future__ import annotations

import math

import torch


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, crop_border: int = 4) -> float:
    """Calculate PSNR on [0, 1] tensors with shape [C, H, W]."""
    pred = pred.detach().clamp(0.0, 1.0)
    target = target.detach().clamp(0.0, 1.0)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")

    if crop_border > 0:
        pred = pred[:, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)
