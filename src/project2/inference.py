from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .utils import load_image_rgb, save_image_rgb


@torch.no_grad()
def run_model(
    model: torch.nn.Module,
    lr: torch.Tensor,
    tile: int = 0,
    tile_pad: int = 16,
    scale: int = 4,
) -> torch.Tensor:
    if tile <= 0:
        return model(lr.unsqueeze(0)).squeeze(0)

    c, h, w = lr.shape
    tile = min(tile, h, w)
    stride = max(1, tile - tile_pad * 2)

    out = torch.zeros(c, h * scale, w * scale, device=lr.device)
    weight = torch.zeros_like(out)

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))

    for y in ys:
        for x in xs:
            y0 = max(y - tile_pad, 0)
            x0 = max(x - tile_pad, 0)
            y1 = min(y + tile + tile_pad, h)
            x1 = min(x + tile + tile_pad, w)

            patch = lr[:, y0:y1, x0:x1].unsqueeze(0)
            patch_sr = model(patch).squeeze(0)

            oy0 = y0 * scale
            ox0 = x0 * scale
            oy1 = y1 * scale
            ox1 = x1 * scale

            cy0 = (y - y0) * scale
            cx0 = (x - x0) * scale
            cy1 = cy0 + min(tile, h - y) * scale
            cx1 = cx0 + min(tile, w - x) * scale

            ty0 = y * scale
            tx0 = x * scale
            ty1 = ty0 + min(tile, h - y) * scale
            tx1 = tx0 + min(tile, w - x) * scale

            out[:, ty0:ty1, tx0:tx1] += patch_sr[:, cy0:cy1, cx0:cx1]
            weight[:, ty0:ty1, tx0:tx1] += 1.0

    return out / weight.clamp_min(1e-8)


@torch.no_grad()
def infer_folder(
    model: torch.nn.Module,
    input_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    tile: int = 0,
    tile_pad: int = 16,
    scale: int = 4,
    names: Iterable[str] | None = None,
) -> list[str]:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if names is None:
        image_paths = sorted(in_dir.glob("*.png"))
    else:
        image_paths = [in_dir / n for n in names]

    saved: list[str] = []
    model.eval()
    for p in image_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing input image: {p}")
        lr = load_image_rgb(p).to(device)
        sr = run_model(model, lr, tile=tile, tile_pad=tile_pad, scale=scale)
        save_image_rgb(sr, out_dir / p.name)
        saved.append(p.name)
    return saved
