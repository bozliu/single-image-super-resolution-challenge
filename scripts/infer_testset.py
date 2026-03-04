#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project2.config import load_config
from project2.inference import infer_folder
from project2.model import build_model
from project2.trainer import load_weights
from project2.utils import select_device


NAME_RE = re.compile(r"^\d{4}\.png$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run testset inference for Project 2.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--tile", type=int, default=None)
    parser.add_argument("--tile-pad", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device)

    input_dir = Path(args.input_dir or cfg["data"]["test_lr_dir"])
    output_dir = Path(args.output_dir or cfg["run_dirs"]["outputs_dir"])
    scale = int(cfg["data"].get("scale", 4))

    tile_cfg = cfg.get("inference", {})
    tile = int(args.tile if args.tile is not None else tile_cfg.get("tile", 0))
    tile_pad = int(args.tile_pad if args.tile_pad is not None else tile_cfg.get("tile_pad", 16))

    model = build_model(cfg).to(device)
    load_weights(model, args.checkpoint, prefer_ema=True)

    names = sorted([p.name for p in input_dir.glob("*.png") if NAME_RE.match(p.name)])
    if len(names) != 80:
        raise RuntimeError(f"Expected 80 test images in {input_dir}, got {len(names)}")

    saved = infer_folder(
        model,
        input_dir=input_dir,
        output_dir=output_dir,
        device=device,
        tile=tile,
        tile_pad=tile_pad,
        scale=scale,
        names=names,
    )

    # Sanity checks.
    if len(saved) != 80:
        raise RuntimeError(f"Expected 80 outputs, got {len(saved)}")

    for name in saved:
        in_size = Image.open(input_dir / name).size
        out_size = Image.open(output_dir / name).size
        if out_size[0] != in_size[0] * scale or out_size[1] != in_size[1] * scale:
            raise RuntimeError(f"Size mismatch for {name}: {in_size} -> {out_size}")

    print(f"Saved {len(saved)} images to {output_dir}")


if __name__ == "__main__":
    main()
