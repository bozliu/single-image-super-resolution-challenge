#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project2.config import load_config
from project2.dataset import PairedSISRDataset
from project2.model import build_model
from project2.trainer import PARAMETER_LIMIT, load_weights, validate
from project2.utils import count_parameters, json_dump, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSNR on validation set.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(args.device)

    model = build_model(cfg).to(device)
    params = count_parameters(model)
    if params >= PARAMETER_LIMIT:
        raise RuntimeError(f"Model params {params} violate limit < {PARAMETER_LIMIT}")

    state = load_weights(model, args.checkpoint, prefer_ema=True)

    data_cfg = cfg["data"]
    val_set = PairedSISRDataset(
        hr_dir=data_cfg["val_hr_dir"],
        lr_dir=data_cfg["val_lr_dir"],
        scale=int(data_cfg.get("scale", 4)),
        train=False,
        cache_images=bool(data_cfg.get("cache_images", False)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(data_cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=max(0, int(data_cfg.get("num_workers", 2)) // 2),
    )

    psnr = validate(
        model,
        val_loader,
        device=device,
        crop_border=int(cfg.get("metrics", {}).get("crop_border", 4)),
        sample_dir=None,
        max_images=args.max_images,
    )

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": int(state.get("step", -1)),
        "best_psnr_in_checkpoint": float(state.get("best_psnr", -1.0)),
        "evaluated_val_psnr": float(psnr),
        "param_count": int(params),
        "param_limit": int(PARAMETER_LIMIT),
        "device": str(device),
    }

    out = Path(cfg["run_dirs"]["results_dir"]) / "eval_summary.json"
    json_dump(out, result)
    print(f"Validation PSNR: {psnr:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
