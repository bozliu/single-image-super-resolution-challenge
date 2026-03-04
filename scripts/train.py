#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project2.config import load_config
from project2.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Project 2 SISR model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(
        cfg,
        device_preference=args.device,
        seed=args.seed,
        max_iters_override=args.max_iters,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
