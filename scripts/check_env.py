#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path

import torch


def _check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Project 2 runtime environment.")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if datasets are missing/mismatched (default: warn only).",
    )
    parser.add_argument(
        "--create-run-dirs",
        action="store_true",
        help="Create checkpoints/results/output directories while checking writability.",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    required_modules = [
        "torch",
        "cv2",
        "yaml",
        "numpy",
        "PIL",
        "tqdm",
        "matplotlib",
        "pandas",
    ]

    print(f"Project root: {root}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    ok = True
    for m in required_modules:
        exists = _check_module(m)
        print(f"module {m}: {'OK' if exists else 'MISSING'}")
        ok = ok and exists

    checks = {
        "train_hr": root / "data/Mini-DIV2K/Train/HR",
        "train_lr": root / "data/Mini-DIV2K/Train/LR_x4",
        "val_hr": root / "data/Mini-DIV2K/Val/HR",
        "val_lr": root / "data/Mini-DIV2K/Val/LR_x4",
        "test_lr": root / "data/test/LR",
    }

    expected_counts = {
        "train_hr": 500,
        "train_lr": 500,
        "val_hr": 80,
        "val_lr": 80,
        "test_lr": 80,
    }

    dataset_ok = True
    for key, path in checks.items():
        cnt = len(list(path.glob("*.png"))) if path.exists() else 0
        exp = expected_counts[key]
        hit = cnt == exp

        if hit:
            status = "OK"
        else:
            status = "MISSING" if not path.exists() else "MISMATCH"
            dataset_ok = False

        print(f"dataset {key}: {path} -> {cnt} png (expected {exp}) {status}")

    if not dataset_ok:
        msg = (
            "Dataset files are missing or incomplete. For setup instructions, see:\n"
            f"  {root / 'data' / 'README.md'}"
        )
        if args.strict:
            print(msg)
            ok = False
        else:
            print("WARNING: " + msg)

    for d in [root / "checkpoints", root / "results", root / "Generated upscaled images from testset"]:
        if args.create_run_dirs:
            d.mkdir(parents=True, exist_ok=True)
        if d.exists():
            probe_dir = d
        else:
            probe_dir = d.parent
            print(f"writable {d}: NOT_CREATED (pass --create-run-dirs to create it)")

        probe = probe_dir / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            print(f"writable {probe_dir}: OK")
        except Exception as e:  # noqa: BLE001
            print(f"writable {probe_dir}: FAIL ({e})")
            ok = False

    if not ok:
        raise SystemExit(1)

    print("Environment check passed.")


if __name__ == "__main__":
    main()
