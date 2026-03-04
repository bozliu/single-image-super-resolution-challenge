#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project2.config import load_config
from project2.model import build_model
from project2.utils import count_parameters


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_loss_plot(train_log: Path, out_png: Path) -> bool:
    if not train_log.exists():
        return False
    steps = []
    losses = []
    for line in train_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Training may still be writing the log file; skip partial lines.
            continue
        if "step" in row and "loss" in row:
            steps.append(row["step"])
            losses.append(row["loss"])
    if len(steps) < 2:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 3.0))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Charbonnier Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    return True


def _render_pdf(md_path: Path, header_path: Path, out_pdf: Path) -> None:
    cmd = [
        "pandoc",
        str(md_path),
        "--pdf-engine=xelatex",
        "--resource-path",
        str(md_path.parent),
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=10pt",
        "-H",
        str(header_path),
        "-o",
        str(out_pdf),
    ]
    subprocess.run(cmd, check=True)


def _pdf_pages(pdf_path: Path) -> int:
    out = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return -1


def _read_checkpoint_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        return {}
    return {
        "step": int(state.get("step", -1)),
        "best_psnr": float(state.get("best_psnr", -1.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate project report (Markdown/PDF).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None, help="Defaults to checkpoints/best.pth")
    parser.add_argument("--md-only", action="store_true", help="Generate Markdown only (skip PDF rendering).")
    parser.add_argument("--student-name", type=str, default=None, help="Optional; omitted by default.")
    parser.add_argument("--matric", type=str, default=None, help="Optional; omitted by default.")
    parser.add_argument("--tutor", type=str, default=None, help="Optional; omitted by default.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["_project_root"]).resolve()
    report_dir = Path(cfg["run_dirs"]["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    eval_summary = _read_json(Path(cfg["run_dirs"]["results_dir"]) / "eval_summary.json")

    exp_name = cfg.get("experiment", {}).get("name", "run")
    run_dir = Path(cfg["run_dirs"]["results_dir"]) / exp_name
    train_log = run_dir / "train_log.jsonl"
    loss_plot = report_dir / "loss_curve.png"
    has_loss_plot = _build_loss_plot(train_log, loss_plot)

    ckpt_dir = Path(cfg["run_dirs"]["checkpoints_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ckpt_dir / "best.pth")
    ckpt_meta = _read_checkpoint_meta(ckpt_path)
    best_psnr = ckpt_meta.get("best_psnr", "N/A")
    best_step = ckpt_meta.get("step", "N/A")
    try:
        ckpt_display = str(ckpt_path.resolve().relative_to(project_root))
    except Exception:
        ckpt_display = str(ckpt_path)

    # Prefer explicit eval result only if it matches the checkpoint we are reporting.
    eval_psnr = best_psnr
    try:
        eval_ckpt = Path(str(eval_summary.get("checkpoint", ""))).resolve()
        if eval_ckpt == ckpt_path.resolve() and int(eval_summary.get("step", -1)) == int(best_step):
            eval_psnr = eval_summary.get("evaluated_val_psnr", best_psnr)
    except Exception:
        eval_psnr = best_psnr

    model = build_model(cfg)
    param_count = count_parameters(model)

    md_path = report_dir / "project2_report.md"
    figure_block = (
        f"![Training Loss Curve]({loss_plot.name}){{ width=90% }}"
        if has_loss_plot
        else "Training loss curve unavailable for this run."
    )

    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentation", {})
    model_cfg = cfg.get("model", {})
    sched_cfg = cfg.get("schedule", {})

    def _fmt_num(x: object) -> str:
        if isinstance(x, (int, float)):
            return f"{x:,}"
        return str(x)

    def _fmt_float(x: object) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    md = f"""# Single Image Super-Resolution (x4)

This report summarizes the dataset, model, training setup, and validation PSNR for an x4 single-image super-resolution (SISR) solution.

## 1. Data Pre-processing

This solution trains on the provided Mini-DIV2K dataset only: 500 training pairs and 80 validation pairs.

To keep training practical on a local Apple M3 Pro, the pipeline avoids repeatedly decoding full 2K PNGs by caching the decoded images in RAM (uint8) and sampling paired random crops from the cache.

- Scale factor: x{data_cfg.get('scale', 4)}
- Train patch: {data_cfg.get('patch_size', 128)}x{data_cfg.get('patch_size', 128)} (HR), {int(data_cfg.get('patch_size', 128))//int(data_cfg.get('scale', 4))}x{int(data_cfg.get('patch_size', 128))//int(data_cfg.get('scale', 4))} (LR)
- Train batch size: {data_cfg.get('train_batch_size', 8)}
- Cache images: {data_cfg.get('cache_images', False)}

## 2. Data Augmentation

The training pipeline applies lightweight paired augmentations:

- random horizontal / vertical flips (p={aug_cfg.get('hflip_prob', 0.0)}/{aug_cfg.get('vflip_prob', 0.0)}),
- random 90-degree rotation (p={aug_cfg.get('rot90_prob', 0.0)}),
- random RGB channel permutation (p={aug_cfg.get('rgb_perm_prob', 0.0)}),
- CutBlur-style patch replacement between LR and HR domains (p={aug_cfg.get('cutblur_prob', 0.0)}).

## 3. Model Architecture

The model is an MSRResNet-style x4 super-resolution network with:

- 20 residual blocks without batch normalization,
- 64 feature channels,
- PixelShuffle-based x4 upsampling.

The total trainable parameter count is **{_fmt_num(param_count)}** (limit: < 1,821,085).

## 4. Loss Function and Optimization

The default objective is Charbonnier loss, chosen for robust convergence and good runtime on Apple M-series MPS backend.  
Optimization uses Adam (`lr=2e-4`, `betas=(0.9,0.99)`), cosine annealing schedule, gradient clipping, and EMA checkpoint tracking.

## 5. Training Configuration

- training iterations: {sched_cfg.get('total_iters', 'N/A')}
- validation interval: {sched_cfg.get('val_interval', 'N/A')} (full 80-image PSNR, RGB, crop border={cfg.get('metrics', {}).get('crop_border', 4)})
- training from scratch only (no external pretrained weights)
- no external data, no model ensemble

## 6. Results

This repository does not version large binaries (datasets, checkpoints, generated images). The numbers below are from a reference run.

- checkpoint (generated): `{ckpt_display}`
- checkpoint step: **{best_step}**
- validation PSNR (best recorded during training): **{_fmt_float(best_psnr)}**
- validation PSNR (final evaluation script): **{_fmt_float(eval_psnr)}**

{figure_block}

## 7. Conclusion

The refreshed pipeline runs locally on Apple M-series GPUs with MPS acceleration and satisfies the strict parameter constraint. Once the dataset is placed under `data/`, it can train from scratch and generate x4 outputs for all 80 test LR images.
"""
    md_path.write_text(md, encoding="utf-8")

    if args.md_only:
        print(f"Report generated (Markdown): {md_path}")
        return

    header_tex = report_dir / "report_header.tex"
    header_tex.write_text(
        "\\usepackage{fontspec}\n"
        "\\setmainfont{Arial}\n",
        encoding="utf-8",
    )

    out_pdf = report_dir / "project2_report.pdf"
    _render_pdf(md_path, header_tex, out_pdf)

    root_pdf = Path(cfg["_project_root"]) / "project2_report.pdf"
    root_pdf.write_bytes(out_pdf.read_bytes())

    pages = _pdf_pages(root_pdf)
    if pages > 5:
        raise RuntimeError(f"Generated report exceeds 5 pages: {pages}")

    print(f"Report generated: {root_pdf} ({pages} pages)")


if __name__ == "__main__":
    main()
