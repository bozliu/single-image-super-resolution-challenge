from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import PairedSISRDataset
from .ema import ExponentialMovingAverage
from .losses import CharbonnierLoss
from .metrics import calculate_psnr
from .model import build_model
from .utils import count_parameters, ensure_dir, json_dump, now_ts, save_image_rgb, seed_everything, select_device

PARAMETER_LIMIT = 1_821_085


def create_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    cache_images = bool(data_cfg.get("cache_images", False))

    train_set = PairedSISRDataset(
        hr_dir=data_cfg["train_hr_dir"],
        lr_dir=data_cfg["train_lr_dir"],
        scale=int(data_cfg.get("scale", 4)),
        train=True,
        patch_size=int(data_cfg.get("patch_size", 128)),
        hflip_prob=float(aug_cfg.get("hflip_prob", 0.5)),
        vflip_prob=float(aug_cfg.get("vflip_prob", 0.5)),
        rot90_prob=float(aug_cfg.get("rot90_prob", 0.5)),
        rgb_perm_prob=float(aug_cfg.get("rgb_perm_prob", 0.0)),
        cutblur_prob=float(aug_cfg.get("cutblur_prob", 0.0)),
        cutblur_alpha=float(aug_cfg.get("cutblur_alpha", 0.4)),
        cache_images=cache_images,
    )

    val_set = PairedSISRDataset(
        hr_dir=data_cfg["val_hr_dir"],
        lr_dir=data_cfg["val_lr_dir"],
        scale=int(data_cfg.get("scale", 4)),
        train=False,
        patch_size=int(data_cfg.get("patch_size", 128)),
        cache_images=cache_images,
    )

    workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader = DataLoader(
        train_set,
        batch_size=int(data_cfg.get("train_batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=int(data_cfg.get("val_batch_size", 1)),
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    crop_border: int = 4,
    sample_dir: str | Path | None = None,
    sample_limit: int = 4,
    max_images: int | None = None,
) -> float:
    model.eval()
    psnrs: list[float] = []
    saved = 0

    if sample_dir is not None:
        sample_dir = ensure_dir(sample_dir)

    seen = 0
    for batch in val_loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        names = batch["name"]

        sr = model(lr).clamp(0.0, 1.0)
        for i in range(sr.shape[0]):
            psnrs.append(calculate_psnr(sr[i], hr[i], crop_border=crop_border))
            if sample_dir is not None and saved < sample_limit:
                save_image_rgb(sr[i].cpu(), sample_dir / f"{names[i]}_sr.png")
                save_image_rgb(hr[i].cpu(), sample_dir / f"{names[i]}_hr.png")
                saved += 1
            seen += 1
            if max_images is not None and seen >= max_images:
                break
        if max_images is not None and seen >= max_images:
            break

    if not psnrs:
        return 0.0
    return float(sum(psnrs) / len(psnrs))


def _save_checkpoint(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ckpt_path: str | Path,
    step: int,
    best_psnr: float,
) -> None:
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        ckpt_path,
    )


def load_checkpoint_state(checkpoint: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(checkpoint, map_location=map_location)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {checkpoint}")
    return state


def load_weights(model: torch.nn.Module, checkpoint: str | Path, prefer_ema: bool = True) -> dict[str, Any]:
    state = load_checkpoint_state(checkpoint, map_location="cpu")
    key = "ema_model" if prefer_ema and "ema_model" in state else "model"
    model.load_state_dict(state[key], strict=True)
    return state


def train(
    cfg: dict[str, Any],
    device_preference: str = "auto",
    seed: int | None = None,
    max_iters_override: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    runtime_cfg = cfg.get("runtime", {})
    seed = int(runtime_cfg.get("seed", 3407) if seed is None else seed)
    seed_everything(seed, deterministic=bool(runtime_cfg.get("deterministic", False)))

    device = select_device(device_preference if device_preference else runtime_cfg.get("device", "auto"))
    train_loader, val_loader = create_dataloaders(cfg)

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    if n_params >= PARAMETER_LIMIT:
        raise RuntimeError(
            f"Parameter count {n_params} violates limit < {PARAMETER_LIMIT}."
        )

    ema_decay = float(cfg.get("schedule", {}).get("ema_decay", 0.999))
    ema = ExponentialMovingAverage(model, decay=ema_decay)
    ema.model.to(device)

    criterion = CharbonnierLoss(eps=float(cfg.get("loss", {}).get("charbonnier_eps", 1e-3)))
    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 2e-4)),
        betas=tuple(float(x) for x in optim_cfg.get("betas", [0.9, 0.99])),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    sched_cfg = cfg.get("schedule", {})
    total_iters = int(sched_cfg.get("total_iters", 12_000))
    if max_iters_override is not None:
        total_iters = int(max_iters_override)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iters),
        eta_min=float(sched_cfg.get("eta_min", 1e-7)),
    )

    val_interval = int(sched_cfg.get("val_interval", 500))
    log_interval = int(sched_cfg.get("log_interval", 20))
    save_interval = int(sched_cfg.get("save_interval", 500))
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))
    crop_border = int(cfg.get("metrics", {}).get("crop_border", 4))
    max_val_images = sched_cfg.get("val_max_images")
    if max_val_images is not None:
        max_val_images = int(max_val_images)

    exp_name = run_name or cfg.get("experiment", {}).get("name", "run")
    result_root = ensure_dir(Path(cfg["run_dirs"]["results_dir"]) / exp_name)
    logs_path = result_root / "train_log.jsonl"

    ckpt_dir = ensure_dir(cfg["run_dirs"]["checkpoints_dir"])
    best_ckpt = ckpt_dir / "best.pth"
    latest_ckpt = ckpt_dir / "latest.pth"

    data_iter = iter(train_loader)
    best_psnr = -1.0
    start = time.time()
    history: list[dict[str, Any]] = []

    for step in range(1, total_iters + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        model.train()
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        optimizer.zero_grad(set_to_none=True)
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if step % log_interval == 0 or step == 1:
            row = {
                "time": now_ts(),
                "step": step,
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": float(time.time() - start),
            }
            history.append(row)
            with logs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[train] step={step}/{total_iters} loss={row['loss']:.6f} "
                f"lr={row['lr']:.3e} elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )

        if step % val_interval == 0 or step == total_iters:
            _save_checkpoint(model, ema.model, optimizer, scheduler, latest_ckpt, step, best_psnr)
            sample_dir = result_root / "val_samples" / f"step_{step:06d}"
            val_psnr = validate(
                ema.model,
                val_loader,
                device=device,
                crop_border=crop_border,
                sample_dir=sample_dir,
                sample_limit=4,
                max_images=max_val_images,
            )
            print(f"[val] step={step} psnr={val_psnr:.4f}", flush=True)
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                _save_checkpoint(model, ema.model, optimizer, scheduler, best_ckpt, step, best_psnr)

        if step % save_interval == 0:
            _save_checkpoint(model, ema.model, optimizer, scheduler, latest_ckpt, step, best_psnr)

    summary = {
        "experiment": exp_name,
        "device": str(device),
        "seed": seed,
        "total_iters": total_iters,
        "param_count": n_params,
        "param_limit": PARAMETER_LIMIT,
        "best_psnr": float(best_psnr),
        "best_checkpoint": str(best_ckpt.resolve()),
        "latest_checkpoint": str(latest_ckpt.resolve()),
        "result_dir": str(result_root.resolve()),
        "finished_at": now_ts(),
        "elapsed_sec": float(time.time() - start),
    }
    json_dump(result_root / "summary.json", summary)
    json_dump(Path(cfg["run_dirs"]["results_dir"]) / "last_train_summary.json", summary)
    print(f"[done] best_psnr={best_psnr:.4f} checkpoint={best_ckpt}", flush=True)
    return summary
