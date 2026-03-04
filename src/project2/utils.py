from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(prefer: str = "auto") -> torch.device:
    prefer = prefer.lower()
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def load_image_uint8_rgb(path: str | Path) -> np.ndarray:
    """Load an RGB image as uint8 HWC.

    Uses OpenCV if available (often faster on macOS), otherwise falls back to PIL.
    """
    path = str(path)
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cv2.imread returned None for {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb)
    except Exception:
        img = Image.open(path).convert("RGB")
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def load_image_rgb(path: str | Path) -> torch.Tensor:
    arr_u8 = load_image_uint8_rgb(path)
    arr = arr_u8.astype(np.float32) / 255.0
    # HWC -> CHW
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image_rgb(t: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = t.detach().cpu().clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def json_dump(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"
