from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import load_image_rgb, load_image_uint8_rgb


def _list_pngs(path: str | Path) -> list[Path]:
    return sorted(Path(path).glob("*.png"))


def _name_map(path: str | Path) -> dict[str, Path]:
    return {p.name: p for p in _list_pngs(path)}


class PairedSISRDataset(Dataset):
    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        scale: int = 4,
        train: bool = True,
        patch_size: int = 128,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        rgb_perm_prob: float = 0.0,
        cutblur_prob: float = 0.0,
        cutblur_alpha: float = 0.5,
        cache_images: bool = False,
    ) -> None:
        self.hr_map = _name_map(hr_dir)
        self.lr_map = _name_map(lr_dir)
        self.names = sorted(set(self.hr_map.keys()) & set(self.lr_map.keys()))
        if not self.names:
            raise RuntimeError(f"No paired PNG files found between {hr_dir} and {lr_dir}.")

        self.scale = scale
        self.train = train
        self.patch_size = patch_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.rgb_perm_prob = rgb_perm_prob
        self.cutblur_prob = cutblur_prob
        self.cutblur_alpha = cutblur_alpha
        self.cache_images = bool(cache_images)

        self._hr_cache: dict[str, np.ndarray] | None = None
        self._lr_cache: dict[str, np.ndarray] | None = None
        if self.cache_images:
            # Cache decoded uint8 arrays to avoid re-reading 2K PNGs every iteration.
            # This is much faster on macOS but uses more RAM (~7GB for the full train set).
            self._hr_cache = {n: load_image_uint8_rgb(self.hr_map[n]) for n in self.names}
            self._lr_cache = {n: load_image_uint8_rgb(self.lr_map[n]) for n in self.names}

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _u8_to_tensor(arr_u8: np.ndarray) -> torch.Tensor:
        # HWC uint8 -> CHW float32 in [0, 1]
        arr = arr_u8.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _random_crop(self, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_lr, w_lr = lr.shape
        lq_patch = self.patch_size // self.scale
        if h_lr < lq_patch or w_lr < lq_patch:
            raise ValueError(
                f"LR image too small for patch. Need >= {lq_patch} got {(h_lr, w_lr)}"
            )
        top = random.randint(0, h_lr - lq_patch)
        left = random.randint(0, w_lr - lq_patch)

        lr = lr[:, top : top + lq_patch, left : left + lq_patch]
        top_hr = top * self.scale
        left_hr = left * self.scale
        hr = hr[:, top_hr : top_hr + self.patch_size, left_hr : left_hr + self.patch_size]
        return lr, hr

    def _random_crop_u8(self, lr_u8: np.ndarray, hr_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h_lr, w_lr, _ = lr_u8.shape
        lq_patch = self.patch_size // self.scale
        if h_lr < lq_patch or w_lr < lq_patch:
            raise ValueError(
                f"LR image too small for patch. Need >= {lq_patch} got {(h_lr, w_lr)}"
            )
        top = random.randint(0, h_lr - lq_patch)
        left = random.randint(0, w_lr - lq_patch)
        lr_patch = lr_u8[top : top + lq_patch, left : left + lq_patch, :]

        top_hr = top * self.scale
        left_hr = left * self.scale
        hr_patch = hr_u8[
            top_hr : top_hr + self.patch_size,
            left_hr : left_hr + self.patch_size,
            :,
        ]
        return lr_patch, hr_patch

    def _augment(self, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.hflip_prob:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        if random.random() < self.vflip_prob:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        if random.random() < self.rot90_prob:
            lr = lr.transpose(1, 2)
            hr = hr.transpose(1, 2)

        if random.random() < self.rgb_perm_prob:
            perm = torch.randperm(3)
            lr = lr[perm]
            hr = hr[perm]

        if random.random() < self.cutblur_prob:
            lr, hr = self._cutblur(lr, hr)

        return lr.contiguous(), hr.contiguous()

    def _cutblur(self, lr: torch.Tensor, hr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h_lr, w_lr = lr.shape
        ratio = min(max(abs(random.gauss(self.cutblur_alpha, 0.05)), 0.08), 1.0)
        patch_h = max(1, int(h_lr * ratio))
        patch_w = max(1, int(w_lr * ratio))
        y = random.randint(0, h_lr - patch_h)
        x = random.randint(0, w_lr - patch_w)

        y_hr = y * self.scale
        x_hr = x * self.scale
        ph_hr = patch_h * self.scale
        pw_hr = patch_w * self.scale

        if random.random() < 0.5:
            # Replace HR patch with upsampled LR patch.
            lr_up = F.interpolate(
                lr.unsqueeze(0),
                scale_factor=self.scale,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            hr[:, y_hr : y_hr + ph_hr, x_hr : x_hr + pw_hr] = lr_up[
                :, y_hr : y_hr + ph_hr, x_hr : x_hr + pw_hr
            ]
        else:
            # Replace LR patch with downsampled HR patch.
            hr_patch = hr[:, y_hr : y_hr + ph_hr, x_hr : x_hr + pw_hr].unsqueeze(0)
            lr_patch = F.interpolate(
                hr_patch,
                size=(patch_h, patch_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            lr[:, y : y + patch_h, x : x + patch_w] = lr_patch

        return lr, hr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        name = self.names[idx]
        if self.cache_images:
            assert self._lr_cache is not None
            assert self._hr_cache is not None
            lr_u8 = self._lr_cache[name]
            hr_u8 = self._hr_cache[name]
            h_lr, w_lr, _ = lr_u8.shape
            h_hr, w_hr, _ = hr_u8.shape
            if h_hr != h_lr * self.scale or w_hr != w_lr * self.scale:
                raise ValueError(
                    f"Scale mismatch for {name}: LR {(h_lr,w_lr)} vs HR {(h_hr,w_hr)} for scale {self.scale}"
                )

            if self.train:
                lr_patch_u8, hr_patch_u8 = self._random_crop_u8(lr_u8, hr_u8)
                lr = self._u8_to_tensor(lr_patch_u8)
                hr = self._u8_to_tensor(hr_patch_u8)
                lr, hr = self._augment(lr, hr)
                return {"lr": lr, "hr": hr, "name": name}

            lr = self._u8_to_tensor(lr_u8)
            hr = self._u8_to_tensor(hr_u8)
        else:
            lr = load_image_rgb(self.lr_map[name])
            hr = load_image_rgb(self.hr_map[name])

        _, h_lr, w_lr = lr.shape
        _, h_hr, w_hr = hr.shape
        if h_hr != h_lr * self.scale or w_hr != w_lr * self.scale:
            raise ValueError(
                f"Scale mismatch for {name}: LR {(h_lr,w_lr)} vs HR {(h_hr,w_hr)} for scale {self.scale}"
            )

        if self.train:
            lr, hr = self._random_crop(lr, hr)
            lr, hr = self._augment(lr, hr)

        return {"lr": lr, "hr": hr, "name": name}


class SingleImageDataset(Dataset):
    def __init__(self, lr_dir: str) -> None:
        self.paths = _list_pngs(lr_dir)
        if not self.paths:
            raise RuntimeError(f"No PNG files found in {lr_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        p = self.paths[idx]
        return {"lr": load_image_rgb(p), "name": p.name}
