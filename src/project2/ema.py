from __future__ import annotations

from copy import deepcopy

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_model: torch.nn.Module) -> None:
        ema_params = dict(self.model.named_parameters())
        model_params = dict(online_model.named_parameters())

        for k, v in ema_params.items():
            v.mul_(self.decay).add_(model_params[k].detach(), alpha=1.0 - self.decay)

        ema_buffers = dict(self.model.named_buffers())
        model_buffers = dict(online_model.named_buffers())
        for k, v in ema_buffers.items():
            v.copy_(model_buffers[k])
