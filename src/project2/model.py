from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_init_weights(module_list, scale: float = 1.0) -> None:
    if not isinstance(module_list, (list, tuple)):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat: int = 64, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = float(res_scale)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class MSRResNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 20,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        if upscale != 4:
            raise ValueError("This project implementation targets x4 only.")

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResidualBlockNoBN(num_feat=num_feat) for _ in range(num_block)]
        )
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        default_init_weights(
            [self.conv_first, self.upconv1, self.upconv2, self.conv_hr, self.conv_last],
            0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(
            x, scale_factor=4, mode="bilinear", align_corners=False
        )
        out = out + base
        return out


def build_model(cfg: dict) -> MSRResNet:
    model_cfg = cfg.get("model", {})
    return MSRResNet(
        num_in_ch=int(model_cfg.get("num_in_ch", 3)),
        num_out_ch=int(model_cfg.get("num_out_ch", 3)),
        num_feat=int(model_cfg.get("num_feat", 64)),
        num_block=int(model_cfg.get("num_block", 20)),
        upscale=int(model_cfg.get("scale", 4)),
    )
