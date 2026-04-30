from typing import Literal

import torch
from torch import nn

NormType = Literal["batch", "instance", "layer", "none"]


def _make_norm(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm_type == "layer":
        return nn.GroupNorm(1, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class CSPup(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: NormType = "batch") -> None:
        super().__init__()
        if in_channels < 2:
            raise ValueError("in_channels must be at least 2")

        left_channels = in_channels // 2
        right_channels = in_channels - left_channels
        hidden_channels = max(out_channels // 2, 1)

        self.left = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(left_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(hidden_channels, norm_type),
            nn.ReLU(True),
        )
        self.right = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(right_channels, hidden_channels, kernel_size=1, bias=False),
            _make_norm(hidden_channels, norm_type),
            nn.ReLU(True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = torch.chunk(x, 2, dim=1)
        return self.fuse(torch.cat([self.left(left), self.right(right)], dim=1))


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        base_channels: int = 64,
        image_channels: int = 3,
        norm_type: NormType = "batch",
    ) -> None:
        super().__init__()
        if base_channels <= 0 or base_channels % 2 != 0:
            raise ValueError("base_channels must be positive and even")
        _make_norm(1, norm_type)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, kernel_size=4, stride=1, padding=0, bias=False),
            _make_norm(base_channels * 8, norm_type),
            nn.ReLU(True),
            CSPup(base_channels * 8, base_channels * 4, norm_type),
            CSPup(base_channels * 4, base_channels * 2, norm_type),
            CSPup(base_channels * 2, base_channels, norm_type),
            CSPup(base_channels, base_channels, norm_type),
            nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base_channels: int = 64, image_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), 1)


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname or "InstanceNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
