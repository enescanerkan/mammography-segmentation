"""The plain U-Net (Ronneberger et al., 2015) and its checkpoint helpers, for inference.

Double 3x3 conv + BatchNorm + ReLU, max-pool down, transposed-conv up, concatenated skips; no attention, no residuals,
no pre-training. It lives in the inference package because the desktop build leaves ``breast_seg.training`` out
(:mod:`breast_seg.training.nets` re-exports it for training). Imported only when a ``framework: unet`` model is
loaded.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """(3x3 conv -> BN -> ReLU) x 2."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _widths(base: int) -> List[int]:
    return [base, base * 2, base * 4, base * 8]


def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class UNet(nn.Module):
    """Four downsamplings, a bottleneck, four upsamplings with skip concatenation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 64):
        super().__init__()
        g = _widths(base_channels)
        self.down = nn.ModuleList()
        prev = in_channels
        for k in g:
            self.down.append(DoubleConv(prev, k))
            prev = k
        self.pool = nn.MaxPool2d(2)
        self.bottom = DoubleConv(g[-1], g[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = g[-1] * 2
        for k in reversed(g):
            self.up.append(nn.ConvTranspose2d(prev, k, 2, stride=2))
            self.dec.append(DoubleConv(k * 2, k))
            prev = k
        self.head = nn.Conv2d(g[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, block, s in zip(self.up, self.dec, reversed(skips)):
            x = _align(up(x), s)
            x = block(torch.cat([s, x], dim=1))
        return self.head(x)


#: Parameter-name prefixes of the released U-Net checkpoint (``segmentation/unet.pt``) -> the names used here.
CHECKPOINT_PREFIXES = (("inis.", "down."), ("dip.", "bottom."), ("yukselt.", "up."), ("cikis.", "dec."),
                   ("bas.", "head."))


def upgrade_state_dict(state: dict) -> dict:
    """Rename the parameters of the released checkpoint so it loads into :class:`UNet` (other names pass through)."""
    out = {}
    for key, value in state.items():
        for old, new in CHECKPOINT_PREFIXES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    return out


def load_checkpoint(path) -> dict:
    """``torch.load`` restricted to tensors and plain values (the checkpoints written here hold nothing else)."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(str(path), map_location="cpu")
