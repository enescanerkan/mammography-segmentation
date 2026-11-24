"""Network definitions of the from-scratch PyTorch families (imported only when such a model is built).

* :class:`UNet` - the plain U-Net (Ronneberger et al., 2015), defined in :mod:`breast_seg.inference.unet` (the
  desktop build leaves this training package out) and re-exported here with its checkpoint helpers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from breast_seg.inference.unet import (  # noqa: F401 - re-exported
    CHECKPOINT_PREFIXES,
    DoubleConv,
    UNet,
    load_checkpoint,
    upgrade_state_dict,
)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over the foreground classes, pooled over the batch (class 0 = background is excluded)."""
    prob = logits.softmax(1)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)[1:]
    denom = prob.sum(dims)[1:] + oh.sum(dims)[1:]
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()
