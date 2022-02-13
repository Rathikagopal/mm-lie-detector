# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch.nn as nn

from mmcv.cnn import trunc_normal_init

from ..registry import registry


@registry.register_module()
class TimeSformerHead(nn.Module):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        init_std (float): Std value for Initiation. Defaults to 0.02.
    """

    def __init__(self, num_classes: int, in_channels: int = 768, init_std: float = 0.02, **kwargs) -> None:
        super().__init__()

        self.init_std = init_std
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x) -> nn.Module:
        # [N, in_channels]
        cls_score = self.fc_cls(x)

        # [N, num_classes]
        return cls_score
