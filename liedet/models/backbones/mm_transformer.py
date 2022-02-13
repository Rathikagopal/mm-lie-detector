from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..registry import build, registry


@registry.register_module()
class AttentionBottleneckTransformer(nn.Module):
    def __init__(
        self,
        transformers: list[dict[str, Any]] | tuple[dict[str, Any]],
        embed_dims: int = 768,
        neck_size: int = 4,
        cls_only: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.transformers = [build(cfg=transformer) for transformer in transformers]
        self.bottleneck = nn.Parameter(data=torch.zeros(1, neck_size, embed_dims))

        self.cls_only = cls_only

    def forward(self, *per_transformer_x):
        batch_size = per_transformer_x[0].size(0)

        shared_neck = self.bottleneck.expand(batch_size, -1, -1)
        next_shared_neck = torch.zeros(shared_neck.size())

        for x, transformer in zip(per_transformer_x, self.transformers):
            x = torch.cat((x, shared_neck), dim=1)
            x = transformer(x)
            next_shared_neck += x[:, -shared_neck.size(1) :]

            if self.cls_only:
                x = x[:, 0]
            else:
                x = x[:, : -shared_neck.size(1)]

        next_shared_neck /= len(per_transformer_x)
        self.bottleneck.copy_(next_shared_neck)

        return per_transformer_x
