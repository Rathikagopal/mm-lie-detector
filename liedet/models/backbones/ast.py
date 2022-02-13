# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import Any

import numpy as np

import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram

from mmcv import ConfigDict
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import _load_checkpoint, load_state_dict
from mmdet.utils.logger import get_root_logger

from ..registry import registry
from .timesformer import PatchEmbed as PE


class PatchEmbed(PE):
    def forward(self, x) -> nn.Module:
        return self.projection(x).flatten(2).transpose(1, 2)


@registry.register_module()
class AST(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = 10,
        embed_dims: int = 768,
        num_heads: int = 12,
        num_transformer_layers: int = 12,
        in_channels: int = 1,
        dropout_ratio: float = 0.0,
        transformer_layers=None,
        pretrained: str | None = None,
        norm_cfg: dict[str, Any] = dict(type="LN", eps=1e-6),
        **kwargs,
    ) -> None:
        super().__init__()

        assert transformer_layers is None or isinstance(transformer_layers, (dict, list))

        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        if transformer_layers is None:
            # stochastic depth decay rule
            dpr = np.linspace(0, 0.1, num_transformer_layers)

            _transformerlayers_cfg = [
                dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=embed_dims,
                            num_heads=num_heads,
                            batch_first=True,
                            dropout_layer=dict(type="DropPath", drop_prob=dpr[i]),
                        )
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims * 4,
                        num_fcs=2,
                        act_cfg=dict(type="GELU"),
                        dropout_layer=dict(type="DropPath", drop_prob=dpr[i]),
                    ),
                    operation_order=("norm", "self_attn", "norm", "ffn"),
                    norm_cfg=dict(type="LN", eps=1e-6),
                    batch_first=True,
                )
                for i in range(num_transformer_layers)
            ]

            transformer_layers = ConfigDict(
                dict(
                    type="TransformerLayerSequence",
                    transformerlayers=_transformerlayers_cfg,
                    num_layers=num_transformer_layers,
                )
            )

        self.transformer_layers = build_transformer_layer_sequence(transformer_layers)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f"load model from: {self.pretrained}")

            state_dict = _load_checkpoint(self.pretrained)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        # x [batch_size, num_patches, embed_dims]
        x = self.patch_embed(x)

        # x [batch_size, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        x = self.transformer_layers(x, None, None)

        x = self.norm(x)

        return x
