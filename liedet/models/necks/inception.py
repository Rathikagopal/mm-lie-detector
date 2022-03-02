from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from mmcv.cnn import ConvModule, xavier_init

from ..registry import registry


@registry.register_module()
class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_levels: int,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        dcn_cfg: Optional[dict] = None,
        share: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn_cfg = dcn_cfg
        self.with_dcn = True if dcn_cfg is not None else False
        self.share = share

        self.level_convs = nn.ModuleList()
        loop = 1 if self.share else self.num_levels

        for i in range(loop):
            convs = nn.ModuleList()

            conv = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.in_channels // 2,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
            )
            convs.append(conv)

            conv = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.in_channels // 4,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )
            convs.append(conv)

            for i in range(3):
                act_cfg = dict(type="ReLU") if i % 2 == 1 else None
                conv = ConvModule(
                    in_channels=self.in_channels // 4,
                    out_channels=self.in_channels // 4,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=act_cfg,
                )
                convs.append(conv)

            self.level_convs.append(convs)

        if self.with_dcn:
            self.dcn = ConvModule(
                in_channels=self.in_channels,
                out_channels=self.in_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=self.dcn_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=act_cfg,
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, input):
        outs = []
        for i in range(self.num_levels):
            x = input[i]
            if self.share:
                x_3 = self.level_convs[0][0](x)

                x_5_1 = self.level_convs[0][1](x)
                x_5 = self.level_convs[0][2](x_5_1)

                x_7_2 = self.level_convs[0][3](x_5_1)
                x_7 = self.level_convs[0][4](x_7_2)
            else:
                x_3 = self.level_convs[i][0](x)

                x_5_1 = self.level_convs[i][1](x)
                x_5 = self.level_convs[i][2](x_5_1)

                x_7_2 = self.level_convs[i][3](x_5_1)
                x_7 = self.level_convs[i][4](x_7_2)
            out = F.relu(torch.cat([x_3, x_5, x_7], dim=1))

            if self.with_dcn:
                out = self.dcn(out)

            outs.append(out)

        return tuple(outs)
