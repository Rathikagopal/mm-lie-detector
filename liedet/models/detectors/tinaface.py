from collections import OrderedDict

import torch

from mmcv.parallel import collate
from mmcv.utils import ConfigDict

from ..registry import build, registry
from .bbox import SingleStageDetector


@registry.register_module()
class Tinaface(SingleStageDetector):
    def __init__(self, frame_to_result: ConfigDict, extract_bboxes: ConfigDict, **kwargs):
        super().__init__(**kwargs)

        self.frame_to_result = build(cfg=frame_to_result)
        self.extract_bboxes = build(cfg=extract_bboxes)
        
        # TODO: remove me
        state_dict = torch.load("weights/tinaface_r50_fpn_gn_dcn.pth", map_location="cpu")
        new_state_dict = OrderedDict()
        for old_key, value in state_dict.items():
            new_key = old_key
            if "neck.0" in old_key:
                new_key = old_key.replace("neck.0", "neck.fpn")
            elif "neck.1" in old_key:
                new_key = old_key.replace("neck.1", "neck.inception")
            elif "retina" in old_key:
                new_key = old_key.replace("retina", "conv")
            new_state_dict[new_key] = state_dict[old_key]
        self.load_state_dict(new_state_dict)

    @torch.no_grad()
    def forward(self, x):
        results = []
        for frame in x:
            h = collate([frame])
            h = super().forward(h)

            h = self.frame_to_result(h)
            results.append(h)

        bboxes = self.extract_bboxes(x, results)

        return bboxes
