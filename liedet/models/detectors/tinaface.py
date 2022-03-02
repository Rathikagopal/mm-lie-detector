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

    def forward(self, x):
        results = []
        for frame in x:
            h = collate([frame])
            h = super().forward(h)

            h = self.frame_to_result(h)
            results.append(h)

        bboxes = self.extract_bboxes(x, results)

        return bboxes
