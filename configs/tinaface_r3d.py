from collections import OrderedDict

from einops import rearrange

import torch
import torchaudio.functional as AF
import torchvision.transforms as V

from mmcv.utils import Config

tinaface = Config.fromfile("configs/models/tinaface.py")

batch_size = 32


fps = 30
window_secs = 3
window = fps * window_secs
window_div = 3

window_frames = int(window / window_div)
video_fps = int(fps / window_div)


post_vtransform = V.Compose(
    [
        V.Lambda(lambd=lambda x: x[::window_div]),
        # BHWC -> BCHW
        V.Lambda(lambd=lambda x: x.permute(0, 3, 1, 2)),
        # resize to (320, 320)
        V.Resize(size=(320, 320)),
        # uint8 to float
        V.Lambda(lambd=lambda x: x.float()),
        # bgr to rgb
        V.Lambda(lambd=lambda x: x[:, (2, 1, 0), :, :]),
        # normalize
        V.Normalize(mean=[123.675, 116.28, 103.53], std=[1.0, 1.0, 1.0]),
    ],
)


dataset = dict(
    type="Interviews",
    path="data/interviews",
    window=window,
    vtransform=post_vtransform,
    split=dict(
        valid_size=0.2,
        by_file=True,
    ),
)

features_dims = 6048
embed_dims = 512
num_classes = 2
model = dict(
    type="LieDetector",
    video_model=dict(
        type="Sequential",
        modules=[
            OrderedDict(
                rearrange1=dict(type="Rearrange", pattern="b t c h w -> (b t) c h w"),
                tinaface=tinaface.cfg,
                resize=V.Resize(size=(224, 224)),
                rearrange2=dict(type="Rearrange", pattern="(b t) c h w -> b c t h w", b=batch_size, c=3, h=224, w=224),
            ),
        ],
    ),
    features_dims=features_dims,
    embed_dims=embed_dims,
    embed_mode="identity",
    time_model=dict(
        type="Sequential",
        modules=[
            OrderedDict(
                r3d=dict(type="ResNet3d", pretrained=None, depth=18),
                pool=dict(type="AdaptiveAvgPool3d", output_size=(1, 1, 1)),
                flat=dict(type="Flatten", start_dim=1),
            )
        ],
    ),
    cls_head=dict(
        type="Linear",
        in_features=embed_dims,
        out_features=num_classes,
    ),
)

runner = dict(
    type="LieDetectorRunner",
)
