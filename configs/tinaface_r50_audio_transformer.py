from collections import OrderedDict

import torch.nn as nn
import torchaudio.functional as AF
import torchvision.transforms as V

from mmcv.utils import Config

batch_size = 1

fps = 30
window_secs = 3
window = fps * window_secs
window_div = 10

window_frames = int(window / window_div)
video_fps = int(fps / window_div)
audio_fps = 22050

vtransform = V.Compose(
    [
        # HWC -> CHW
        V.Lambda(lambd=lambda x: x.permute(2, 0, 1)),
        # resize to (320, 320)
        V.Resize(size=(320, 320)),
        # uint8 to float
        V.Lambda(lambd=lambda x: x.float()),
        # bgr to rgb
        V.Lambda(lambd=lambda x: x[(2, 1, 0), :, :]),
        # normalize
        V.Normalize(mean=[123.675, 116.28, 103.53], std=[1.0, 1.0, 1.0]),
    ],
)
vtransform_post = V.Compose(
    [
        V.Lambda(lambda x: x[::window_div]),
    ]
)
atransform = V.Compose(
    [
        V.Lambda(lambda x: AF.resample(waveform=x, orig_freq=int(x.size(-1) / window_secs), new_freq=audio_fps)),
    ]
)

dataset = dict(
    type="Interviews",
    path="data/interviews",
    window=window,
    vtransform=vtransform_post,
    vr_kwargs=dict(
        vtransform=vtransform,
        atransform=atransform,
    ),
    split=dict(
        valid_size=0.2,
        by_file=True,
    ),
)

features_dims = 2066
embed_dims = 512
num_classes = 2

tinaface = Config.fromfile("configs/models/tinaface.py")
model = dict(
    type="LieDetector",
    video_model=dict(
        type="Sequential",
        modules=[
            OrderedDict(
                rearrange1=dict(type="Rearrange", pattern="b t c h w -> (b t) c h w"),
                tinaface=tinaface.cfg,
                resize=V.Resize(size=(224, 224)),
                r50=dict(
                    type="mmdet.ResNet",
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                    norm_eval=False,
                    dcn=dict(type="DCN", deform_groups=1, im2col_step=1, fallback_on_stride=False),
                    stage_with_dcn=(False, False, True, True),
                    style="pytorch",
                ),
                select=dict(type="Select", index=-1),
                pool=dict(type="AdaptiveAvgPool2d", output_size=(1, 1)),
                flat=dict(type="Flatten", start_dim=1),
                rearrange2=dict(type="Rearrange", pattern="(b t) f -> b t f", b=batch_size, t=window_frames),
            ),
        ],
    ),
    audio_model=dict(type="AudioFeatures", fps=video_fps, chunk_length=1, sr=audio_fps, normalization=True),
    features_dims=features_dims,
    embed_dims=embed_dims,
    window=window_frames,
    time_model=dict(
        type="TransformerEncoder",
        encoder_layer=dict(
            type="TransformerEncoderLayer",
            d_model=embed_dims,
            nhead=8,
            dim_feedforward=embed_dims * 4,
            dropout=0.0,
            batch_first=True,
        ),
        num_layers=6,
        norm=dict(type="LayerNorm", normalized_shape=embed_dims),
    ),
    cls_head=dict(
        type="Linear",
        in_features=embed_dims,
        out_features=num_classes,
    ),
)
