from collections import OrderedDict

import torch.nn as nn
import torchaudio.functional as AF
import torchvision.transforms as V

from mmcv.utils import Config

batch_size = 1

fps = 30
window_secs = 10
window = fps * window_secs
window_div = 3

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

audio_features_dims = 18 * window_secs
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
                rearrange2=dict(
                    type="Rearrange", pattern="(b t) c h w -> b c t h w", b=batch_size, t=window_frames, h=224, w=224
                ),
                timesformer=dict(
                    type="TimeSformer",
                    img_size=(224, 224),
                    patch_size=(16, 16),
                    num_frames=window_frames,
                    embed_dims=embed_dims,
                    num_heads=8,
                ),
            ),
        ],
    ),
    audio_model=dict(
        type="Sequential",
        modules=[
            OrderedDict(
                feas=dict(type="AudioFeatures", fps=1, chunk_length=1, sr=audio_fps, normalization=True),
                flat=dict(type="Flatten", start_dim=1),
            )
        ],
    ),
    features_dims=embed_dims + audio_features_dims,
    embed_dims=embed_dims,
    pos_embed=False,
    window=window_frames,
    time_model=dict(type="Identity"),
    cls_head=dict(
        type="Linear",
        in_features=embed_dims,
        out_features=num_classes,
    ),
)
