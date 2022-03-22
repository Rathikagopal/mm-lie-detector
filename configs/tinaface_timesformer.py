from collections import OrderedDict

import torchvision.transforms as V

from mmcv.utils import Config

tinaface = Config.fromfile("configs/models/tinaface.py")

batch_size = 1


# target video fps (real video fps --> target video fps)
video_fps = 10
window_secs = 10
# target window size (number of frames)
window = video_fps * window_secs

# video transforms
video_transform = V.Normalize(mean=[123.675, 116.28, 103.53], std=[1.0, 1.0, 1.0])


# dataset config
dataset = dict(
    type="Interviews",
    root="data/converted",
    window=window,
    video_fps=video_fps,
    # audio_fps=1 to drop audio
    audio_fps=1,
    # train valid split
    split=dict(valid_size=0.2, balanced_valid_set=True, by_file=True),
)

# number of target classes (binary == 2)
num_classes = 2

# model pipeline
model = dict(
    type="LieDetector",
    # model to extract features from video
    video_model=dict(
        type="Sequential",
        modules=[
            OrderedDict(
                # move time dim to batch dim
                rearrange1=dict(type="Rearrange", pattern="b t c h w -> (b t) c h w"),
                # model to detect and extract faces
                tinaface=tinaface.cfg,
                # resize output images with different sizes
                resize=V.Resize(size=(224, 224)),
                # normalize
                norm=V.Normalize(mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], inplace=True),
                # extract time dim back and move it
                rearrange2=dict(type="Rearrange", pattern="(b t) c h w -> b c t h w", b=batch_size),
            ),
        ],
    ),
    # time model to extract time-dependent features from time-independent ones
    time_model=dict(
        type="TimeSformer",
        num_frames=window,
        img_size=(224, 224),
        patch_size=(16, 16),
        embed_dims=768,
        pretrained="https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth",
    ),
    # classifier
    cls_head=dict(
        type="TimeSformerHead",
        in_channels=768,
        num_classes=num_classes,
    ),
    init_cfg=dict(type="PretrainedInit", checkpoint="weights/tinaface_timesformer.pth"),
)

runner = dict(
    type="LieDetectorRunner",
)
