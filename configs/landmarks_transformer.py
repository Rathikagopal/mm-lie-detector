import torch
import torchaudio.functional as AF
import torchvision.transforms as V

batch_size = 32

fps = 30
window_secs = 10
window = fps * window_secs
window_div = 3

video_fps = int(window / window_div)

vtransform = V.Compose(
    [
        # fps -> fps // window_div
        V.Lambda(lambda x: x[::window_div]),
        # BHWC -> BCHW
        V.Lambda(lambd=lambda x: x.permute(0, 3, 1, 2)),
        # H,W -> 240, 320
        V.Resize(size=(240, 320)),
        # BCHW -> BHWC
        V.Lambda(lambd=lambda x: x.permute(0, 2, 3, 1)),
    ]
)


dataset = dict(
    type="Interviews",
    path="data/interviews",
    window=window,
    vtransform=vtransform,
    split=dict(
        valid_size=0.2,
        by_file=True,
    ),
)

features_dims = 1437
embed_dims = 512
num_classes = 2
model = dict(
    type="LieDetector",
    video_model=dict(type="Landmarks", fps=video_fps),
    features_dims=features_dims,
    embed_dims=embed_dims,
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
        num_layers=3,
        norm=dict(type="LayerNorm", normalized_shape=embed_dims),
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
