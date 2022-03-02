import torch
import torchaudio.functional as AF
import torchvision.transforms as V

batch_size = 20

fps = 30
window_secs = 10
window = fps * window_secs
window_div = 3

video_fps = int(window / window_div)
audio_fps = 22050

vtransform = V.Compose(
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
    vtransform=vtransform,
    vr_kwargs=dict(
        atransform=atransform,
    ),
    split=dict(
        valid_size=0.2,
        by_file=True,
    ),
)

features_dims = 1455
embed_dims = 512
num_classes = 2
model = dict(
    type="LieDetector",
    video_model=dict(type="Landmarks", fps=video_fps),
    audio_model=dict(type="AudioFeatures", fps=video_fps, chunk_length=1, sr=audio_fps, normalization=True),
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
        num_layers=6,
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
