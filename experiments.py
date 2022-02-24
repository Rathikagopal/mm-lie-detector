import os

from catalyst import dl, metrics

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from liedet.datasets.interviews import InterviewsDataset
from liedet.models.detectors.audio import AudioFeatures
from liedet.models.detectors.landmarks import LandmarksFeatures

batch_size = 2
window = int(3 * 30.0)
dataset = InterviewsDataset(path="data/interviews", window=window)
train_set, valid_set = dataset.split(valid_size=0.2, by_file=True, balanced="under")
train_loader = DataLoader(dataset, batch_size=batch_size)
valid_loader = DataLoader(dataset, batch_size=batch_size)
loaders = dict(train_loader=train_loader, valid_loader=valid_loader)

audio_model = AudioFeatures()
video_model = LandmarksFeatures()
features_dims = 1455
embed_dims = 512
linear = nn.Linear(in_features=features_dims, out_features=embed_dims)
transformer_encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dims,
    nhead=6,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True,
)
time_model = nn.TransformerEncoder(
    encoder_layer=transformer_encoder_layer,
    num_layers=6,
    norm=nn.LayerNorm(normalized_shape=embed_dims),
)
cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
pos_embed = nn.Parameter(torch.zeros(1, window+1, embed_dims))
cls_head = nn.Linear(in_features=embed_dims, out_features=1)

model = nn.ModuleDict(dict(
    video_model=video_model,
    audio_model=audio_model,
    linear=linear,
    time_model=time_model,
    cls_token=cls_token,
    pos_embed=pos_embed,
    cls_head=cls_head
))

optimizer = dict(
    video_model=optim.Adam(video_model.parameters()),
    audio_model=optim.Adam(audio_model.parameters()),
    linear=optim.Adam(linear.parameters()),
    time_model=optim.Adam(time_model.parameters()),
    cls_token=optim.Adam(cls_token.parameters()),
    pos_embed=optim.Adam(pos_embed.parameters()),
    cls_head=optim.Adam(cls_head.parameters()),
)
criterion = nn.CrossEntropyLoss()


class Runner(dl.Runner):
    def handle_batch(self, batch):
        vframes, aframes, labels = batch["vframes"], batch["aframes"], batch["labels"]

        vfeatures = self.model["video_model"](vframes)
        afeatures = self.model["audio_model"](aframes)

        logits = torch.cat([vfeatures, afeatures], dim=-1)
        del vfeatures, afeatures

        logits = self.model["linear"](logits)
        cls_tokens = self.model["cls_token"].expand(logits.size(0), -1, -1)
        logits = torch.cat((cls_tokens, logits), dim=1)
        logits = logits + self.model["pos_embed"]
        logits = self.model["time_model"](logits)
        logits = self.model["cls_head"](logits)

        self.batch = dict(logits=logits, labels=labels)


runner = Runner()


runner = Runner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir="./logs",
    num_epochs=200,
    valid_loader="valid_loader",
    valid_metric="accuracy01",
    minimize_valid_metric=False,
    callbacks=[
        dl.CriterionCallback(input_key="logits", target_key="labels", metric_key="loss"),
        dl.OptimizerCallback(metric_key="loss", accumulation_steps=int(64 // batch_size)),
        dl.AccuracyCallback(input_key="logits", target_key="labels", num_classes=2),
        dl.EarlyStoppingCallback(patience=15, loader_key="valid_loader", metric_key="loss", minimize=True),
        dl.CheckpointCallback(logdir="./logs", loader_key="valid_loader", metric_key="loss", minimize=True, save_n_best=1),
    ],
    load_best_on_end=True,
    verbose=True,
)
