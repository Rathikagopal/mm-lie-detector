import os

from catalyst import dl, metrics

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mmcv.utils import Config

from liedet.datasets import build_dataset
from liedet.models.e2e import LieDetector, LieDetectorRunner
from liedet.models.registry import build

# cfg = "configs/train/landmarks_audio_transformer.py"
# cfg = "configs/train/tinaface_r50_audio_transformer.py"
cfg = "configs/train/tinaface_audio_transformer.py"
cfg = Config.fromfile(cfg)

dataset = build_dataset(cfg.dataset)
dataset = dataset.split(**cfg.dataset.split)
loaders = dict(
    train_loader=DataLoader(dataset["train_set"], batch_size=cfg.batch_size, num_workers=0, drop_last=True),
    valid_loader=DataLoader(dataset["valid_set"], batch_size=cfg.batch_size, num_workers=0),
)


model = build(cfg.model)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

runner = LieDetectorRunner(model=model)

for batch in loaders["valid_loader"]:
    with torch.no_grad():
        batch_prediciton = runner.predict_batch(batch)
