from __future__ import annotations

from catalyst import dl

import torch
import torch.nn as nn

from mmcv.utils import ConfigDict

from .registry import build, registry


@registry.register_module()
class LieDetector(nn.Module):
    def __init__(
        self,
        *,
        time_model: nn.Module | dict,
        cls_head: nn.Module | dict,
        features_dims: int,
        embed_dims: int,
        video_model: nn.Module | dict | None = None,
        audio_model: nn.Module | dict | None = None,
        window: int = 100,
        embed_mode: str | None = "linear",
        pos_embed: bool = True,
        **kwargs,
    ):
        super().__init__()

        if video_model is None and audio_model is None:
            raise ValueError("At least one of video_model or audio_model should be specified.")

        self.video_model = video_model
        self.audio_model = audio_model
        self.time_model = time_model
        self.cls_head = cls_head

        self.window = window

        if isinstance(self.video_model, ConfigDict):
            self.video_model = build(cfg=self.video_model, registry=registry)
        if isinstance(self.audio_model, ConfigDict):
            self.audio_model = build(cfg=self.audio_model, registry=registry)
        if isinstance(self.time_model, ConfigDict):
            self.time_model = build(cfg=self.time_model, registry=registry)
        if isinstance(self.cls_head, ConfigDict):
            self.cls_head = build(cfg=self.cls_head, registry=registry)

        if embed_mode == "linear":
            self.embed = nn.Linear(in_features=features_dims, out_features=embed_dims)
        else:
            self.embed = nn.Identity()

        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = pos_embed
        self.pos_embeds = nn.Parameter(torch.zeros(1, window + 1, embed_dims))

    def forward(self, batch):
        vframes, aframes = batch["vframes"], batch["aframes"]

        if self.video_model is not None:
            vfeatures = self.video_model(vframes)
        if self.audio_model is not None:
            afeatures = self.audio_model(aframes)

        if self.video_model is not None and self.audio_model is not None:
            logits = torch.cat((vfeatures, afeatures), dim=-1)
            del vfeatures, afeatures
        elif self.model is not None:
            logits = vfeatures
        else:
            logits = afeatures

        logits = self.embed(logits)

        shape_size = len(logits.shape)

        if shape_size == 3:
            cls_tokens = self.cls_tokens.expand(logits.size(0), -1, -1)
            logits = torch.cat((logits, cls_tokens), dim=1)

        if shape_size == 3 and self.pos_embed:
            logits = logits + self.pos_embeds

        logits = self.time_model(logits)
        if shape_size == 3:
            logits = logits[:, -1]

        logits = self.cls_head(logits)

        return logits


class LieDetectorRunner(dl.Runner):
    @torch.no_grad()
    def predict_batch(self, batch):
        logits = self.model(batch)
        probs = torch.sigmoid(logits).argmax(dim=1)

        return probs

    def handle_batch(self, batch):
        _, _, labels = batch["vframes"], batch["aframes"], batch["labels"]

        logits = self.model(batch)

        self.batch = dict(logits=logits, labels=labels)
