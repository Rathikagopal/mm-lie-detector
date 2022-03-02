import os
import uuid

import torch
import torch.nn as nn
import torchaudio

from ...registry import registry
from .main import main


@registry.register_module()
class AudioFeatures(nn.Module):
    def __init__(self, fps: int = 30, chunk_length=1, sr=48000, normalization: bool = True):
        super().__init__()

        self.video_fps = fps
        self.chunk_length = chunk_length
        self.sample_rate = sr
        self.normalization = normalization

    @torch.no_grad()
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        x = x.cpu()
        h = []
        for i in range(batch_size):
            tmp_path = uuid.uuid4()
            tmp_path = f"/tmp/{tmp_path}.wav"
            torchaudio.save(tmp_path, x[i], self.sample_rate)

            hi = main(
                audio_path=tmp_path,
                fps=self.video_fps,
                normalization=self.normalization,
                sr=self.sample_rate,
                chunk_length=self.chunk_length,
                csv=False,
            )
            h.append(hi)

            os.remove(tmp_path)

        return torch.stack(h, dim=0).to(device)
