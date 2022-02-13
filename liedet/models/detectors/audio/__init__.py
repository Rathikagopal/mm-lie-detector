import os
import uuid

import torch
import torch.nn as nn
import torchaudio

from .main import main


class AudioFeatures(nn.Module):
    def __init__(self, fps: int = 30, chunk_length=1, sr=48000, normalization: bool = True):
        super().__init__()

        self.video_fps = fps
        self.chunk_length = chunk_length
        self.sample_rate = sr
        self.normalization = normalization

    def forward(self, x):
        h = []
        for i in range(x.size(0)):
            tmp_path = uuid.uuid4()
            tmp_path = f"/tmp/{tmp_path}.wav"
            torchaudio.save(tmp_path, x[i], self.sample_rate)

            hi = main(
                audio_path=tmp_path,
                fps=self.video_fps,
                normalization=self.normalization,
                chunk_length=self.chunk_length,
                csv=False,
            )
            h.append(hi)

            os.remove(tmp_path)

        return torch.stack(h, dim=0)
