from __future__ import annotations

import decord

import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision import transforms as T


class VideoReader(Dataset):
    def __init__(
        self,
        uri: str,
        mono: bool = True,
        bridge: str = "torch",
        vtransform: T.Compose | None = None,
        atransform: T.Compose | None = None,
        **kwargs,
    ):
        decord.bridge.set_bridge(new_bridge=bridge)

        self.uri = uri
        self.mono = mono

        meta = io.read_video(filename=uri, start_pts=0, end_pts=1)[-1]
        video_fps, audio_fps = meta["video_fps"], meta["audio_fps"]

        self.reader = decord.AVReader(uri=uri, sample_rate=audio_fps, mono=self.mono, num_threads=1)
        num_frames = len(self.reader)
        num_audio_frames = self.reader._AVReader__audio_reader._num_samples_per_channel  # noqa: WPS437

        self.meta = dict(
            video_fps=int(video_fps),
            audio_fps=int(audio_fps),
            sample_rate=int(audio_fps),
            num_frames=int(num_frames),
            num_video_frames=int(num_frames),
            num_audio_frames=int(num_audio_frames),
        )

        self.vtransform = vtransform
        self.atransform = atransform

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int | slice) -> dict:
        if isinstance(idx, slice):
            aframes, vframes = self.reader[idx]
            aframes = torch.cat(aframes, dim=-1)
            C, L = aframes.shape
            dL = int(vframes.shape[0] / self.meta["video_fps"] * self.meta["audio_fps"] - L)
            dL = dL if dL > 0 else 0
            aframes = torch.cat((aframes, torch.zeros(C, dL)), dim=1)

            if self.vtransform is not None:
                vframes = torch.stack([self.vtransform(vframe) for vframe in vframes])
            if self.atransform is not None:
                aframes = torch.stack([self.atransform(aframe) for aframe in aframes])
        else:
            aframes, vframes = self.reader[idx]

            if self.vtransform is not None:
                vframes = self.vtransform(vframes)
            if self.atransform is not None:
                aframes = self.atransform(aframes)

        return {"vframes": vframes, "aframes": aframes}
