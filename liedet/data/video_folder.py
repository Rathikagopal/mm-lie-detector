from __future__ import annotations

import copy
from glob import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as V

from liedet.data.video_reader import VideoReader


class VideoFolder(Dataset):
    def __init__(
        self,
        root: str,
        pattern: str = ".mp4",
        window: int = 1,
        video_fps: int = 30,
        audio_fps: int = 48000,
        label_key: str = "label",
        video_transforms: V.Compose | None = None,
        audio_transforms: V.Compose | None = None,
        video_per_window_transforms: V.Compose | None = None,
        audio_per_window_transforms: V.Compose | None = None,
        height: int | None = None,
        width: int | None = None,
        mono: bool = True,
        bridge: str = "torch",
        valid_size: float | None = None,
        split_by_file: bool = True,
        balanced_valid_set: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        self.root = root
        self.pattern = pattern

        self.files = self.get_files()
        self.metas = self.get_metas()
        self.label_key = label_key
        self.labels = self.get_labels()
        self.per_file_frames_num = self.get_per_files_frames_num()

        self.window = window
        self.video_fps = video_fps
        self.audio_fps = audio_fps

        self.order = self.get_order()

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.video_transforms = video_transforms
        self.audio_transforms = audio_transforms
        self.video_per_window_transforms = video_per_window_transforms
        self.audio_per_window_transforms = audio_per_window_transforms

        self.bridge = bridge
        self.mono = mono

        self.height = height
        self.width = width

        self.valid_size: float | None = valid_size
        self._train_set: VideoFolder | None = None
        self._valid_set: VideoFolder | None = None
        self.split_by_file = split_by_file
        self.balanced_valid_set = balanced_valid_set
        if self.valid_size is not None:
            self._train_set, self._valid_set = self.split(
                valid_size=self.valid_size, by_file=split_by_file, balanced_valid_set=balanced_valid_set
            )

    def get_files(self) -> list[str]:
        pathname = f"{self.root}/**/*{self.pattern}"
        files = sorted(glob(pathname=pathname, recursive=True))

        return list(files)

    def get_metas(self) -> list[dict[str, Any]]:
        return [self.get_meta(path=path) for path in self.files]

    def get_meta(self, path: str) -> dict[str, Any]:
        return dict(label=Path(path).name.split("_")[-1].split(".")[0])

    def get_labels(self) -> Tensor:
        return torch.tensor([int(meta[self.label_key]) for meta in self.metas], dtype=torch.long)

    def get_order(self) -> Tensor:
        return torch.tensor(
            [
                (i, j * self.window)
                for i, num_frames in enumerate(self.per_file_frames_num)
                for j in range(int(num_frames // self.window) - 1)
            ]
        )

    def get_per_files_frames_num(self) -> list[int]:
        per_file_frames_num = []
        for file in self.files:
            cap = cv2.VideoCapture(file)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            per_file_frames_num.append(n_frames)
            cap.release()

        return per_file_frames_num

    def split(
        self,
        valid_size: float = 0.2,
        by_file: bool = True,
        balanced_valid_set: bool = True,
        seed: int | None = None,
    ) -> tuple[VideoFolder, VideoFolder]:
        _rng = self.rng
        if seed is not None:
            _rng = np.random.default_rng(seed=seed)

        train_set = copy.deepcopy(self)
        valid_set = copy.deepcopy(self)

        labels: Tensor
        if by_file:
            labels = self.labels
        else:
            labels = self.labels[self.order[..., 0]]

        total_samples = self.order.size(0)

        if balanced_valid_set:
            counts: Tensor
            unique_labels, counts = labels.unique(return_counts=True)
            min_size = counts.min()
            valid_samples = int(min_size * valid_size)

            train_indices: list[Tensor] | Tensor = []
            valid_indices: list[Tensor] | Tensor = []
            for label in unique_labels:
                indices = torch.argwhere(labels == label).T[0]
                samples = indices.size(0)

                _indices = np.arange(samples)
                _rng.shuffle(_indices)

                valid_indices.extend(indices[_indices[:valid_samples]])  # type: ignore
                train_indices.extend(indices[_indices[valid_samples:]])  # type: ignore
        else:
            valid_samples = int(total_samples * valid_size)

            indices = np.arange(total_samples)
            _rng.shuffle(indices)

            valid_indices = indices[:valid_samples]
            train_indices = indices[valid_samples:]

        train_indices = torch.tensor(train_indices, dtype=torch.long)
        valid_indices = torch.tensor(valid_indices, dtype=torch.long)
        if by_file:
            train_indices = torch.isin(train_set.order[..., 0], train_indices)
            valid_indices = torch.isin(valid_set.order[..., 0], valid_indices)
            train_set.order = train_set.order[train_indices]
            valid_set.order = valid_set.order[valid_indices]
        else:
            train_set.order = train_set.order[train_indices]
            valid_set.order = valid_set.order[valid_indices]

        self._train_set, self._valid_set = train_set, valid_set

        return self._train_set, self._valid_set

    @property
    def train_set(self) -> VideoFolder:
        if self._train_set is not None:
            return self._train_set

        self._train_set, self._valid_set = self.split()

        return self._train_set

    @property
    def valid_set(self) -> VideoFolder:
        if self._valid_set is not None:
            return self._valid_set

        self._train_set, self._valid_set = self.split()

        return self._valid_set

    def __len__(self) -> int:
        return self.order.size(0)

    def __getitem__(self, idx) -> dict[str, Any]:
        file_idx, start_frame_idx = self.order[idx, :]

        video_reader = VideoReader(
            uri=self.files[file_idx],
            video_fps=self.video_fps,
            audio_fps=self.audio_fps,
            video_transforms=self.video_per_window_transforms,
            audio_transforms=self.audio_per_window_transforms,
            mono=self.mono,
            bridge=self.bridge,
            height=self.height,
            width=self.width,
        )

        frames = video_reader[start_frame_idx : start_frame_idx + self.window]

        if self.video_transforms is not None:
            frames["video_frames"] = self.video_transforms(frames["video_frames"])
        if self.audio_transforms is not None:
            frames["audio_frames"] = self.audio_transforms(frames["audio_frames"])

        label = self.labels[file_idx]

        return {"labels": label, **frames}
