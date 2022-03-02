from __future__ import annotations

import copy
from glob import glob
from typing import Any, Optional

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as V

from .video_reader import VideoReader


class VideoFolder(Dataset):
    def __init__(
        self,
        path: str,
        window: int,
        pattern: str = ".mp4",
        shuffle: bool = False,
        seed: int | None = None,
        vtransform: V.Compose | None = None,
        atransform: V.Compose | None = None,
        vr_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.folder_path = path
        self.pattern = pattern

        self.files = self.get_files()
        self.labels = self.get_labels()
        self.per_file_frames_num = self.get_per_files_frames_num()
        self.window = window

        self.order = self.get_order()

        self.vr_kwargs = vr_kwargs if vr_kwargs is not None else {}

        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed=seed)
        if self.shuffle:
            self.order = self.reshuffle()
        self.order = self.order[:5000]

        self.vtransform = vtransform
        self.atransform = atransform

    def get_files(self) -> list[str]:
        pathname = f"{self.folder_path}/**/*{self.pattern}"
        files = sorted(glob(pathname=pathname, recursive=True))

        return list(files)

    def get_labels(self) -> list[int]:
        return [self.get_label(path=path) for path in self.files]

    def get_label(self, path: str) -> int | dict[str, Any]:
        raise NotImplementedError

    def get_order(self) -> np.ndarray:
        return np.asarray(
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

    def reshuffle(self) -> np.ndarray:
        ids_order = [i for i in range(self.order.shape[0])]
        self.rng.shuffle(ids_order)

        return self.order[ids_order, :]

    def split(
        self,
        test_size: float = 0.0,
        valid_size: float = 0.2,
        by_file: bool = False,
        balanced: str = "under",
        shuffle: bool = True,
    ) -> dict[str, "VideoFolder"]:
        train_set = copy.deepcopy(self)
        test_set = copy.deepcopy(self)
        valid_set = copy.deepcopy(self)

        if by_file:
            labels = self.labels
        else:
            labels = [self.labels[i] for i in self.order[:, 0]]
        per_class_ids: dict[str, list[int]] = {}
        for idx, label in enumerate(labels):
            if label not in per_class_ids:
                per_class_ids[label] = []
            per_class_ids[label].append(idx)

        per_class_size = [len(ids) for ids in per_class_ids.values()]
        min_size, max_size = min(per_class_size), max(per_class_size)

        min_test_size = int(min_size * test_size)
        min_valid_size = int(min_size * valid_size)
        min_train_size = min_size - min_test_size - min_valid_size

        max_test_size = int(max_size * test_size)
        max_valid_size = int(max_size * valid_size)
        max_train_size = max_size - max_test_size - max_valid_size

        train_ids: list[int] = []
        test_ids: list[int] = []
        valid_ids: list[int] = []
        for label, ids in per_class_ids.items():
            train_class_ids, test_class_ids, valid_class_ids = [], [], []
            if shuffle:
                self.rng.shuffle(ids)

            mult = 1 if balanced == "under" else int(len(ids) // min_size)
            train_class_ids = ids[: mult * min_train_size]
            test_class_ids = ids[mult * min_train_size : mult * (min_train_size + min_test_size)]
            valid_class_ids = ids[mult * (min_train_size + min_test_size) : mult * min_size]

            if balanced == "over" and shuffle:
                train_class_ids = train_class_ids + list(
                    self.rng.choice(train_class_ids, max_train_size - min_train_size)
                )
                test_class_ids = test_class_ids + list(self.rng.choice(test_class_ids, max_test_size - min_test_size))
                valid_class_ids = valid_class_ids + list(
                    self.rng.choice(valid_class_ids, max_valid_size - min_valid_size)
                )
            elif balanced == "over":
                train_class_ids = train_class_ids + train_class_ids[: max_train_size - min_train_size]
                test_class_ids = test_class_ids + test_class_ids[: max_test_size - min_test_size]
                valid_class_ids = valid_class_ids + valid_class_ids[: max_valid_size - min_valid_size]

            train_ids.extend(train_class_ids)
            test_ids.extend(test_class_ids)
            valid_ids.extend(valid_class_ids)

        if by_file:
            train_set.order = self.order[np.isin(self.order[:, 0], train_ids), :]
            test_set.order = self.order[np.isin(self.order[:, 0], test_ids), :]
            valid_set.order = self.order[np.isin(self.order[:, 0], valid_ids), :]
        else:
            train_set.order = self.order[train_ids, :]
            test_set.order = self.order[test_ids, :]
            valid_set.order = self.order[valid_ids, :]

        return {
            "train_set": train_set,
            "test_set": test_set,
            "valid_set": valid_set,
        }

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx) -> dict[str, Any]:
        file_idx, start_frame_idx = self.order[idx, :]

        vr = VideoReader(uri=self.files[file_idx], **self.vr_kwargs)

        frames = vr[start_frame_idx : start_frame_idx + self.window]
        if len(frames["vframes"]) < self.window:
            frames = vr[-self.window :]

        if self.vtransform is not None:
            frames["vframes"] = self.vtransform(frames["vframes"])
        if self.atransform is not None:
            frames["aframes"] = self.atransform(frames["aframes"])
        label = self.labels[file_idx]

        return {"labels": label, **frames}
