from __future__ import annotations

from pathlib import Path

from ..data import VideoFolder
from .builder import datasets


@datasets.register_module(name="RAVDESS")
class RAVDESSDataset(VideoFolder):
    def get_label(self, path: str) -> dict:
        filename = Path(path).name.split(".")[0]
        tokens = filename.split("-")

        return dict(
            modality=tokens[0],
            voice=tokens[1],
            emotion=tokens[2],
            intensity=tokens[3],
            statement=tokens[4],
            repetition=tokens[5],
            actor=tokens[6],
        )
