from __future__ import annotations

from pathlib import Path

from ..data import VideoFolder
from .builder import datasets


@datasets.register_module(name="Interviews")
class InterviewsDataset(VideoFolder):
    def get_label(self, path: str) -> dict[str, int]:
        filename = Path(path)
        user, interview_id = filename.parent.name.split("_")
        question_id, label = filename.name.split("_")
        label = label.split(".")[0]

        return dict(user=int(user), interview_id=int(interview_id), question_id=int(question_id), label=int(label))
