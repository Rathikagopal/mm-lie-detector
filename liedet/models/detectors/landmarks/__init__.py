import os
import uuid

import numpy as np

import torch
import torch.nn as nn
from torchvision import io

from ...registry import registry
from .utils import landmarks


@registry.register_module(name="Landmarks")
class LandmarksFeatures(nn.Module):
    def __init__(self, fps: int = 30):
        super().__init__()

        self.video_fps = fps

    @torch.no_grad()
    def forward(self, x):
        batch_size = x.size(0)
        time_size = x.size(1)
        device = x.device
        x = x.cpu().numpy()

        h = []

        for i in range(batch_size):
            tmp_path = uuid.uuid4()
            tmp_path = f"/tmp/{tmp_path}.mp4"

            io.write_video(tmp_path, x[i], fps=self.video_fps)

            detector = landmarks.FaceDetector(video_path=tmp_path, normalize=True)

            landmarks_df = detector.get_landmarks()
            landmarks_df = detector.rotate_landmarks(df=landmarks_df, x="face_x", y="face_y", z="face_z")

            landmarks_data = []
            for idx, row in landmarks_df.iterrows():
                landmarks_row = []

                for item in row:
                    if isinstance(item, np.ndarray):
                        landmarks_row.extend(list(item))
                    else:
                        landmarks_row.append(float(item))

                landmarks_data.append(landmarks_row)

            os.remove(tmp_path)

            dT = time_size - len(landmarks_data)

            if dT > 0:
                landmarks_data = landmarks_data + [landmarks_data[-1]] * dT
            landmarks_data = landmarks_data[:time_size]

            h.append(landmarks_data)

        return torch.tensor(h, dtype=torch.float).to(device)
