from __future__ import annotations

from einops import rearrange
from mediapipe.python.solutions import face_mesh

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import io

from liedet.models.base_module import BaseModule

from ...registry import registry
from .rotate_regressor import Regressor


@registry.register_module()
class FaceLandmarks(BaseModule):
    def __init__(
        self,
        window: int | None = None,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        normalize: bool = True,
        rotate: bool = True,
        init: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.window = window

        self.static_image_mode = static_image_mode if window is None else False
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.norm = normalize
        self.rot = rotate

        self.regressor = Regressor()

        if init:
            self.init_weights()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        device = x.device

        self.regressor.to(device)
        if self.window is not None:
            h = rearrange(x, "b t c h w -> b t h w c")
            # h = rearrange(tensor=x, pattern="(b t) c h w -> b t h w c", t=self.window)
        else:
            h = rearrange(tensor=x, pattern="(b t) c h w -> b t h w c", b=1)

        h = h.cpu().numpy().astype("uint8")

        batch_landmarks = []
        for chunk in h:
            with face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as fm:
                chunk_landmarks = []
                # FIXME: It will be better to use .zeros and clip gradient,
                #   but catalyst's BackwardCallback with grad_clip_fn has bag: it uses undefined variable `model`
                prev_landmarks = torch.rand(size=(3, 478), dtype=torch.float).to(device)
                for frame in chunk:
                    landmarks = fm.process(frame)
                    if landmarks.multi_face_landmarks:
                        landmarks = landmarks.multi_face_landmarks[0].landmark
                        landmarks = [
                            torch.tensor((landmark.x, landmark.y, landmark.z), dtype=torch.float)
                            for landmark in landmarks
                        ]
                        landmarks = torch.stack(landmarks).T.to(device)

                        prev_landmarks = landmarks
                    else:
                        landmarks = prev_landmarks

                    chunk_landmarks.append(landmarks)
                batch_landmarks.append(torch.stack(chunk_landmarks))

        h = torch.stack(batch_landmarks)

        del batch_landmarks

        if self.norm:
            h = self.normalize(x=h)
        if self.rot:
            h = self.rotate(x=h)

        return h

    def normalize(self, x: Tensor) -> Tensor:
        min_value, max_value = x.min(dim=-1, keepdim=True).values, x.max(dim=-1, keepdim=True).values

        if min_value.allclose(max_value):
            return torch.zeros(x.size(), dtype=torch.float).to(x.device)

        return (x - min_value).abs() / (max_value - min_value).abs()

    def _rotate(self, x: Tensor, axis: Tensor, angles: Tensor) -> Tensor:
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        dot_products = torch.inner(axis, x)
        dot_products = rearrange(dot_products, "c k m b t f -> b t f (c k m)")
        cross_products = torch.linalg.cross(axis, x, dim=-1)

        return cos_angles * x + sin_angles * cross_products + (1 - cos_angles) * dot_products * axis

    def rotate(self, x: Tensor) -> Tensor:
        h: Tensor = x - x.mean(dim=1, keepdim=True)
        h = h / torch.norm(h, dim=-1, keepdim=True)

        angles = self.regressor(h.flatten(start_dim=2))
        angles[..., 0] = -angles[..., 0]
        angles[..., 1] = -angles[..., 1]

        h = rearrange(h, "b t c f -> b t f c")
        angles = rearrange(angles, "b t (c k f) -> b t f k c", f=1, k=1)

        basis = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=torch.float,
        )
        basis = basis.view(1, 1, 1, 3, 3).to(x.device)

        h = self._rotate(h, basis[..., 2], angles[..., 2])
        h = self._rotate(h, basis[..., 1], angles[..., 1])
        h = self._rotate(h, basis[..., 0], angles[..., 0])

        h = torch.cat((h, angles[..., 0, :]), dim=-2)
        h = rearrange(h, "b t f c -> b t c f").flatten(start_dim=-2)

        return h
