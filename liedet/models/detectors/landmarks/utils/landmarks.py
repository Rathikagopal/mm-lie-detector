import os
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import vg

import torch
import torch.nn as nn

from . import models


class FaceDetector(object):
    """Performs detection and analysis of facial landmarks on video.

    Parameters
    ----------
    video_path : path
        Path to the video file.
    normalize : bool
        Method for linear normalization of facial landmarks values.

    See Also
    --------
    mp.solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES : int
        Returns number of x, y, and z coordinates of facial landmarks (478 values each).
    mp.solutions.face_mesh.FACEMESH_CONTOURS : frozenset
        Returns set of x, y, and z coordinates of face mesh landmarks (128 values each).
    mp.solutions.face_mesh.FACEMESH_FACE_OVAL : frozenset
        Returns set of x, y, and z coordinates of face oval landmarks (36 values each).
    mp.solutions.face_mesh.FACEMESH_LEFT_EYE : frozenset
        Returns set of x, y, and z coordinates of left eye landmarks (16 values each).
    mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW : frozenset
        Returns set of x, y, and z coordinates of left eyebrow landmarks (10 values each).
    mp.solutions.face_mesh.FACEMESH_LEFT_IRIS : frozenset
        Returns set of x, y, and z coordinates of left iris landmarks (4 values each).
    mp.solutions.face_mesh.FACEMESH_RIGHT_EYE : frozenset
        Returns set of x, y, and z coordinates of right eye landmarks (16 values each).
    mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW : frozenset
        Returns set of x, y, and z coordinates of right eyebrow landmarks (10 values each).
    mp.solutions.face_mesh.FACEMESH_RIGHT_IRIS : frozenset
        Returns set of x, y, and z coordinates of right iris landmarks (4 values each).
    mp.solutions.face_mesh.FACEMESH_LIPS : frozenset
        Returns set of x, y, and z coordinates of lip landmarks (40 values each).
    """

    def __init__(self, video_path, normalize=False):
        self.video_path = video_path
        self.mp_face_mesh = mp.solutions.face_mesh
        self.normalize = normalize
        self.df = pd.DataFrame(
            columns=[
                "FACE_X",
                "FACE_Y",
                "FACE_Z",
                "FACEMESH_CONTOURS_X",
                "FACEMESH_CONTOURS_Y",
                "FACEMESH_CONTOURS_Z",
                "FACEMESH_FACE_OVAL_X",
                "FACEMESH_FACE_OVAL_Y",
                "FACEMESH_FACE_OVAL_Z",
                "FACEMESH_LEFT_EYE_X",
                "FACEMESH_LEFT_EYE_Y",
                "FACEMESH_LEFT_EYE_Z",
                "FACEMESH_LEFT_EYEBROW_X",
                "FACEMESH_LEFT_EYEBROW_Y",
                "FACEMESH_LEFT_EYEBROW_Z",
                "FACEMESH_LEFT_IRIS_X",
                "FACEMESH_LEFT_IRIS_Y",
                "FACEMESH_LEFT_IRIS_Z",
                "FACEMESH_RIGHT_EYE_X",
                "FACEMESH_RIGHT_EYE_Y",
                "FACEMESH_RIGHT_EYE_Z",
                "FACEMESH_RIGHT_EYEBROW_X",
                "FACEMESH_RIGHT_EYEBROW_Y",
                "FACEMESH_RIGHT_EYEBROW_Z",
                "FACEMESH_RIGHT_IRIS_X",
                "FACEMESH_RIGHT_IRIS_Y",
                "FACEMESH_RIGHT_IRIS_Z",
                "FACEMESH_LIPS_X",
                "FACEMESH_LIPS_Y",
                "FACEMESH_LIPS_Z",
            ]
        )

        self.mp_name = [
            "FACEMESH_CONTOURS",
            "FACEMESH_FACE_OVAL",
            "FACEMESH_LEFT_EYE",
            "FACEMESH_LEFT_EYEBROW",
            "FACEMESH_LEFT_IRIS",
            "FACEMESH_RIGHT_EYE",
            "FACEMESH_RIGHT_EYEBROW",
            "FACEMESH_RIGHT_IRIS",
            "FACEMESH_LIPS",
        ]

        self.mp_method = [
            self.mp_face_mesh.FACEMESH_CONTOURS,
            self.mp_face_mesh.FACEMESH_FACE_OVAL,
            self.mp_face_mesh.FACEMESH_LEFT_EYE,
            self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
            self.mp_face_mesh.FACEMESH_LEFT_IRIS,
            self.mp_face_mesh.FACEMESH_RIGHT_EYE,
            self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
            self.mp_face_mesh.FACEMESH_RIGHT_IRIS,
            self.mp_face_mesh.FACEMESH_LIPS,
        ]

    def get_properties(self):
        """Returns name, frame rate, and duration of the video."""

        cap = cv.VideoCapture(self.video_path)
        basename = os.path.basename(self.video_path)
        ret, frame = cap.read()

        if not ret:
            return print("No such video in the path!")

        frame_count = np.around(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = np.around(cap.get(cv.CAP_PROP_FPS))
        duration = np.around(frame_count / fps)

        print("Video {}".format(basename))
        print("FPS {}".format(fps))
        print("Duration {}".format(duration))

    def get_indices(self, mp_array):
        """Returns list of unique indexes of facial landmarks.

        Parameters
        ----------
        mp_array : frozenset
            Set of indexes of facial landmarks.
        """

        indices_list = np.array(list(mp_array))
        indices_list = indices_list.reshape(-1)
        return np.unique(indices_list)

    def get_coordinates(self, frame, index, column, mp_indices, normalize):
        """Returns x, y, z coordinates of facial lanmarks.

        Parameters
        ----------
        frame : DataFrame
            Spreadsheet for storing coordinates of facial lanmarks.
        index : int
            Spreadsheet index.
        column : str
            Spreadsheet column name.
        mp_indices : list
            List of unique indexes of facial landmarks.
        normalize : bool
            Method for linear normalization of facial landmarks values.
        """

        x = []
        y = []
        z = []

        for i in mp_indices:
            x.append(frame.multi_face_landmarks[0].landmark[i].x)
            y.append(frame.multi_face_landmarks[0].landmark[i].y)
            z.append(frame.multi_face_landmarks[0].landmark[i].z)

        # If normalize then performs linear normalization of facial landmarks values.
        if normalize:
            x = self.normalize_landmarks(x)
            y = self.normalize_landmarks(y)
            z = self.normalize_landmarks(z)

        self.df.loc[index, "{}_X".format(column)] = x
        self.df.loc[index, "{}_Y".format(column)] = y
        self.df.loc[index, "{}_Z".format(column)] = z

    def normalize_landmarks(self, array):
        """Performs linear normalization of facial landmarks values.

        Parameters
        ----------
        array : ndarray
            List of coordinates of facial landmarks.
        """

        max_value = np.max(array)
        min_value = np.min(array)

        # If constant then returns zero.
        if max_value == min_value:
            return np.zeros(len(array)).tolist()

        array = np.absolute(array - min_value) / np.absolute(max_value - min_value)
        return array

    def rotate_landmarks(self, df, x, y, z):
        """Performs orthogonal transformation of coordinates of facial landmarks.

        Parameters
        ----------
        df : DataFrame
            Spreadsheet for storing coordinate features.
        x : ndarray
            List of x coordinates of facial landmarks.
        y : ndarray
            List of y coordinates of facial landmarks.
        z : ndarray
            List of z coordinates of facial landmarks.
        """

        sample = pd.DataFrame(columns=["x_rotated", "y_rotated", "z_rotated", "x_angle", "y_angle", "z_angle"])

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = models.Regressor().to(device)
        path = Path(__file__).parent.parent / "weights/regressor.pth"
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        for index in range(len(df)):
            initial = np.array([df[x][index], df[y][index], df[z][index]])

            pre_initial = stack_landmarks(initial)
            pre_initial = center_landmarks(pre_initial)
            pre_initial = sphere_landmarks(pre_initial)
            pre_initial = flat_landmarks(pre_initial)

            pre_initial = torch.tensor(pre_initial).flatten().float().to(device)

            output = model(pre_initial).detach().cpu().numpy()

            x_angle = -output[0]
            y_angle = -output[1]
            z_angle = output[2]

            post_initial = stack_landmarks(initial)
            post_initial = center_landmarks(post_initial)

            post_initial = vg.rotate(post_initial, vg.basis.z, z_angle, units="rad", assume_normalized=True)
            post_initial = vg.rotate(post_initial, vg.basis.y, y_angle, units="rad", assume_normalized=True)
            post_initial = vg.rotate(post_initial, vg.basis.x, x_angle, units="rad", assume_normalized=True)

            post_initial = flat_landmarks(post_initial)

            sample.loc[index, "x_rotated"] = post_initial[0]
            sample.loc[index, "y_rotated"] = post_initial[1]
            sample.loc[index, "z_rotated"] = post_initial[2]
            sample.loc[index, "x_angle"] = x_angle
            sample.loc[index, "y_angle"] = y_angle
            sample.loc[index, "z_angle"] = z_angle
        return sample

    def get_features(self, df, x, y, z):
        """Calculates absolute value of difference between the coordinates and the center of mass.

        Parameters
        ----------
        df : DataFrame
            Spreadsheet for storing coordinate features.
        x : ndarray
            List of x coordinates of facial landmarks.
        y : ndarray
            List of y coordinates of facial landmarks.
        z : ndarray
            List of z coordinates of facial landmarks.
        """

        df_features = pd.DataFrame(columns=["features"])

        for row in range(len(df)):
            sub_features = []
            for index in range(478):
                length_x = np.sqrt(np.power(df[x][row][index] - np.mean(df[x][row]), 2))
                length_y = np.sqrt(np.power(df[y][row][index] - np.mean(df[y][row]), 2))
                length_z = np.sqrt(np.power(df[z][row][index] - np.mean(df[z][row]), 2))
                sub_features.append(np.sqrt(np.power(length_x, 2) + np.power(length_y, 2) + np.power(length_z, 2)))

            df_features.loc[row, "features"] = self.normalize_landmarks(sub_features)

        return df_features

    def get_landmarks(self, count=False):
        """Returns spreadsheet with x, y, z coordinates of facial lanmarks."""

        cap = cv.VideoCapture(self.video_path)

        with self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:
            """Runs MediaPipe Face Mesh.

            Parameters
            ----------
            max_num_faces : int
                Maximum number of faces to detect.
            refine_landmarks : bool
                Method for detecting landmarks of lips and eyes.
            min_detection_confidence : float
                Confidence level for face detection.
            min_tracking_confidence : float
                Confidence level for facial landmarks detection.
            """

            index = 0
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                output = face_mesh.process(frame)

                # If face then stores coordinates of facial landmarks.
                if output.multi_face_landmarks:

                    mp_indices = range(self.mp_face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES)
                    self.get_coordinates(
                        frame=output, index=index, column="FACE", mp_indices=mp_indices, normalize=self.normalize
                    )

                    for (
                        method,
                        name,
                    ) in zip(self.mp_method, self.mp_name):
                        mp_indices = self.get_indices(mp_array=method)
                        self.get_coordinates(
                            frame=output, index=index, column=name, mp_indices=mp_indices, normalize=self.normalize
                        )
                    index = index + 1
                    if count:
                        if count == index:
                            break

        self.df.rename(columns=str.lower, inplace=True)
        return self.df


def test(path):
    x = []
    y = []
    z = []

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    ) as face_mesh:

        image = cv.imread(path)
        output = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if output.multi_face_landmarks:
            mp_indices = range(mp_face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES)

            for idx in mp_indices:
                x.append(output.multi_face_landmarks[0].landmark[idx].x)
                y.append(output.multi_face_landmarks[0].landmark[idx].y)
                z.append(output.multi_face_landmarks[0].landmark[idx].z)

            return np.array([x, y, z])


def show(array):
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(projection="3d")
    ax.view_init(elev=-90, azim=-90)

    ax.scatter(array[0], array[1], array[2], c="black", s=50, alpha=1)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.axis("off")
    plt.show()


def draw(path, color="#15B01A", size=10):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    array = test(path)

    figure = plt.figure(figsize=(15, 15))

    plt.scatter(array[0] * width, array[1] * height, c=color, s=size, alpha=1)

    plt.axis("off")
    plt.imshow(image)


def stack_landmarks(array):
    array = np.column_stack([np.array(array[0]).flatten(), np.array(array[1]).flatten(), np.array(array[2]).flatten()])
    return array


def flat_landmarks(array):
    array = np.column_stack(array)
    return array


def center_landmarks(array):
    array = array - np.mean(array)
    return array


def sphere_landmarks(array):
    norm = np.linalg.norm(array, axis=-1)
    array = array / np.tile(np.expand_dims(norm, 1), [1, 3])
    return array
