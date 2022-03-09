# Mutimodal Lie Detector

- [Mutimodal Lie Detector](#mutimodal-lie-detector)
  - [1. Introduction](#1-introduction)
  - [2. Requirements](#2-requirements)
  - [3. Dependencies](#3-dependencies)
    - [3.1. System dependencies](#31-system-dependencies)
    - [3.2. Python dependencies](#32-python-dependencies)
    - [3.3. Docker containers](#33-docker-containers)
    - [3.4. Dev-dependencies](#34-dev-dependencies)
  - [4. Installation](#4-installation)
  - [5. Usage Examples](#5-usage-examples)
    - [5.0. Download dataset and weights](#50-download-dataset-and-weights)
    - [5.1. Train / Valid](#51-train--valid)
    - [5.2. Infer](#52-infer)
  - [6. Pre-defined configs](#6-pre-defined-configs)
  - [7. Customization](#7-customization)
    - [7.1. Custom dataset](#71-custom-dataset)
    - [7.2. Modify train/valid/infer parameters](#72-modify-trainvalidinfer-parameters)

## 1. Introduction


## 2. Requirements

- `Linux` or `WSL2` (`Windows` and `MacOS` will be supported when `mmcv` will be dropped)
- [Docker](https://www.docker.com/) or [Conda](conda.io)
- (Optional) `CUDA 10.2` or `11.3`


## 3. Dependencies

### 3.1. System dependencies

- `ffmpeg`
- `libgl1`
- `libsndfile1-dev`

### 3.2. Python dependencies

- `python`==3.9.x
- [torch](https://github.com/pytorch/pytorch)==1.10.2
- [torchvision](https://github.com/pytorch/vision)==0.11.3
- [torchaudio](https://github.com/pytorch/audio)==0.10.2
- [mmcv-full](https://github.com/open-mmlab/mmcv)==1.4.6 (will be dropped in future)
- [mmdet](https://github.com/open-mmlab/mmdetection)==2.22.0 (will be dropped in future)
- [mmaction2](https://github.com/open-mmlab/mmaction2)==0.21.0  (will be dropped in future)
- [catalyst](https://github.com/catalyst-team/catalyst)==22.2.1
- [einops](https://github.com/arogozhnikov/einops)==0.4.0
- [decord](https://github.com/dmlc/decord)==0.6.0
- [av](https://github.com/PyAV-Org/PyAV)==8.1.0
- [mediapipe](https://github.com/google/mediapipe)==0.8.9.1 (super slow, because uses tf lite XNNPACK, drop in replacement should be found)
- ~~[vg](https://github.com/lace/vg)==2.0.0~~ (dropped by pure torch operations and now support batching)
- [librosa](https://github.com/librosa/librosa)==0.9.0 (will be replaced by `torchaudio` in future)
- [pydub](https://github.com/jiaaro/pydub)==0.25.1 (wil be dropped in future)
- [soundfile](https://github.com/bastibe/python-soundfile)==0.10.3
- [noisereduce](https://github.com/timsainb/noisereduce)==2.0.0
- [moviepy](https://github.com/zulko/moviepy)==1.0.3  (wil be dropped in future)
- [Signal_Analysis](https://github.com/brookemosby/Speech_Analysis)==0.1.26 (will be dropped as soon as possible)

### 3.3. Docker containers

- Containers are based on:

  - CPU: [ubuntu:20.04](https://hub.docker.com/_/ubuntu)
  - CUDA 10.2: [nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04](https://hub.docker.com/r/nvidia/cuda) (`devel` will be dropped with `mmcv`)
  - CUDA 11.3: [nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04](https://hub.docker.com/r/nvidia/cuda) (`devel` will be dropped with `mmcv`)

- Visual Studio Code `dev-container` is supported

  1. Install [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extention and reopen the editor
  2. Click `Re-open in contrainer`
     - Use `.devcontainer/devcontainer.cpu.json` for `CPU` only container
     - Use `.devcontainer/devcontainer.cu102.json` for `CUDA 10.2` container
     - (default) Use `.devcontainer/devcontainer.json` for `CUDA 11.3` container

### 3.4. Dev-dependencies

<details>
    <summary>click to show</summary>

- [wemake-python-styleguide](https://github.com/wemake-services/wemake-python-styleguide)==0.16.0
  - flake8
  - flake8-commas
  - flake8-quotes
  - flake8-comprehensions
  - flake8-docstrings
  - flake8-string-format
  - flake8-bugbear
  - flake8-debugger
  - flake8-isort
  - flake8-eradicate
  - flake8-bandit
  - flake8-broken-line
  - flake8-rst-docstrings
  - pep8-naming
  - darglint
  - \+ extra rules
- [black](https://github.com/psf/black)==22.1.0
- [isort](https://github.com/PyCQA/isort)==5.10.1
- [mypy](https://github.com/python/mypy)==0.910
- [colorama](https://github.com/tartley/colorama)==0.4.4

</details>

## 4. Installation

<details>
    <summary>with docker (recommended)</summary>

- CPU

```bash
    docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cpu .
    docker run --rm -it --init -v `pwd`:/workspace/ <DOCKER IMAGE NAME> bash
```

- or CUDA 10.2

```bash
    docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cu102 .
    docker run --rm -it --init --gpus=all -v `pwd`:/workspace/ <DOCKER IMAGE NAME> bash
```

- or CUDA 11.3

```bash
    docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cu113 .
    docker run --rm -it --init --gpus=all -v `pwd`:/workspace/ <DOCKER IMAGE NAME> bash
```

</details>

<details>
    <summary>with conda</summary>

- Pre-requirements (GL1 и Soundfile)

```bash
    sudo apt install libgl1 libsndfile1-dev
```

- CPU

```bash
    conda <ENV NAME> create -f requirements/environment-cpu.yml
    conda activate <ENV NAME>
```

- or CUDA 10.2

```bash
    conda <ENV NAME> create -f requirements/environment-cu102.yml
    conda activate <ENV NAME>
```

- or CUDA 11.3

```bash
    conda <ENV NAME> create -f requirements/environment-cu113.yml
    conda activate <ENV NAME>
```

</details>

---

## 5. Usage Examples

### 5.0. Download dataset and weights

- [Download](https://drive.google.com/drive/folders/1CqCrIbcopweNAIHj6TwipXhNwAxHj49E) the dataset and place it into `data/` folder

- [Download](https://drive.google.com/drive/folders/1n3z2ID4qispJ77wDYcgtgxmLexGcnbpY?usp=sharing) all weights or specific ones and place them into `weights/` folder

- The dirs tree should be as follows:

```txt
...
assets/
configs
...
data/
    |
    interviews/
        |
        184937_143/
        |   |
        |   1_0.mp4
        |   ...
        2059949_12/
        ...
    ...
...
liedet/
requirements/
weights/
...
```


### 5.1. Train / Valid

```bash
    python {train|valid}.py {--config|-C} /path/to/config.py
```

### 5.2. Infer

```bash
    python infer.py {--config|-C} /path/to/config.py {--video|-V} /path/to/input_file.mp4
```

<br>

for example, to train Transformer Encoder on landmarks and angles features:

```bash
    python train.py -C configs/landmarks_transformer.py
```

to validate it:

```bash
    python valid.py -C configs/landmarks_transformer.py
```

to use it to generate prediction for a custom video example

```bash
    python infer.py -C configs/landmarks_transformer.py -V assets/example.mp4
```

## 6. Pre-defined configs

| Config | Video features                    | Audio Features  | Time features | Classifier
| --- |------------------------ | ---- | -------------- | ---
| configs/landmarks_transformer.py |Landmarks    | - | TransformerEncoder | 1x Linear
| configs/landmarks_audio_transformer.py |Landmarks    | Volume, Tone, ... | TransformerEncoder | 1x Linear
| x | Tinaface (face detection)<br>+ ResNet50 (feature extraction)    | - | TransformerEncoder | 1x Linear
| configs/tinaface_r50_audio_transformer.py | Tinaface (face detection)<br>+ ResNet50 (feature extraction)| Volume, Tone, ... | TransformerEncoder | 1x Linear
| x | Tinaface (face detection) | - | TimeSformer | 1x Linear
| configs/tinaface_r3d.py | Tinaface (face detection) | - | ResNet3D (r3d) | 1x Linear
| x | Tinaface (face detection) | Volume, Tone, ... | TimeSformer | 1x Linear
| x | Tinaface (face detection) | MelSpectrogram | Video: TimeSformer<br>Audio: AST | 1x Linear

</details>

## 7. Customization

### 7.1. Custom dataset

If custom dataset has following structure, you can use `VideoFolder` class. It extracts target class name (`int`) from a file name

```txt
...
data/custom_dataset/
    |
    any_folder_name1/
    |   |
    |   any_file_name1_<TARGET CLASS>.mp4
    |   |
    |   any_file_name2_<TARGET CLASS>.mp4
    |   ...
    any_folder_name1/
    ...
...
```

If custom dataset has other structure, you can inherit `VideoFolder` class and overwrite `get_meta(self, path) -> dict[str, Any]` method. For example, `RAVDESS` dataset uses the following filename structure, where `emotion` is a label:

`<MODALITY>-<VOICE>-<EMOTION>-<INTENSITY>-<STATEMEN>-<REPETITION>-<ACTOR>.mp4`

Use can overwrite `get_meta` as follows. It stores metadata of each file in `self.metas` attribute and uses `self.label_key` to extract target class from metadata. By default `self.label_key == "label"`, so your method output dict should contains key `label`, or you can pass `label_key` kwarg to the `CustomDataset` constructor.

```python
from pathlib import Path

from liedet.data import VideoFolder

class CustomDataset(VideoFolder):
    def get_meta(self, path: str) -> dict[str, Any]:
        filename = Path(path).name.split(".")[0]
        tokens = filename.split("-")

        return dict(
            modality=tokens[0],
            voice=tokens[1],
            emotion=tokens[2],
            label=tokens[2],
            intensity=tokens[3],
            statement=tokens[4],
            repetition=tokens[5],
            actor=tokens[6],
        )
```

If you would like to use this class in config files then you need to register it in registry by decorator as follows:

```python
# liedet/datasets/custom_dataset.py'

from liedet.data import VideoFolder
from liedet.datasets import datasets

@datasets.register_module()
class CustomDataset(VideoFolder):
    ...
```

and import it in `liedet/datasets/__init__.py` file

```python
# liedet/datasets/__init__.py'
...
from liedet.datasets.custom_dataset import CustomDataset
...
```

### 7.2. Modify train/valid/infer parameters

All configurations are stored in separate files in `config/` dirrectory. To modity something, for example `batch_size`, dataset `root`, preprocessing pipeline, model configuration and parts, `learning rate`, `metrics`, `callbacks`, `criterion` and other and other, just open and edit existing config file or copy it and edit.

To use custom dataset, modify `dataset` attribute in configuration file. If you register your dataset class you use it like:

```python
# configs/custom_config.py
...
dataset = dict(
    type="CustomDataset",
    root="path/to/custom_dataset",
    ...
)
...
```

otherwise, if you still want to use config, you can just import it

```python
# configs/custom_config.py
...
from path.to.custom_module import CustomDataset
...
dataset = dict(
    type=CustomDataset,
    root="path/to/custom_dataset",
    ...
)
...
```

See configuration examples in `configs/` folder for more information
