# Mutimodal Lie Detector

## Installation

<details>
    <summary>with conda</summary>

- CPU

```bash
conda <ENV NAME> create -f /workspace/requirements/environment-cpu.yml
conda activate <ENV NAME>
```

- cuda

```bash
conda <ENV NAME> create -f /workspace/requirements/environment-cuda.yml
conda activate <ENV NAME>
```

</details>

<details>
    <summary>with docker</summary>

- CPU

```bash
docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cpu .
docker run --rm -it --init -v `pwd`:/workspace/ <DOCKER IMAGE NAME> /bin/bash
```

- cuda

```bash
docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cuda .
docker run --rm -it --init --gpus=all -v `pwd`:/workspace/ <DOCKER IMAGE NAME> /bin/bash
```

</details>
