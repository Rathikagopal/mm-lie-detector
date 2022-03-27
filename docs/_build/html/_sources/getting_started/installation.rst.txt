Quickstart
===========

Step 1 - Installation: Docker
------------------------------

The preffered way is to use `Docker`_ container.

`Docker`_ containers are based on:

- CPU: `ubuntu:20.04`_

.. code-block:: bash

    docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cpu .
    docker run --rm -it -p 7999:7999 --init -v `pwd`:/workspace/<DOCKER IMAGE NAME> bash

- CUDA 10.2: `nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04`_

.. code-block:: bash

    docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cu102 .
    docker run --rm -it -p 7999:7999 --init --gpus=all -v `pwd`:/workspace/ <DOCKER IMAGE NAME> bash

.. note::

    Your host system must have the appropriate nvidia driver and CUDA version.

- CUDA 11.3: `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`_

.. code-block:: bash

    docker build -t <DOCKER IMAGE NAME> -f .devcontainer/Docker.cu113 .
    docker run --rm -it -p 7999:7999 --init --gpus=all -v `pwd`:/workspace/ <DOCKER IMAGE NAME> bash

.. note::

    Your host system should have the appropriate nvidia driver and CUDA version.

This commands start the container and attach shell to it.
They also serve Jupyter Notebook on 7999 port
which can be accessed via browser via `127.0.0.1:7999`_ url
without password or token.

.. _`Docker`: https://www.docker.com/
.. _`ubuntu:20.04`: https://hub.docker.com/_/ubuntu
.. _`nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04`: https://hub.docker.com/r/nvidia/cuda
.. _`nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`: https://hub.docker.com/r/nvidia/cuda
.. _`127.0.0.1:7999`: http://127.0.0.1:7999/


or Step 1 - Installation: Conda
--------------------------------

Using `conda`_ is also a good way as it helps to isolate production environment
and have consistent dependencies.
But in out case it still requires some extra libraries
which should be installed on the host system.

First of all you need to install `gl1`_ and `soundfile`_ libraries.
It can be done by a default package manager of your system.

.. code-block:: bash

    # Ubuntu
    sudo apt-get install libgl1 libsndfile-dev


Then you can bootstrap the dependencies
depending on the target environment.

- CPU

.. code-block:: bash

    conda <ENV NAME> create -f requirements/environment-cpu.yml

- CUDA 10.2

.. code-block:: bash

    conda <ENV NAME> create -f requirements/environment-cu102.yml

- CUDA 11.3

.. code-block:: bash

    conda <ENV NAME> create -f requirements/environment-cu113.yml


.. note::

    Please note that this installation method has only been tested on Ubuntu 20.04. If you are using a other system, you may need to install some additional dependencies.

.. _`conda`: https://conda.io/
.. _`gl1`: https://github.com/bastibe/python-soundfile
.. _`soundfile`: https://github.com/bastibe/python-soundfile


Step 2 - Download datasets and weights of the pretrained models
----------------------------------------------------------------

- `Download datasets`_ and place them into `data/` folder.
- `Download weights`_  and place them into `weights/` folder
- The dirs three should be as follows:

.. code-block:: text

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


.. _`Download datasets`: https://drive.google.com/drive/folders/1CqCrIbcopweNAIHj6TwipXhNwAxHj49E
.. _`Download weights`: https://drive.google.com/drive/folders/1n3z2ID4qispJ77wDYcgtgxmLexGcnbpY?usp=sharing

Step 3 - Train / Validate / Inference models
---------------------------------------------

For training/validation, you can use the appropriate scripts:

.. code-block:: text

    python {train|valid}.py {--config|-C} <PATH TO CONFIG>

and for inference:

.. code-block:: text

    python infer.py {--config|-C} <PATH TO CONFIG> {--video|-V} <PATH TO VIDEO>

For example, for training Transformer Encoder on face landmarks and angles,
you can use the following command:

.. code-block:: text

    python train.py -C configs/landmarks_transformer.py

similarly, for its validation:

.. code-block:: text

    python valid.py -C configs/landmarks_transformer.py

and similarly, for generating predictions for a custom video:

.. code-block:: text

    python infer.py -C configs/landmarks_transformer.py -V assets/example.mp4

Step 4 - Customization
-----------------------

4.1 Custom dataset
~~~~~~~~~~~~~~~~~~~

If a custom dataset has the following structure,
you can use `VideoFolder`_ class.
It extracts target class name (`int`) from a file name.

.. code-block:: text

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

If a custom dataset has an other structure,
you can inherit `VideoFolder`_ class and overwrite `get_meta` method.

For example, `RAVDESS`_ dataset uses the following filename structure,
where `emotion` is a label:

.. code-block:: text

    <MODALITY>-<VOICE>-<EMOTION>-<INTENSITY>-<STATEMEN>-<REPETITION>-<ACTOR>.mp4

In this case, you can overwrite the `get_meta` as follows:

.. code-block:: python

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

It stores tags of each file in `self.metas` attribute
and uses `self.label_key` to extract target class label from metadata.
By default `self.label_key == "label"`,
so your method should return a dict object which contains key `label`,
or you can pass `label_key` kwarg to a `CustomDataset` constructor.

.. _`VideoFolder`: liedet/data/video_folder.py
.. _`RAVDESS`: https://paperswithcode.com/dataset/ravdess

If you would like to use this class in the config files
when you need to register it in registry by decorator as follows:

.. code-block:: python

    # liedet/datasets/custom_dataset.py'

    from liedet.data import VideoFolder
    from liedet.datasets import datasets

    @datasets.register_module()
    class CustomDataset(VideoFolder):
        ...

and import it in `liedet/datasets/__init__.py` file:

.. code-block:: python

    ...
    from liedet.datasets.custom_dataset import CustomDataset
    ...

After that you can use your `CustomDataset` in config files as follows:

.. code-block:: python

    # configs/custom_config.py

    ...
    dataset = dict(
        type="CustomDataset",
        root="<PATH TO CUSTOM DATASET>",
        ...
    )
    ...

Another way is to import a `CustomDataset` directly to a config file:

.. code-block:: python

    # configs/custom_config.py

    ...
    from path.to.custom.module import CustomDataset
    ...
    dataset = dict(
        type=CustomDataset,
        root="<PATH TO CUSTOM DATASET>",
        ...
    )
    ...

4.2 Custom model
~~~~~~~~~~~~~~~~~

Similarly to custom dataset, you can implement and use your own model.
The model should be the submodule of `torch.nn.Model`_ class.


4.3 Modifying training/validation/inference parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All pre-defined configurations are stored in separate files
in `configs/`_ directory.
For modifying parameters like batch_size, dataset,
pre-processing pipeline, model configuration and its parts,
learning rate, metrics, callbacks, critetion and other and other,
you can just open and edit existing configuration file or copy it and edit.

You can use any libraries and their components
as long as they are compatible with pytorch.
For example, `torchvision`_, `torchaudio`_, `torchtext`_,
`timm`_, `open-mmlab ecosystem`_, `pytorch-toolbelt`_, and others.

See configuration examples in `configs/`_ folder for more information.

.. _`torch.nn.Model`: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
.. _`configs/`: configs/
.. _`torchvision`: https://github.com/pytorch/vision
.. _`torchaudio`: https://github.com/pytorch/audio
.. _`torchtext`: https://github.com/pytorch/text
.. _`timm`: https://github.com/rwightman/pytorch-image-models
.. _`open-mmlab ecosystem`: https://github.com/open-mmlab
.. _`pytorch-toolbelt`: https://github.com/BloodAxe/pytorch-toolbelt
