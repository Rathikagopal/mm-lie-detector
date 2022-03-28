Model Training / Validation / Inference Example
================================================

CLI way
--------------------

The simplest way is to use `train.py`_ / `valid.py`_  / `infer.py`_ files
and one of pre-defined `configuration files`_.

Train model
~~~~~~~~~~~~

.. code-block:: bash

    # SYNTAX: python train.py --config <PATH TO CONFIG FILE>

    # for example
    python train.py --config configs/landmarks_transformer.py

Evaluate model
~~~~~~~~~~~~~~~

.. code-block:: bash

    # SYNTAX: python valid.py --config <PATH TO CONFIG FILE>

    # for example
    python valid.py --config configs/landmarks_transformer.py


Inference model on custom video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # SYNTAX: python infer.py --config <PATH TO CONFIG FILE> --video <PATH TO VIDEO FILE>

    # for example
    python infer.py --config configs/landmarks_transformer.py --video assets/example.mp4

.. _`train.py`: https://github.com/digital-profiling/mm-lie-detector/blob/master/train.py
.. _`valid.py`: https://github.com/digital-profiling/mm-lie-detector/blob/master/valid.py
.. _`infer.py`: https://github.com/digital-profiling/mm-lie-detector/blob/master/infer.py
.. _`configuration files`: https://github.com/digital-profiling/mm-lie-detector/tree/master/configs

Manual way
-----------

For example, let's try to classify truth and lies on video interviews.
To solve this problem, let's train, validate and inference Transformer Encoder
on face landmarks and rotation angles extracted from 10 seconds of video
(stream of images) with 10 video frames per second, 320 pixels height
and 480 pixels in width and
audio features extracted from corresponding audio waves
with 16000 sample rate and single mono channel.

Let's divide dataset on training and validation subsets
with 0.2 ratio of validation part.

Build dataset
~~~~~~~~~~~~~~

- Config way

.. code-block:: python

    from mmcv.utils import Config

    from liedet.datasets import build_dataset

    dataset_cfg = Config.fromfile("configs/datasets/interviews.py")
    dataset = build_dataset(dataset_cfg)
    train_set, valid_set = dataset.split(dataset_cfg.split)


- Class way

.. code-block:: python

    from liedet.datasets import Interviews

    #
    # Default parameters
    #

    # target video fps, real video fps is adjusted to this value
    video_fps = 10
    # target audio fps (sample rate), real audio fps is adjusted to this value
    audio_fps = 16000
    # time window in seconds
    window_secs = 10
    # time window in video frames
    window = video_fps * window_secs

    # height of video frame
    height = 320
    # width of video frame
    width = 480

    dataset = Interviews(
        root="data/interviews",
        video_fps=video_fps,
        audio_fps=audio_fps,
        window=window,
        # target height of video frame,
        #  real height is adjusted to this value
        height=320,
        # target width of video frame,
        #  real width is adjusted to this value
        width=480,
        # target number of audio channel,
        #  True is single mono channel,
        #  otherwise two stereo channels are used
        mono=True,
    )

    train_set, valid_set = dataset.split(
        valid_size=0.2,
        # return balanced validation subset with respect to target labels
        balanced_valid_set=True,
        # use whole files to split
        by_file=True,
    )


The `Interviews` dataset is already implement `torch.utils.data.Dataset`,
so it can be directly passed to `torch.utils.data.Dataloader`.

.. code-block:: python

    from torch.utils.data import Dataloader

    batch_size = 16

    loaders = dict(
        train_loader=Dataloader(train_set, batch_size=batch_size, shuffle=True),
        valid_loader=Dataloader(valid_set, batch_size=batch_size, shuffle=False),
    )


Build model
~~~~~~~~~~~~

Next, let's build our model, criterion and optimizer.

- Config way

.. code-block:: python

    import torch.nn as nn
    from torch import optim

    from liedet.models.registry import build

    model_cfg = Config.fromfile("configs/landmarks_audio_transformer.py")
    model = build(model_cfg.model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())


- Class way

.. code-block:: python

    import torch
    from torch.nn import TransformerEncoderLayer

    from liedet.models import LieDetector, FaceLandmarks, AudioFeatures, TransformerEncoder

    # number of features (landmarks + angles + audio features == features_dims)
    features_dims = 1455
    # size of embedding  (features_dims --(Linear)--> embed_dims)
    embed_dims = 512
    # number of classes  (binary == 2)
    num_classes = 2

    landmarks_model = FaceLandmarks(window=window)
    # load pre-trained weights of rotation angles regressor
    landmarks_model.load_state_dict(torch.load("weights/angles_regressor.pth", map_location="cpu"))

    audio_model = AudioFeatures(fps=window_secs, chunk_length=1, sr=audio_fps, normalization=True)

    time_model = TransformerEncoder(
        encoder_layer=TransformerEncoderLayer(
            d_model=embed_dims,
            # number of parallel computed self-attentions
            nhead=16,
            # size of embedding inside feed-forward network
            dim_feedforward=512*4,
            # probability of dropout
            dropout=0.5,
            # batch is the first dimension
            batch_first=True,
        )
    )

    cls_head = nn.Linear(in_features=embed_dims, out_features=num_classes)

    # model pipeline
    model = LieDetector(
        video_model=landmarks_model,
        audio_model=audio_model,
        features_dims=features_dims,
        embed_dims=embed_dims,
        time_model=time_model,
        cls_head=cls_head,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())


Train model
~~~~~~~~~~~~

Finally, let's use `catalyst`_ framework to train model.

- Catalyst way

.. code-block:: python

    from catalyst import dl

    from liedet.models import LieDetectorRunner

    runner = LieDetectorRunner(model=model)
    runner.train(
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=200,
        callbacks=[
            # use criterion and calculate loss
            dl.CriterionCallback(input_key="logits", target_key="labels", metric_key="loss"),
            # backward loss
            dl.BackwardCallback(metric_key="loss"),
            # step optimizer
            dl.OptimizerCallback(metric_key="loss"),
            # calculate runtime accuracy
            dl.AccuracyCallback(input_key="logits", target_key="labels", num_classes=2),
            # early stop training if last 15 epochs models is outfitted
            #   use validation loss to monitor it
            dl.EarlyStoppingCallback(patience=15, loader_key="valid_loader", metric_key="loss", minimize=True),
            # checkpoint best model and model after last epoch
            dl.CheckpointCallback(logdir="./logs", loader_key="valid_loader", metric_key="loss", minimize=True, topk=1),
        ],
        load_best_on_end=True,
    )


.. _`catalyst`: https://github.com/catalyst-team/catalyst

- Manual way (#TODO)


Evaluate model
~~~~~~~~~~~~~~~

When the model is trained, let's evaluate it on validation subset
to assess final accuracy.

- Catalyst way

.. code-block:: python

    runner.evaluate_loader(
        loader=loaders["valid_loader"],
        callbacks=[
            # transform logits to probabilities using sigmoid function
            dl.BatchTransformCallback(
                input_key="logits",
                output_key="scores",
                scope="on_batch_end",
                transform=torch.sigmoid,
            ),
            # calculate final accuracy
            dl.AccuracyCallback(input_key="scores", target_key="labels", num_classes=2),
        ],
    )


- Manual way (#TODO)


Inference model on custom video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's try to inference obtained model on custom video file.

- Catalyst way

.. code-block:: python

    from liedet.data import VideoReader

    # Config way
    video_reader = VideoReader(uri="assets/example.mp4", **dataset_cfg)

    # Class way
    video_reader = VideoReader(
        uri="assets/example.mp4",
        video_fps=video_fps,
        audio_fps=audio_fps,
        window=window,
        # target height of video frame,
        #  real height is adjusted to this value
        height=320,
        # target width of video frame,
        #  real width is adjusted to this value
        width=480,
        # target number of audio channel,
        #  True is single mono channel,
        #  otherwise two stereo channels are used
        mono=True,
    )

    for start in range(0, length, window):
        # extract single window
        sample = video_reader[start : start + window]

        # predict label for extracted window
        window_label = runner.predict_sample(sample)

- Manual way (#TODO)
