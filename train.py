import argparse

from catalyst import dl

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from mmcv.utils import Config

from liedet.datasets import build_dataset
from liedet.models.e2e import LieDetectorRunner
from liedet.models.registry import build


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", type=str, help="path to config", metavar="CONFIG_PATH", dest="config")

    args, _ = parser.parse_known_args()

    return vars(args)


def main():
    cfg = parse_args()["config"]

    cfg = Config.fromfile(cfg)
    cfg["model"].pop("init_cfg")

    dataset = build_dataset(cfg.dataset)
    train_set, valid_set = dataset.split(**cfg.dataset.split)
    loaders = dict(
        train_loader=DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True),
        valid_loader=DataLoader(valid_set, batch_size=cfg.batch_size, num_workers=0),
    )

    model = build(cfg.model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    runner = LieDetectorRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=200,
        valid_loader="valid_loader",
        valid_metric="accuracy01",
        minimize_valid_metric=False,
        callbacks=[
            dl.CriterionCallback(input_key="logits", target_key="labels", metric_key="loss"),
            dl.BackwardCallback(metric_key="loss"),
            dl.OptimizerCallback(metric_key="loss", accumulation_steps=int(16 // cfg.batch_size)),
            dl.AccuracyCallback(input_key="logits", target_key="labels", num_classes=2),
            dl.EarlyStoppingCallback(patience=15, loader_key="valid_loader", metric_key="loss", minimize=True),
            dl.CheckpointCallback(
                logdir="./logs",
                loader_key="valid_loader",
                metric_key="loss",
                minimize=True,
                topk=1,
            ),
        ],
        load_best_on_end=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
