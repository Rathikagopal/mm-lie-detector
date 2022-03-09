import argparse

from catalyst import dl

import torch
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

    dataset = build_dataset(cfg.dataset)
    _, valid_set = dataset.split(**cfg.dataset.split)
    loader = DataLoader(valid_set, batch_size=cfg.batch_size, num_workers=0, drop_last=True)

    model = build(cfg.model)
    runner = LieDetectorRunner(model=model)

    runner.evaluate_loader(
        loader=loader,
        callbacks=[
            dl.BatchTransformCallback(
                input_key="logits", output_key="scores", scope="on_batch_end", transform=torch.sigmoid
            ),
            dl.AccuracyCallback(input_key="scores", target_key="labels", num_classes=2),
        ],
        verbose=True,
    )


if __name__ == "__main__":
    main()
