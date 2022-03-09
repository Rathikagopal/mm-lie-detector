import argparse

from catalyst import dl

import torch
from torch.utils.data import DataLoader

from mmcv.utils import Config

from liedet.data import VideoReader
from liedet.datasets import build_dataset
from liedet.models.e2e import LieDetectorRunner
from liedet.models.registry import build


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-C", type=str, help="path to config", metavar="CONFIG_PATH", dest="config")
    parser.add_argument("--video", "-V", type=str, help="path to video file", metavar="VIDEO_PATH", dest="video_path")

    args, _ = parser.parse_known_args()

    return vars(args)


def main():
    args = parse_args()
    cfg = args["config"]
    video_path = args["video_path"]

    cfg = Config.fromfile(cfg)

    vr = VideoReader(uri=video_path, **cfg.dataset)
    length = len(vr)

    model = build(cfg.model)
    runner = LieDetectorRunner(model=model)

    for start in range(0, length, cfg.window):
        sample = vr[start : start + cfg.window]

        print(runner.predict_sample(sample))


if __name__ == "__main__":
    main()
