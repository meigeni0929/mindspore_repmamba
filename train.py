"""Minimal FP32 MindSpore training-step entry point for the reference model."""

import argparse
import mindspore as ms
from mindspore import nn, ops

from .model import SMSR


class L1LossStep(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.loss = nn.L1Loss()

    def construct(self, image, target):
        return self.loss(self.network(image), target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--selective-scan-backend', choices=('reference', 'cpu_optimized'),
                        default='reference')
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = SMSR(upscale=args.scale,
               selective_scan_backend=args.selective_scan_backend)
    train_step = nn.TrainOneStepCell(L1LossStep(net), nn.Adam(net.trainable_params(), args.lr))
    print('created reference training step:', train_step)


if __name__ == '__main__':
    main()
