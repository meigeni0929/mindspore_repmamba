"""Small dependency-free forward smoke entry point."""

import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor

from .model import SMSR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--selective-scan-backend', choices=('reference', 'cpu_optimized'),
                        default='reference')
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    net = SMSR(upscale=args.scale,
               selective_scan_backend=args.selective_scan_backend)
    output = net(Tensor(np.zeros((1, 3, args.height, args.width), np.float32)))
    print('output shape:', output.shape)
    print('finite:', bool(np.isfinite(output.asnumpy()).all()))


if __name__ == '__main__':
    main()
