"""FP32 CPU losses used by the original project or useful for validation."""

import math
import mindspore as ms
from mindspore import nn, ops, Tensor


class L1Loss(nn.Cell):
    def construct(self, prediction, target): return ops.mean(ops.abs(prediction - target))


class MSELoss(nn.Cell):
    def construct(self, prediction, target): return ops.mean(ops.square(prediction - target))


class CharbonnierLoss(nn.Cell):
    def __init__(self, eps=1e-3): super().__init__(); self.eps = float(eps)
    def construct(self, prediction, target):
        difference = prediction - target
        return ops.mean(ops.sqrt(difference * difference + self.eps * self.eps))


class EdgeLoss(nn.Cell):
    def __init__(self, eps=1e-3):
        super().__init__(); one = Tensor([[.05, .25, .4, .25, .05]], ms.float32)
        kernel = ops.matmul(ops.transpose(one, (1, 0)), one).reshape((1, 1, 5, 5))
        self.kernel = ops.tile(kernel, (3, 1, 1, 1)); self.loss = CharbonnierLoss(eps)
    def laplacian(self, image):
        # MindSpore CPU reflect/replicate padding support varies; symmetric padding is explicit.
        filtered = ops.conv2d(ops.pad(image, (2, 2, 2, 2)), self.kernel, groups=3)
        down = filtered[:, :, ::2, ::2]
        up = ops.zeros_like(filtered); up[:, :, ::2, ::2] = down * 4
        filtered2 = ops.conv2d(ops.pad(up, (2, 2, 2, 2)), self.kernel, groups=3)
        return image - filtered2
    def construct(self, prediction, target): return self.loss(self.laplacian(prediction), self.laplacian(target))


def psnr(prediction, target, max_value=1.0):
    mse = ops.mean(ops.square(prediction - target))
    return 10.0 * ops.log(Tensor(max_value * max_value, ms.float32) / mse) / math.log(10.0)
