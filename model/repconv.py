"""Training/deployment re-parameterizable depthwise convolution."""

import mindspore as ms
from mindspore import nn, ops


def _fuse_conv_bn(conv, bn):
    scale = bn.gamma / ops.sqrt(bn.moving_variance + bn.eps)
    weight = conv.weight * ops.reshape(scale, (-1, 1, 1, 1))
    bias = bn.beta - bn.moving_mean * scale
    return weight, bias


class RepConv(nn.Cell):
    """RepDW-style block: depthwise 3x3+BN, 1x1, identity, outer BN.

    Input and output are ``(B,C,H,W)``. ``switch_to_deploy`` replaces all
    branches with one equivalent depthwise 3x3 convolution using frozen BN
    statistics; call it after ``set_train(False)``.
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = int(channels)
        self.deploy = False
        self.rbr_dense_conv = nn.Conv2d(channels, channels, 3, pad_mode="pad",
                                        padding=1, group=channels, has_bias=False)
        self.rbr_dense_bn = nn.BatchNorm2d(channels, eps=eps)
        self.rbr_1x1 = nn.Conv2d(channels, channels, 1, pad_mode="valid",
                                group=channels, has_bias=True)
        self.rbr_post_bn = nn.BatchNorm2d(channels, eps=eps)
        self.rbr_reparam = None

    def construct(self, x):
        if self.deploy:
            return self.rbr_reparam(x)
        identity = x
        dense = self.rbr_dense_bn(self.rbr_dense_conv(x))
        point = self.rbr_1x1(x)
        return self.rbr_post_bn(dense + point + identity)

    def _identity_kernel(self):
        center = ms.Tensor([[0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0]], ms.float32)
        return ops.tile(ops.reshape(center, (1, 1, 3, 3)),
                        (self.channels, 1, 1, 1))

    def switch_to_deploy(self):
        if self.deploy:
            return self
        w3, b3 = _fuse_conv_bn(self.rbr_dense_conv, self.rbr_dense_bn)
        w1 = ops.pad(self.rbr_1x1.weight, (1, 1, 1, 1))
        b1 = self.rbr_1x1.bias
        identity = self._identity_kernel()
        kernel = w3 + w1 + identity
        bias = b3 + b1
        scale = self.rbr_post_bn.gamma / ops.sqrt(
            self.rbr_post_bn.moving_variance + self.rbr_post_bn.eps)
        kernel = kernel * ops.reshape(scale, (-1, 1, 1, 1))
        bias = self.rbr_post_bn.beta + (bias - self.rbr_post_bn.moving_mean) * scale

        self.rbr_reparam = nn.Conv2d(self.channels, self.channels, 3,
                                     pad_mode="pad", padding=1, group=self.channels,
                                     has_bias=True)
        self.rbr_reparam.weight.set_data(kernel)
        self.rbr_reparam.bias.set_data(bias)
        self.deploy = True
        self.rbr_dense_conv = None
        self.rbr_dense_bn = None
        self.rbr_1x1 = None
        self.rbr_post_bn = None
        return self
