"""Basic MindSpore layers with PyTorch-compatible tensor semantics."""

import math

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal


class DropPath(nn.Cell):
    """Stochastic depth on residual paths; input shape is ``(B, ...)``."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.uniform = ops.UniformReal()

    def construct(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = ops.floor(self.uniform(shape) + keep_prob)
        return x * ops.cast(mask, x.dtype) / keep_prob


class PatchEmbed(nn.Cell):
    """Convert ``(B,C,H,W)`` to row-major tokens ``(B,H*W,C)``."""

    def __init__(self, embed_dim, patch_norm=True, epsilon=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm((embed_dim,), epsilon=epsilon) if patch_norm else None

    def construct(self, x):
        b, c, h, w = x.shape
        x = ops.transpose(ops.reshape(x, (b, c, h * w)), (0, 2, 1))
        return self.norm(x) if self.norm is not None else x


class PatchUnEmbed(nn.Cell):
    """Convert row-major tokens ``(B,H*W,C)`` to ``(B,C,H,W)``."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = int(embed_dim)

    def construct(self, x, height, width):
        b = x.shape[0]
        x = ops.transpose(x, (0, 2, 1))
        return ops.reshape(x, (b, self.embed_dim, height, width))


class ChannelAttention(nn.Cell):
    """RCAN channel attention; input/output shape ``(B,C,H,W)``."""

    def __init__(self, channels, squeeze_factor=16):
        super().__init__()
        hidden = channels // squeeze_factor
        if hidden < 1:
            raise ValueError("channels // squeeze_factor must be positive")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, hidden, 1, pad_mode="valid", has_bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden, channels, 1, pad_mode="valid", has_bias=True)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        gate = self.sigmoid(self.conv2(self.relu(self.conv1(self.pool(x)))))
        return x * gate


class CAB(nn.Cell):
    """Channel-attention block; input/output shape ``(B,C,H,W)``."""

    def __init__(self, channels, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        hidden = channels // compress_ratio
        self.body = nn.SequentialCell(
            nn.Conv2d(channels, hidden, 3, pad_mode="pad", padding=1, has_bias=True),
            nn.GELU(approximate=False),
            nn.Conv2d(hidden, channels, 3, pad_mode="pad", padding=1, has_bias=True),
            ChannelAttention(channels, squeeze_factor),
        )

    def construct(self, x):
        return self.body(x)


class MLP(nn.Cell):
    """Token MLP mapping ``(B,L,C) -> (B,L,C)``."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.gelu = nn.GELU(approximate=False)
        self.drop1 = nn.Dropout(p=drop)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop2 = nn.Dropout(p=drop)

    def construct(self, x):
        return self.drop2(self.fc2(self.drop1(self.gelu(self.fc1(x)))))


class ConvFFN(nn.Cell):
    """Rep-Mamba FFN with a residual depthwise 5x5 branch on ``(B,L,C)``."""

    def __init__(self, channels, hidden_channels, drop=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.fc1 = nn.Dense(channels, hidden_channels)
        self.gelu = nn.GELU(approximate=False)
        self.dwconv = nn.Conv2d(
            hidden_channels, hidden_channels, 5, pad_mode="pad", padding=2,
            group=hidden_channels, has_bias=True,
        )
        self.fc2 = nn.Dense(hidden_channels, channels)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x, height, width):
        # x: (B,H*W,C) -> hidden: (B,H*W,hidden)
        hidden = self.gelu(self.fc1(x))
        b = hidden.shape[0]
        image = ops.reshape(ops.transpose(hidden, (0, 2, 1)),
                            (b, self.hidden_channels, height, width))
        image = self.gelu(self.dwconv(image))
        image = ops.transpose(ops.reshape(image, (b, self.hidden_channels, height * width)),
                              (0, 2, 1))
        return self.drop(self.fc2(self.drop(hidden + image)))


class Upsample(nn.Cell):
    """Classical Conv+PixelShuffle upsampler for scales 2^n or 3."""

    def __init__(self, scale, channels):
        super().__init__()
        layers = []
        if scale > 0 and (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.extend([
                    nn.Conv2d(channels, 4 * channels, 3, pad_mode="pad", padding=1,
                              has_bias=True),
                    nn.PixelShuffle(2),
                ])
        elif scale == 3:
            layers.extend([
                nn.Conv2d(channels, 9 * channels, 3, pad_mode="pad", padding=1,
                          has_bias=True),
                nn.PixelShuffle(3),
            ])
        else:
            raise ValueError("scale must be a positive power of two or 3")
        self.body = nn.SequentialCell(layers)

    def construct(self, x):
        return self.body(x)


class UpsampleOneStep(nn.Cell):
    """Direct reconstruction head ``(B,C,H,W) -> (B,out,H*s,W*s)``."""

    def __init__(self, scale, channels, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, scale * scale * out_channels, 3,
                              pad_mode="pad", padding=1, has_bias=True)
        self.shuffle = nn.PixelShuffle(scale)

    def construct(self, x):
        return self.shuffle(self.conv(x))
