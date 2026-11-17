import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindspore_repmamba.ops.basic import (
    CAB, ChannelAttention, MLP, PatchEmbed, PatchUnEmbed, UpsampleOneStep,
)


def setup_module():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_seed(7)
    np.random.seed(7)


def test_patch_roundtrip():
    x = Tensor(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(np.float32))
    tokens = PatchEmbed(3, patch_norm=False)(x)
    restored = PatchUnEmbed(3)(tokens, 4, 5)
    assert tokens.shape == (2, 20, 3)
    np.testing.assert_array_equal(restored.asnumpy(), x.asnumpy())


def test_basic_shapes():
    x = Tensor(np.random.randn(2, 60, 8, 8).astype(np.float32))
    assert ChannelAttention(60, 30)(x).shape == x.shape
    assert CAB(60, 3, 30)(x).shape == x.shape
    tokens = Tensor(np.random.randn(2, 64, 16).astype(np.float32))
    assert MLP(16, 32)(tokens).shape == tokens.shape
    assert UpsampleOneStep(4, 60)(x).shape == (2, 3, 32, 32)


def test_pixelshuffle_exact_order():
    # Four channels at one pixel map to a row-major 2x2 patch.
    shuffle = ms.nn.PixelShuffle(2)
    x = Tensor(np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1))
    expected = np.array([[[[0, 1], [2, 3]]]], dtype=np.float32)
    np.testing.assert_array_equal(shuffle(x).asnumpy(), expected)
