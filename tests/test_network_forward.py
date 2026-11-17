import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindspore_repmamba.model import SMSR


def test_network_x4_forward_64():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    net = SMSR(upscale=4, depths=(1,), selective_scan_backend="cpu_optimized")
    x = Tensor(np.random.default_rng(43).standard_normal((1, 3, 64, 64)).astype(np.float32))
    y = net(x)
    assert y.shape == (1, 3, 256, 256)
    assert np.isfinite(y.asnumpy()).all()


def test_network_x2_x3_shapes():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = Tensor(np.random.default_rng(44).standard_normal((1, 3, 8, 8)).astype(np.float32))
    assert SMSR(upscale=2, depths=(), selective_scan_backend="cpu_optimized")(x).shape == (1, 3, 16, 16)
    assert SMSR(upscale=3, depths=(), selective_scan_backend="cpu_optimized")(x).shape == (1, 3, 24, 24)


def test_network_accepts_reference_backend():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = Tensor(np.zeros((1, 3, 4, 4), np.float32))
    net = SMSR(upscale=2, depths=(1,), selective_scan_backend="reference")
    assert net(x).shape == (1, 3, 8, 8)
