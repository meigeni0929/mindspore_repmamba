import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor, nn

from mindspore_repmamba.model import SMSR
from mindspore_repmamba.ops.selective_scan import selective_scan
from mindspore_repmamba.tools.pytorch_reference import selective_scan as torch_scan


def test_network_single_backward_graph_mode():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    net = SMSR(upscale=2, depths=(), selective_scan_backend="cpu_optimized")
    x = Tensor(np.random.default_rng(45).standard_normal((1, 3, 8, 8)).astype(np.float32))
    class MeanOutput(nn.Cell):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def construct(self, z):
            return self.model(z).mean()

    value, grad = ms.value_and_grad(MeanOutput(net), grad_position=0)(x)
    assert value.shape == ()
    assert grad.shape == x.shape
    assert np.isfinite(grad.asnumpy()).all()


def test_selective_scan_gradients_match_torch():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    rng = np.random.default_rng(46)
    arrays = [
        rng.standard_normal((1, 2, 4), dtype=np.float32),  # u
        rng.standard_normal((1, 2, 4), dtype=np.float32),  # delta
        -np.exp(rng.standard_normal((2, 2), dtype=np.float32)),  # A
        rng.standard_normal((1, 2, 4), dtype=np.float32),  # B
        rng.standard_normal((1, 2, 4), dtype=np.float32),  # C
        rng.standard_normal((2,), dtype=np.float32),  # D
        rng.standard_normal((2,), dtype=np.float32),  # delta bias
    ]

    def ms_loss(u, delta, a, b, c, d, bias):
        return selective_scan(u, delta, a, b, c, d, bias, True).sum()

    ms_inputs = tuple(Tensor(item) for item in arrays)
    _, ms_grads = ms.value_and_grad(ms_loss, grad_position=tuple(range(7)))(*ms_inputs)
    torch_inputs = tuple(torch.tensor(item, requires_grad=True) for item in arrays)
    torch_scan(*torch_inputs, delta_softplus=True).sum().backward()
    for actual, expected in zip(ms_grads, torch_inputs):
        error = np.max(np.abs(actual.asnumpy() - expected.grad.numpy()))
        assert error <= 1e-4, "gradient max abs error: {}".format(error)
