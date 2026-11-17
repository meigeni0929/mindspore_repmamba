import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor

from mindspore_repmamba.ops.selective_scan import selective_scan
from mindspore_repmamba.tools.pytorch_reference import selective_scan as torch_scan


def _run(u, delta, a, b, c, d=None, bias=None, softplus=False, mode=ms.PYNATIVE_MODE):
    ms.set_context(mode=mode, device_target="CPU")
    return selective_scan(Tensor(u), Tensor(delta), Tensor(a), Tensor(b), Tensor(c),
                          None if d is None else Tensor(d),
                          None if bias is None else Tensor(bias), softplus).asnumpy()


def test_selective_scan_matches_reference_variants():
    rng = np.random.default_rng(23)
    bsz, channels, length, state = 2, 4, 5, 3
    u = rng.standard_normal((bsz, channels, length), dtype=np.float32)
    delta = rng.standard_normal((bsz, channels, length), dtype=np.float32)
    a = -np.exp(rng.standard_normal((channels, state), dtype=np.float32))
    d = rng.standard_normal(channels, dtype=np.float32)
    bias = rng.standard_normal(channels, dtype=np.float32)
    for b, c in [
        (rng.standard_normal((channels, state), dtype=np.float32),
         rng.standard_normal((channels, state), dtype=np.float32)),
        (rng.standard_normal((bsz, state, length), dtype=np.float32),
         rng.standard_normal((bsz, state, length), dtype=np.float32)),
        (rng.standard_normal((bsz, 2, state, length), dtype=np.float32),
         rng.standard_normal((bsz, 2, state, length), dtype=np.float32)),
    ]:
        actual = _run(u, delta, a, b, c, d, bias, True)
        expected = torch_scan(torch.from_numpy(u), torch.from_numpy(delta),
                              torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c),
                              torch.from_numpy(d), torch.from_numpy(bias), True).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_selective_scan_graph_mode_and_gradient():
    rng = np.random.default_rng(24)
    u = Tensor(rng.standard_normal((1, 2, 4), dtype=np.float32))
    delta = Tensor(rng.standard_normal((1, 2, 4), dtype=np.float32))
    a = Tensor(-np.ones((2, 2), dtype=np.float32))
    b = Tensor(rng.standard_normal((2, 2), dtype=np.float32))
    c = Tensor(rng.standard_normal((2, 2), dtype=np.float32))
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    fn = lambda x: selective_scan(x, delta, a, b, c, delta_softplus=True).sum()
    value, grad = ms.value_and_grad(fn, grad_position=0)(u)
    assert value.shape == ()
    assert grad.shape == u.shape
    assert np.isfinite(grad.asnumpy()).all()


def test_reference_and_optimized_backends_match():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    rng = np.random.default_rng(25)
    arrays = [
        rng.standard_normal((1, 4, 9), dtype=np.float32),
        rng.standard_normal((1, 4, 9), dtype=np.float32),
        -np.exp(rng.standard_normal((4, 3), dtype=np.float32)),
        rng.standard_normal((1, 2, 3, 9), dtype=np.float32),
        rng.standard_normal((1, 2, 3, 9), dtype=np.float32),
        rng.standard_normal((4,), dtype=np.float32),
        rng.standard_normal((4,), dtype=np.float32),
    ]
    tensors = tuple(Tensor(item) for item in arrays)

    def loss_for(backend):
        def loss(*items):
            return selective_scan(*items, delta_softplus=True,
                                  backend=backend).sum()
        return loss

    ref_value, ref_grads = ms.value_and_grad(
        loss_for("reference"), grad_position=tuple(range(7)))(*tensors)
    opt_value, opt_grads = ms.value_and_grad(
        loss_for("cpu_optimized"), grad_position=tuple(range(7)))(*tensors)
    np.testing.assert_allclose(opt_value.asnumpy(), ref_value.asnumpy(),
                               atol=1e-4, rtol=1e-5)
    for actual, expected in zip(opt_grads, ref_grads):
        np.testing.assert_allclose(actual.asnumpy(), expected.asnumpy(),
                                   atol=1e-4, rtol=1e-5)


def test_backend_validation_and_fp16_reportable():
    with np.testing.assert_raises(ValueError):
        from mindspore_repmamba.ops.selective_scan import SelectiveScan
        SelectiveScan(backend="unknown")
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    rng = np.random.default_rng(26)
    u = rng.standard_normal((1, 4, 7), dtype=np.float32)
    delta = rng.standard_normal((1, 4, 7), dtype=np.float32)
    a = -np.exp(rng.standard_normal((4, 2), dtype=np.float32))
    b = rng.standard_normal((1, 2, 2, 7), dtype=np.float32)
    c = rng.standard_normal((1, 2, 2, 7), dtype=np.float32)
    d = rng.standard_normal((4,), dtype=np.float32)
    bias = rng.standard_normal((4,), dtype=np.float32)
    fp32 = selective_scan(*[Tensor(item) for item in (u, delta, a, b, c, d, bias)],
                          delta_softplus=True, backend="reference").asnumpy()
    fp16 = selective_scan(*[Tensor(item.astype(np.float16))
                            for item in (u, delta, a, b, c, d, bias)],
                          delta_softplus=True, backend="cpu_optimized").asnumpy()
    assert np.isfinite(fp16).all()
    assert np.max(np.abs(fp16.astype(np.float32) - fp32)) < 2e-2


def test_optimized_gradient_all_bc_layouts():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    rng = np.random.default_rng(27)
    batch, channels, length, state = 2, 4, 7, 3
    common = (
        rng.standard_normal((batch, channels, length), dtype=np.float32),
        rng.standard_normal((batch, channels, length), dtype=np.float32),
        -np.exp(rng.standard_normal((channels, state), dtype=np.float32)),
    )
    layouts = (
        (rng.standard_normal((channels, state), dtype=np.float32),
         rng.standard_normal((channels, state), dtype=np.float32)),
        (rng.standard_normal((batch, state, length), dtype=np.float32),
         rng.standard_normal((batch, state, length), dtype=np.float32)),
        (rng.standard_normal((batch, 2, state, length), dtype=np.float32),
         rng.standard_normal((batch, 2, state, length), dtype=np.float32)),
    )
    d = rng.standard_normal((channels,), dtype=np.float32)
    bias = rng.standard_normal((channels,), dtype=np.float32)
    for b_param, c_param in layouts:
        tensors = tuple(Tensor(item) for item in common + (b_param, c_param, d, bias))

        def gradients(backend):
            return ms.value_and_grad(
                lambda *xs: selective_scan(*xs, delta_softplus=True,
                                            backend=backend).sum(),
                grad_position=tuple(range(7)))(*tensors)[1]

        expected, actual = gradients("reference"), gradients("cpu_optimized")
        for reference_grad, optimized_grad in zip(expected, actual):
            error = np.max(np.abs(reference_grad.asnumpy() -
                                  optimized_grad.asnumpy()))
            assert error <= 1e-4, "gradient max abs error: {}".format(error)


def test_optimized_exact_bprop_graph_mode():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    rng = np.random.default_rng(28)
    arrays = (
        rng.standard_normal((1, 4, 5), dtype=np.float32),
        rng.standard_normal((1, 4, 5), dtype=np.float32),
        -np.exp(rng.standard_normal((4, 2), dtype=np.float32)),
        rng.standard_normal((1, 2, 2, 5), dtype=np.float32),
        rng.standard_normal((1, 2, 2, 5), dtype=np.float32),
        rng.standard_normal((4,), dtype=np.float32),
        rng.standard_normal((4,), dtype=np.float32),
    )
    inputs = tuple(Tensor(item) for item in arrays)

    class Loss(ms.nn.Cell):
        def construct(self, u, delta, a, b_param, c_param, d, bias):
            return selective_scan(u, delta, a, b_param, c_param, d, bias,
                                  delta_softplus=True,
                                  backend="cpu_optimized").sum()

    value, gradients = ms.value_and_grad(
        Loss(), grad_position=tuple(range(7)))(*inputs)
    assert value.shape == ()
    assert all(np.isfinite(item.asnumpy()).all() for item in gradients)
