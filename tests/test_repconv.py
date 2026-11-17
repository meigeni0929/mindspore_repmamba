import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindspore_repmamba.model.repconv import RepConv


def test_repconv_deploy_matches_frozen_training_graph():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_seed(31)
    block = RepConv(4)
    block.set_train(False)
    x = Tensor(np.random.default_rng(31).standard_normal((2, 4, 7, 5)).astype(np.float32))
    before = block(x).asnumpy()
    block.switch_to_deploy()
    after = block(x).asnumpy()
    np.testing.assert_allclose(before, after, atol=1e-5, rtol=1e-5)
    assert block.deploy


def test_repconv_gradient():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    block = RepConv(2)
    x = Tensor(np.ones((1, 2, 4, 4), np.float32))
    loss = lambda z: block(z).sum()
    value, grad = ms.value_and_grad(loss, grad_position=0)(x)
    assert value.shape == ()
    assert grad.shape == x.shape
    assert np.isfinite(grad.asnumpy()).all()
