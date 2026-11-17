import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindspore_repmamba.model.repmamba_ms import SS2D, MambaLayer, CCmamba, HierarchicalTransformerBlock


def test_ssm_block_shapes_and_finite():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = Tensor(np.random.default_rng(41).standard_normal((1, 8, 6, 5)).astype(np.float32))
    y = MambaLayer(8, selective_scan_backend="cpu_optimized")(x)
    assert y.shape == x.shape
    assert np.isfinite(y.asnumpy()).all()


def test_hierarchical_block_graph_mode():
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    x = Tensor(np.random.default_rng(42).standard_normal((1, 15, 64)).astype(np.float32))
    y = HierarchicalTransformerBlock(64, selective_scan_backend="cpu_optimized")(x, 3, 5)
    assert y.shape == x.shape
    assert np.isfinite(y.asnumpy()).all()
