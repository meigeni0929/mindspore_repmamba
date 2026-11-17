import numpy as np
import mindspore as ms
from mindspore import Tensor
import torch

from mindspore_repmamba.ops.scans import CrossMerge, CrossScan
from mindspore_repmamba.tools.pytorch_reference import cross_merge


def test_cross_merge_matches_torch():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    rng = np.random.default_rng(11)
    data = rng.standard_normal((2, 4, 3, 3, 5), dtype=np.float32)
    actual = CrossMerge()(Tensor(data)).asnumpy()
    expected = cross_merge(torch.from_numpy(data)).numpy()
    assert actual.shape == (2, 3, 15)
    assert np.max(np.abs(actual - expected)) <= 1e-5


def test_cross_merge_of_scan_is_four_x():
    rng = np.random.default_rng(12)
    data = rng.standard_normal((1, 2, 3, 5), dtype=np.float32)
    scanned = CrossScan()(Tensor(data))
    merged = CrossMerge()(ms.ops.reshape(scanned, (1, 4, 2, 3, 5))).asnumpy()
    np.testing.assert_allclose(merged, 4 * data.reshape(1, 2, -1), atol=1e-5)
