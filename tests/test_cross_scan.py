import numpy as np
import mindspore as ms
from mindspore import Tensor
import torch

from mindspore_repmamba.ops.scans import (
    CrossScan, MultiMerge, MultiScan, cross_scale_reverse, cross_scale_scan,
)
from mindspore_repmamba.tools.pytorch_reference import cross_scan


def test_cross_scan_matches_torch():
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    data = np.arange(2 * 3 * 3 * 5, dtype=np.float32).reshape(2, 3, 3, 5)
    actual = CrossScan()(Tensor(data)).asnumpy()
    expected = cross_scan(torch.from_numpy(data)).numpy()
    assert actual.shape == (2, 4, 3, 15)
    assert np.max(np.abs(actual - expected)) <= 1e-5


def test_multiscan_inverse_sum_non_square():
    data = np.arange(1 * 2 * 5 * 7, dtype=np.float32).reshape(1, 2, 5, 7)
    directions = ("w4", "w4_flip", "c_w4", "c_w4_flip")
    scanned = MultiScan(directions)(Tensor(data))
    restored = MultiMerge(5, 7, directions)(scanned).asnumpy()
    np.testing.assert_array_equal(restored, 4 * data.reshape(1, 2, -1))


def test_multiscan_graph_mode_and_cross_scale_roundtrip():
    data = np.arange(1 * 2 * 8 * 8, dtype=np.float32).reshape(1, 2, 8, 8)
    down = data[:, :, ::2, ::2].copy()
    x = Tensor(data)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    scanned = MultiScan(("w4", "w4_flip", "c_w4", "c_w4_flip"))(x)
    assert scanned.shape == (1, 4, 2, 64)
    for flip, column in [(False, False), (True, False), (False, True), (True, True)]:
        sequence = cross_scale_scan(x, Tensor(down), window=4, flip=flip,
                                    column_first=column)
        restored = cross_scale_reverse(sequence, 8, 8, window=4, flip=flip,
                                       column_first=column)
        np.testing.assert_array_equal(restored.asnumpy(), data.reshape(1, 2, -1))
