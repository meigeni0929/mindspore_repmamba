"""Offline same-weight PyTorch/MindSpore numerical comparison.

This tool imports PyTorch by design. No runtime model module imports it.
"""

import numpy as np
import torch
import mindspore as ms
from mindspore import Tensor

from model.repmamba import SMSR as TorchSMSR
from mindspore_repmamba.model import SMSR
from mindspore_repmamba.tools.convert_weights import load_torch_state
from mindspore_repmamba.tools.pytorch_reference import selective_scan


def main():
    np.random.seed(123)
    torch.manual_seed(123)
    ms.set_seed(123)
    torch_net = TorchSMSR(upscale=4).eval()
    for module in torch_net.modules():
        if module.__class__.__name__ == 'SS2D':
            module.selective_scan = selective_scan
    ms_net = SMSR(upscale=4)
    report = load_torch_state(ms_net, torch_net.state_dict())
    ms_net.set_train(False)
    data = np.random.randn(1, 3, 8, 8).astype(np.float32)
    with torch.no_grad():
        expected = torch_net(torch.from_numpy(data)).numpy()
    actual = ms_net(Tensor(data)).asnumpy()
    difference = np.abs(actual - expected)
    print(report)
    print('max_abs:', float(difference.max()))
    print('mean_abs:', float(difference.mean()))


if __name__ == '__main__':
    main()
