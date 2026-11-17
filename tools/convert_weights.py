"""Strict PyTorch SMSR state-dict to MindSpore parameter conversion."""

from pathlib import Path
import re

import mindspore as ms
from mindspore import Tensor, Parameter


def mindspore_to_torch_name(name):
    """Map one phase-one MindSpore parameter name to the original SMSR key."""
    name = name.replace('patch_norm.', 'patch_embed.norm.')
    name = re.sub(r'^blocks\.(\d+)\.', r'layers.\1.', name)
    name = re.sub(r'^(layers\.\d+)\.blocks\.', r'\1.residual_group.blocks.', name)
    name = re.sub(r'^(layers\.\d+)\.conv\.', r'\1.conv.', name)
    name = name.replace('.norm.', '.norm2.') if '.residual_group.blocks.' in name and '.attn.' not in name else name
    name = name.replace('.attn.', '.attn_layer.')
    name = name.replace('.ssm.', '.ss2d.')
    name = name.replace('.mlp.dwconv.', '.mlp.dwconv.depthwise_conv.0.')

    # RepConv branch names.
    name = name.replace('.rbr_dense_conv.', '.conv.c.')
    name = name.replace('.rbr_dense_bn.', '.conv.bn.')
    name = name.replace('.rbr_1x1.', '.conv1.')
    name = name.replace('.rbr_post_bn.', '.bn.')
    # MambaLayer's final Conv2d_BN.
    name = name.replace('.proj_bn.', '.conv.bn.')
    name = name.replace('.proj.', '.conv.c.')

    if name.startswith('upsample.conv.'):
        name = name.replace('upsample.conv.', 'upsample.0.', 1)

    suffixes = {
        '.gamma': '.weight', '.beta': '.bias',
        '.moving_mean': '.running_mean', '.moving_variance': '.running_var',
    }
    for source, target in suffixes.items():
        if name.endswith(source):
            name = name[:-len(source)] + target
            break
    return name


def convert_state_dict(torch_state, network):
    """Return strict MindSpore params and a conversion report dictionary."""
    converted = {}
    used = set()
    missing = []
    mismatched = []
    for parameter in network.get_parameters():
        torch_name = mindspore_to_torch_name(parameter.name)
        if torch_name not in torch_state:
            missing.append((parameter.name, torch_name))
            continue
        array = torch_state[torch_name].detach().cpu().numpy()
        if tuple(array.shape) != tuple(parameter.shape):
            mismatched.append((parameter.name, tuple(parameter.shape), torch_name,
                               tuple(array.shape)))
            continue
        converted[parameter.name] = Tensor(array, dtype=parameter.dtype)
        used.add(torch_name)
    unexpected = sorted(name for name in torch_state
                        if name not in used and not name.endswith('num_batches_tracked'))
    report = {'missing': missing, 'mismatched': mismatched, 'unexpected': unexpected,
              'converted': len(converted)}
    if missing or mismatched or unexpected:
        raise ValueError('strict conversion failed: {}'.format(report))
    return converted, report


def load_torch_state(network, torch_state):
    params, report = convert_state_dict(torch_state, network)
    loadable = {name: Parameter(value, name=name, requires_grad=False)
                for name, value in params.items()}
    not_loaded, _ = ms.load_param_into_net(network, loadable, strict_load=True)
    if not_loaded:
        raise ValueError('MindSpore did not load parameters: {}'.format(not_loaded))
    return report


def convert_checkpoint(source, destination, network):
    """Convert a `.pth` file to `.ckpt`; importing torch is tool-only."""
    import torch
    source = Path(source)
    payload = torch.load(str(source), map_location='cpu')
    state = payload.get('state_dict', payload) if isinstance(payload, dict) else payload
    params, report = convert_state_dict(state, network)
    ms.save_checkpoint([{'name': name, 'data': value} for name, value in params.items()],
                       str(destination))
    return report
