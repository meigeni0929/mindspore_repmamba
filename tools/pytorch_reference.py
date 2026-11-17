"""PyTorch-only numerical oracle. Never imported by the MindSpore model."""

import torch


def cross_scan(x):
    b, c, h, w = x.shape
    row = x.flatten(2, 3)
    column = x.transpose(2, 3).flatten(2, 3)
    return torch.stack((row, column, row.flip(-1), column.flip(-1)), dim=1)


def cross_merge(ys):
    b, _, c, h, w = ys.shape
    ys = ys.view(b, 4, c, -1)
    row = ys[:, 0] + ys[:, 2].flip(-1)
    column = ys[:, 1] + ys[:, 3].flip(-1)
    column = column.view(b, c, w, h).transpose(2, 3).contiguous().view(b, c, -1)
    return row + column


def selective_scan(u, delta, a, b_param, c_param, d=None, delta_bias=None,
                   delta_softplus=False, z=None, return_last_state=False):
    """Pure PyTorch recurrence oracle using no CUDA extension."""
    dtype = u.dtype
    u, delta, a = u.float(), delta.float(), a.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    b_param, c_param = b_param.float(), c_param.float()
    batch, channels, length = u.shape
    state_size = a.shape[1]
    groups = b_param.shape[1] if b_param.dim() == 4 else 1
    channels_per_group = channels // groups
    state = torch.zeros(batch, channels, state_size, dtype=torch.float32)
    outputs = []
    for index in range(length):
        delta_t = delta[:, :, index]
        transition = torch.exp(delta_t[:, :, None] * a[None, :, :])
        if b_param.dim() == 2:
            bt = b_param[None, :, :]
            ct = c_param[None, :, :]
        elif b_param.dim() == 3:
            bt = b_param[:, :, index][:, None, :].expand(-1, channels, -1)
            ct = c_param[:, :, index][:, None, :].expand(-1, channels, -1)
        else:
            bt = b_param[:, :, :, index].repeat_interleave(channels_per_group, dim=1)
            ct = c_param[:, :, :, index].repeat_interleave(channels_per_group, dim=1)
        state = transition * state + delta_t[:, :, None] * bt * u[:, :, index, None]
        outputs.append((state * ct).sum(-1))
    out = torch.stack(outputs, dim=-1)
    if d is not None:
        out = out + u * d.float()[None, :, None]
    out = out.to(dtype)
    if z is not None:
        out = out * torch.nn.functional.silu(z)
    if return_last_state:
        return out, state
    return out
