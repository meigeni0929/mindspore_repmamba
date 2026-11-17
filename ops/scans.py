"""MindSpore-native spatial scan and merge operators.

All operators are ordinary differentiable tensor graphs. No custom backward is
used. Spatial flattening is row-major, matching the original PyTorch code.
"""

import math

from mindspore import nn, ops


def _reverse_last(x):
    """Reverse the sequence axis of a ``(..., L)`` tensor."""
    return ops.ReverseV2(axis=[-1])(x)


class CrossScan(nn.Cell):
    """Four-direction scan: ``(B,C,H,W) -> (B,4,C,H*W)``.

    Direction order is row-forward, column-forward, row-reverse,
    column-reverse.
    """

    def construct(self, x):
        b, c, h, w = x.shape
        row = ops.reshape(x, (b, c, h * w))
        column = ops.reshape(ops.transpose(x, (0, 1, 3, 2)), (b, c, h * w))
        return ops.stack((row, column, _reverse_last(row), _reverse_last(column)), axis=1)


class CrossMerge(nn.Cell):
    """Merge four directions: ``(B,4,C,H,W) -> (B,C,H*W)``."""

    def construct(self, ys):
        b, _, c, h, w = ys.shape
        flat = ops.reshape(ys, (b, 4, c, h * w))
        row = flat[:, 0] + _reverse_last(flat[:, 2])
        column = flat[:, 1] + _reverse_last(flat[:, 3])
        column = ops.reshape(column, (b, c, w, h))
        column = ops.reshape(ops.transpose(column, (0, 1, 3, 2)), (b, c, h * w))
        return row + column


def _pad_bottom_right(x, target_h, target_w):
    """Zero-pad NCHW tensor on bottom/right to ``target_h/target_w``."""
    pad_h = target_h - x.shape[2]
    pad_w = target_w - x.shape[3]
    if pad_h == 0 and pad_w == 0:
        return x
    # MindSpore 2.0 uses (left, right, top, bottom) for 4-D constant padding.
    return ops.pad(x, (0, pad_w, 0, pad_h))


def local_scan_bchw(x, window=8, flip=False, column_first=False):
    """Window scan ``(B,C,H,W) -> (B,C,ceil(H/w)*ceil(W/w)*w*w)``."""
    b, c, h, width = x.shape
    hg, wg = math.ceil(h / window), math.ceil(width / window)
    x = _pad_bottom_right(x, hg * window, wg * window)
    x = ops.reshape(x, (b, c, hg, window, wg, window))
    if column_first:
        x = ops.transpose(x, (0, 1, 4, 2, 5, 3))
    else:
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5))
    x = ops.reshape(x, (b, c, -1))
    return _reverse_last(x) if flip else x


def local_reverse(x, height, width, window=8, flip=False, column_first=False):
    """Inverse local scan ``(B,C,Lpad) -> (B,C,H*W)``."""
    b, c, _ = x.shape
    hg, wg = math.ceil(height / window), math.ceil(width / window)
    if flip:
        x = _reverse_last(x)
    if column_first:
        x = ops.reshape(x, (b, c, wg, hg, window, window))
        x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
    else:
        x = ops.reshape(x, (b, c, hg, wg, window, window))
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5))
    x = ops.reshape(x, (b, c, hg * window, wg * window))
    x = x[:, :, :height, :width]
    return ops.reshape(x, (b, c, height * width))


def nested_local_scan_bchw(x, outer=8, inner=2, flip=False, column_first=False):
    """Nested window scan ``(B,C,H,W) -> (B,C,Lpad)``."""
    b, c, h, width = x.shape
    hg, wg = math.ceil(h / outer), math.ceil(width / outer)
    x = _pad_bottom_right(x, hg * outer, wg * outer)
    if column_first:
        x = ops.reshape(x, (b, c, hg, outer, wg, outer))
        x = ops.transpose(x, (0, 1, 4, 2, 5, 3))
    else:
        x = ops.reshape(x, (b, c, hg, outer, wg, outer))
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5))
    x = ops.reshape(x, (b, c, hg * wg, outer // inner, inner, outer // inner, inner))
    x = ops.transpose(x, (0, 1, 2, 3, 5, 4, 6))
    x = ops.reshape(x, (b, c, -1))
    return _reverse_last(x) if flip else x


def reverse_nested_local_scan_bchw(x, height, width, outer=8, inner=2,
                                   flip=False, column_first=False):
    """Inverse nested scan ``(B,C,Lpad) -> (B,C,H*W)``."""
    b, c, _ = x.shape
    hg, wg = math.ceil(height / outer), math.ceil(width / outer)
    if flip:
        x = _reverse_last(x)
    x = ops.reshape(x, (b, c, hg * wg, outer // inner, outer // inner, inner, inner))
    x = ops.transpose(x, (0, 1, 2, 3, 5, 4, 6))
    x = ops.reshape(x, (b, c, hg, wg, outer, outer))
    if column_first:
        x = ops.transpose(x, (0, 1, 3, 5, 2, 4))
    else:
        x = ops.transpose(x, (0, 1, 2, 4, 3, 5))
    x = ops.reshape(x, (b, c, hg * outer, wg * outer))[:, :, :height, :width]
    return ops.reshape(x, (b, c, height * width))


def merge_cross_scale(low_sequence, high_sequence):
    """Interleave one low token and four high tokens into groups of five."""
    b, c, low_len = low_sequence.shape
    if high_sequence.shape[-1] != 4 * low_len:
        raise ValueError("high-resolution sequence length must equal 4 * low length")
    high = ops.reshape(high_sequence, (b, c, low_len, 4))
    low = ops.expand_dims(low_sequence, -1)
    return ops.reshape(ops.concat((low, high), axis=-1), (b, c, 5 * low_len))


def cross_scale_scan(x, x_down, window=8, flip=False, column_first=False):
    """Cross-scale scan returning ``(B,C,5*H*W/4)`` for 2x spatial scales."""
    high = nested_local_scan_bchw(x, outer=window, inner=2, flip=flip,
                                   column_first=column_first)
    low = local_scan_bchw(x_down, window=window // 2, flip=flip,
                           column_first=column_first)
    return merge_cross_scale(low, high)


def cross_scale_reverse(sequence, height, width, window=8, flip=False,
                        column_first=False):
    """Restore the high-resolution ``(B,C,H*W)`` stream from 5-token groups."""
    b, c, length = sequence.shape
    grouped = ops.reshape(sequence, (b, c, length // 5, 5))
    high = ops.reshape(grouped[:, :, :, 1:], (b, c, -1))
    return reverse_nested_local_scan_bchw(high, height, width, outer=window, inner=2,
                                          flip=flip, column_first=column_first)


class MultiScan(nn.Cell):
    """Apply configured window directions to ``(B,C,H,W)``.

    Output shape is ``(B,K,C,Lpad)``. Supported directions are ``wN``,
    ``wN_flip``, ``c_wN`` and ``c_wN_flip``.
    """

    def __init__(self, directions=("w8", "w8_flip", "c_w8", "c_w8_flip")):
        super().__init__()
        self.directions = tuple(directions)
        # Parse Python strings during initialization. MindSpore 2.0 GRAPH_MODE
        # cannot slice/split strings inside ``construct``.
        self.windows = tuple(self._decode(item)[0] for item in self.directions)
        self.flips = tuple(self._decode(item)[1] for item in self.directions)
        self.columns = tuple(self._decode(item)[2] for item in self.directions)

    @staticmethod
    def _decode(direction):
        column = direction.startswith("c_w")
        prefix = "c_w" if column else "w"
        window = int(direction[len(prefix):].split("_")[0])
        return window, direction.endswith("flip"), column

    def construct(self, x):
        sequences = []
        for window, flip, column in zip(self.windows, self.flips, self.columns):
            sequences.append(local_scan_bchw(x, window, flip, column))
        return ops.stack(sequences, axis=1)


class MultiMerge(nn.Cell):
    """Inverse and sum ``(B,K,C,Lpad) -> (B,C,H*W)`` directions."""

    def __init__(self, height, width,
                 directions=("w8", "w8_flip", "c_w8", "c_w8_flip")):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.directions = tuple(directions)
        self.windows = tuple(MultiScan._decode(item)[0] for item in self.directions)
        self.flips = tuple(MultiScan._decode(item)[1] for item in self.directions)
        self.columns = tuple(MultiScan._decode(item)[2] for item in self.directions)

    def construct(self, xs):
        restored = []
        for index, (window, flip, column) in enumerate(
                zip(self.windows, self.flips, self.columns)):
            restored.append(local_reverse(xs[:, index], self.height, self.width,
                                          window, flip, column))
        return ops.addn(restored)
