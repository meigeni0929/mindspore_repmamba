"""Pure MindSpore SMSR/Rep-Mamba reference network.

Tensor layouts are explicit: convolutions use ``(B,C,H,W)`` and token blocks
use ``(B,H*W,C)``. The implementation is intentionally reference-oriented;
selective scan uses the differentiable recurrence in ``ops.selective_scan``.
"""

import math
import mindspore as ms
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import TruncatedNormal, initializer

from ..ops.basic import ConvFFN, Upsample, UpsampleOneStep
from ..ops.selective_scan import SelectiveScan
from .repconv import RepConv


class SS2D(nn.Cell):
    """Four-direction low-resolution SSM with local high-frequency branch."""

    def __init__(self, d_model, d_state=16, expand=2.0,
                 selective_scan_backend="reference"):
        super().__init__()
        ratios = {8: 1, 16: 4, 32: 16, 64: 32}
        if d_model not in ratios:
            raise ValueError("SS2D supports channel dimensions 8, 16, 32, 64")
        self.d_model = d_model
        self.d_model2 = ratios[d_model]
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_inner2 = int(expand * self.d_model2)
        self.dt_rank = max(1, math.ceil(self.d_model2 / 16))
        self.in_proj = nn.Dense(d_model, self.d_inner * 2, has_bias=False)
        self.conv2d = RepConv(self.d_inner)
        self.rewconv = RepConv(self.d_inner - self.d_inner2)
        self.pool = nn.AvgPool2d(2, 2)
        self.act = nn.SiLU()
        self.gate_act = nn.SiLU()
        self.x_proj_weight = Parameter(initializer(
            TruncatedNormal(0.02), (4, self.dt_rank + 2 * d_state, self.d_inner2), ms.float32))
        self.dt_projs_weight = Parameter(initializer(
            TruncatedNormal(0.02), (4, self.d_inner2, self.dt_rank), ms.float32))
        self.dt_projs_bias = Parameter(initializer(
            TruncatedNormal(0.02), (4, self.d_inner2), ms.float32))
        logs = [[math.log(float(n + 1)) for n in range(d_state)]
                for _ in range(4 * self.d_inner2)]
        self.A_logs = Parameter(Tensor(logs, ms.float32).reshape((4 * self.d_inner2, d_state)))
        self.Ds = Parameter(ops.ones((4 * self.d_inner2,), ms.float32))
        self.scan = SelectiveScan(delta_softplus=True,
                                  backend=selective_scan_backend)
        self.out_norm = nn.LayerNorm((self.d_inner,))
        self.out_proj = nn.Dense(self.d_inner, d_model, has_bias=False)

    def _core(self, x):
        b, _, h, w = x.shape
        l = h * w
        row = ops.reshape(x, (b, self.d_inner2, l))
        col = ops.reshape(ops.transpose(x, (0, 1, 3, 2)), (b, self.d_inner2, l))
        xs = ops.stack((row, col, ops.ReverseV2(axis=[-1])(row),
                        ops.ReverseV2(axis=[-1])(col)), axis=1)
        # (B,K,D,L) @ (K,C,D) -> (B,K,C,L), expressed with matmul because
        # MindSpore 2.0's Einsum backend is GPU-only.
        xs_ld = ops.transpose(xs, (0, 1, 3, 2))
        proj = ops.matmul(xs_ld, ops.transpose(self.x_proj_weight, (0, 2, 1)))
        proj = ops.transpose(proj, (0, 1, 3, 2))
        parts = ops.split(proj, (self.dt_rank, self.d_state, self.d_state), axis=2)
        dt_lr = ops.transpose(parts[0], (0, 1, 3, 2))
        dts = ops.matmul(dt_lr, ops.transpose(self.dt_projs_weight, (0, 2, 1)))
        dts = ops.transpose(dts, (0, 1, 3, 2))
        scan_x = ops.reshape(xs, (b, 4 * self.d_inner2, l))
        scan_dt = ops.reshape(dts, (b, 4 * self.d_inner2, l))
        bb = parts[1]
        cc = parts[2]
        a = -ops.exp(self.A_logs)
        out = self.scan(scan_x, scan_dt, a, bb, cc, self.Ds, self.dt_projs_bias)
        out = ops.reshape(out, (b, 4, self.d_inner2, l))
        inv = ops.ReverseV2(axis=[-1])(out[:, 2:4])
        col_out = ops.reshape(out[:, 1], (b, self.d_inner2, w, h))
        col_out = ops.reshape(ops.transpose(col_out, (0, 1, 3, 2)), (b, self.d_inner2, l))
        inv_col = ops.reshape(inv[:, 1], (b, self.d_inner2, w, h))
        inv_col = ops.reshape(ops.transpose(inv_col, (0, 1, 3, 2)), (b, self.d_inner2, l))
        return out[:, 0] + inv[:, 0] + col_out + inv_col

    def construct(self, x):
        b, h, w, _ = x.shape
        xz = self.in_proj(x)
        x, z = ops.chunk(xz, 2, axis=-1)
        x = self.act(self.conv2d(ops.transpose(x, (0, 3, 1, 2))))
        low, high = ops.split(x, (self.d_inner2, self.d_inner - self.d_inner2), axis=1)
        high = self.rewconv(high)
        low0 = low
        low = self.pool(low)
        low_h, low_w = low.shape[2], low.shape[3]
        scan = self._core(low)
        scan = ops.reshape(ops.transpose(scan, (0, 2, 1)), (b, low_h, low_w, self.d_inner2))
        scan = ops.transpose(scan, (0, 3, 1, 2))
        scan = ops.interpolate(scan, size=(h, w), mode='bilinear', align_corners=False)
        low0 = low0 - ops.interpolate(low, size=(h, w), mode='nearest')
        y = ops.concat((scan + low0, high), axis=1)
        y = ops.transpose(y, (0, 2, 3, 1))
        y = self.out_norm(y) * self.gate_act(z)
        return self.out_proj(y)


class MambaLayer(nn.Cell):
    def __init__(self, channels, selective_scan_backend="reference"):
        super().__init__()
        self.norm = nn.LayerNorm((channels,))
        self.ssm = SS2D(channels, selective_scan_backend=selective_scan_backend)
        self.proj = nn.Conv2d(channels, channels, 1, pad_mode='valid', group=channels,
                              has_bias=False)
        self.proj_bn = nn.BatchNorm2d(channels)

    def construct(self, x):
        b, c, h, w = x.shape
        tokens = ops.transpose(x, (0, 2, 3, 1))
        tokens = self.norm(tokens)
        y = self.ssm(tokens) + ops.transpose(x, (0, 2, 3, 1))
        return self.proj_bn(self.proj(ops.transpose(y, (0, 3, 1, 2))))


class CCmamba(nn.Cell):
    def __init__(self, dim, selective_scan_backend="reference"):
        super().__init__()
        self.m8 = MambaLayer(8, selective_scan_backend)
        self.m16 = MambaLayer(16, selective_scan_backend)
        self.m32 = MambaLayer(32, selective_scan_backend)
        self.m64 = MambaLayer(dim, selective_scan_backend)

    def construct(self, x):
        x = ops.transpose(x, (0, 3, 1, 2))
        c0, c1, c2, c3 = ops.split(x, (8, 8, 16, 32), axis=1)
        x = ops.concat((c0 + self.m8(c0), c1), axis=1)
        x = ops.concat((self.m16(x), c2), axis=1)
        x = ops.concat((self.m32(x), c3), axis=1)
        return self.m64(x)


class HierarchicalTransformerBlock(nn.Cell):
    def __init__(self, dim, mlp_ratio=2.0, selective_scan_backend="reference"):
        super().__init__()
        self.attn = CCmamba(dim, selective_scan_backend)
        self.norm = nn.LayerNorm((dim,))
        self.mlp = ConvFFN(dim, int(dim * mlp_ratio))

    def construct(self, x, height, width):
        b, l, c = x.shape
        image = ops.reshape(x, (b, height, width, c))
        image = self.attn(image)
        x = x + ops.reshape(ops.transpose(image, (0, 2, 3, 1)), (b, l, c))
        return x + self.mlp(self.norm(x), height, width)


class RHTB(nn.Cell):
    """Residual group matching ``conv(unembed(BasicLayer(x))) + x``."""

    def __init__(self, dim, depth=1, mlp_ratio=2.0,
                 selective_scan_backend="reference"):
        super().__init__()
        self.dim = int(dim)
        self.blocks = nn.CellList([HierarchicalTransformerBlock(
            dim, mlp_ratio, selective_scan_backend)
                                   for _ in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, pad_mode='pad', padding=1, has_bias=True)

    def construct(self, x, height, width):
        shortcut = x
        for block in self.blocks:
            x = block(x, height, width)
        b = x.shape[0]
        image = ops.reshape(ops.transpose(x, (0, 2, 1)),
                            (b, self.dim, height, width))
        image = self.conv(image)
        x = ops.transpose(ops.reshape(image, (b, self.dim, height * width)),
                          (0, 2, 1))
        return x + shortcut


class SMSR(nn.Cell):
    """MindSpore Rep-Mamba ×2/×3/×4 reference network."""

    def __init__(self, upscale=4, in_chans=3, embed_dim=64, depths=(1, 1, 1, 1, 1, 1),
                 mlp_ratio=2.0, img_range=1.0,
                 selective_scan_backend="reference"):
        super().__init__()
        self.upscale = int(upscale)
        self.img_range = float(img_range)
        self.mean = Tensor([0.4488, 0.4371, 0.4040], ms.float32).reshape((1, 3, 1, 1))
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, pad_mode='pad', padding=1,
                                    has_bias=True)
        self.patch_norm = nn.LayerNorm((embed_dim,))
        self.selective_scan_backend = selective_scan_backend
        self.blocks = nn.CellList([RHTB(embed_dim, depth, mlp_ratio,
                                        selective_scan_backend)
                                   for depth in depths])
        self.norm = nn.LayerNorm((embed_dim,))
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, pad_mode='pad', padding=1,
                                         has_bias=True)
        self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)

    def construct(self, x):
        _, _, h, w = x.shape
        x = (x - self.mean) * self.img_range
        shallow = self.conv_first(x)
        b, c, _, _ = shallow.shape
        tokens = ops.reshape(ops.transpose(shallow, (0, 2, 3, 1)), (b, h * w, c))
        tokens = self.patch_norm(tokens)
        for block in self.blocks:
            tokens = block(tokens, h, w)
        body = ops.transpose(ops.reshape(self.norm(tokens), (b, h, w, c)), (0, 3, 1, 2))
        return self.upsample(self.conv_after_body(body) + shallow) / self.img_range + self.mean
