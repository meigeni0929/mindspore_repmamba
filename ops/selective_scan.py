"""Reference and pure-MindSpore optimized selective scan implementations.

The state recurrence, evaluated in float32, is::

    delta_t = softplus(delta_t + delta_bias)  # when requested
    state_t = exp(delta_t * A) * state_{t-1}
              + delta_t * B_t * u_t
    y_t = sum_n(C_t * state_t) + D * u_t

The optimized forward uses an associative Hillis-Steele prefix scan. Its
explicit reverse-time ``bprop`` preserves the reference recurrence's FP32
gradient association. Both paths use ordinary MindSpore tensor operations.
"""

import mindspore as ms
from mindspore import nn, ops


class _SelectiveScanBase(nn.Cell):
    """Selective scan mapping ``u, delta: (B,D,L)`` to ``(B,D,L)``.

    ``A`` has shape ``(D,N)``. ``B`` and ``C`` may be fixed ``(D,N)``,
    batch-varying ``(B,N,L)``, or grouped ``(B,K,N,L)`` where ``D % K == 0``.
    ``D`` and ``delta_bias`` are optional vectors of shape ``(D,)``.
    """

    def __init__(self, delta_softplus=False):
        super().__init__()
        self.delta_softplus = bool(delta_softplus)
        self.softplus = ops.Softplus()

    def _prepare(self, u, delta, a, b_param, c_param, delta_bias):
        output_dtype = u.dtype
        u = ops.cast(u, ms.float32)
        delta = ops.cast(delta, ms.float32)
        a = ops.cast(a, ms.float32)
        b_param = ops.cast(b_param, ms.float32)
        c_param = ops.cast(c_param, ms.float32)
        channels = u.shape[1]
        if delta_bias is not None:
            bias = ops.reshape(ops.cast(delta_bias, ms.float32), (1, channels, 1))
            delta = delta + bias
        if self.delta_softplus:
            delta = self.softplus(delta)
        return output_dtype, u, delta, a, b_param, c_param

    @staticmethod
    def _parameter_at(parameter, index, channels):
        """Expand a B/C parameter at time ``index`` to ``(B,D,N)``."""
        if parameter.ndim == 2:
            # (D,N) -> (1,D,N), broadcast over batch in the recurrence.
            return ops.expand_dims(parameter, 0)
        if parameter.ndim == 3:
            # (B,N,L) -> (B,1,N), broadcast over channels.
            return ops.expand_dims(parameter[:, :, index], 1)
        if parameter.ndim == 4:
            # (B,K,N,L) -> (B,D,N), repeating each group contiguously.
            grouped = parameter[:, :, :, index]
            groups = grouped.shape[1]
            if channels % groups != 0:
                raise ValueError("channel count must be divisible by B/C groups")
            grouped = ops.expand_dims(grouped, 2)
            grouped = ops.tile(grouped, (1, 1, channels // groups, 1))
            return ops.reshape(grouped, (grouped.shape[0], channels, grouped.shape[-1]))
        raise ValueError("B and C must have rank 2, 3, or 4")

    @staticmethod
    def _parameter_slice(parameter, start, end, channels):
        """Expand a B/C sequence slice to ``(B,D,T,N)``."""
        if parameter.ndim == 2:
            return ops.expand_dims(ops.expand_dims(parameter, 0), 2)
        if parameter.ndim == 3:
            sliced = ops.transpose(parameter[:, :, start:end], (0, 2, 1))
            return ops.expand_dims(sliced, 1)
        if parameter.ndim == 4:
            sliced = ops.transpose(parameter[:, :, :, start:end], (0, 1, 3, 2))
            groups = sliced.shape[1]
            if channels % groups != 0:
                raise ValueError("channel count must be divisible by B/C groups")
            sliced = ops.expand_dims(sliced, 2)
            sliced = ops.tile(sliced, (1, 1, channels // groups, 1, 1))
            return ops.reshape(sliced, (sliced.shape[0], channels,
                                        sliced.shape[-2], sliced.shape[-1]))
        raise ValueError("B and C must have rank 2, 3, or 4")

    @staticmethod
    def _add_skip_and_cast(output, u, d, output_dtype):
        if d is not None:
            skip = ops.reshape(ops.cast(d, ms.float32), (1, u.shape[1], 1))
            output = output + u * skip
        return ops.cast(output, output_dtype)


class SelectiveScanReference(_SelectiveScanBase):
    """Sequential mathematical oracle; ``L`` Python-loop iterations."""

    def construct(self, u, delta, a, b_param, c_param, d=None, delta_bias=None):
        output_dtype, u, delta, a, b_param, c_param = self._prepare(
            u, delta, a, b_param, c_param, delta_bias)
        batch, channels, length = u.shape
        state_size = a.shape[1]
        state = ops.zeros((batch, channels, state_size), ms.float32)
        outputs = []
        for index in range(length):
            delta_t = delta[:, :, index]
            transition = ops.exp(ops.expand_dims(delta_t, -1) * ops.expand_dims(a, 0))
            b_t = self._parameter_at(b_param, index, channels)
            c_t = self._parameter_at(c_param, index, channels)
            state = transition * state + ops.expand_dims(
                delta_t * u[:, :, index], -1) * b_t
            outputs.append(ops.sum(state * c_t, dim=-1))
        output = ops.stack(outputs, axis=-1)
        return self._add_skip_and_cast(output, u, d, output_dtype)


class SelectiveScanOptimized(_SelectiveScanBase):
    """Associative Hillis-Steele prefix scan; ``ceil(log2(L))`` rounds."""

    def construct(self, u, delta, a, b_param, c_param, d=None, delta_bias=None):
        output_dtype, u, delta, a, b_param, c_param = self._prepare(
            u, delta, a, b_param, c_param, delta_bias)
        batch, channels, length = u.shape
        state_size = a.shape[1]
        state = ops.zeros((batch, channels, state_size), ms.float32)
        # Associative prefix scan over affine transforms. A recurrence step is
        # represented by (transition, injection), and composition is
        # (a2,b2) o (a1,b1) = (a2*a1, a2*b1+b2). Hillis-Steele doubling needs
        # only ceil(log2(L)) ordinary tensor rounds.
        transition = ops.exp(ops.expand_dims(delta, -1) *
                             ops.expand_dims(ops.expand_dims(a, 0), 2))
        b_sequence = self._parameter_slice(b_param, 0, length, channels)
        c_sequence = self._parameter_slice(c_param, 0, length, channels)
        injection = ops.expand_dims(delta * u, -1) * b_sequence
        prefix_a = transition
        prefix_b = injection
        offset = 1
        while offset < length:
            identity = ops.ones_like(prefix_a[:, :, :offset, :])
            zeros = ops.zeros_like(prefix_b[:, :, :offset, :])
            previous_a = ops.concat((identity, prefix_a[:, :, :-offset, :]), axis=2)
            previous_b = ops.concat((zeros, prefix_b[:, :, :-offset, :]), axis=2)
            prefix_b = prefix_b + prefix_a * previous_b
            prefix_a = prefix_a * previous_a
            offset *= 2
        states = prefix_a * ops.expand_dims(state, 2) + prefix_b
        output = ops.sum(states * c_sequence, dim=-1)
        return self._add_skip_and_cast(output, u, d, output_dtype)

    @staticmethod
    def _reduce_parameter_gradient(gradient, parameter, channels):
        """Reduce expanded ``(B,D,L,N)`` gradient to the B/C input shape."""
        if parameter.ndim == 2:
            return ops.sum(gradient, dim=(0, 2))
        if parameter.ndim == 3:
            return ops.transpose(ops.sum(gradient, dim=1), (0, 2, 1))
        groups = parameter.shape[1]
        grouped = ops.reshape(
            gradient, (gradient.shape[0], groups, channels // groups,
                       gradient.shape[2], gradient.shape[3]))
        grouped = ops.sum(grouped, dim=2)
        return ops.transpose(grouped, (0, 1, 3, 2))

    def bprop(self, u, delta, a, b_param, c_param, d, delta_bias, out, dout):
        """Exact reverse-time recurrence for the optimized forward graph.

        This intentionally preserves the sequential mathematical backward
        association. It removes the large prefix-autodiff tape and is the
        pure-MindSpore CPU implementation while avoiding a large autodiff tape.
        """
        input_dtypes = (u.dtype, delta.dtype, a.dtype, b_param.dtype,
                        c_param.dtype, d.dtype, delta_bias.dtype)
        u32 = ops.cast(u, ms.float32)
        delta_pre = ops.cast(delta, ms.float32) + ops.reshape(
            ops.cast(delta_bias, ms.float32), (1, u.shape[1], 1))
        delta32 = self.softplus(delta_pre) if self.delta_softplus else delta_pre
        a32 = ops.cast(a, ms.float32)
        b32 = ops.cast(b_param, ms.float32)
        c32 = ops.cast(c_param, ms.float32)
        dout32 = ops.cast(dout, ms.float32)
        batch, channels, length = u32.shape
        state = ops.zeros((batch, channels, a32.shape[1]), ms.float32)
        previous_states = []
        states = []
        transitions = []
        for index in range(length):
            previous_states.append(state)
            dt = delta32[:, :, index]
            transition = ops.exp(ops.expand_dims(dt, -1) * ops.expand_dims(a32, 0))
            bt = self._parameter_at(b32, index, channels)
            state = transition * state + ops.expand_dims(
                dt * u32[:, :, index], -1) * bt
            transitions.append(transition)
            states.append(state)

        adjoint = ops.zeros_like(state)
        du_reverse = []
        ddelta_reverse = []
        d_a = ops.zeros_like(a32)
        d_b_expanded_reverse = []
        d_c_expanded_reverse = []
        d32 = ops.cast(d, ms.float32)
        for index in range(length - 1, -1, -1):
            dt = delta32[:, :, index]
            u_t = u32[:, :, index]
            bt = self._parameter_at(b32, index, channels)
            ct = self._parameter_at(c32, index, channels)
            dy = dout32[:, :, index]
            adjoint = adjoint + ops.expand_dims(dy, -1) * ct
            previous = previous_states[index]
            transition = transitions[index]
            d_transition = adjoint * previous
            d_injection = adjoint
            d_a = d_a + ops.sum(
                d_transition * transition * ops.expand_dims(dt, -1), dim=0)
            d_dt = ops.sum(
                d_transition * transition * ops.expand_dims(a32, 0) +
                d_injection * bt * ops.expand_dims(u_t, -1), dim=-1)
            du_t = (dy * d32 + ops.sum(
                d_injection * bt * ops.expand_dims(dt, -1), dim=-1))
            du_reverse.append(du_t)
            ddelta_reverse.append(d_dt)
            d_b_expanded_reverse.append(
                d_injection * ops.expand_dims(dt * u_t, -1))
            d_c_expanded_reverse.append(
                ops.expand_dims(dy, -1) * states[index])
            adjoint = adjoint * transition

        reverse = ops.ReverseV2(axis=[2])
        du = reverse(ops.stack(du_reverse, axis=2))
        d_delta_processed = reverse(ops.stack(ddelta_reverse, axis=2))
        d_b_expanded = reverse(ops.stack(d_b_expanded_reverse, axis=2))
        d_c_expanded = reverse(ops.stack(d_c_expanded_reverse, axis=2))
        if self.delta_softplus:
            d_delta = d_delta_processed * ops.Sigmoid()(delta_pre)
        else:
            d_delta = d_delta_processed
        # delta_bias is stored as (K, inner) in SMSR, while the recurrence
        # flattens those axes into the scan channel dimension.
        d_bias = ops.reshape(ops.sum(d_delta, dim=(0, 2)), delta_bias.shape)
        d_d = ops.sum(dout32 * u32, dim=(0, 2))
        d_b = self._reduce_parameter_gradient(d_b_expanded, b32, channels)
        d_c = self._reduce_parameter_gradient(d_c_expanded, c32, channels)
        gradients = (du, d_delta, d_a, d_b, d_c, d_d, d_bias)
        return tuple(ops.cast(value, dtype)
                     for value, dtype in zip(gradients, input_dtypes))


class SelectiveScan(nn.Cell):
    """Backend-selecting Cell with ``reference`` and ``cpu_optimized`` modes."""

    def __init__(self, delta_softplus=False, backend="reference"):
        super().__init__()
        if backend not in ("reference", "cpu_optimized"):
            raise ValueError("selective scan backend must be 'reference' or 'cpu_optimized'")
        self.backend = backend
        if backend == "reference":
            self.implementation = SelectiveScanReference(delta_softplus)
        else:
            self.implementation = SelectiveScanOptimized(delta_softplus)

    def construct(self, u, delta, a, b_param, c_param, d=None, delta_bias=None):
        # The exact optimized bprop receives D and delta_bias as tensors. Keep
        # the public optional-argument contract by using the reference Cell
        # when either optional input is omitted.
        if self.backend == "cpu_optimized" and (d is None or delta_bias is None):
            return SelectiveScanReference(
                self.implementation.delta_softplus)(
                    u, delta, a, b_param, c_param, d, delta_bias)
        return self.implementation(u, delta, a, b_param, c_param, d, delta_bias)


def selective_scan(u, delta, a, b_param, c_param, d=None, delta_bias=None,
                   delta_softplus=False, backend="reference"):
    """Functional wrapper around :class:`SelectiveScan`."""
    return SelectiveScan(delta_softplus, backend)(
        u, delta, a, b_param, c_param, d, delta_bias)


def selective_scan_reference(u, delta, a, b_param, c_param, d=None,
                             delta_bias=None, delta_softplus=False):
    return SelectiveScanReference(delta_softplus)(
        u, delta, a, b_param, c_param, d, delta_bias)


def selective_scan_optimized(u, delta, a, b_param, c_param, d=None,
                             delta_bias=None, delta_softplus=False):
    return SelectiveScanOptimized(delta_softplus)(
        u, delta, a, b_param, c_param, d, delta_bias)
