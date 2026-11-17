"""Benchmark reference and optimized selective scan implementations.

Use ``--quick`` for a bounded smoke benchmark. The full requested matrix is
the Cartesian product B={1,2}, H=W={32,64,128}, D={48,96,192}, K={4,8}.
"""

import argparse
import csv
import math
import os
import threading
import time

import numpy as np
import mindspore as ms
from mindspore import Tensor

from mindspore_repmamba.ops.selective_scan import selective_scan


def synchronize(tensor=None):
    """Force queued device work to finish without relying on a GPU-only API."""
    if tensor is not None:
        tensor.asnumpy()


class RssSampler:
    """Poll current-process RSS while an asynchronous MindSpore call runs."""

    def __init__(self):
        import psutil
        self.process = psutil.Process(os.getpid())
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.002)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return max(0, self.peak - self.baseline)


def make_inputs(batch, height, channels, state, groups, dtype, seed):
    length = height * height
    rng = np.random.default_rng(seed)
    arrays = (
        rng.standard_normal((batch, channels, length), dtype=np.float32),
        rng.standard_normal((batch, channels, length), dtype=np.float32),
        -np.exp(rng.standard_normal((channels, state), dtype=np.float32)),
        rng.standard_normal((batch, groups, state, length), dtype=np.float32),
        rng.standard_normal((batch, groups, state, length), dtype=np.float32),
        rng.standard_normal((channels,), dtype=np.float32),
        rng.standard_normal((channels,), dtype=np.float32),
    )
    np_dtype = np.float16 if dtype == "float16" else np.float32
    return tuple(Tensor(item.astype(np_dtype)) for item in arrays)


def forward(inputs, backend):
    return selective_scan(*inputs, delta_softplus=True, backend=backend)


def loss(inputs, backend):
    return forward(inputs, backend).sum()


def run_timed(call, warmup, repeats):
    result = None
    first_sampler = RssSampler()
    first_sampler.start()
    first_start = time.perf_counter()
    result = call()
    synchronize(result[0] if isinstance(result, tuple) else result)
    first_ms = (time.perf_counter() - first_start) * 1000.0
    first_rss = first_sampler.stop()
    for _ in range(warmup):
        result = call()
        synchronize(result[0] if isinstance(result, tuple) else result)
    times = []
    sampler = RssSampler()
    sampler.start()
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        synchronize(result[0] if isinstance(result, tuple) else result)
        times.append((time.perf_counter() - start) * 1000.0)
    peak_rss = sampler.stop()
    return (first_ms, float(np.median(times)), max(first_rss, peak_rss), result)


def theoretical_bytes(batch, channels, length, state, backend):
    element = 4
    state_bytes = batch * channels * state * element
    sequence_bytes = batch * channels * length * element
    expanded = batch * channels * length * state * element
    if backend == "reference":
        # recurrent state + inputs/output; autograd may retain all L states.
        forward = 2 * state_bytes + 3 * sequence_bytes
        backward_upper = forward + length * state_bytes
    else:
        # transition, injection, prefix A/B, shifted A/B, states and B/C views.
        forward = 7 * expanded + 3 * sequence_bytes
        # Explicit reverse-time bprop stores previous states, states and
        # transitions, then emits expanded B/C gradients.
        backward_upper = 8 * expanded + 5 * sequence_bytes
    return forward, backward_upper


def benchmark_case(batch, height, channels, state, groups, dtype, warmup, repeats,
                   benchmark_backward=True, backends=("reference", "cpu_optimized")):
    inputs = make_inputs(batch, height, channels, state, groups, dtype,
                         seed=batch * 100000 + height * 1000 + channels * 10 + groups)
    outputs = {}
    rows = {}
    for backend in backends:
        forward_first_ms, forward_ms, forward_rss, forward_result = run_timed(
            lambda b=backend: forward(inputs, b), warmup, repeats)
        outputs[backend] = forward_result.asnumpy().astype(np.float32)
        backward_ms = float("nan")
        backward_rss = 0
        if benchmark_backward:
            grad_fn = ms.value_and_grad(
                lambda *xs, b=backend: loss(xs, b),
                grad_position=tuple(range(7)))
            backward_first_ms, backward_ms, backward_rss, backward_result = run_timed(
                lambda fn=grad_fn: fn(*inputs), warmup, repeats)
            rows[backend + "_grads"] = backward_result[1]
        theoretical_forward, theoretical_backward = theoretical_bytes(
            batch, channels, height * height, state, backend)
        rows[backend] = {
            "forward_first_ms": forward_first_ms,
            "forward_ms": forward_ms,
            "backward_first_ms": backward_first_ms if benchmark_backward else float("nan"),
            "backward_ms": backward_ms,
            "rss_delta_bytes": max(forward_rss, backward_rss),
            "theoretical_forward_bytes": theoretical_forward,
            "theoretical_backward_upper_bytes": theoretical_backward,
        }
    paired = "reference" in rows and "cpu_optimized" in rows
    forward_error = (float(np.max(np.abs(outputs["cpu_optimized"] - outputs["reference"])))
                     if paired else float("nan"))

    if benchmark_backward and paired:
        ref_grad = rows.pop("reference_grads")
        opt_grad = rows.pop("cpu_optimized_grads")
        gradient_error = max(float(np.max(np.abs(x.asnumpy().astype(np.float32) -
                                               y.asnumpy().astype(np.float32))))
                             for x, y in zip(ref_grad, opt_grad))
    else:
        gradient_error = float("nan")
    return {
        "batch": batch, "height": height, "width": height,
        "channels": channels, "state": state, "groups": groups,
        "dtype": dtype,
        "reference_forward_ms": rows.get("reference", {}).get("forward_ms", float("nan")),
        "optimized_forward_ms": rows.get("cpu_optimized", {}).get("forward_ms", float("nan")),
        "reference_forward_first_ms": rows.get("reference", {}).get("forward_first_ms", float("nan")),
        "optimized_forward_first_ms": rows.get("cpu_optimized", {}).get("forward_first_ms", float("nan")),
        "forward_speedup": (rows["reference"]["forward_ms"] /
                            rows["cpu_optimized"]["forward_ms"] if paired else float("nan")),
        "reference_backward_ms": rows.get("reference", {}).get("backward_ms", float("nan")),
        "optimized_backward_ms": rows.get("cpu_optimized", {}).get("backward_ms", float("nan")),
        "reference_backward_first_ms": rows.get("reference", {}).get("backward_first_ms", float("nan")),
        "optimized_backward_first_ms": rows.get("cpu_optimized", {}).get("backward_first_ms", float("nan")),
        "backward_speedup": (rows["reference"]["backward_ms"] /
                             rows["cpu_optimized"]["backward_ms"]
                             if paired and benchmark_backward else float("nan")),
        "reference_peak_rss_delta_bytes": rows.get("reference", {}).get("rss_delta_bytes", 0),
        "optimized_peak_rss_delta_bytes": rows.get("cpu_optimized", {}).get("rss_delta_bytes", 0),
        "reference_theoretical_forward_bytes": rows.get("reference", {}).get("theoretical_forward_bytes", 0),
        "optimized_theoretical_forward_bytes": rows.get("cpu_optimized", {}).get("theoretical_forward_bytes", 0),
        "reference_theoretical_backward_upper_bytes": rows.get("reference", {}).get("theoretical_backward_upper_bytes", 0),
        "optimized_theoretical_backward_upper_bytes": rows.get("cpu_optimized", {}).get("theoretical_backward_upper_bytes", 0),
        "forward_max_abs": forward_error,
        "gradient_max_abs": gradient_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-target", default="CPU")
    parser.add_argument("--mode", choices=("pynative", "graph"), default="pynative")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--batches", default="1,2")
    parser.add_argument("--heights", default="32,64,128")
    parser.add_argument("--channels", default="48,96,192")
    parser.add_argument("--groups", default="4,8")
    parser.add_argument("--backends", choices=("both", "reference", "cpu_optimized"),
                        default="both")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csv", default="")
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args()
    mode = ms.PYNATIVE_MODE if args.mode == "pynative" else ms.GRAPH_MODE
    ms.set_context(mode=mode, device_target=args.device_target)
    matrix = ([(1, 8, 48, 16, 4), (1, 16, 48, 16, 4),
               (2, 16, 96, 16, 8)] if args.quick else
              [(b, h, d, 16, k)
               for b in [int(x) for x in args.batches.split(',')]
               for h in [int(x) for x in args.heights.split(',')]
               for d in [int(x) for x in args.channels.split(',')]
               for k in [int(x) for x in args.groups.split(',')]])
    rows = []
    backends = (("reference", "cpu_optimized") if args.backends == "both"
                else (args.backends,))
    print("device_target={}, mode={}, mindspore={}".format(
        args.device_target, args.mode, ms.__version__))
    for case in matrix:
        row = benchmark_case(*case, args.dtype, args.warmup, args.repeats,
                             benchmark_backward=not args.no_backward,
                             backends=backends)
        rows.append(row)
        print(row, flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
