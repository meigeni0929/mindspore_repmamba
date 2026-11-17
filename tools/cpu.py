"""CPU-only runtime configuration shared by tools and entry points."""

import os


def configure_cpu_threads(num_threads=None):
    """Configure CPU thread pools before importing MindSpore/PyTorch.

    A missing value preserves the scheduler/environment defaults.  Explicit
    values configure each library to the same bounded number to make runs
    reproducible and avoid nested use of all logical CPUs.
    """
    if num_threads is None:
        value = os.environ.get("REPMAMBA_NUM_THREADS")
        if value is None:
            return None
        num_threads = int(value)
    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError("num_threads must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(num_threads)
    return num_threads


def set_mindspore_cpu(mode, num_threads=None):
    """Select the only supported target and apply MindSpore 2.0 thread knobs."""
    configured = configure_cpu_threads(num_threads)
    import mindspore as ms
    kwargs = {"device_target": "CPU", "mode": mode}
    if configured is not None:
        kwargs.update(inter_op_parallel_num=configured,
                      runtime_num_threads=configured)
    ms.set_context(**kwargs)
    if ms.get_context("device_target") != "CPU":
        raise RuntimeError("MindSpore did not select the CPU backend")
    return configured

