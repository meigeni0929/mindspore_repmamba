# Pure MindSpore Rep-Mamba (phase one reference)

This directory is independent of the original PyTorch implementation. The
reference selective scan is written with ordinary MindSpore tensor operations
and automatic differentiation; it does not import or call `torch`,
`mamba_ssm`, or `selective_scan_cuda` during model execution.

The current network supports PYNATIVE_MODE and GRAPH_MODE on the validated
MindSpore 2.0 CPU environment, with ×2/×3/×4 reconstruction heads.

Selective Scan is selected with `backend="reference"` or
`backend="cpu_optimized"`. The safe default is `reference`. The optimized forward uses an
associative prefix scan and its backward uses an exact reverse-time recurrence,
so the tested FP32 gradient error remains below `1e-4`. `reference` remains
the sequential numerical oracle. Both implementations are CPU-only MindSpore
graphs and do not call device extensions.

Run from the repository root:

```bash
/root/anaconda3/envs/repmamba/bin/python -m pytest -q mindspore_repmamba/tests
/root/anaconda3/envs/repmamba/bin/python -m mindspore_repmamba.eval --scale 4
/root/anaconda3/envs/repmamba/bin/python -m mindspore_repmamba.tools.compare_pytorch
/root/anaconda3/envs/repmamba/bin/python benchmark_selective_scan.py --quick
```

Detailed CPU benchmark results and memory limits are recorded in
[`SELECTIVE_SCAN_PERFORMANCE.md`](SELECTIVE_SCAN_PERFORMANCE.md).

Run PYNATIVE_MODE and GRAPH_MODE 64×64 checks in separate Python processes on
the 12 GiB CPU environment. MindSpore 2.0 retains compiled graphs when modes
are switched in-process, and the combined cache can exceed the memory limit.

CPU workflow:

```bash
/root/anaconda3/envs/repmamba/bin/python tools/check_cpu_environment.py --num-threads 1
/root/anaconda3/envs/repmamba/bin/python tools/convert_pytorch_to_mindspore.py \
  repmambax4/models/DID-Data_new/model_best.pth artifacts/smsr_x4.ckpt
/root/anaconda3/envs/repmamba/bin/python tools/compare_data_pipeline.py /path/to/DRealSR
/root/anaconda3/envs/repmamba/bin/python train_ms.py --data /path/to/DRealSR --steps 10 --patch-size 8
/root/anaconda3/envs/repmamba/bin/python eval_ms.py --data /path/to/DRealSR \
  --checkpoint artifacts/smsr_x4.ckpt --max-images 10
/root/anaconda3/envs/repmamba/bin/python tools/benchmark_cpu.py --threads 1
```

The original training entry is not an authoritative SMSR CPU entry: it imports
the missing `model.DVMSR` and contains device-specific training code. `train_ms.py`
exposes the confirmed L1/Adam setup and leaves unconfirmed scheduler or clipping
settings explicit instead of silently inventing them.
