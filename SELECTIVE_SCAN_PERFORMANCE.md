# Selective Scan CPU performance report

## Environment

- MindSpore 2.0.0 CPU backend, Python 3.8.20
- 12 GiB host memory, no swap
- PYNATIVE_MODE and FP32 unless noted
- Timing forces synchronization with `asnumpy()`
- Peak memory uses 2 ms process-RSS sampling

This report contains no NVIDIA, CUDA, Ascend or CANN measurements.

## Implementations and complexity

- `reference`: original recurrence with L sequential Python/graph steps.
- `optimized`: Hillis-Steele prefix forward with `ceil(log2(L))` rounds,
  plus an exact reverse-time MindSpore `bprop`.

For `B=2,D=192,N=16,L=16384`, static analysis gives 16,384 reference
iterations versus 14 prefix rounds. The optimized sequence state contains
100,663,296 FP32 elements (384 MiB per tensor), so it trades memory for
parallel forward execution. Explicit backward avoids differentiating through
the entire prefix graph.

## Representative paired FP32 results

| B | H | D | K | ref fwd | opt fwd | fwd speedup | ref bwd | opt bwd | bwd speedup | fwd error | grad error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 48 | 4 | 86.7 ms | 12.6 ms | 6.90× | 162 ms | 277 ms | 0.59× | 3.81e-6 | 2.29e-5 |
| 1 | 16 | 48 | 4 | 382 ms | 18.1 ms | 21.1× | 5290 ms | 989 ms | 5.35× | 9.54e-6 | 3.81e-5 |
| 2 | 16 | 96 | 8 | 367 ms | 40.0 ms | 9.18× | 6549 ms | 1217 ms | 5.38× | 7.63e-6 | 5.34e-5 |
| 1 | 32 | 96 | 8 | 1998 ms | 82.3 ms | 24.3× | 3791 ms | 4484 ms | 0.85× | 7.63e-6 | 9.16e-5 |
| 1 | 64 | 48 | 4 | 8204 ms | 182 ms | 45.2× | — | — | — | 1.34e-5 | — |

All rows above use the final exact reverse-time `bprop`. Timings vary with
shape and MindSpore compilation/cache state; first-call timings are emitted by
the benchmark separately and are not mixed into steady timing. Prefix
autodiff was rejected because its A-gradient exceeded the strict threshold.

Additional exact-backward maximum gradient errors:

- `B=2,H=16,D=96,K=8`: 5.34e-5
- `B=2,H=16,D=192,K=8`: 6.87e-5
- `B=1,H=32,D=48,K=4`: 8.96e-5
- fixed `(D,N)`, batch `(B,N,L)`, and grouped `(B,K,N,L)` B/C all pass tests

## Larger optimized forward results

| B | H | D | K | steady | first call | sampled RSS increase |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 48 | 4 | 731 ms | 1.92 s | 624 MiB |
| 1 | 128 | 96 | 4 | 1.42 s | 3.85 s | 1.69 GiB |
| 1 | 128 | 192 | 4 | 3.26 s | 8.42 s | 4.03 GiB |
| 2 | 32 | 192 | 8 | 358 ms | 1.04 s | 1.15 GiB† |

`†` includes optimized backward in that isolated run. The full 36-case
forward matrix was partially run and then stopped before high-memory
`B=2,H=128,D=192` cases could destabilize the shared 12 GiB process. Run large
cases in separate processes for reliable memory isolation.

## FP16 and BF16

FP16 quick cases reported forward max-absolute error around `9.77e-4` to
`1.95e-3`. Prefix-autodiff gradient errors were `4.88e-4` to `6.25e-2`; the
final exact `bprop` computes in FP32 and casts back, but FP16 remains a separate
precision mode and is not covered by the FP32 `1e-4` requirement.

This MindSpore 2.0.0 build has no `bfloat16` dtype. BF16 was not measured.

## Default backend

`optimized` is the CPU default because it materially accelerates forward and
its exact backward passes the FP32 gradient tolerance. `reference` remains the
numerical oracle. CPU backward speed is shape-dependent: quick cases improved,
while the `H=32,D=96` paired run was slightly slower. Backward performance is
therefore still an open optimization target.

## Commands

```bash
/root/anaconda3/envs/repmamba/bin/python -m pytest -q mindspore_repmamba/tests
/root/anaconda3/envs/repmamba/bin/python benchmark_selective_scan.py --quick
/root/anaconda3/envs/repmamba/bin/python benchmark_selective_scan.py \
  --batches 1 --heights 32 --channels 96 --groups 8 \
  --warmup 0 --repeats 1
/root/anaconda3/envs/repmamba/bin/python benchmark_selective_scan.py \
  --batches 1 --heights 128 --channels 48 --groups 4 \
  --backends optimized --no-backward --warmup 0 --repeats 1
```
