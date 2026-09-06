# Qwen quantized KV cache at 65K (TG512), and a correction to 872ff3f

## Correction

The commit message of 872ff3f ("Add FP8 dequant-once, INT8 per-token-head, and TQ4NC
quantizers", PR #83) contains two throughput claims that were not backed by a run record:

- "65K prefill goes 57.9 -> 1138.1 tok/s" for the FP8 path;
- "Prefill reuses the dequant-once path and lands at 1142.3 tok/s" for INT8.

Neither number came from a 65536-token run of the format it was attached to. `1138.1` and
the neighbouring TurboQuant/FP16 figures circulating in notes at the time were 32768-token
measurements that had been relabelled as 65536, and every long-context record on disk
predating this document used `--max-new-tokens 4`, which cannot measure decode at all. No
run record for `turboquant_k8v4` or `int8_per_token_head` above 32768 tokens existed when
that commit was written.

The commit is merged, so its message stands as written. This file is the correction of
record. The table below replaces those claims. The architectural claims in 872ff3f
(dequant-once removes the O(rows x context) redundancy; prefill reaches FP16 parity; DP4A
cannot amortize a per-position scale; the Triton kernel matches the PyTorch reference to
8e-6 but is not wired into `cpp_engine`) are unaffected and still hold.

## Measured results

Qwen3.8-27B-FP8, 4x RTX 2080 Ti, TP4, NCCL-enabled build, rev `3aa1484` (tree-identical to
`872ff3f`). Four cases run serially, each generating 512 tokens, all with 4-rank token
parity PASS.

```
python scripts/bench_qwen_long_context.py \
  --ckpt /mnt/data2/Qwen3.8-27B-FP8 \
  --binary build/pocketllm_engine \
  --lengths 65536 --max-new-tokens 512 \
  --prefill-chunk-tokens 4096 \
  --kv-cache-dtype <fp16|fp8|turboquant_k8v4|int8_per_token_head> \
  --tp-world 4
```

| KV dtype | prefill s | prefill tok/s | decode s | decode tok/s | KV MiB/rank | pf vs fp16 | dc vs fp16 |
|---|---|---|---|---|---|---|---|
| fp16 | 56.46 | 1160.7 | 25.41 | 20.11 | 1032 | 1.000 | 1.000 |
| fp8 | 56.29 | 1164.3 | 204.22 | 2.50 | 532 | 1.003 | 0.124 |
| turboquant_k8v4 | 56.39 | 1162.1 | 68.57 | 7.45 | 391 | 1.001 | 0.370 |
| int8_per_token_head | 56.32 | 1163.6 | 487.87 | 1.05 | 520 | 1.002 | 0.052 |

`decode_token_count` is 511 in every case (the first token belongs to prefill, per
`docs/benchmarking.md`). TQ4NC is absent because it has no CUDA kernel and is not wired
into attention, so it cannot run.

## Reading of the results

Prefill parity is real: all three quantized formats land within 0.3% of FP16, confirming
dequant-once works. Prefill is also insensitive to the KV format, which is what a single
bulk dequant into a dense FP16 workspace predicts.

Decode is where the gap lives, and it is large. Ranking by decode throughput,
`turboquant_k8v4` (0.370x FP16) is the only quantized format that is practical, and it is
also the most memory-efficient at 391 MiB/rank, a 62% saving. FP8 at 0.124x and
`int8_per_token_head` at 0.052x are opt-in curiosities at this context length.

## Against vLLM-2080Ti

vLLM-2080Ti's own PP65536/TG512 sweep (`docs/qwen36-kv-throughput-sweep.zh-CN.md` in that
tree) reports, for Qwen3.6 27B FP8 TP2 without MTP: FP16 1303.9/29.1, INT8 1274.8/33.7,
TQK8V4 1277.9/20.7, TQ4NC 1273.9/19.6.

This is a different model (Qwen3.6 vs 3.8) at a different TP width (2 vs 4), so the ratios
below are indicative, not a controlled comparison:

- prefill: 0.89x (fp16), 0.91x (tqk8v4), 0.91x (int8)
- decode: 0.69x (fp16), 0.36x (tqk8v4), 0.031x (int8)

The FP16 and TQK8V4 decode ratios are plausible as TP4 communication overhead plus model
differences. The INT8 ratio is not: on vLLM, INT8 decode is *faster* than FP16 (33.7 vs
29.1), while ours is 1/19th of our own FP16. The sign is reversed, which points at the
implementation rather than at hardware or model differences. The suspect is Phase 3 of
`cpp_engine/backends/cuda/kernels/qwen_int8_per_token_head_ops.cu`, which walks all 65536 positions serially
per output channel, dequantizing one value at a time. Wiring in the already-validated
Triton kernel (`src/kernels/int8_per_token_head_triton.py`) is the open follow-up; whether
it reaches vLLM's 33.7 tok/s is untested.
