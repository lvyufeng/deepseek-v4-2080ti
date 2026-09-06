# Qwen3.8-27B-NVFP4

## Runtime status

**Validated native C++/CUDA TP2 text runtime on RTX 2080 Ti (SM75).** PocketLLM loads the mixed-precision `unsloth/Qwen3.8-27B-NVFP4` checkpoint, keeps the NVFP4 dense-MLP weights resident as packed 4-bit blocks with their FP8 group scales, and executes them through integer kernels. Generation is bit-identical to the FP8 checkpoint on the validated fixtures.

RTX 2080 Ti has no native FP4 tensor-core instruction. Nothing here executes Blackwell FP4 MMA. The NVFP4 weights are unpacked in registers to INT8 and consumed by DP4A and INT8 WMMA, which is why NVFP4 buys memory rather than speed on this hardware. Read the performance section before choosing this format.

Like the FP8 page, this covers text prompt/token-ID smoke and timed greedy generation through `pocketllm_engine`. The vision tower is not executed and the OpenAI-compatible server is not wired up.

## Checkpoint specification

The text architecture is identical to [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md): 64 text layers (48 Gated DeltaNet + 16 full GQA), hidden 5,120, dense MLP intermediate 17,408, vocabulary 248,320, 24 query heads over 4 KV heads at head dimension 256.

What differs is the quantization, and the checkpoint is **not** uniformly NVFP4:

| Field | Value |
| --- | --- |
| HF repository | `unsloth/Qwen3.8-27B-NVFP4` |
| Revision | `9e3d73c76eddb75f795cc24ccfbc5affe41c66bd` |
| `quantization_config.format` | `mixed-precision` |
| NVFP4 group (`group_1`) | `re:.*mlp\.(gate|up|down)_proj$` |
| NVFP4 weight scheme | 4-bit float, `tensor_group`, group size 16, scale dtype `float8_e4m3fn`, static actorder |
| FP8 group (`group_0`) | attention/linear-attn projections, `lm_head`, and `layers.56–63` MLP |
| FP8 weight scheme | 8-bit float, per-channel, symmetric |
| Ignored | the complete `model.visual.*` stack |

Layers 56–63 keep FP8 MLPs, so "NVFP4" is a per-tensor property, not a model-wide one. Rank-local telemetry on TP2 reports 168 active NVFP4 group-16 linears, 233 active FP8 per-channel linears, and 96 active dense FP16 linears.

### NVFP4 numerics as implemented

Each NVFP4 weight tensor carries three levels:

1. `weight_packed` — `uint8`, two E2M1 values per byte. The **low** nibble is the earlier logical weight.
2. `weight_scale` — E4M3, one scale per **16** consecutive input channels.
3. `weight_global_scale` — one F32 per tensor. The runtime dequantization factor is `1 / weight_global_scale`.

The host packs these into a fixed 36-byte record covering 64 logical weights:

```cpp
struct QwenNvfp4Block64 {
    uint8_t d[4];    // four E4M3 group-16 scales
    uint8_t qs[32];  // 64 packed E2M1 values
};
```

This is deliberately **not** the DeepSeek-V4-Flash FP4 layout. DeepSeek uses E2M1 with an E8M0 scale over 32 elements, which lets a kernel apply one scale across two K=16 halves. Qwen's group is 16 with an E4M3 scale, so every K=16 boundary needs its own rescale. The DeepSeek kernels were reused for their *scheduling* ideas — PRMT nibble unpacking, DP4A accumulation, INT8 WMMA, activation-tile reuse, N-splitting — with Qwen's own scale math. No DeepSeek scale semantics are applied to Qwen tensors.

## Implemented execution path

Everything from the FP8 runtime (DeltaNet kernels, GQA prefill/decode, chunked prefill, prefix reuse, MTP/DSpark/DFlash2 speculation, TP sharding) is shared. NVFP4-specific additions:

- Packed NVFP4 weight residency: 4-bit blocks and E4M3 scales stay on device. No full FP16/FP32 weight expansion in any dispatched path; the FP32 expansion exists only as the reference/test kernel.
- Per-token dynamic INT8 activation quantization into a small reusable workspace (2.5 MB peak on TP2), matching the checkpoint's dynamic-token activation scheme.
- Decode (`rows == 1`): `q8_group32_dp4a`. PRMT-unpacks the nibbles and accumulates with `__dp4a`, reapplying the E4M3 group scale at every K=16 boundary.
- Prefill (`rows >= 128`): `q8_group32_wide_n64_shared_gate_up_q8`. A 128×64 output tile reuses each INT8 activation tile across 64 output rows and each unpacked weight tile across 128 activation rows, loading A as packed `int32`. Both K=16 scale boundaries are consumed inside one 32-value load/compute step, so the scale application collapses to one float multiply per K=16 subgroup instead of a per-subgroup store/barrier round trip.
- Narrow batches keep the 16×16×16 INT8 WMMA fallback, which is cheaper below the wide tile's row span.
- Gate and up share one INT8 activation quantization per token block.
- Mixed dispatch inside a single layer: NVFP4 MLPs for layers 0–55 and FP8 per-channel MLPs for layers 56–63, plus FP8 attention projections and FP8 `lm_head` throughout.

Gates, all defaulting to the fast path:

| Variable | Default | Effect |
| --- | --- | --- |
| `POCKETLLM_QWEN_NVFP4` | `auto` | `dp4a`, `wmma`, or `reference` force a single kernel family |
| `POCKETLLM_QWEN_NVFP4_WIDE_N64` | on | `0` falls back to the narrow WMMA prefill tile |
| `POCKETLLM_QWEN_NVFP4_WIDE_N64_MIN_ROWS` | `128` | row count at which the wide tile engages |
| `POCKETLLM_QWEN_NVFP4_SHARED_Q8_SWIGLU` | on | share one activation quantization between gate and up |
| `QWEN_PHASE_PROFILE` | off | per-phase timing breakdown |

## Validated performance

Hardware: 2×RTX 2080 Ti 22 GiB on the NVLink-connected physical GPUs 2 and 3 (`NV2`), CPU affinity `22-43,66-87`, single request, 16 generated tokens, real checkpoints and real tokenizer fixtures, three serial fresh-process repetitions per length, medians reported. All three configurations below ran on the **same** engine binary `4707381072...43cb1048` with `POCKETLLM_QWEN_GQA_OPTIMIZED=1`, the same prompt fixture, prefill chunk 512, and FP16 KV cache.

| Prompt | NVFP4 TP2 prefill / decode | FP8 TP2 prefill / decode | NVFP4 memory / rank | FP8 memory / rank |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 292.84 / 14.01 tok/s | 646.01 / 23.15 tok/s | 10.80 GiB | 14.77 GiB |
| 8,192 | 240.40 / 11.11 tok/s | 483.70 / 17.03 tok/s | 11.15 GiB | 15.11 GiB |

**NVFP4 is slower than FP8 on this hardware.** Against FP8 TP2 it reaches 0.45x prefill and 0.61x decode at 512, and 0.50x prefill and 0.65x decode at 8,192. What it buys is memory: 0.73x per-rank device usage, from `resident_weight_bytes=10,354,076,160` plus `resident_scale_bytes=470,206,976` against FP8's larger resident set. On SM75 that trade is the whole story — there is no FP4 tensor-core path to recover the arithmetic, so every NVFP4 GEMM pays nibble unpacking and a group-16 rescale that FP8 does not.

Deployment reference on all four GPUs, same binary and fixture:

| Prompt | FP8 TP4 prefill / decode | Memory / rank |
| ---: | ---: | ---: |
| 512 | 730.28 / 37.30 tok/s | 7.54 GiB |
| 8,192 | 732.81 / 24.86 tok/s | 7.75 GiB |

*FP8 TP4 — four GPUs; not a same-GPU-count speedup comparison.* It is included only because it is the existing deployment configuration. NVFP4 TP2 reaches 0.40x/0.38x of it at 512 and 0.33x/0.45x at 8,192.

### Wide-N64 prefill tile

The wide tile is the one change that moved NVFP4 prefill materially. Same binary, same fixture, only `POCKETLLM_QWEN_NVFP4_WIDE_N64` differs:

| Prompt | Wide off (narrow WMMA) | Wide on | Prefill gain |
| ---: | ---: | ---: | ---: |
| 512 | 103.13 / 13.93 tok/s | 292.84 / 14.01 tok/s | 2.84x |
| 8,192 | 90.43 / 10.61 tok/s | 240.40 / 11.11 tok/s | 2.66x |

Generated tokens are identical between the two settings at both lengths. Decode is unchanged, as expected: decode has `rows == 1` and stays on the DP4A matvec.

Kernel-level microbenchmark at the TP2-local MLP shapes, batch 512, 20 iterations, GPU 2 (`bench_qwen_nvfp4`). These are synthetic kernel timings and do not extrapolate to end-to-end throughput:

```
rows=8704 cols=5120  reference=106.107ms  dp4a=68.763  wmma=23.387  wide_n64=4.455  (23.3x vs reference)
rows=5120 cols=8704  reference=111.278ms  dp4a=69.289  wmma=21.895  wide_n64=4.693  (23.0x vs reference)
```

At batch 1 the ordering inverts — `dp4a=0.197ms`, `wmma=0.979ms`, `wide_n64=1.353ms` — which is why the row threshold exists and why decode keeps DP4A.

### Rejected and neutral experiments

- A fused NVFP4 SwiGLU projection regressed and is not on the default path.
- Sharing the INT8 activation quantization between gate and up is roughly break-even; it is kept because it lowers workspace pressure at no measured cost.
- Forcing `POCKETLLM_QWEN_NVFP4=dp4a` for prefill measured 28.57 tok/s at 8,192, about 8.4x below the wide tile. Pure DP4A is a decode kernel.

## Correctness and precision

- Both TP ranks generated identical greedy tokens in every repetition at both lengths (`rank_token_parity=PASS`, 3/3 per length in all three configurations).
- NVFP4 TP2 generated the **same** 16 tokens as FP8 TP2 and FP8 TP4 on the identical fixture at both lengths:
  - 512: `[1973, 9722, 54102, 9191, 13, 5546, 61485, 42903, 383, 87567, 69736, 68022, 369, 5222, 1132, 948]`
  - 8,192: `[10723, 6326, 8307, 6829, 777, 1056, 279, 3568, 6327, 1821, 13, 1061, 27502, 5533, 69892, 44424]`
- `test_qwen_nvfp4` passes: host packing exact (`half=0` `float=0` `q8_codes=0` `q8_scale=0`), INT8 activation quantization `1.458e-2`, and all three device kernels agreeing at `9.072e-4` half / `9.537e-7` float. The identical error across DP4A, WMMA, and wide confirms the residual is activation quantization, not a kernel defect.
- The wide tile masks partial tiles rather than shrinking its launch, so a dedicated tail case uses `batch=129 rows=67 cols=320` — a multiple of neither tile side — and compares against the FP32 WMMA output of the same INT8 input: `relative=4.763e-04` at `magnitude=2.317e+00`. FP16 output cannot resolve the absolute reference at this magnitude, and the check requires non-trivial magnitude so a zero result cannot pass.
- Regression suite run on the final binary: `test_fp4_matvec`, `test_qwen_config`, `test_qwen_weights`, `test_qwen_half_ops`, `test_qwen_nvfp4`, `test_qwen_fp8_online`, `test_qwen_sampler`, `test_qwen_dspark_ops`, `test_qwen_dflash2_ops`, `test_qwen_engine`, `test_qwen_dspark`, `test_qwen_dflash2`, `test_safetensors_reader`.
- NCCL smoke on GPUs 2/3 passes (`rank=0 value=1 sum=3`, `rank=1 value=2 sum=3`), and `ldd` resolves a real `libnccl.so.2`. TP results are meaningless without this check: when NCCL is absent the all-reduce is compiled out and TP still produces plausible output.
- DFlash2 TP2 parity over NVFP4 passes with both ranks agreeing, with and without the wide tile.

## Reproduction

Two ranks pinned to the NVLink pair, one shared NCCL ID:

```bash
rm -f /tmp/pocketllm_qwen_nvfp4_nccl.id
for rank in 0 1; do
  physical=$((rank + 2))
  CUDA_VISIBLE_DEVICES=$physical POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
  taskset -c 22-43,66-87 \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-NVFP4 \
    --tp-world 2 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nvfp4_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --generate-token 123 --max-new-tokens 16 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_nvfp4_rank${rank}.log 2>&1 &
done
wait
```

The fastest validated measurement command, which produced the NVFP4 rows above:

```bash
taskset -c 22-43,66-87 /path/to/deepseek/bin/python \
  scripts/bench_qwen_long_context.py \
  --ckpt /path/to/Qwen3.8-27B-NVFP4 \
  --binary build/cpp_engine/pocketllm_engine \
  --lengths 512,8192 --repetitions 3 \
  --tp-world 2 --devices 2,3 \
  --env POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
  --topology-label "NVFP4 TP2 GPUs2,3 NV2 wide-n64" \
  --work-dir .tmp/fair_nvfp4_tp2_wide_gqaopt
```

The FP8 A/B is the same command with `--ckpt /path/to/Qwen3.8-27B-FP8`; the FP8 TP4 reference adds `--tp-world 4 --devices 0,1,2,3`. Comparisons are valid only when the binary SHA, tokenizer fixture SHA, prefill chunk, KV dtype, and generated length all match — the harness records each of these in `results.json`.

Kernel-level sweep:

```bash
CUDA_VISIBLE_DEVICES=2 build/cpp_engine/tests/bench_qwen_nvfp4 \
  --batch 512 --rows 8704 --cols 5120 --iterations 20
```

Measure on idle GPUs. Contention from unrelated jobs was repeatedly visible in these runs as inflated `gpu_used_bytes` with decode collapsing to roughly 60% of its clean value, and in the worst case as an allocation failure of the NVFP4 device linear. Discard any repetition whose `gpu_used_bytes` deviates from the others.

## Known limitations

- **Slower than FP8 on SM75.** NVFP4 is a memory-footprint option here (0.73x per rank), not a throughput option. Choose it only when 4 GiB per rank matters more than roughly half the throughput.
- No native FP4 tensor-core execution. All NVFP4 math is emulated through INT8 DP4A/WMMA after register-level nibble unpacking.
- Text-only, CLI-only, greedy generation only — the same limits as the FP8 page. The vision tower is skipped, and Qwen is rejected by the OpenAI server path.
- The wide tile only engages at 128 rows and above. Decode and very narrow prefill chunks see none of the 2.7–2.8x prefill gain.
- Validated at 512 and 8,192 on TP2 only. Longer contexts, other TP widths, and FP8 KV cache over NVFP4 weights are not measured here.
- Layers 56–63 are FP8 in this checkpoint, so any claim about NVFP4 coverage must be read as a per-tensor, not a whole-model, property.
- Speculative decoding (MTP, DSpark, DFlash2) loads and passes parity over NVFP4 but its acceleration was not re-measured on this format.

## Evidence and related notes

- `cpp_engine/backends/cuda/kernels/qwen_nvfp4_ops.cu`
- `cpp_engine/include/qwen_cuda_ops.hpp`
- `cpp_engine/engine/qwen_engine.cpp`
- `cpp_engine/engine/qwen_weights.cpp`
- `cpp_engine/tests/test_qwen_nvfp4.cpp`
- `cpp_engine/tests/bench_qwen_nvfp4.cpp`
- `scripts/bench_qwen_long_context.py`
- [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) for the shared text runtime
- [DeepSeek-V4](deepseek-v4.md) for the FP4 INT8 kernels this borrowed scheduling from
- [Benchmark reporting rules](../benchmarking.md)
