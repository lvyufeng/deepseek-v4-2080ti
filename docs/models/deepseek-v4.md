# DeepSeek-V4-Flash

## Runtime status

**Validated generation.** DeepSeek-V4 is PocketLLM's most mature model family and has both PyTorch and native C++/CUDA execution paths. Safetensors FP4/FP8 and GGUF Q2/IQ2/IQ1 variants have separate loading, kernel, and placement policies.

- C++ Safetensors path: persistent generation and OpenAI-compatible serving.
- PyTorch Safetensors path: heterogeneous routed-expert serving and performance experiments.
- C++ GGUF path: TP4 generation, grouped prefill, active-expert decode, and low-bit raw-block kernels.
- DSpark: available as an experimental speculative path; the current C++ sequential verify implementation is a correctness path, not an e2e speedup claim.

## Model specification

The validated DeepSeek-V4-Flash configuration in this repository uses:

| Field | Value |
| --- | ---: |
| Transformer layers | 43 |
| Hidden size | 4096 |
| Attention heads | 64 |
| Head dimension | 512 |
| RoPE dimension | 64 |
| Routed experts | 256 |
| Active experts | top-6 |
| Shared experts | 1 |
| Expert intermediate size | 2048 |
| Original sequence length | 65,536 |
| Attention | MLA, sparse/indexed attention, C4 indexer/compressor |
| Safetensors dense dtype | FP8-style checkpoint tensors |
| Safetensors expert dtype | FP4 |

GGUF variants mix dense Q8_0 tensors with IQ1/IQ2/Q2 routed expert formats depending on the checkpoint.

## Implemented execution paths

### FP4 Safetensors

Dense/attention work runs on GPU. Routed experts may be resident or kept in host memory and staged as active quantized blocks. The implementation includes TP4 NCCL reductions, compressed/indexed attention, grouped prefill MoE, deterministic expert reduction, and an embedded C++ OpenAI-compatible server.

### GGUF Q2/IQ2/IQ1

The C++ GGUF runtime reads raw quantized blocks and uses dedicated DP4A/MMQ-style kernels where supported. Prefill and decode use different dispatches. Routed experts can remain host-resident and are copied through bounded staging buffers; the runtime deliberately does not pin an entire file-backed GGUF mmap.

## Validated performance

Hardware unless noted: 4×RTX 2080 Ti 22 GiB, TP4, single request, real DeepSeek-V4 checkpoints.

### Native C++ FP4

| Prompt | Prefill time | Prefill | Decode | Peak GPU/rank |
| ---: | ---: | ---: | ---: | ---: |
| 2,101 tokens | ~7.6 s | ~275 tok/s | ~3.7 tok/s | ~7 GiB |
| 32,768 tokens | ~82 s | ~402 tok/s | not reported | ~11.2 GiB |
| 65,536 tokens | ~164 s | ~401 tok/s | ~3.7 tok/s | ~14.5 GiB |

The long-context batched-attention path improved 32K/64K prefill by approximately 8.6–8.8× over the earlier per-position implementation. Decode remains limited primarily by PCIe expert staging rather than Python overhead.

### PyTorch FP4 heterogeneous path

| Scenario | Prompt tokens | Decode tokens | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: |
| Maximum validated context | 65,536 | 2 | ~255 tok/s | n/a |
| Long prompt | 2,148 | 63 | ~321 tok/s after warmup | 3.49 tok/s mean |
| Short prompt | 29 | 127 | short timing is noisy | 3.16 tok/s mean |

### GGUF Q2/IQ2 heterogeneous serving snapshot

| Case | Prompt | Decode tokens | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: |
| Warm `long_short` | 2,148 | 7 | 216.17 tok/s | 4.44 tok/s |
| Warm `long_long` | 2,148 | 63 | 213.05 tok/s | 3.75 tok/s |

A separate C++ kernel milestone measured 512-token GGUF Q2 prefill at 60.42 tok/s after sparse window/head-pair attention, with decode unchanged at 4.50 tok/s for that benchmark. It is not directly comparable to the OpenAI serving table because the runtime/fixture differs.

## Correctness and precision

- TP4 GGUF Q2 uses FP32 all-reduce to preserve parity; the BF16 round-trip reduction compounded quantization drift.
- FP4 MoE deterministic reduction is enabled by default to avoid `atomicAdd` run-to-run drift.
- Long-context prefill must use the validated batched attention path; earlier alternative continuation kernels regressed performance substantially.
- DSpark's multi-token verify can change floating-point reduction order and token selection. See the dedicated note before making parity claims.

## Reproduction

Build the C++ engine as documented in the repository root, then launch the Safetensors server:

```bash
CKPT=/path/to/DeepSeek-V4-Flash \
PORT=8000 \
MAX_CONTEXT=8192 \
PYTHON=python \
bash scripts/run_cpp_serve_tp4.sh
```

PyTorch OpenAI-compatible serving:

```bash
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 \
bash scripts/run_openai_server.sh
```

GGUF Q2/IQ2 serving:

```bash
CKPT_PATH=/path/to/deepseek-v4.gguf \
TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash-tokenizer \
bash scripts/run_gguf_q2_layer_pp.sh
```

Inspect a GGUF checkpoint:

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/deepseek-v4.gguf \
  --summary --validate-ds4-q2
```

## Known limitations

- Host-resident routed experts make decode sensitive to PCIe, NUMA, page-cache, and CPU behavior.
- One-GPU GGUF Q2 mode is functional for short smoke tests but is not practical for long prompts.
- FlashMemory runtime scoring and 1M-context functionality have their own enablement and validation constraints.
- The C++ executable still carries the compatibility name `pocketllm_engine`.

## Evidence and related notes

- [DSpark speculative decoding](../dspark.md)
- [FlashMemory 1M context](../FLASHMEMORY_1M_CONTEXT.md)
- [Phase 2 validation](../PHASE2_VALIDATION_RESULTS.md)
- [GGUF Q2 single-GPU history](deepseek-v4-gguf-q2-single-gpu.md)
- [Historical 4×RTX 2080 Ti report](../reports/dsv4_2080ti_report.pdf)
- [Benchmark reporting rules](../benchmarking.md)
