# PocketLLM C++/CUDA engine

[Project home](../README.md) · [Model guides](../docs/models/README.md) · [Benchmarking](../docs/benchmarking.md)

`cpp_engine/` is PocketLLM's native C++/CUDA runtime and model-inspection layer. It currently contains performance-oriented DeepSeek-V4 paths, a validated Qwen3.8-27B-FP8 TP4 text runtime, GGUF/Safetensors readers, NCCL helpers, and model-specific CUDA tests.

The executable is still named `pocketllm_engine` for compatibility with existing scripts. The name does not mean every code path is DeepSeek-specific.

## Current scope

| Model/format | C++ capability |
| --- | --- |
| DeepSeek-V4 Safetensors FP4/FP8 | Persistent generation and OpenAI-compatible TP4 server |
| DeepSeek-V4 GGUF Q2/IQ2/IQ1 | Inspect, low-bit kernels, TP4 generation/smoke paths |
| Qwen3.8-27B-FP8 Safetensors | Config/tensor audit, GPU-resident TP4 text prefill/decode, timed greedy CLI |
| Other GGUF architectures | Generic reader/inspection tools; generation may live in the Python raw-block runtime |

Qwen vision execution and Qwen OpenAI serving are not implemented. Model metadata support should not be interpreted as full generation support.

## Build

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

The default CUDA architecture is SM75 for the RTX 2080 Ti/Turing baseline.

## Executable

```text
build/cpp_engine/pocketllm_engine
```

Inspect or dump a DeepSeek GGUF config:

```bash
build/cpp_engine/pocketllm_engine \
  --model /path/to/model.gguf \
  --dump-config
```

Inspect a Safetensors checkpoint:

```bash
build/cpp_engine/pocketllm_engine \
  --ckpt /path/to/checkpoint \
  --dump-config
```

Audit a Qwen TP shard without running generation:

```bash
build/cpp_engine/pocketllm_engine \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --tp-rank 0 \
  --qwen-audit
```

See the [Qwen model guide](../docs/models/qwen3.8-27b-fp8.md) for the four-rank timed generation command and the [DeepSeek model guide](../docs/models/deepseek-v4.md) for server usage.

## Standalone inspection tools

```bash
build/cpp_engine/tools/inspect_gguf /path/to/model.gguf --limit 20
build/cpp_engine/tools/inspect_safetensors /path/to/checkpoint
```

Available targets and exact paths are defined in `cpp_engine/CMakeLists.txt`.

## Tests and benchmarks

CMake builds focused CUDA/numerical tests under `build/cpp_engine/tests/`, including:

- Qwen config, weight sharding, FP8 online unpack, DeltaNet, GQA, convolution-tail, and engine smoke tests;
- DeepSeek FP4/FP8/Q2 kernels, MoE determinism, attention, tokenizer, persistent-engine, and NCCL tests;
- DSpark draft/verify parity and scheduling tests;
- `bench_qwen_fp8_kernels` and `bench_dspark_decode` microbenchmarks.

A kernel microbenchmark is not a model TPS result. End-to-end numbers must follow the repository's [benchmark reporting rules](../docs/benchmarking.md).

## Naming

The runtime uses the `pocket` C++ namespace, `POCKETLLM_*` environment variables, and the
`pocketllm_engine` executable. `DeepSeek-V4` still appears wherever an actual upstream model,
checkpoint, or architecture is meant — that is a model name, not a project name.

This is a rename from the project's former `dsv4` identity and it is a **breaking change**: see
[the migration note](../docs/migration/dsv4-to-pocket-rename.md) for the full mapping and for how to
translate an existing deployment command.
