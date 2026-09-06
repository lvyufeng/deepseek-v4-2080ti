# cpp_engine vs vLLM/SGLang Architecture Comparison

> **Historical baseline:** Sections in this document describe the pre-Phase-1 `cpp_engine` surface
> where explicitly noted. Phase 1 adds the `pocketllm` control plane without merging the Torch and
> native C++ data planes.

## Phase 1 / Phase 2.1 / Phase 2.2 status

Phase 2.2 adds a local tensor-parallel process supervisor to the CLI. For `--tensor-parallel-size N`
the default `pocketllm serve` path creates per-run rendezvous state, assigns rank environment
variables, starts all local ranks, waits for a flushed readiness marker from every rank, exposes HTTP
only from rank 0, forwards termination signals, propagates rank failures, and reaps all children.
`--no-tensor-parallel-supervisor` preserves existing `torchrun`/manual rank launch workflows.

Automatic supervision is backend-gated: Torch delegates nonzero ranks to the existing
`src.server.openai._worker_loop`, while the Python C++ Qwen adapter still has no worker-loop binding
and therefore requires an external/legacy native launcher. The supervisor controls process lifecycle
only; it is not a request scheduler and does not add request-local KV state or continuous batching.

The supervisor uses backend-specific rendezvous protocols unchanged. Physical KV layouts, recurrent
state, NCCL/Gloo/HCCL collectives, and CUDA/Ascend kernels remain private to each execution plane.

## Phase 1 / Phase 2.1 baseline status

The new `pocketllm` package provides:

- `EngineArgs`, `SamplingParams`, `LLM`, and `AsyncLLM` public interfaces;
- explicit `torch` / `cpp` / `auto` backend selection;
- a native C++ adapter restricted to Qwen3.5 safetensors checkpoints;
- shared `/health`, `/alive`, `/ready`, `/metrics`, `/v1/chat/completions`, and
  `/v1/completions` endpoints;
- request IDs, safe-boundary cancellation, streaming token events, and usage metrics;
- library-first `LLM.chat()` / `chat_stream()` and matching `AsyncLLM` methods;
- one shared chat request builder for offline and HTTP entry points;
- optional pybind11 bindings for the token-oriented native Qwen engine.

The C++ compatibility adapter is intentionally serialized around one mutable native session. True
continuous batching, complete OpenAI semantics across both backends, and real-model Torch/C++ parity
remain follow-up work.

## 1. API entry points

### vLLM and SGLang

Both projects are library-first and expose a Python API in addition to an OpenAI-compatible server.
For example, vLLM exposes `LLM` and `SamplingParams`, while SGLang provides a native generation DSL
and server launcher.

### Pre-Phase-1 cpp_engine

The native runtime was primarily CLI-oriented:

```bash
./pocketllm_engine --serve --port 8000 --model /path/to/checkpoint --tp-world 4
```

`QwenEngine` and `PersistentEngine` were C++ classes for native callers, not a supported Python
library surface. The existing executable and its CLI remain supported.

### PocketLLM Phase 2.1

```python
from pocketllm import EngineArgs, LLM, SamplingParams

llm = LLM(EngineArgs(model="/path/to/qwen3.5", backend="cpp"))
outputs = llm.chat(
    [{"role": "user", "content": "hello"}],
    SamplingParams(max_tokens=32),
)
```

The public facade accepts raw text, pre-tokenized IDs, and OpenAI-style chat messages. `chat()` and
`chat_stream()` use the same backend-neutral normalization and prompt-construction path as the
HTTP chat endpoint; async variants mirror them through the existing executor wrapper. Results and
stream events remain backend-neutral. The native binding remains token-oriented and does not expose
Torch tensors, CUDA handles, ACL handles, or a shared physical KV-cache layout.

This is a library-first request surface, not a continuous-batching implementation: the C++ adapter
still serializes access to one mutable native session and reports `supports_batch=False`.

## 2. Request processing and batching

### vLLM/SGLang

- concurrent request admission;
- continuous batching during decode;
- request-aware KV-cache allocation and reclamation;
- scheduler-level fairness, backpressure, and (depending on configuration) preemption.

### cpp_engine and Phase 1

The legacy native HTTP server serialized requests with an in-flight mutex. The Phase-1 C++ adapter
preserves this safety property by serializing access to its one mutable Qwen session. Cancellation is
checked between native decode steps and never interrupts a device kernel.

**Remaining gap:** no true C++ multi-request batching, paged logical KV pool, fair queue, or
preemption. These require a request-aware scheduler and must not be implemented by sharing the
current mutable session unsafely. The Torch adapter continues to reuse its existing serving queue.

Phase 1.1 hardened protocol and termination behavior only. Chat requests now carry normalized
messages through a shared prompt boundary: checkpoint-owned `chat_template` is preferred, the
validated DeepSeek encoder is used for its no-template checkpoints, and deterministic `role: content`
is only a generic last resort. The C++ adapter stops at EOS with a correct `stop`/`length` finish reason
instead of always reporting `length`. Neither change touches scheduling: the native path remains one
serialized session, and `supports_batch` stays `False`.

## 3. Configuration

### vLLM/SGLang

Configuration is primarily validated Python arguments or CLI flags, including model path, tensor
parallel size, context length, cache dtype, and prefix caching.

### cpp_engine before Phase 1

Configuration was distributed among C++ structs, CLI flags, and more than 50 environment variables.
This made per-instance configuration and discoverability difficult.

### PocketLLM Phase 1

`EngineArgs` validates common settings such as:

- model and format (`safetensors` or `gguf`);
- backend and device;
- tensor-parallel size/rank;
- maximum context and prefill chunk size;
- KV-cache dtype and prefix caching;
- speculative-decoding method and token count.

Backend-specific tuning remains in `backend_options`. `EngineArgs.from_env()` is a compatibility
bridge; runtime variables are named `POCKETLLM_*` (renamed from `DSV4_*` — see
[the migration note](migration/dsv4-to-pocket-rename.md)), while `QWEN_*` and related names are
unchanged. Explicit API and CLI values take precedence over this bridge.

**Remaining gap:** not every model-specific tuning switch has a portable typed field, by design.
Kernel and scheduler tuning remains backend- and hardware-specific.

## 4. Health and observability

Before Phase 1, native serving exposed mostly log output and a basic health response. The shared
frontend now exposes:

- `/health` for liveness and backend status;
- `/alive` for a minimal liveness probe;
- `/ready` with HTTP 503 until the backend is ready;
- `/metrics` using dependency-free Prometheus text exposition;
- `/v1/models` for model discovery.

The first metrics surface includes total requests, active-request gauge, request duration,
prompt/completion token counters, and disconnect cancellation counters. Backend-specific KV usage,
queue depth, cache-hit, and speculative-decoding metrics remain follow-up work.

## 5. OpenAI compatibility

Phase 1 implements non-streaming and streaming chat completions, text completions, request IDs,
usage accounting, and safe-boundary cancellation through a shared protocol layer. It deliberately
does not claim full parity with vLLM or SGLang.

Remaining protocol work includes:

- completion-specific streaming chunk schemas;
- complete tool-call and reasoning/thinking normalization across both adapters;
- structured outputs and grammar-constrained decoding;
- multiple choices (`n`) on the native C++ path;
- logprobs and stop-string behavior on every backend;
- embeddings and multimodal request endpoints.

Unsupported native controls raise typed errors rather than being silently ignored.

## 6. Python and native integration

Phase 1 adds:

- `pocketllm.LLM` and `pocketllm.AsyncLLM`;
- optional `pocketllm_cpp` pybind11 module;
- lazy imports so `import pocketllm` does not require Torch, CUDA, or the native extension;
- a stable backend-neutral contract over independent execution implementations.

Native compute wrappers release the Python GIL around expensive prefill, decode, generation, and
verification calls. Construction and lifecycle operations remain native-boundary operations and are
not described as universally GIL-free.

## 7. Feature comparison

The native runtime retains strengths that must not be weakened by the common facade:

- MTP, DSpark, and DFlash2 speculative paths where supported;
- FP4, FP8, GGUF Q2, and other validated quantized kernels;
- tensor parallel execution and existing TP worker protocol;
- prefix reuse and recurrent-state snapshots;
- model-specific reasoning support in the existing Torch/native paths.

Compared with mature vLLM/SGLang deployments, PocketLLM still lacks a common implementation of:

- multimodal inputs;
- embeddings;
- LoRA adapter loading and multiplexing;
- structured/grammar-constrained outputs;
- complete multi-choice and logprob semantics;
- a common continuous-batching scheduler.

These are capability-gated follow-up features, not reasons to merge vendor kernels or KV layouts.

## 8. Architecture decision

The intended layering is:

```text
pocketllm control plane
  API types, request IDs, lifecycle, cancellation, streaming, OpenAI frontend, metrics, CLI
       |
       +-- Torch adapter  --> existing PyTorch/Triton runtime and scheduler
       |
       +-- C++ adapter    --> native C++/CUDA or C++/Ascend runtime and scheduler
```

The following remain backend-specific:

- physical KV-cache layout and memory management;
- continuous-batching implementation;
- TP collectives and worker protocol;
- quantization and attention kernels;
- CUDA versus Ascend device code;
- hardware-specific performance switches.

A common API is therefore a product/control-plane unification, not a lowest-common-denominator
data-plane rewrite.

## 9. Follow-up recommendations

1. Implement a request-aware C++ scheduler with per-request logical state and a backend-specific KV
   pool; gate it behind capability and latency regression tests.
2. Finish shared OpenAI semantics for tools, reasoning content, structured outputs, multiple choices,
   and completion streaming.
3. Add real-model greedy-token parity fixtures for Torch and C++ with documented tolerances.
4. Add backend-specific queue, cache, and speculative metrics.
5. Add embeddings, multimodal, LoRA, and grammar features only when the underlying model/runtime
   supports them.

## 10. Compatibility guarantees

Phase 1 preserves:

- existing `src.*` imports and environment variables;
- `pocketllm_engine` executable and legacy CLI/server commands;
- TP worker-loop behavior;
- speculative-decoding and quantized kernel implementations;
- validated single-request latency paths and hardware-specific tuning.

The public API is additive. It does not make a claim of bitwise equality between backend kernels,
and it does not enable continuous batching by default before parity and performance verification.
