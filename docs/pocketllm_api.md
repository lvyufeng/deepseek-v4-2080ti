# PocketLLM API and Backend Guide

PocketLLM presents one user-facing API over two independent execution planes:

- **Torch** uses the existing PyTorch/Triton runtimes under `src/`.
- **C++** uses the native `cpp_engine` runtime and the selected CUDA or Ascend backend.

The common API does not imply shared kernels, KV-cache layouts, or schedulers. Those remain backend- and hardware-specific so that Turing CUDA and Ascend optimizations are not weakened by a lowest-common-denominator abstraction.

## Offline API

```python
from pocketllm import EngineArgs, LLM, SamplingParams

llm = LLM(EngineArgs(
    model="/path/to/checkpoint",
    backend="auto",  # or "torch" / "cpp"
    tensor_parallel_size=4,
    max_model_len=65536,
))

outputs = llm.generate(
    ["Explain speculative decoding.", "Explain continuous batching."],
    SamplingParams(max_tokens=128, temperature=0.0),
)
for output in outputs:
    print(output.text, output.usage.as_dict())
```

Pre-tokenized input is also accepted:

```python
outputs = llm.generate([[1, 42, 17]], SamplingParams(max_tokens=16))
```

For chat-shaped inputs, use the library-first chat surface. It accepts the same normalized message
and optional fields as `/v1/chat/completions`, and returns the same list-shaped result as
`generate()` (one result for the supplied conversation):

```python
messages = [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is 2+2?"},
]
outputs = llm.chat(
    messages,
    SamplingParams(max_tokens=32, temperature=0.0),
    reasoning_effort="low",
)
print(outputs[0].text)
```

`chat()` also accepts `reasoning`, `tools`, `tool_choice`, `response_format`, and an optional
`request_id`. The request body is normalized through the same backend-neutral builder used by the
HTTP endpoint; checkpoint-owned chat templates remain the authority for model-specific prompt
encoding. Caller-owned message and tool structures are not mutated.

Use `generate_stream()` or `chat_stream()` for token events and `cancel(request_id)` to request
cancellation at a safe generation boundary. The initial C++ compatibility adapter is serialized and
exposes native greedy generation; unsupported sampling or request features report
`UnsupportedFeatureError` rather than being silently ignored. Native streaming decodes the cumulative
token sequence before emitting each delta, so BPE and UTF-8 token boundaries are handled by the
tokenizer.

## Async API

`AsyncLLM` mirrors every offline entry point: `generate`, `generate_stream`, `chat`, and
`chat_stream`.

```python
from pocketllm import AsyncLLM, EngineArgs, SamplingParams

async with AsyncLLM(EngineArgs(model="/path/to/checkpoint")) as llm:
    result = (await llm.generate("Hello", SamplingParams(max_tokens=32)))[0]
    async for event in llm.generate_stream("Stream this"):
        print(event.text, end="", flush=True)

    chat_result = (await llm.chat(
        [{"role": "user", "content": "Explain KV caching."}],
        SamplingParams(max_tokens=32),
    ))[0]
    print(chat_result.text)
    async for event in llm.chat_stream(
        [{"role": "user", "content": "Stream a short answer."}],
    ):
        print(event.text, end="", flush=True)
```

`AsyncLLM` currently provides non-blocking application integration around the backend contract. It does not claim device-level continuous batching. Backend schedulers will add that capability independently. The async chat methods reuse the same executor-backed lifecycle and `TokenEvent` contract as the sync facade; they do not add a scheduler.

## CLI and server

```bash
# Installed console script
pocketllm serve \
  --model /path/to/checkpoint \
  --backend auto \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --port 8000

# Source-tree equivalent
python -m pocketllm serve \
  --model /path/to/checkpoint \
  --backend auto \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --port 8000
``` 

For `tensor_parallel_size > 1`, the CLI supervises local tensor-parallel ranks by default. It creates a
private per-run rendezvous directory and NCCL-ID path, assigns `RANK`/`LOCAL_RANK`/`WORLD_SIZE` and
`TP_RANK`/`TP_WORLD`, starts every rank without a shell, and waits for all ranks to finish loading
before rank 0 is considered ready. Only rank 0 binds the HTTP listener. A rank failure, startup
timeout, or received `SIGINT`/`SIGTERM` causes the supervisor to stop and reap the whole group.
Use `--tensor-parallel-startup-timeout SECONDS` and `--tensor-parallel-shutdown-timeout SECONDS`
to tune lifecycle bounds; `--tensor-parallel-master-addr`, `--tensor-parallel-master-port`, and
`--tensor-parallel-rendezvous-dir` are available for deployments that need explicit rendezvous
placement. A caller-provided rendezvous directory is treated as a parent for a fresh private run
directory and is never removed by PocketLLM.

The built-in supervisor currently works with the Torch backend by reusing its existing NCCL/Gloo
worker loop. The Python C++ Qwen adapter does not yet expose a native worker entry point, so
`backend="cpp"` must use the legacy `pocketllm_engine` launcher or opt out with
`--no-tensor-parallel-supervisor`. Existing `torchrun` and manual rank launchers remain compatible
through that opt-out. This process supervisor is not a scheduler and does not provide continuous
batching or request-local native state.

The unified server provides:

- `GET /health`
- `GET /alive`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `DELETE /v1/requests/<request_id>`

`/ready` returns HTTP 503 while model loading is incomplete. `/metrics` uses dependency-free Prometheus text exposition and can later be wrapped by a richer exporter.

## Configuration precedence

Prefer typed `EngineArgs` and explicit CLI options. `EngineArgs.from_env()` exists as a compatibility bridge for legacy deployments. Runtime tuning variables are named `POCKETLLM_*` (renamed from `DSV4_*`, a breaking change — see [the migration note](migration/dsv4-to-pocket-rename.md)); `QWEN_*` and related names are unchanged. Backend-specific tuning belongs in `backend_options` and must not be assumed portable between CUDA and Ascend.

## Native C++ Python module

The native bridge is optional and does not affect CPU-only imports:

```bash
cmake -S cpp_engine -B cpp_engine/build-python \
  -DPOCKET_BACKEND=cuda \
  -DPOCKET_BUILD_PYTHON=ON \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build cpp_engine/build-python --target pocketllm_cpp -j
```

Add `cpp_engine/build-python/python` to `PYTHONPATH` for a build-tree smoke test:

```bash
PYTHONPATH=cpp_engine/build-python/python python -c \
  'import pocketllm_cpp; print(pocketllm_cpp.backend)'
```

The module exposes token-oriented `QwenEngine` and low-level `PersistentEngine` value types. Device-touching calls (prefill, decode, generate, verify, warmup, reset) release the Python GIL; cheap accessors and construction do not. It intentionally does not expose CUDA/ACL handles or Torch tensors.

## Backend selection

`backend="auto"` picks the C++ adapter only when the native module is importable and the checkpoint
is a Qwen3.5 safetensors model; anything else, including GGUF, stays on Torch. An explicit
`backend="cpp"` for an unsupported checkpoint raises `UnsupportedFeatureError` before any CUDA
initialization instead of failing deep inside the native loader.

Capabilities reported by the C++ adapter follow the linked device backend. An Ascend build advertises
only the speculative methods it implements, since the external DSpark and DFlash2 drafters are
CUDA-only.

## Request normalization

`pocketllm.protocol` holds the request normalization shared by the unified server and the legacy
`src.server.openai` server: OpenAI content-block flattening, tool attachment and `tool_choice`
instructions, `reasoning`/`reasoning_effort` handling, tool-call shaping, and stop-string truncation.
There is one implementation, and it imports neither Torch nor the native module.

`/v1/chat/completions` puts the normalized messages, thinking mode, reasoning effort, and tool
metadata in `GenerationRequest.metadata`. The shared prompt boundary first asks the selected
checkpoint tokenizer to apply its own `chat_template` with an assistant generation prompt. This is
the same model-owned-template contract used by vLLM/SGLang and preserves model-specific special
tokens, reasoning controls, and tool formatting. For DeepSeek checkpoints whose tokenizer has no
chat template, the validated legacy `src.encoding.deepseek_v4.encode_messages` format is used instead.
`GenerationRequest.prompt` still carries a deterministic `role: content` rendering only as a last-resort
fallback for generic tokenizers that provide neither format. `/v1/completions` passes `prompt` through
unchanged and validates that a list prompt contains only strings.

The template receives normalized tool definitions and a private compatibility copy of prior tool-call
arguments; public request metadata is never mutated. Template-specific reasoning names are mapped to
the vocabulary accepted by the checkpoint (for example, `high`/`max` map to Qwen's `xhigh`).
Unsupported model-specific template features remain the responsibility of the selected backend.

A backend that separates reasoning from content can set `reasoning_content` and `tool_calls` in its
result or event metadata; those are forwarded to the response and to streamed deltas. A backend that
does not simply omits them.

## Termination semantics

The C++ adapter decides when generation stops. The first EOS token ends the request, is excluded from
the returned token ids and text, and yields `finish_reason="stop"`. `finish_reason="length"` means the
token budget ended first. Usage counts the EOS step the engine executed, so streaming and offline
usage agree.

`QwenEngine.generate` takes no EOS argument and keeps mutating its session for the whole token budget,
so when an EOS id is known both the offline and streamed paths drive `prefill`/`decode_step`
themselves and stop at EOS. Running `generate()` and truncating afterwards would leave the recurrent
state and prefix cache positioned past text the caller never saw, corrupting reuse for the next
request. Native `generate()` is still used when no EOS id is available, where the token budget is the
only stopping rule.

EOS ids are resolved in order: `backend_options["eos_token_id"]`, the native engine's `eos_id`, the
native config, the checkpoint's `generation_config.json`, the checkpoint's `config.json`, then the
tokenizer. `generation_config.json` is preferred over the tokenizer because chat checkpoints commonly
stop on a turn-end token that differs from the tokenizer's EOS. A non-integer override is rejected
rather than guessed. When no EOS is available, `capabilities.details["eos_source"]` reports `none` and
only the token budget can end generation. Streaming never issues another `decode_step` after EOS, and
it never calls `reset()` per request, since `QwenEngine::reset()` would clear the prefix cache that
`prefill()` relies on.

## Cancellation semantics

`cancel(request_id)` returns `True` only for a request that is currently active, and cancellation is
observed at safe boundaries between generation steps. It never interrupts a running device kernel and
never rolls back a partially executed native step. `DELETE /v1/requests/<request_id>` returns HTTP 404
for an unknown or already-finished request.

The existing `pocketllm_engine` executable and its CLI remain supported. The shared Python server is a migration path, not a replacement that invalidates existing production commands.
