# cpp_engine multi-backend refactor plan

Target: one engine implementation running over swappable device backends (CUDA / Ascend), with no
compromise to per-hardware kernel optimization.

Decisions already made:

- **Route A**: a single engine plus a device runtime abstraction (not per-backend engine forks)
- **Directory layout: Option B** — explicit `core/` + `engine/` + `backends/` layering
- **Build-time single backend** selection, no runtime vtable in the hot path
- Scope: **all existing cpp_engine models**, single-card **and** multi-card
- Ascend stack: **ACL + AscendC + HCCL**
- Primary weight format: **per model** (to be pinned per model before kernel work)

## Why this is cheaper than it looks

Three measured facts make the refactor tractable:

1. **Zero kernel launches in the engine layer.** `<<<` count is 0 in `deepseek_v4_engine.cpp`,
   `qwen_engine.cpp`, and `dspark_engine.cpp`. All kernels already sit behind a function-call
   boundary.
2. **The kernel headers are already vendor-neutral.** `cuda_ops.hpp` and `qwen_cuda_ops.hpp` include
   only `<cstddef>` / `<cstdint>`, and 91 of 93 declarations take `void* stream`. Only
   `sampler_ops.hpp` leaks CUDA types.
3. **Device memory is already funnelled.** In `deepseek_v4_engine.cpp`, 892 of ~934 CUDA call sites go
   through the `check_cuda(...)` wrapper; only 42 raw `cudaMalloc` sites bypass it.

So the engine-layer coupling is overwhelmingly **memory management, not compute**:
`cudaMalloc`/`cudaFree` 1333, `cudaMemcpy*` 268, stream/event ~100, kernel launches 0.

## Target layout

```
cpp_engine/
  core/                    # device-agnostic, compiles without any vendor SDK
    json_lite.cpp  gguf_reader.cpp  safetensors_reader.cpp  safetensors_model.cpp
    weight_source.cpp  model_config.cpp  qwen_config.cpp  tokenizer.cpp
    tensor.cpp  sampler.cpp  cmd_channel.cpp  python_sidecar.cpp  openai_server.cpp
  engine/                  # one copy, written against backends/api only
    deepseek_v4_engine.cpp  qwen_engine.cpp  dspark_engine.cpp
    qwen_dflash2.cpp  qwen_dspark.cpp  qwen_weights.cpp  qwen_target_head.cpp
  backends/
    api/                   # vendor-neutral contracts (no vendor headers)
      device_runtime.hpp   # alloc/free, memcpy, stream, event, device mgmt
      device_ops.hpp       # operator contracts (was cuda_ops.hpp)
      device_collective.hpp# all-reduce / all-gather / broadcast / top-k
    cuda/
      runtime/             # CUDA impl of device_runtime
      kernels/             # current cuda/*.cu, moved verbatim
      collective/          # NCCL (current tp_comm.cpp)
    ascend/
      runtime/             # ACL impl
      kernels/             # AscendC
      collective/          # HCCL
  tools/  tests/  third_party/
```

`core/` and `engine/` must never include a vendor header. This is checkable in CI with a grep, which
is the main reason for choosing Option B over a flat split.

## Interface design

### device_runtime.hpp

Covers exactly what the engine layer actually uses today, nothing speculative:

- allocation: device alloc/free, host-pinned alloc/free
- transfer: H2D, D2H, D2D, async variants, memset
- ordering: stream create/destroy/synchronize, event create/record/wait/destroy
- device: set/get current device, device count, peer access query
- memory info: allocated / reserved / peak counters

Streams and events are exposed as opaque handles (`void*`), matching the existing
`void* stream = nullptr` convention. `cudaStream_t` and `aclrtStream` both fit.

### device_ops.hpp

Rename `*_cuda` symbols to vendor-neutral operator names (`q8_0_matvec_cuda` -> `q8_0_matvec`).
Signatures stay **coarse-grained per operator** (`q8_0_matvec`, `fp4_moe_*`, `gqa_*`) rather than
per-tensor-op. This is deliberate: coarse contracts let each backend choose its own tiling, L2
residency, and pipelining internally. A fine-grained abstraction would collapse to a
lowest-common-denominator API and destroy exactly the hardware-specific tuning this project exists
for.

### device_collective.hpp

The collective surface is small: 13 functions in `tp_comm.hpp`, using only
AllReduce / AllGather / Broadcast plus comm init/destroy and group start/end. Every one of these has
an HCCL equivalent. Engine call sites: 54 in `deepseek_v4_engine.cpp`, 52 in `qwen_engine.cpp`.

## Kernel sharing policy

**CUDA and Ascend share interface signatures only — never kernel implementations.**

The 22k lines under `cuda/` are tuned for sm_75 (no bf16 hardware, no FP8 units): `qwen_half_ops.cu`
3513 lines, `qwen_gqa_optimized.cu` 2471 lines. None of that transfers to DaVinci, which has native
bf16 and a completely different Cube matrix unit. Ascend gets its own from-scratch implementations.

Any attempt to share kernel bodies degrades to a lowest-common-denominator and loses the 2080 Ti
optimizations. What is shared is `core/` and `engine/` — the parts that would otherwise be
maintained twice across 4+ model architectures.

## Phases

Each phase ends in a buildable, testable state.

### Phase 0 — Neutralize leaks (no behavior change)

- Remove the CUDA leak from `sampler_ops.hpp`: `#include <cuda_runtime.h>` + 5x `cudaStream_t`
  become `void* stream`, matching the other 91 declarations.
- Rename `*_cuda` operator symbols to vendor-neutral names, keeping thin deprecated aliases so no
  call site breaks in this phase.
- Verifiable purely by CUDA rebuild + existing tests. Zero runtime change.

### Phase 1 — Extract device_runtime, CUDA-only

Scope is now measured: the 7 engine sources still needing a vendor SDK are `deepseek_v4_engine.cpp`,
`qwen_engine.cpp`, `dspark_engine.cpp`, `qwen_dflash2.cpp`, `qwen_dspark.cpp`, `qwen_weights.cpp`
and `main.cpp`. Completing this phase means `check_layering` can be extended to cover `engine/`.

- Define `backends/api/device_runtime.hpp`.
- Implement it in `backends/cuda/runtime/`, initially as thin inline wrappers so the CUDA path stays
  identical in codegen.
- Migrate `check_cuda` call sites to the abstraction. Start with the low-count files
  (`qwen_target_head` 4, `main` 7, `qwen_weights` 18, `qwen_dspark` 50) to validate the interface
  before touching `deepseek_v4_engine.cpp`.
- **The 42 raw `cudaMalloc` sites in `deepseek_v4_engine.cpp` need individual review** — they bypass
  `check_cuda` and will not be caught by changing one helper.
- Safety net: CUDA build must stay green throughout; the compiler catches missed sites once vendor
  headers are removed from `engine/`.

### Phase 2 — Move directories (DONE)

- Moved into the Option B layout with `git mv`. All 43 moves are `R100`, i.e. byte-identical
  renames, so no CUDA kernel content changed.
- Added the `check_layering` target asserting that `include/` and `core/` pull in no vendor SDK
  header. `engine/` is deliberately not yet covered, since it still calls vendor APIs directly.
- Reworked CMake into `POCKET_BACKEND=cuda|ascend` with `cuda` as the default, keeping the
  `pocket_cpp_core` archive, the `pocketllm_engine` binary name and all 87 test targets unchanged.

Verified locally without a CUDA toolchain: all 13 `core/` sources plus `third_party/httplib.cpp`
compile to objects, while the 7 `engine/` sources fail on `cuda_runtime.h` exactly as expected.
That is empirical confirmation the core boundary holds and a precise scope for Phase 1.

### Phase 3 — Ascend runtime skeleton

- Implement `device_runtime` on ACL (`aclrtMalloc` / `aclrtMemcpy` / `aclrtCreateStream` /
  `aclrtCreateEvent`), plus HCCL for `device_collective`.
- Engine compiles and links against Ascend with kernels stubbed, proving the abstraction is complete
  before any kernel work begins.

### Phase 4 — AscendC kernels, per model

- Pin the primary weight format per model first, then implement and tune.
- Target the **first-generation** SoC on this machine (32 MB L2, Cube+Vector combined). See
  CLAUDE.md for the naming trap: `910B` without a digit is first generation.
- Multi-card: measure intra-server topology first. `/etc/hccn.conf` is currently empty, so RDMA
  needs configuration; intra-server SDMA does not depend on it.

## Verification

- **CUDA regression**: this machine has no `nvcc`, so CUDA builds and the 87 test targets must run on
  a separate 2080 Ti machine. Locally verifiable: interface contracts, static layering checks.
- **Ascend correctness**: parity tests against the CUDA path per operator, reusing the existing
  `test_*_parity.cpp` pattern (`test_dspark_attention_parity`, `test_moe_*`, `test_qwen_*`).
- **Layering**: CI grep asserting no vendor headers in `core/` or `engine/`.
- Existing safety net is substantial: 87 test/bench targets under `cpp_engine/tests/`, including
  parity and determinism tests.

## Open items

- Primary weight format per model, needed before Phase 4.
- Intra-server 8-card topology, to be measured in Phase 4.
- Whether existing CJK code comments get swept to English (separate task, keep out of this diff).
