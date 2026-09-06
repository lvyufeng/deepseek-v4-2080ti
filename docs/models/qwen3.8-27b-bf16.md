# Qwen3.8-27B (official BF16)

## Runtime status

**Checkpoint mapping and TP audit validated on the host; CUDA generation not yet validated.** PocketLLM parses the official multimodal root config, maps all 866 text tensors of the official BF16 release, classifies the 333 bundled vision tensors as deliberately ignored, and produces rank-local shard descriptors for any TP world size. This audit runs with no vendor SDK present.

Generation reuses the existing Qwen3.5 text runtime, so the CUDA path is the same one validated for the FP8 and NVFP4 checkpoints. What is new here is the dense BF16 weight source. On-device generation from this checkpoint has not been measured yet; see [Known limitations](#known-limitations).

## Model specification

The official checkpoint (`Qwen/Qwen3.8-27B`) reports:

| Field | Value |
| --- | ---: |
| HF architecture | `Qwen3_5ForConditionalGeneration` |
| Root model type | `qwen3_5` |
| Text model type | `qwen3_5_text` |
| `language_model_only` | `false` |
| Text layers | 64 |
| Gated DeltaNet layers | 48 |
| Full GQA layers | 16 |
| Full-attention interval | 4 |
| Hidden size | 5120 |
| Dense MLP intermediate | 17,408 |
| Vocabulary | 248,320 |
| Maximum positions | 262,144 |
| Query heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| Output-gated attention | `attn_output_gate: true`, swish |
| Partial RoPE | 64 dimensions (factor 0.25) |
| RoPE theta | 1e7, `mrope_section [11, 11, 10]` |
| Linear-attention key heads | 16 × 128 |
| Linear-attention value heads | 48 × 128 |
| Convolution kernel | 4 |
| Native MTP layers | 1, shared embeddings |
| `tie_word_embeddings` | `false` |
| Checkpoint dtype | BF16, no quantization metadata |

Shard layout, verified against the published index:

| Field | Value |
| --- | ---: |
| Shards | 18 |
| Index entries | 1,199 |
| Index `total_size` | 55,562,855,904 B (51.747 GiB) |
| Text tensors | 866 (54,641,395,712 B / 50.889 GiB) |
| Vision tensors (`model.visual.*`) | 333 (921,460,192 B / 0.858 GiB) |

The vision tower ships inside the same 18 shards as the text model, so every shard is required even though PocketLLM executes text only.

## Implemented execution path

- Root/nested config parsing that reads through the multimodal root keys (`vision_config`, `image_token_id`, `language_model_only`) to the nested `text_config`.
- Dense BF16 linear mapping with no scale tensors: all 505 mapped linears per rank classify as `DenseF16`, and the FP8-block/FP8-channel/NVFP4 counts are zero.
- Explicit coverage accounting. Every index entry must be either mapped by the text map, recognized as a vision tensor, or reported as unexpected. Strict mode throws on any unexpected entry, so an unrecognized checkpoint variant fails instead of loading a partial model.
- BF16 storage to FP16 device residency. RTX 2080 Ti has no native BF16 arithmetic, so every BF16 tensor is converted at materialization. FP8 and NVFP4 checkpoints keep their existing compressed paths untouched.
- Host/device split of the loader. The mapping, coverage and host materialization live in `cpp_engine/core/qwen_weight_map.cpp` and link only `pocket_core`; device residency and uploads stay in `cpp_engine/engine/qwen_weights.cpp`.

## TP4 shard contract

Audited per rank at `--tp-world 4`:

| Quantity | Full | Per rank |
| --- | ---: | ---: |
| Vocabulary / `lm_head` rows | 248,320 | 62,080 |
| MLP intermediate rows | 17,408 | 4,352 |
| `down_proj` K | 17,408 | 4,352 |
| Query heads | 24 | 6 |
| KV heads | 4 | 1 |
| `o_proj` K | 6,144 | 1,536 |
| Linear-attention key heads | 16 | 4 |
| Linear-attention value heads | 48 | 12 |

Resident weights per rank: 12.796 GiB total, of which 12.697 GiB is sharded and 0.099 GiB is replicated. Per-rank totals deliberately do not sum to the checkpoint size: norms, `mtp.fc` and other replicated tensors exist on every rank.

Fused `in_proj_qkv` and the depthwise `conv1d` shard as three ordered segments (K, K, V) rather than one contiguous range, so each rank takes its own slice of each segment. `out_proj`, `o_proj` and `down_proj` are row-parallel and shard their input dimension.

## Correctness and precision

Verified on the real 51.7 GiB checkpoint, host-only:

- All 18 shard headers parse and their byte extents validate without reading payloads.
- `mapped=866 visual_ignored=333 unexpected=0` on every rank of a TP4 audit.
- `checkpoint_text_bytes` is identical across ranks, confirming the logical text size is TP-independent.
- Rank metadata (`embed_rows`, `head_rows`, `mlp_intermediate_rows`, `down_proj_k`, `storage_dtype=BF16`, `device_dtype=F16`, `mtp=1`) is identical on all four ranks.
- A synthetic fixture carrying an unrecognized tensor is rejected by strict coverage.
- A BF16 `1.0` payload materializes to FP16 `1.0`.

Numerical parity of BF16 generation against a reference implementation has not been measured.

## Reproduction

Audit the checkpoint with no accelerator required:

```bash
cmake -S cpp_engine -B build/cpp_engine
cmake --build build/cpp_engine --target qwen_audit -j
build/cpp_engine/tools/qwen_audit /path/to/Qwen3.8-27B --tp-world 4 --strict
```

The same audit through the engine CLI, which also stays on the host in audit mode:

```bash
build/cpp_engine/pocketllm_engine \
  --ckpt /path/to/Qwen3.8-27B \
  --tp-world 4 --tp-rank 0 --qwen-audit-strict
```

Host-only tests:

```bash
cmake --build build/cpp_engine --target check_layering test_qwen_config test_qwen_bf16_checkpoint -j
build/cpp_engine/tests/test_qwen_config
QWEN38_CKPT=/path/to/Qwen3.8-27B build/cpp_engine/tests/test_qwen_bf16_checkpoint
```

Without `QWEN38_CKPT` the test runs its synthetic fixture only and reports the real-checkpoint half as skipped.

Generation follows the FP8 page's four-rank NCCL procedure with this checkpoint path substituted; see [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md#reproduction).

## Known limitations

- No on-device validation yet. Generation, TPS, determinism across ranks, and MTP on/off behavior have not been measured for this checkpoint.
- BF16 weights are materialized as FP16 for RTX 2080 Ti. This is a precision-narrowing conversion at load time, not a lossless path, and it is specific to Turing. An accelerator with native BF16 must supply its own dtype policy rather than reusing `qwen_device_dtype`.
- Resident BF16 weights are 12.8 GiB per rank at TP4, well above the FP8 and NVFP4 checkpoints. Long-context headroom on 22 GiB cards is correspondingly smaller.
- Text-only: no image or video preprocessing, and the vision tower is never mapped or uploaded.
- CLI only: Qwen remains rejected by the DeepSeek-V4 OpenAI server path.
- The Ascend backend configures but does not link; no kernels exist yet.

## Evidence and related notes

- `cpp_engine/core/qwen_weight_map.cpp` — mapping, coverage, host materialization
- `cpp_engine/engine/qwen_weights.cpp` — device residency and uploads
- `cpp_engine/include/qwen_weights.hpp` — `QwenCoverage`, shard rules, dtype policy
- `cpp_engine/core/qwen_config.cpp` — root/nested config parsing
- `cpp_engine/tools/qwen_audit.cpp` — host-only audit tool
- `cpp_engine/tests/test_qwen_bf16_checkpoint.cpp` — fixture and real-checkpoint audit
- [Qwen3.8-27B-FP8](qwen3.8-27b-fp8.md) — the validated CUDA runtime this checkpoint reuses
