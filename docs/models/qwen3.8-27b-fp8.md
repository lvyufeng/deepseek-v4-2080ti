# Qwen3.8-27B-FP8

## Runtime status

**Validated native C++/CUDA TP4 text runtime.** PocketLLM detects the nested Qwen3.5 text configuration, maps rank-local Safetensors weights, converts BF16 scales/non-FP8 tensors to FP16 for Turing where required, and keeps local FP8 weights resident on each GPU.

The current integration supports text prompt/token-ID smoke and timed greedy generation through `pocketllm_engine`. It does not execute the checkpoint's vision tower and is not connected to the OpenAI-compatible server.

## Model specification

The validated checkpoint reports:

| Field | Value |
| --- | ---: |
| HF architecture | `Qwen3_5ForConditionalGeneration` |
| Text model type | `qwen3_5_text` |
| Text layers | 64 |
| Gated DeltaNet layers | 48 |
| Full GQA layers | 16 |
| Hidden size | 5120 |
| Dense MLP intermediate | 17,408 |
| Vocabulary | 248,320 |
| Maximum positions | 262,144 |
| Query heads | 24 |
| KV heads | 4 |
| Head dimension | 256 |
| Partial RoPE | 64 dimensions (factor 0.25) |
| Linear-attention key heads | 16 × 128 |
| Linear-attention value heads | 48 × 128 |
| Convolution kernel | 4 |
| Quantization | FP8 E4M3, dynamic activation scheme |
| Weight scale block | 128×128 |

The root config also contains a vision tower, but PocketLLM deliberately dispatches only the text model tensors.

## Implemented execution path

- Nested Qwen config detection and strict tensor/scale shape validation.
- TP4 rank-local embedding, head, attention, and dense MLP sharding.
- FP8 E4M3 weights stored as bytes with FP16 block scales on RTX 2080 Ti.
- Online FP8 unpacking in CUDA tiles/registers; no full FP16/FP32 weight expansion.
- Separate multi-row prefill and single-token decode projection kernels.
- 48-layer Gated DeltaNet sequence/recurrent kernels with persistent state and convolution tails.
- 16-layer GQA prefill and KV-cache decode with local K/V heads.
- FP16 activation storage with FP32 local accumulation/state where required; no prompt-length FP32 activation expansion.
- Chunked prefill (default 512 tokens) that retains only recurrent state, convolution tails, and full-attention KV cache between chunks.
- Exact single-request prefix reuse: the position-indexed GQA KV cache and the DeltaNet recurrent state are retained across sequential `prefill()` calls. Appended prompts execute only their uncached suffix; diverging or compressed prompts restore a device-resident recurrent snapshot at the longest safe common prefix.
- FP16 KV cache by default, plus explicit opt-in FP8 E4M3 cache with per-token/KV-head FP16 scales over 64-channel blocks.
- Decode-only fused FP8 gate/up projection plus SwiGLU.
- Opt-in exact FP16 GQA kernels: tiled prefill and split-context fused decode with compact online-softmax partials. Enable with `POCKETLLM_QWEN_GQA_OPTIMIZED=1`; the default remains the reference full-attention path.
- Opt-in FP16 sink-plus-sliding-window attention through `--qwen-attention-window N` and optional `--qwen-attention-sink-tokens N`. This changes full-attention semantics and is not part of exact parity or default performance claims; FP8 cache is intentionally rejected for this mode.
- TP4 NCCL reductions and global greedy top-1 selection.
- Opt-in native one-layer MTP loading and greedy speculative generation through `--qwen-mtp-tokens K`. The MTP layer reuses the target embedding/LM head, recursively proposes drafts, and verifies `[current_token, draft_1, ..., draft_K]` in one multi-row target forward. Partial rejection restores DeltaNet state/convolution tails and replays only the committed input prefix. MTP remains disabled by default.
- Opt-in external Qwen DSpark loading through `--qwen-dspark PATH`. The real five-layer BF16 drafter is replicated on every TP rank, consumes target post-layer taps `4,16,28,40,52`, proposes the checkpoint's fixed seven-token block, and verifies `[anchor,draft_1,...,draft_7]` in one eight-row target forward. Its position-indexed context K/V follows the target prefix cache across append, shorter-prefix, branch, and compressed-context restores.

## Validated performance

Hardware: 4×RTX 2080 Ti 22 GiB, TP4, single request, real Qwen3.8-27B-FP8 checkpoint and prompts.

| Prompt | Generated tokens | Prefill | Decode | GPU used/rank |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 24 | 138.61–138.69 tok/s | 36.82 tok/s | ~8.04–8.46 GiB |
| 512 tokens | 24 | 416.48 tok/s | 35.87 tok/s | ~8.18–8.60 GiB |

Additional repeat runs on the 512-token fixture measured approximately 411.8–416.4 tok/s prefill and 35.66–35.85 tok/s decode.

### Current memory-safe FP16-activation kernels

The current reference runtime now uses FP16-input, FP32-accumulation FP8 projection kernels without expanding prompt activations or weights. The prefill path uses a 128-token x 64-output N64 tile when alignment and batch size permit; decode uses vectorized single-row FP8 matvec, while the original scalar kernel remains the fallback. Two-row and four-row decode variants remain explicit experiments because their register pressure reduced end-to-end decode throughput. These kernels preserve the default exact full-attention semantics and FP16 KV cache.

A clean serial TP4 run on the same real checkpoint and 512-token fixture measured `453.08 tok/s` prefill and `36.95 tok/s` decode with 24 generated tokens. A prior repeat measured `456.78 / 37.12 tok/s`; both runs produced identical rank-local greedy sequences and `rank_token_parity=PASS`. The resident weight and scale bytes remained `7,367,270,656` and `742,400` per rank, and peak GPU memory was `8,497,528,832` bytes on the highest rank.

The same executable was then run serially over longer prompts with four generated tokens, complete 64-layer execution, 512-token chunks, and FP16 KV cache:

| Prompt | Prefill | Decode | Activation workspace | KV data | Highest rank memory | Rank parity |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4,096 | 386.16 tok/s | 29.82 tok/s | 61.00 MiB | 64.1 MiB | 8.02 GiB | PASS |
| 8,192 | 293.60 tok/s | 24.58 tok/s | 61.00 MiB | 128.1 MiB | 8.09 GiB | PASS |
| 32,768 | 113.92 tok/s | 11.05 tok/s | 61.00 MiB | 512.1 MiB | 8.47 GiB | PASS |
| 65,536 | 65.21 tok/s | 6.60 tok/s | 61.00 MiB | 1,024.1 MiB | 8.97 GiB | PASS |
| 131,072 | 35.19 tok/s | 3.65 tok/s | 62.50 MiB | 2,048.1 MiB | 9.97 GiB | PASS |

The direct FP16-activation FP8 projection gate covers aligned and padded strides, masked rows, tail K tiles, vectorized-versus-scalar decode dispatch, and the wide prefill tile. It reports decode max absolute error `1.459e-2` against the FP32 host reference, with vectorized-versus-scalar output difference `0`; the 4-row experimental path differs by at most `3.906e-3`. The focused FP8 online operator suite and full TP4 rank parity checks also pass.

### Native MTP speculative decoding

The checkpoint declares `mtp_num_hidden_layers=1` and ships the predictor in `mtp.safetensors`. PocketLLM maps this full-attention layer under TP4, uses the required `[normalized_embedding, normalized_hidden]` fusion order, consumes the target's final-normalized hidden state, and uses the MTP layer's normalized output as the recursive hidden. Target verification computes all candidate-row logits together and performs one batched TP global top-1 collective.

Enable it explicitly with:

```bash
--qwen-mtp-tokens 4
```

`--qwen-mtp` is equivalent to enabling the default `K=1`. MTP requires all 64 target layers; combining it with a nonzero partial `--smoke-layers` value is rejected. It is also context-safe: the prompt plus requested output count must fit in `max_context`; each speculative block is capped by the remaining output count, so its temporary verify suffix stays within that bound. The runtime reports `mtp_accept_rate`, proposed/correct draft counts, rollback/replay counts, and separate prefill/draft/verify/replay seconds.

This MTP implementation follows the standard Qwen3.5 shifted-hidden predictor path used by vLLM: prompt target rows are paired with their next-token inputs to prime an independent absolute-position MTP KV cache, after which the predictor advances recursively. SGLang's newer `frozen_kv_mtp` worker is a separate Gemma-oriented optimization that requires a model-specific mapping from assistant layers to target KV-owner layers; Qwen3.5 does not expose such a mapping, so target and MTP KV are not aliased.

The optimized target verifier reuses each FP8 weight row across all 2–8 candidate rows, fuses small-batch gate/up/SwiGLU, reuses transaction buffers, and batches exact GQA while retaining the reference score/softmax/value reduction order. An even faster split online-softmax verifier is available through `QWEN_GQA_VERIFY_SPLIT=1`, but stays experimental: its small numerical drift can change greedy output at near-tie logits even though direct attention error is below `8e-6`.

Real TP4, full 64-layer, 64-token generation results below cover several tokenizer-real 512-token prompts. Every TP rank agreed and each MTP sequence matched its serial plain run. The final adaptive policy starts at `K=1`, doubles K after full acceptance, and backs off after rejection:

| Prompt | Acceptance | Plain TPS | Adaptive K<=4 TPS | Decode speedup | Wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repeated natural language | 100.0% | 37.19 | 79.39 | **2.135x** | **1.331x** |
| Config/model text | 94.0% | 36.75 | 69.61 | **1.894x** | **1.273x** |
| Source code | 70.0% | 36.91 | 50.66 | 1.372x | 1.106x |
| README prose | 70.9% | 36.83 | 48.10 | 1.306x | 1.079x |
| Model documentation | 64.8% | 36.93 | 43.53 | 1.179x | 1.026x |

The optimized path therefore clears 1.5x only when draft acceptance is high (94% or better in the measured cases); it cannot honestly guarantee 1.5x for arbitrary prompts. Starting adaptive mode at K=1 protects mixed prompts better than starting at K=4, but 65–71% acceptance still yields only 1.18–1.37x decode speedup. A target top1-top2 logit-margin gate was also tested and rejected: thresholds 0.5 and 1.0 reduced performance on the difficult prompts, so no margin-gating code or CLI option is retained. The one-shot wall result includes independent-MTP prompt priming; persistent single-concurrency requests with exact prefix reuse remain the intended workload.

The parity-safe exact-GQA verifier now uses a warp-tiled value pass: one warp owns one candidate row, three query heads, and 32 value channels while preserving each output element's left-to-right FP32 accumulation order. Fresh 100%-acceptance measurements show the long-context gain increasing as plain decode becomes attention-bound:

| Prompt | Plain TPS | MTP K=4 TPS | Decode speedup | One-shot wall speedup | Parity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4,096 | 30.10 | 50.17 | **1.667x** | not reported | PASS |
| 8,192 | 24.78 | 61.29 | **2.474x** | 0.961x | PASS |
| 32,768 | 11.30 | 36.26 | **3.208x** | 0.929x | PASS |

The 8K/32K one-shot wall numbers remain below 1x because each separately launched MTP process primes an additional full-prompt predictor KV cache; this is not repeated for an exact-prefix hit in the long-lived workload. An opt-in split-GQA experiment measured 2.38–4.57x at 512/4K/8K/32K, but it is not included in the parity-safe claim because a separate near-tie prompt exposed sequence drift.

Reproduce serial plain/K=1/K=2/K=4 A/B cases with real tokenizer IDs and automatic TP-rank/plain parity checks:

```bash
python scripts/bench_qwen_mtp.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,32768 \
  --mtp-tokens 1,2,4 \
  --max-new-tokens 32 \
  --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

For the recommended adaptive policy, pass one maximum K and `--adaptive`:

```bash
python scripts/bench_qwen_mtp.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,8192,32768 \
  --mtp-tokens 4 --adaptive \
  --max-new-tokens 64 --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

### External DSpark speculative decoding

The external checkpoint `RadixArk/Qwen3.8-27B-DSpark` (`epoch_2_step_4166`) is supported as an explicit opt-in:

```bash
--qwen-dspark /path/to/Qwen3.8-27B-DSpark
```

The directory must contain its `config.json` and single `model.safetensors`. The implementation validates all 62 expected BF16 tensors, materializes them as FP16 on SM75, and adds `2,623,214,594` resident weight bytes per rank. The five-layer draft backbone is replicated rather than tensor-parallel; only target embedding/head operations and the vocabulary-sharded Markov `w2` use TP collectives. Native MTP and external DSpark are mutually exclusive, and DSpark requires all 64 target layers.

The checkpoint fixes `block_size=7`, so each transaction proposes seven drafts and target-verifies eight rows: `[anchor,draft_1,...,draft_7]`. The target post-layer taps are `4,16,28,40,52`. Partial rejection restores DeltaNet recurrent state and convolution tails, crops logical target/DSpark K/V to the committed position, then replays only `[anchor,accepted drafts]`. A remaining output tail shorter than eight tokens uses ordinary exact decode rather than changing the checkpoint's block semantics.

The confidence head is evaluated for every draft and reported as `dspark_confidence_count/mean/min/max`, but it does not currently change the static width-8 schedule: neither the checkpoint nor its published static deployment supplies a validated confidence threshold. Speculative accounting retains the existing `mtp_*` field names for CLI compatibility (`mtp_accept_rate`, proposed/correct drafts, rollback/replay, and stage seconds).

Real TP4, full-model, FP16-target-KV A/B results below use the same tokenizer-real deterministic language fixture, 64 generated tokens, plain then DSpark serial execution, and exact DSpark-versus-plain plus all-rank token checks. The table's `draft match rate` is `correct_drafts / proposed_drafts` and excludes the target bonus token; it is not the model-card `spec_accept_length`, whose numerator includes one bonus token per verification step.

| Prompt | Draft match rate | Plain TPS | DSpark TPS | Decode speedup | Plain wall | DSpark wall | Wall speedup | Highest DSpark rank memory | Parity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 94.64% | 37.20 | 60.02 | **1.614x** | 2.944 s | 2.385 s | **1.235x** | 11.19 GB | PASS |
| 8,192 | 96.43% | 24.80 | 43.79 | **1.766x** | 31.505 s | 31.551 s | 0.999x | 11.61 GB | PASS |
| 32,768 | 96.43% | 11.31 | 21.70 | **1.919x** | 295.812 s | 295.773 s | 1.000x | 12.52 GB | PASS |

Long-context one-shot wall time is prefill-bound. DSpark's target-feature projector and five layers of context K/V projection make prefill slightly slower even though decode gets progressively faster. This is why the intended deployment remains a long-lived, single-request prefix-reusing engine rather than repeatedly paying cold prefill.

Acceptance is workload-sensitive. The earlier 17% figure came from a deliberately bare 16-token prompt, not the DSpark model-card benchmark protocol: it omitted the Qwen chat template, used greedy decoding, and counted only draft matches. Re-running the same wording with the real chat template still produced a difficult greedy case (15.34% draft match without thinking, 23.81% with thinking), so it is a valid stress case but not evidence that the published DSpark acceptance is 17%. The model card instead reports `spec_accept_length` including the bonus token, with a request-weighted mean of 3.39 accepted tokens per verification step over 1,164 sampled requests (macro-average 3.35) at temperature 0.6, top-k 20, top-p 0.95, thinking enabled, and 2,048 generated tokens. Our native path is currently greedy and should not be compared to those stochastic benchmark numbers as if they were the same metric. DSpark remains default-off pending benchmark-protocol-matched measurements.

For reference, the bare greedy stress prompt measured 31 correct drafts out of 182 proposals (`31/182 = 17.03%`), 17.42 tok/s DSpark decode, and exact plain-token parity. The chat-template, thinking-enabled rerun measured 35/147 (`23.81%`).

Reproduce the protocol-sensitive chat-template fixture with `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` before passing token IDs to the C++ runtime; do not use the raw user sentence as the benchmark prompt.

DSpark therefore remains default-off; enable it only after measuring representative traffic with the intended chat template, sampling policy, and generation length.

Reproduce the short/long serial A/B with:

```bash
/path/to/deepseek/bin/python scripts/bench_qwen_dspark.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --dspark /path/to/Qwen3.8-27B-DSpark \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,8192,32768 \
  --max-new-tokens 64
```

The focused prefix/cold-parity suite covers exact repeat, monotonic append, shorter prefix, interior branch, and compressed context under both target KV dtypes. With the default early 256-token snapshot spacing, a shorter/branched prompt may reuse the deepest safe 256-token boundary and recompute the remainder rather than claiming the entire matched prefix:

```bash
for dtype in fp16 fp8; do
  /path/to/deepseek/bin/python scripts/bench_qwen_dspark_prefix_cache.py \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --dspark /path/to/Qwen3.8-27B-DSpark \
    --kv-cache-dtype "$dtype" \
    --max-context 1024 --max-new-tokens 16
done
```

### External DFlash2 speculative decoding

The external draft checkpoint is supported as an explicit opt-in:

```bash
--qwen-dflash2 /path/to/Qwen3.8-27B-DFlash2
```

The directory must contain its `config.json` and single `model.safetensors`. The runtime validates all 81 expected tensors and adds `1,450,191,360` sharded plus `3,848,808,960` replicated device bytes per rank. `--qwen-dspark` and `--qwen-dflash2` are mutually exclusive, and neither can be combined with native MTP.

The checkpoint fixes `block_size=8` with `target_layer_ids = [5,19,33,47,61]`, a five-layer sliding-attention backbone (`sliding_window=2048`), `selector_rank=256`, and `selector_top_k=16`. Each transaction drafts up to seven tokens and verifies eight target rows. Partial rejection restores DeltaNet recurrent state and convolution tails, crops logical K/V to the committed position, and replays only the accepted prefix.

Unlike DSpark, DFlash2's residual and MLP `down` outputs exceed the FP16 range: a pure FP16 residual overflows in the first layer and produces NaN. The working SM75 mixed precision keeps the residual, `down` output, and finish convolution in FP32 while `gate`/`up`/SwiGLU stay in cuBLAS FP16, converting back to FP16 after each layer norm for attention and dynamic projection. All DFlash2 RMSNorms are standard direct-gamma, not the target's `(1 + gamma)`.

Real TP4, full-model, FP16-target-KV A/B results with serial plain-then-DFlash2 execution and exact cross-mode plus all-rank token checks:

| Fixture | Plain wall | DFlash2 wall | Wall speedup | Decode speedup | Accept length | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Synthetic 512, 512 new | 14.510 s | 5.228 s | **2.776x** | **3.017x** | 8 / 8 | PASS |
| Synthetic 4,096, 512 new | 21.019 s | 9.307 s | 2.259x | 3.104x | 8 / 8 | PASS |
| Synthetic 8,192, 512 new | 30.539 s | 15.677 s | 1.951x | 3.400x | 8 / 8 | PASS |
| GSM8K, 8 prompts, 256 new | 57.595 s | 43.356 s | **1.328x** | 1.300x | 2.87 – 4.40 | PASS 8/8 |

Decode-phase speedup lands inside upstream's published 2.67–3.43x band on all three synthetic lengths. Upstream defines speedup as a per-token decoding latency ratio, so a full-request wall ratio is not the same metric: prefill is shared identically by both modes and caps the 8,192 case at 1.95x regardless of drafter quality.

GSM8K acceptance is far lower (0.55–0.71 per-draft rate) and per-prompt wall speedup tracks it directly, from 1.13x to 1.57x. Verify cost is roughly linear in block width because the 48 gated-delta layers recur sequentially over rows, so a full-width block pays for and discards the rejected tail. `POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1` tracks an EWMA of accepted count and verifies `ewma + 1.5` rows with a floor of two, which keeps the fully-accepted synthetic cases at width 8 while lifting the low-acceptance GSM8K case. A fixed width cannot serve both: at width 7 GSM8K regresses to 0.86x, and at width 2 the synthetic gains are discarded.

Four flags are opt-in and all four were enabled for the results above:

| Flag | Effect |
| --- | --- |
| `POCKETLLM_DFLASH2_CUBLAS_FP32=1` | Routes the target LM head through cuBLAS FP32; the head is 55% of draft cost and this makes it ~10x faster |
| `POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1` | EWMA-driven verify width, as above |
| `POCKETLLM_DFLASH2_SPLIT_TOPK=1` | Partitions each row's local top-16 shard and merges with the identical comparator |
| `POCKETLLM_QWEN_GQA_OPTIMIZED=1` | Selects the tiled GQA prefill kernel; long prefill is 64% full-attention, not FP8 GEMM |

`POCKETLLM_DFLASH2_VITERBI_SELECTOR=1` computes an exact MAP over the selector chain instead of greedy argmax. It scores higher but accepts worse (3.02 to 2.95 accept length, 0.555 to 0.279 rate), so draft search is not an acceptance lever.

Reproduce the synthetic serial A/B with:

```bash
POCKETLLM_DFLASH2_CUBLAS_FP32=1 POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1 \
POCKETLLM_DFLASH2_SPLIT_TOPK=1 POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
/path/to/deepseek/bin/python scripts/bench_qwen_dflash2.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --dflash2 /path/to/Qwen3.8-27B-DFlash2 \
  --lengths 512,4096,8192 \
  --max-new-tokens 512 --prefill-chunk-tokens 4096 --snapshot-interval 0
```

And the dataset-shaped single-request workload, which matches upstream's aggregate-completion protocol rather than a fixed-token microbenchmark:

```bash
POCKETLLM_DFLASH2_CUBLAS_FP32=1 POCKETLLM_DFLASH2_ADAPTIVE_WIDTH=1 \
POCKETLLM_DFLASH2_SPLIT_TOPK=1 POCKETLLM_QWEN_GQA_OPTIMIZED=1 \
/path/to/deepseek/bin/python scripts/bench_qwen_dflash2_upstream.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --dflash2 /path/to/Qwen3.8-27B-DFlash2 \
  --dataset gsm8k --num-prompts 8 --max-new-tokens 256
```

### Exact prefix reuse

`QwenEngineOptions::prefix_cache` is enabled by default for a long-lived engine. The full-attention KV cache is already indexed by absolute position, while the 48 DeltaNet layers carry a small recurrent state (`state` plus convolution tail). The engine retains both across sequential prefill requests and reports `prefix_reused_tokens`, `prefix_computed_tokens`, `prefix_matched_tokens`, and `prefix_resume_source` in the persistent stdin result.

The runtime stores device-resident recurrent snapshots at dense 256-token boundaries through the first 4K, then at the configured 4K interval through the 262K limit. This keeps a compressed or diverging request exact while limiting recomputation to the suffix after the selected snapshot. The default 82 snapshots use about 3.0 GiB per rank on the full 48-layer model; this is additional KV/state working memory and is included in the reported GPU memory. `--qwen-no-prefix-cache`, `--qwen-snapshot-interval N`, and `--qwen-max-snapshots N` provide explicit A/B controls.

A long-lived TP4 stdin worker can be started with `--qwen-persistent-stdin --max-context N`; rank 0 reads lines of the form `<max_new_tokens> token0 token1 ...`, and rank 1..3 receive the same requests over the Qwen command socket. This mode is intended for single-concurrency clients and preserves the cache between lines. Each request returns exact greedy tokens and prefix accounting. The ordinary one-shot CLI creates a fresh engine, so it cannot reuse a cache across processes and therefore keeps snapshots disabled; the reported one-shot long-context TPS and memory are unaffected by this feature. `--qwen-no-prefix-cache` fully disables snapshots, prompt history, and cached results even in persistent mode.

A real TP4 continuous-request test on the Qwen3.8-27B-FP8 checkpoint produced:

| Request | Prompt | Reused | Computed | Resume | Request prefill TPS |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 512 | 0 | 512 | empty | 409.9 |
| 2 | 1,028 | 515 | 513 | live | 642.8 |
| 3 | 1,544 | 1,031 | 513 | live | 908.9 |
| 4 | 768 (256 common prefix + compressed suffix) | 256 | 512 | snapshot | 627.5 |

The cache-on and cold A/B runs generated identical tokens on all four TP ranks. In the cold run, requests 2/3/4 computed 1,028/1,544/768 tokens respectively, and every cold request reported `prefix_snapshots=0` with `prefix_snapshot_bytes=0`. Prefix reuse is exact: it does not claim that newly compressed content is cached; only the unchanged token prefix is reused.

Native MTP was also exercised through this same long-lived protocol with adaptive `K<=4`, 64 generated tokens, two appends, and an interior compression branch. A separate serial plain run used the identical four requests. All generated sequences matched, every request had TP-rank parity, and both modes reported identical prefix accounting:

| Request | Prompt | Reused | Computed | Resume | MTP acceptance | Plain decode TPS | MTP decode TPS | Decode speedup | Plain wall | MTP wall | Wall speedup |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0 | 512 | empty | 100.0% | 37.13 | 79.67 | **2.146x** | 2.959 s | 2.219 s | **1.334x** |
| 2 | 1,088 | 575 | 513 | live | 77.8% | 35.75 | 50.42 | 1.410x | 3.426 s | 3.120 s | 1.098x |
| 3 | 1,664 | 1,151 | 513 | live | 70.2% | 34.55 | 48.65 | 1.408x | 3.484 s | 3.169 s | 1.099x |
| 4 | 768 (256 common prefix + compressed suffix) | 256 | 512 | snapshot | 63.2% | 36.30 | 42.45 | 1.169x | 2.955 s | 2.897 s | 1.020x |

This validates the intended single-concurrency cache behavior and shows a wall win on all four requests, but it also confirms that the 1.5x requirement remains acceptance-dependent: only the 100%-acceptance request clears 1.5x decode and wall throughput. Appends reuse the target recurrent/KV state and the MTP shifted boundary; compression restores a target-hidden snapshot and rewrites the MTP boundary before priming the new suffix. The persistent harness reports wall, prefill, and decode timing separately; prefill includes MTP predictor priming when enabled.

The same harness was then run from a 32,768-token cold prompt with 512-token appends and a 4,096-token compression boundary:

| Request | Prompt | Reused | Computed | Resume | Request prefill TPS | Snapshot bytes/rank |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 32,768 | 0 | 32,768 | empty | 113.7 | 885,178,368 |
| 2 | 33,284 | 32,771 | 513 | live | 4,083.9 | 923,664,384 |
| 3 | 33,800 | 33,287 | 513 | live | 4,105.8 | 962,150,400 |
| 4 | 4,608 (4,096 common prefix + compressed suffix) | 4,096 | 512 | snapshot | 2,361.3 | 654,262,272 |

Request 1 matches the cold 32K prefill baseline. The two appends each execute only the 513 uncached tokens, and the compressed request recomputes only its 512-token suffix after restoring the 4,096-token snapshot. Snapshot memory shrinks when a shorter prompt invalidates later rollback points.

### Long-context TP4 baseline

The following recent serial runs use the real checkpoint, deterministic natural-language tokenizer IDs, four generated tokens, complete 64-layer execution, 512-token prefill chunks, and greedy-token parity across all four ranks. Decode TPS excludes the first generated token, which is produced by prefill. The activation workspace is the peak capacity of the reusable chunk workspace, not a prompt-length buffer.

| Cache | Prompt | Prefill | Decode | Activation workspace | KV data / scales | Highest rank memory | Rank parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 32,768 | 99.32 tok/s | 10.39 tok/s | 63.97 MB | 512.0 / 0 MB | 8.47 GiB | PASS |
| FP8 | 32,768 | 81.58 tok/s | 4.44 tok/s | 63.97 MB | 256.0 / 8.00 MB | 8.22 GiB | PASS |
| FP16 | 65,536 | 60.42 tok/s | 6.48 tok/s | 63.97 MB | 1,024.0 / 0 MB | 8.97 GiB | PASS |
| FP8 | 65,536 | 47.91 tok/s | 2.41 tok/s | 63.97 MB | 512.0 / 16.00 MB | 8.48 GiB | PASS |
| FP16 | 131,072 | 33.80 tok/s | 3.58 tok/s | 62.50 MB | 2,048.0 / 0 MB | 9.97 GiB | PASS |
| FP8 | 131,072 | 26.03 tok/s | 1.25 tok/s | 62.50 MB | 1,024.0 / 32.00 MB | 9.01 GiB | PASS |
| FP16 | 262,140 | 17.88 tok/s | 1.92 tok/s | 65.50 MB | 4,096.0 / 0 MB | 11.91 GiB | PASS |
| FP8 | 262,140 | 13.29 tok/s | 0.64 tok/s | 65.50 MB | 2,048.0 / 64.0 MB | 9.97 GiB | PASS |

The 32K, 64K, and 128K FP16/FP8 runs produced identical four-token sequences for each cache-dtype pair. The FP16 and FP8 262,140-token boundary runs both generated `[321, 5979, 13914, 13]` with `max_context=262144`, completed without OOM, and preserved TP-rank token parity. FP8 cache is retained as an explicit memory-saving option, not the default: on this RTX 2080 Ti setup, online cache dequantization materially reduces prefill and decode throughput. At the 262K boundary it halves KV data from 4,096 MiB to 2,048 MiB and reduces the highest observed rank memory from 11.91 GiB to 9.97 GiB, while prefill falls from 17.88 to 13.29 tok/s and decode from 1.92 to 0.64 tok/s. FP16 KV cache remains the precision/performance baseline.

These measurements establish that chunked prefill removes the previous prompt-length FP32 activation allocation and that a 262,140-token prompt plus four generated positions completes within the 22 GiB/rank budget with either FP16 or FP8 KV cache. The FP8 boundary run took approximately 19,729.6 seconds wall time with the complete 64-layer runtime.

### Decode Context Parallelism feasibility on four GPUs

**Rejected for the current four-GPU topology.** DCP can directly shard only the 16 full-GQA layers. The 48 Gated DeltaNet layers retain complete recurrent state on every replica and do not benefit from ordinary KV position sharding.

Keeping four GPUs constrains the proposed topology to TP2xDCP2. For a context length `C`, the per-GPU full-attention decode work is unchanged: TP4 performs `6` local Q heads over `C` positions, while TP2xDCP2 performs `12` local Q heads over `C/2` positions. Both equal `6C` head-position evaluations per device; the DCP topology then adds two DCP all-reduces per full-attention layer and doubles the local TP2 weights.

This was tested with a deliberately favorable upper bound: plain TP2 at half the TP4 context length, without the two DCP collectives or cache compaction. It was already slower and used substantially more memory:

| TP4 context / result | TP2 half-context, no DCP communication | Result |
| --- | --- | --- |
| 512 / 35.84 decode tok/s | 256 / 22.56 decode tok/s | 37% slower upper bound |
| 4,096 / 29.43 decode tok/s | 2,048 / 20.87 decode tok/s | 29% slower upper bound |
| 8,192 / 24.24 decode tok/s | 4,096 / 19.07 decode tok/s | 21% slower upper bound |
| 32,768 / 11.57 decode tok/s | 16,384 / OOM before prefill | infeasible |

TP2 local resident weights measured 13.72 GiB per rank, versus 6.86 GiB under TP4. Therefore an actual TP2xDCP2 implementation would be slower than these already-negative upper bounds and would introduce numerical/communicator complexity without reducing per-device attention work. The default TP4 path remains unchanged; no DCP code is enabled.

A useful context-parallel experiment requires at least eight ranks/GPUs for TP4xDCP2, which preserves the TP4 weight shard while halving per-device full-attention context. Even there, it would apply only to the 16 full-GQA layers and must beat the added two DCP collectives per such layer.

Per rank, the engine reported:

```text
resident_weight_bytes=7367270656
resident_scale_bytes=742400
gpu_memory_total_bytes=23068868608
```

The stable pre-optimization decode baseline was approximately 22.4 tok/s. Rank-local full-attention K/V projection, grouped GQA value aggregation, and fused decode SwiGLU raised the measured result into the 35–37 tok/s range while preserving the separate prefill path.

## Correctness and precision

- All four TP ranks generated identical token sequences on both the 64-token and 512-token fixtures.
- GQA decode matched the CPU reference with worst absolute error `1.192e-7`.
- Fused FP8 SwiGLU matched the separate projection path with `max_abs=0` and `max_rel=0` in its test fixture.
- Qwen RMSNorm, gated RMSNorm, L2 normalization, online FP8 matvec/matmul, DeltaNet, convolution tail, GQA, and TP weight-sharding tests pass.
- Parallel GQA softmax changes reduction association. CPU-reference error remains near `1e-7`, and real generated token sequences were unchanged in the validated runs.
- The existing DeepSeek FP8 matvec/matmul and minimum-layer smoke tests were also run to protect the older path.

## Reproduction

Build the C++ engine, then start four ranks with one shared NCCL ID file:

```bash
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "Explain tensor parallelism in one paragraph." \
    --generate-token 123 --max-new-tokens 24 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

The numeric value passed to `--generate-token` is ignored once `--prompt` supplies the prompt IDs; it currently activates the generation mode in the compatibility CLI parser. Rank 0 prints the timed result and all ranks print their local runtime/accounting lines. The CLI defaults to one smoke layer; use `--smoke-layers 0` for a complete 64-layer performance claim.

For reproducible serial long-context TP4 measurements:

```bash
python scripts/bench_qwen_long_context.py \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --devices 0,1,2,3 \
  --lengths 512,4096,8192,32768,65536 \
  --max-new-tokens 4 \
  --prefill-chunk-tokens 512 \
  --kv-cache-dtype fp16 \
  --layers 0 \
  --tokenizer-python /path/to/deepseek/bin/python
```

The harness persists one log per rank, records rank-local timing and memory fields, checks greedy-token parity across TP ranks, and writes `results.json` after every successful context length. FP16-versus-FP8 cache parity is a separate comparison of the generated sequences from two serial runs.

For the exact optimized FP16 GQA path, set `POCKETLLM_QWEN_GQA_OPTIMIZED=1` around the engine command or benchmark process. It keeps full attention and uses a tiled prefill kernel. The engine uses compact split-context fused decode partials from context 16,384 onward on SM75; shorter contexts retain the reference score/value decode path because it is faster there. A clean TP4 run with 24 generated tokens measured the following opt-in results, with token parity at every length:

| Prompt | Reference prefill / decode | Optimized prefill / decode |
| ---: | ---: | ---: |
| 512 | 295.46 / 31.06 tok/s | 294.84 / 30.98 tok/s |
| 4,096 | 259.69 / 25.96 tok/s | 282.47 / 25.72 tok/s |
| 8,192 | 211.02 / 21.19 tok/s | 253.70 / 21.01 tok/s |
| 16,384 | 154.58 / 15.58 tok/s | 208.24 / 17.57 tok/s |
| 32,768 | 97.75 / 10.66 tok/s | 159.52 / 17.36 tok/s |

The 4,096 and 8,192 optimized rows use the tiled prefill but reference decode dispatch; the 16,384 row is the fused-decode crossover validation, and the 32,768 row shows the long-context gain. The direct CUDA gate covers causal offsets through 333 tokens, head dimensions 64/256, contexts 4,096/8,192/32,768, and a 262,144-token compact-partial boundary check. The optimized path preserves the default token sequence in the clean TP4 runs.

Sparse experiments require an explicit `--qwen-attention-window N` and may add `--qwen-attention-sink-tokens N`; `N=0` is exact full attention. The sparse kernel attends to the leading sink prefix plus the newest window positions without changing KV-cache storage. This is an experimental semantic change, not an exact full-attention optimization claim. Window values that cover the complete context are directly checked against exact output; long-context quality and throughput are not reported here until measured on clean GPUs.

Audit only the rank-local weight mapping:

```bash
build/cpp_engine/pocketllm_engine \
  --ckpt /path/to/Qwen3.8-27B-FP8 \
  --tp-world 4 --tp-rank 0 \
  --qwen-audit
```

## Known limitations

- Text-only: no image/video preprocessing or vision-tower execution.
- CLI/smoke integration only: Qwen is explicitly rejected by the current DeepSeek-V4 OpenAI server path.
- Greedy generation only in the current Qwen engine API.
- The model limit is 262,144 positions; with four generated tokens, the longest valid benchmark prompt is 262,140 tokens. This boundary is validated with both the default FP16 KV cache and the explicit FP8 cache mode; FP8 uses less memory but is slower on this RTX 2080 Ti setup.
- CUDA Graph and a decode megakernel remain future work; neither is included in the reported TPS.
- Native MTP is opt-in. Parity-safe high-acceptance cases accelerate decode by 1.67x at 4K, 2.47x at 8K, and 3.21x at 32K; 65–71% acceptance gives only 1.18–1.37x on 512-token varied prompts. Persistent exact-prefix workloads are the intended use case; `--qwen-mtp-adaptive` starts at K=1 and limits but does not eliminate low-acceptance overhead.
- External Qwen DSpark is opt-in and always uses its fixed seven-draft/eight-row transaction. It accelerates high-draft-match decode by 1.61–1.92x in measured 512/8K/32K cases, while a bare greedy stress prompt achieved only 31/182 draft matches and regressed to about 17.4 tok/s. This draft-match ratio excludes bonus tokens and is not comparable to the model card's bonus-inclusive `spec_accept_length=3.39` sampled-workload mean. Confidence is telemetry only; no unvalidated threshold is used to gate transactions.
- Split exact-GQA verification is experimental and enabled only with `QWEN_GQA_VERIFY_SPLIT=1`; direct numerical checks pass, but near-tie greedy output can drift. General long-prefill/decode optimized GQA and sparse attention remain separate opt-in paths; sparse attention changes model semantics.

## Evidence and related notes

- `cpp_engine/include/qwen_config.hpp`
- `cpp_engine/core/qwen_config.cpp`
- `cpp_engine/engine/qwen_weights.cpp`
- `cpp_engine/engine/qwen_engine.cpp`
- `cpp_engine/backends/cuda/kernels/qwen_fp8_ops.cu`
- `cpp_engine/backends/cuda/kernels/qwen_half_ops.cu`
- `cpp_engine/backends/cuda/kernels/qwen_attention_ops.cu`
- `cpp_engine/tests/test_qwen_config.cpp`
- `cpp_engine/tests/test_qwen_fp8_online.cpp`
- `cpp_engine/tests/test_qwen_gqa_attention.cpp`
- `cpp_engine/tests/test_qwen_half_ops.cpp`
- `cpp_engine/tests/test_qwen_weights.cpp`
- `cpp_engine/tests/test_qwen_engine.cpp`
- `cpp_engine/include/qwen_dspark.hpp`
- `cpp_engine/engine/qwen_dspark.cpp`
- `cpp_engine/backends/cuda/kernels/qwen_dspark_ops.cu`
- `cpp_engine/tests/test_qwen_dspark.cpp`
- `cpp_engine/tests/test_qwen_dspark_ops.cpp`
- `scripts/bench_qwen_mtp.py`
- `scripts/bench_qwen_dspark.py`
- `scripts/bench_qwen_dspark_prefix_cache.py`
- [Qwen3.8-27B-NVFP4](qwen3.8-27b-nvfp4.md) for the mixed NVFP4/FP8 checkpoint on the same text runtime
- [Benchmark reporting rules](../benchmarking.md)
