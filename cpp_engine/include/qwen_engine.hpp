#pragma once

#include "inference_engine.hpp"
#include "qwen_config.hpp"
#include "qwen_dflash2.hpp"
#include "qwen_weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace pocket {

enum class QwenKvCacheDType {
    Fp16,
    Fp8,
    // TurboQuant K8V4: one combined byte slot per token and KV head holding an
    // FP8 E5M2 key, a 4-bit uniformly quantized value, and the value's FP16
    // scale and minimum. This is a lossy cache and stays opt-in; the FP16 and
    // separate-array FP8 paths remain the exact defaults.
    TurboQuantK8V4,
    // INT8 per-token-head: INT8 K/V arrays with per-token per-head FP16 scales.
    // Dynamic quantization per token and KV head. Proved faster than FP16 in
    // vLLM-2080Ti benchmarks (+16% decode at 65K context).
    Int8PerTokenHead,
};

const char* qwen_kv_cache_dtype_name(QwenKvCacheDType dtype);
QwenKvCacheDType parse_qwen_kv_cache_dtype(const std::string& value);

struct QwenEngineOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    // The FP8 projections dequantize the weight into an FP16 scratch buffer once
    // per call, a fixed cost that amortises over the chunk: measured per-call
    // dequant overhead on the real projection shapes is 29-34% at 512 rows but
    // only 9-12% at 2048. Raising the chunk from 512 recovers that on the real
    // 64-layer TP4 checkpoint at a cost of 0.42 GiB/rank, well inside the 22 GB
    // budget, with generated tokens unchanged:
    //
    //   context   chunk 512   chunk 4096   prefill
    //     8192     1125.7      1229.9      1.09x
    //    32768     1059.1      1244.3      1.17x
    //    65536      886.8      1077.7      1.21x
    //
    // The DeepSeek-V4 engine already defaults to 4096 for the same reason.
    //
    // 8192 became reachable only once the snapshot grid stopped splitting the
    // chunk (see `snapshot_interval_tokens`). Re-measured on the real 64-layer
    // TP4 model with that fix, identical generated tokens, FP16 cache:
    //
    //   context   chunk 4096   chunk 8192   chunk 16384
    //     8192      1798.7       1811.1        -
    //    65536      1457.0       1481.8       1490.2
    //
    // 16384 is within noise of 8192 at 65K but costs another 1.06 GB of
    // activation workspace per rank, so 8192 is the default.
    int prefill_chunk_tokens = 8192;
    QwenKvCacheDType kv_cache_dtype = QwenKvCacheDType::Fp16;
    // 0 preserves exact full attention. Nonzero values enable the explicit
    // sink-plus-sliding-window attention path in the optimized FP16 kernels.
    int attention_window = 0;
    int attention_sink_tokens = 0;
    // Exact prefix reuse across sequential requests. The full-attention KV
    // cache is already position-indexed, so only the recurrent DeltaNet state
    // has to be carried forward or rolled back.
    bool prefix_cache = true;
    // Recurrent-state snapshots let a diverging prompt roll back to an interior
    // position instead of recomputing from zero. 0 disables snapshots; live
    // continuation of an appended prompt still applies.
    int state_snapshot_interval_tokens = 4096;
    // 82 covers dense 256-token checkpoints through 4K, every 4K boundary
    // through the 262,144-token limit, and a few request-boundary snapshots.
    // This uses about 3.0 GiB/rank for 48-layer Qwen3.8 recurrent snapshots.
    int max_state_snapshots = 82;
    // Native Qwen MTP is opt-in until real TP4 parity/performance validation
    // proves that speculative verification is a win on the target GPU.
    bool mtp = false;
    int mtp_speculative_tokens = 1;
    // When enabled, start with one draft token, double K after full acceptance,
    // and back off after rejection. This protects low-acceptance prompts while
    // quickly reaching the configured maximum on predictable continuations.
    bool mtp_adaptive = false;
    // External Qwen DSpark drafter. Empty keeps the feature disabled. DSpark and
    // native MTP are mutually exclusive because both own the speculative target
    // transaction and hidden-state side channel.
    std::string dspark_checkpoint;
    // External Qwen DFlash2 block-diffusion drafter. This remains opt-in and
    // is mutually exclusive with native MTP and DSpark.
    std::string dflash2_checkpoint;
    std::string nccl_id_path;
    // Stochastic sampling. temperature <= 1e-5 keeps the exact greedy argmax
    // path, so existing greedy results stay reproducible by default. Sampling
    // draws its uniforms on the host and shares them across TP ranks, which is
    // what keeps every rank committing the same token.
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 20;
    unsigned long long sampling_seed = 0;
    // Maximum concurrent requests for batched execution (Phase 3.2).
    // 1 = single-session mode (default, backward compatible, zero overhead)
    // 2-8 = multi-slot mode (enables continuous batching)
    // KV cache is allocated as [max_batch_size, max_seq_len, kv_heads, head_dim]
    // at construction time. Memory scales linearly: 2-slot ≈ 2× single-session.
    // Recommended: 2 for most workloads, 4-8 for high-concurrency scenarios.
    int max_batch_size = 1;
    // Paged KV cache. The arena above reserves max_context per slot whether the
    // request holds 100 tokens or 100K, so at 128K context the reservation alone
    // is 2 GiB per slot per rank and concurrency is capped by the reservation
    // rather than by real token use. With this enabled the cache is one pool of
    // fixed-size blocks handed out as tokens arrive, so batch size scales with
    // what the requests actually hold.
    //
    // FP16 only: batched decode already routes to the FP16 kernels before the
    // dtype branches are reached, so paging FP16 covers all of batched decode
    // while the quantized caches keep their validated scale and packed-slot
    // arithmetic untouched. Construction rejects the other dtypes rather than
    // silently falling back.
    bool kv_paged = false;
    // Tokens per block. 16 follows vLLM; at kv_heads * head_dim = 1024 FP16
    // elements that is 32 KiB of contiguity per block, so reads stay coalesced
    // within a block.
    int kv_block_size = 16;
    // Pool budget in bytes for the paged K and V arenas together, across all
    // full-attention layers on this rank. 0 derives it from what
    // max_batch_size * max_context would have reserved, which makes enabling
    // paging alone memory-neutral and lets the block count be raised
    // deliberately.
    uint64_t kv_cache_bytes = 0;
};

// Accounting for one native-MTP or external-DSpark generate call.
struct QwenMtpStats {
    uint64_t verify_count = 0;
    uint64_t proposed_drafts = 0;
    uint64_t correct_drafts = 0;
    uint64_t rollback_count = 0;
    uint64_t replay_tokens = 0;
    uint64_t confidence_count = 0;
    double confidence_sum = 0.0;
    float confidence_min = 0.0f;
    float confidence_max = 0.0f;
    double prefill_seconds = 0.0;
    double draft_seconds = 0.0;
    double verify_seconds = 0.0;
    double replay_seconds = 0.0;
    // Mean committed tokens per speculative verification, including the target
    // bonus token. This matches DSpark's published spec_accept_length metric.
    double accept_length() const {
        return verify_count == 0
            ? 0.0
            : static_cast<double>(correct_drafts + verify_count) /
                  static_cast<double>(verify_count);
    }

    // Draft-token match ratio only. This excludes the target bonus and must not
    // be compared directly with bonus-inclusive spec_accept_length results.
    double accept_rate() const {
        return proposed_drafts == 0
            ? 0.0
            : static_cast<double>(correct_drafts) /
                  static_cast<double>(proposed_drafts);
    }

    double mean_confidence() const {
        return confidence_count == 0
            ? 0.0 : confidence_sum / static_cast<double>(confidence_count);
    }
};

struct QwenRuntimeTelemetry {
    QwenLinearKindCounts checkpoint_linear_kinds;
    QwenLinearKindCounts active_linear_kinds;
    std::string nvfp4_decode_path = "unused";
    std::string nvfp4_prefill_path = "unused";
    std::string fp8_channel_decode_path = "unused";
    std::string fp8_channel_prefill_path = "unused";
    std::string target_head_path = "unknown";
    uint64_t host_global_metadata_bytes = 0;
    uint64_t nvfp4_q8_workspace_peak_bytes = 0;
    // Paged KV cache geometry. 0 blocks means the contiguous arena is in use.
    int kv_paged_blocks = 0;
    int kv_paged_block_size = 0;
};

struct QwenPrefixCacheStats {
    // Tokens whose KV cache and recurrent state were reused unchanged.
    int reused_tokens = 0;
    // Tokens actually pushed through the network by the last prefill.
    int computed_tokens = 0;
    // Total prompt length of the last prefill.
    int prompt_tokens = 0;
    // Recurrent state source: "empty", "live", or "snapshot".
    std::string resume_source = "empty";
    // Longest common prefix with the previous prompt, before snapshot rounding.
    int matched_tokens = 0;
    int snapshots = 0;
    uint64_t snapshot_bytes = 0;
    int hits = 0;
    int misses = 0;
};

// Independent Qwen3.5 hybrid dense runtime. Checkpoint BF16 tensors are
// materialized as FP16 on Turing; FP8 projections retain their compressed
// codes and BF16 block scales are uploaded as FP16 for online unpack.
//
// The batched request and result types this engine speaks -- ForwardResult,
// BatchSamplingParams, BatchedRequest, BatchPrefillResult, BatchDecodeResult,
// PartialPrefillResult -- live in inference_engine.hpp, because a scheduler has
// to name them without naming a model. What stays here is what is genuinely
// Qwen3.5's: its options, weight map, prefix-cache and MTP accounting, TP
// worker protocol, and speculative-decoding entry points.
class QwenEngine : public InferenceEngine {
public:
    QwenEngine(const std::string& ckpt_dir, const QwenEngineOptions& options,
               int layer_count = 0, int max_context = 8192);
    ~QwenEngine() override;

    QwenEngine(const QwenEngine&) = delete;
    QwenEngine& operator=(const QwenEngine&) = delete;

    const QwenConfig& config() const { return config_; }
    const QwenWeightMap& weight_map() const { return weights_; }
    const QwenEngineOptions& options() const { return options_; }
    Capabilities caps() const override;
    int max_context() const override { return max_context_; }
    int device() const override { return options_.device; }
    int position() const { return position_; }
    uint64_t resident_weight_bytes() const { return resident_weight_bytes_; }
    uint64_t resident_scale_bytes() const { return resident_scale_bytes_; }
    uint64_t verify_weight_bytes() const;
    uint64_t activation_workspace_peak_bytes() const;
    uint64_t kv_cache_bytes() const;
    uint64_t kv_cache_scale_bytes() const;
    QwenRuntimeTelemetry runtime_telemetry() const;

    const QwenPrefixCacheStats& prefix_cache_stats() const {
        return prefix_stats_;
    }
    const QwenMtpStats& mtp_stats() const { return mtp_stats_; }

    void set_dflash2_debug_callback(QwenDFlash2DebugCallback callback);
    // Runs target prefill with the normal native kernels while exporting the
    // captured [row, tap, hidden] DFlash2 feature matrix through the callback.
    // This path does not load the drafter and is therefore usable in TP=1 within
    // one 22 GiB card.
    ForwardResult debug_prefill_dflash2(
        const std::vector<int>& token_ids,
        const std::vector<int>& target_layer_ids,
        QwenDFlash2DebugCallback callback);

    void reset();
    // Drops every cached prefix so the next prefill recomputes from zero.
    void clear_prefix_cache();
    void warmup_tp();
    ForwardResult prefill(const std::vector<int>& token_ids, int slot_id = 0);
    // Prefill that consumes at most `max_tokens` prompt tokens and returns,
    // leaving the rest for a later call. This is what lets a scheduler keep a
    // long prompt from monopolising the GPU: a 65K prompt run in 8192-token
    // steps yields control roughly every 4.3s at TP4 instead of holding it for
    // the full ~56s, so running decodes still make progress.
    //
    // Resumption rides the existing growing-prompt prefix path, so callers pass
    // the same full `token_ids` every time and the engine skips what this slot
    // already holds. `max_tokens <= 0` consumes the whole remainder.
    PartialPrefillResult prefill_partial(const std::vector<int>& token_ids,
                                             int slot_id, int max_tokens);
    ForwardResult decode_step(int token_id, int slot_id = 0);
    // `stop_at_eos` ends generation on one of the checkpoint's eos ids, keeping
    // that token as the last element. It defaults to false so existing callers
    // and benchmarks keep returning exactly max_new_tokens results; under TP
    // every rank runs this loop and decides independently, so the flag has to be
    // the same on all of them or they stop at different lengths.
    std::vector<ForwardResult> generate(const std::vector<int>& prompt_ids,
                                             int max_new_tokens,
                                             bool stop_at_eos = false);

    // ========== Batched API (Phase 3.1) ==========

    // Allocate KV cache slots for batched execution
    // Must be called before using batch_prefill/batch_decode_step
    // max_batch_size: maximum number of concurrent requests
    void allocate_batch_slots(int max_batch_size) override;

    // Allocate a KV cache slot for a new request
    // Returns slot_id on success, -1 if no slots available
    int allocate_slot(uint64_t request_id) override;

    // Free a KV cache slot when request completes
    void free_slot(uint64_t request_id) override;

    // ========== Paged KV admission (for the scheduler) ==========

    // Whether the paged block pool is in use. False means the contiguous arena
    // is, in which case a slot already owns max_context tokens and the block
    // accounting below carries no information.
    bool kv_paged() const override;

    // Blocks free right now, and the pool total. Under the contiguous arena both
    // are 0.
    int kv_free_blocks() const override;
    int kv_total_blocks() const override;

    // Blocks a sequence of `tokens` logical positions needs in total. This is
    // the unit admission has to reason in: a request's cost is set by the blocks
    // its context will span, not by the single slot it occupies.
    int kv_blocks_for_tokens(int tokens) const override;

    // Batch prefill: process multiple requests in their prefill phase.
    // Each request may have a different prompt length, and requests are still
    // processed one at a time rather than merged into one variable-length
    // forward -- see the note in the definition for the measurements behind that.
    //
    // `token_budget` bounds how many prompt tokens each request advances in this
    // call; 0 runs every prompt to completion as before. With a budget, a row
    // whose prompt is unfinished is reported through `result.incomplete` and its
    // logits must not be sampled. Call again with the same requests to continue.
    BatchPrefillResult batch_prefill(
        const std::vector<BatchedRequest*>& requests,
        int token_budget) override;

    // Batch decode step: process one decode step for all active requests
    // Every request advances one token in a single batched forward, so the
    // requests may sit at different positions; they must occupy distinct slots.
    // Returns next token for each request
    BatchDecodeResult batch_decode_step(
        const std::vector<BatchedRequest*>& requests) override;

    // One batched decode step over raw tokens and the slots they belong to.
    // batch_decode_step is the request-oriented wrapper around this; tests and
    // the TP worker loop use this form because they carry no request objects.
    // Does not announce anything to the TP workers: callers on rank 0 drive that
    // themselves so a batch is announced exactly once.
    std::vector<ForwardResult> batch_decode_tokens(
        const std::vector<int>& tokens, const std::vector<int>& slot_ids);

    // Check if batched API is supported (depends on build config)
    bool supports_batching() const;

    // TP rank > 0 entry point. Blocks on a small NCCL int32 broadcast channel
    // driven by rank 0; runs the requested op until SHUTDOWN.
    void run_worker_loop();

    // Rank 0 utilities to drive the worker loop. No-op for tp_world == 1.
    enum class WorkerCommand : int32_t {
        Prefill = 0,
        DecodeStep = 1,
        Reset = 2,
        Shutdown = 3,
        BatchDecodeStep = 4,
        // Releases one slot's paged blocks on every rank. Under the contiguous
        // arena a slot owns max_context implicitly, so there is nothing to
        // broadcast and this command is only sent when paging is on.
        FreeSlot = 5,
    };
    // slot_id selects the KV cache slot the workers must use, so it has to match
    // the slot rank 0 computes into.  It defaults to 0 for the single-session
    // path, where only slot 0 ever exists.
    void worker_command_prefill(const std::vector<int>& token_ids,
                                int32_t slot_id = 0, int32_t token_budget = 0);
    void worker_command_decode(int32_t last_token, int32_t slot_id = 0);
    // A batched step's slots do not fit the single slot_id header field, so the
    // tokens and their slots travel together as one interleaved payload.
    void worker_command_batch_decode(const std::vector<int>& tokens,
                                     const std::vector<int>& slot_ids);
    void worker_command_reset();
    void worker_command_shutdown();
    void worker_command_free_slot(int32_t slot_id);

private:
    struct Impl;

    // Shared body behind prefill() and prefill_partial(). `max_tokens` of 0
    // means unbounded, which is what makes the full-prompt path byte-for-byte
    // the same work it was before the bounded entry point existed.
    PartialPrefillResult prefill_bounded(const std::vector<int>& token_ids,
                                             int slot_id, int max_tokens);

    // Whether `token` ends generation under these params: the request's own
    // stop_token_ids when set, otherwise the checkpoint's eos ids.
    bool is_stop_token(const BatchSamplingParams& sampling, int token) const;

    std::string ckpt_dir_;
    QwenEngineOptions options_;
    QwenConfig config_;
    SafeTensorsIndex index_;
    QwenWeightMap weights_;
    int active_layers_ = 0;
    int max_context_ = 0;
    int position_ = 0;
    uint64_t resident_weight_bytes_ = 0;
    uint64_t resident_scale_bytes_ = 0;
    QwenPrefixCacheStats prefix_stats_;
    QwenMtpStats mtp_stats_;
    // The cached prompt/result and snapshot ring live per KV slot inside Impl,
    // so that a batch cannot match one sequence's prefix against another's.
    Impl* impl_ = nullptr;
};

}  // namespace pocket
