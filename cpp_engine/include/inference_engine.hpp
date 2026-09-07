#pragma once

#include <cstdint>
#include <vector>

namespace pocket {

// What an engine can actually do, declared rather than inferred.
//
// The scheduler used to read `kv_total_blocks() == 0` as "this engine uses a
// contiguous arena", which happens to be true for the one engine that existed
// but is an inference from an accounting field, not a statement of capability.
// An engine that supports neither paging nor batching reports the same zero,
// and the scheduler would have gone on to treat it as a wide batched engine
// with an unpaged cache. Declaring the answer keeps the scheduler honest about
// engines it was not written against.
struct Capabilities {
    // KV cache is a pool of fixed-size blocks handed out as tokens arrive,
    // rather than a per-slot arena reserving max_context up front. When false,
    // the kv_*_blocks accounting carries no information and admission is bound
    // by slots alone.
    bool paged_kv = false;
    // More than one request may occupy the engine at once. When false the
    // scheduler must serialize, whatever max_slots says.
    bool continuous_batching = false;
    // batch_prefill() honours its token budget and can return a prompt that is
    // still incomplete. When false the budget is ignored and every prompt runs
    // to completion in one call, so a long prompt holds the device throughout.
    bool chunked_prefill = false;
    // Concurrent request slots. 1 means one mutable session at a time.
    int max_slots = 1;
};

// One forward pass over one sequence.
struct ForwardResult {
    int token = 0;
    int layers = 0;
    int dim = 0;
    int logits = 0;
    int top_token = 0;
    // Filled by native MTP speculative steps; plain forwards leave these zero.
    int correct_drafts = 0;
    int bonus_token = 0;
    std::vector<int> accept_tokens;
    std::vector<float> accept_logits;
    std::vector<float> accept_checksums;
    float top_logit = 0.0f;
    float checksum = 0.0f;
    int position = 0;
};

// Sampling parameters for one batched request.
struct BatchSamplingParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 20;
    unsigned long long seed = 0;
    int max_new_tokens = 128;
    // Tokens that end this request. Left empty, the engine falls back to the
    // checkpoint's eos ids from generation_config.json; set it to override for
    // this request only. `ignore_eos` runs to max_new_tokens regardless, which
    // is what benchmarks want so their token counts stay fixed.
    std::vector<int> stop_token_ids;
    bool ignore_eos = false;
};

// Per-request state for batched continuous execution.
struct BatchedRequest {
    uint64_t request_id = 0;
    std::vector<int> prompt_tokens;
    int seq_len = 0;
    int slot_id = -1;
    int cached_prefix_len = 0;
    BatchSamplingParams sampling;
    bool finished = false;
    int last_token = 0;
    std::vector<int> generated_tokens;
    ForwardResult last_result;
};

// Result from batch_prefill.
struct BatchPrefillResult {
    std::vector<ForwardResult> results;
    int total_tokens = 0;
    double seconds = 0.0;
    // Rows whose prompt is not yet fully consumed, parallel to `results`. A
    // partial row's `results` entry holds the last chunk's logits, which are
    // not a prediction for the prompt's final token and must not be sampled.
    std::vector<bool> incomplete;
};

// Outcome of one bounded prefill step over a single sequence.
struct PartialPrefillResult {
    ForwardResult result;
    // Prompt tokens consumed so far, i.e. where the next step resumes.
    int consumed_tokens = 0;
    // False while tokens remain; `result` is only a sampleable prediction for
    // the prompt once this is true.
    bool complete = false;
};

// Result from batch_decode_step.
struct BatchDecodeResult {
    std::vector<int> next_tokens;
    std::vector<bool> finished;
    // Rows that stopped on a stop token rather than on max_new_tokens, parallel
    // to `next_tokens`. Callers need the distinction to report finish_reason,
    // which cannot be recovered from `finished` alone.
    std::vector<bool> hit_stop_token;
    double seconds = 0.0;
};

// The surface a scheduler needs from a model runtime.
//
// This is deliberately the smallest set that BatchScheduler actually calls, not
// a general model API: slot lifecycle, paged-KV accounting for admission, the
// two batched forward entry points, and a capability declaration. Everything
// model-specific -- weight layout, attention kind, speculative decoding, prefix
// reuse, TP worker protocol -- stays on the concrete engine, where it can keep
// its own types.
//
// Every method here is called once per scheduler iteration, never per token and
// never per layer, so dispatching them virtually cannot show up against a
// batched forward pass. Layer-level components are concrete types for the
// opposite reason.
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;

    virtual Capabilities caps() const = 0;

    // Per-sequence context ceiling.
    virtual int max_context() const = 0;

    // Device this engine bound at construction. The current device is
    // per-thread, so a scheduler thread has to re-select it before touching
    // device memory.
    virtual int device() const = 0;

    // ---- Slot lifecycle ----

    // Prepare `max_batch_size` concurrent KV slots. Must precede any batched
    // call. An engine declaring max_slots == 1 may reject a larger request.
    virtual void allocate_batch_slots(int max_batch_size) = 0;

    // Bind a request to a slot. Returns the slot id, or -1 when none is free.
    virtual int allocate_slot(uint64_t request_id) = 0;

    // Release the slot a request holds, and any blocks with it.
    virtual void free_slot(uint64_t request_id) = 0;

    // ---- Paged KV admission ----

    // Whether the paged block pool is in use. Mirrors caps().paged_kv for
    // engines that can be configured either way; false means the block counts
    // below carry no information.
    virtual bool kv_paged() const = 0;

    // Blocks free right now, and the pool total. Both 0 when not paged.
    virtual int kv_free_blocks() const = 0;
    virtual int kv_total_blocks() const = 0;

    // Blocks a sequence of `tokens` logical positions needs in total. This is
    // the unit admission reasons in: a request's cost is set by the blocks its
    // context will span, not by the single slot it occupies.
    virtual int kv_blocks_for_tokens(int tokens) const = 0;

    // ---- Batched execution ----

    // Advance each request's prompt by at most `token_budget` tokens; 0 runs
    // every prompt to completion. A row left unfinished is reported through
    // `incomplete` and its logits must not be sampled. Engines that do not
    // declare chunked_prefill ignore the budget.
    virtual BatchPrefillResult batch_prefill(
        const std::vector<BatchedRequest*>& requests, int token_budget) = 0;

    // Advance every request one token. Requests may sit at different positions
    // but must occupy distinct slots.
    virtual BatchDecodeResult batch_decode_step(
        const std::vector<BatchedRequest*>& requests) = 0;
};

}  // namespace pocket
