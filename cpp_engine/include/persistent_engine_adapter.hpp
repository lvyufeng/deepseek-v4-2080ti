#pragma once

#include "inference_engine.hpp"
#include "persistent_engine.hpp"

#include <memory>
#include <string>
#include <vector>

namespace pocket {

// Presents the DeepSeek-V4 PersistentEngine through the InferenceEngine surface.
//
// PersistentEngine is a single mutable session: one KV cache, one position
// counter the caller maintains (see persistent_engine.hpp), no slots, no block
// pool, no bounded prefill. None of that is a defect to be papered over here --
// porting it onto paged KV and slots is a separate effort -- so this adapter does
// not pretend. It declares `paged_kv = false`, `continuous_batching = false`,
// `chunked_prefill = false`, `max_slots = 1`, and the scheduler reads that
// declaration instead of inferring capability from an accounting field that
// happens to read zero.
//
// What the adapter does supply is the position bookkeeping the batched API hides
// and PersistentEngine requires: `decode_step` takes the absolute position of the
// token it is fed, which every existing caller has had to track by hand.
//
// TP is handled the way QwenEngine's batched entry points handle it: each forward
// announces itself on the worker command channel first, so a scheduler driving
// this adapter is TP-safe without knowing the protocol exists. Rank 0 drives;
// worker ranks stay in PersistentEngine::run_worker_loop(), which the owner
// reaches through engine().
class PersistentEngineAdapter : public InferenceEngine {
public:
    // Owning form: constructs the PersistentEngine. This is what the model
    // registry's factory uses.
    PersistentEngineAdapter(const std::string& ckpt_dir,
                            const ForwardSmokeOptions& opts,
                            int layer_count,
                            int max_context);

    // Borrowing form, for a caller that already owns a PersistentEngine and
    // still needs its non-batched entry points (speculative decoding, the TP
    // worker loop). The engine must outlive the adapter.
    explicit PersistentEngineAdapter(PersistentEngine& engine);

    ~PersistentEngineAdapter() override;

    PersistentEngineAdapter(const PersistentEngineAdapter&) = delete;
    PersistentEngineAdapter& operator=(const PersistentEngineAdapter&) = delete;

    // The wrapped engine, for the model-specific surface the interface does not
    // carry: warmup_tp(), run_worker_loop(), worker_command_shutdown(),
    // speculative decoding, the tokenizer.
    PersistentEngine& engine() { return *engine_; }
    const PersistentEngine& engine() const { return *engine_; }

    Capabilities caps() const override;
    int max_context() const override;
    int device() const override;

    // Rejects anything above one slot rather than accepting it and overwriting
    // one request's KV cache with another's.
    void allocate_batch_slots(int max_batch_size) override;
    int allocate_slot(uint64_t request_id) override;
    void free_slot(uint64_t request_id) override;

    // Not paged; all three report 0, which caps().paged_kv == false is what makes
    // meaningful.
    bool kv_paged() const override;
    int kv_free_blocks() const override;
    int kv_total_blocks() const override;
    int kv_blocks_for_tokens(int tokens) const override;

    // Single-request degradations. Both reject a batch wider than one row: the
    // adapter declared max_slots == 1, so a wider call is a scheduler that
    // ignored the declaration, and failing loudly beats silently interleaving two
    // sequences through one KV cache.
    //
    // `token_budget` is ignored, as caps().chunked_prefill == false announces:
    // the prompt runs to completion in one call.
    BatchPrefillResult batch_prefill(const std::vector<BatchedRequest*>& requests,
                                     int token_budget) override;
    BatchDecodeResult batch_decode_step(
        const std::vector<BatchedRequest*>& requests) override;

private:
    bool is_stop_token(const BatchSamplingParams& sampling, int token) const;

    std::unique_ptr<PersistentEngine> owned_;
    PersistentEngine* engine_ = nullptr;
    // The one slot, and whose it is. -1 means free.
    bool slot_taken_ = false;
    uint64_t slot_request_id_ = 0;
    // Absolute position of `last_token` for the request holding the slot. Prefill
    // of n tokens leaves this at n, which is where the token it sampled sits.
    int position_ = 0;
};

}  // namespace pocket
