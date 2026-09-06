#pragma once

#include "dspark.hpp"
#include "deepseek_v4_engine.hpp"
#include "tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pocket {

struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 1.0f;
    bool greedy = true;
    uint64_t seed = 0;
};

// Persistent inference engine that owns SafeForwardContext (weights, resident
// device caches, NCCL handle, FP4 host pinned buffers) for the full lifetime
// of a server process. Lets multiple requests reuse all heavy resident state.
class PersistentEngine {
public:
    // ctor allocates SafeForwardContext, prepares resident caches sized for
    // `max_context` tokens (prompt + generation combined). Throws on failure.
    PersistentEngine(const std::string& ckpt_dir,
                     const ForwardSmokeOptions& opts,
                     int layer_count,
                     int max_context);
    ~PersistentEngine();

    PersistentEngine(const PersistentEngine&) = delete;
    PersistentEngine& operator=(const PersistentEngine&) = delete;

    // Clear KV / indexer caches and reset internal position counter. Cheap.
    void reset_session();

    // Run prefill on token_ids (which must include the prompt's last token).
    // Returns the sampled token id for the next position (rank 0 valid; on
    // worker ranks the return value is the rank-local argmax, which the
    // server should discard).
    int prefill(const std::vector<int>& token_ids, const SamplingParams& sp);

    // Run one decode step using `last_token` as the input embedding at
    // position `position`. Returns the sampled next token (rank 0 valid).
    int decode_step(int last_token, int position, const SamplingParams& sp);

    // Speculative-decode verify: forward `draft_tokens` starting at
    // `start_position` and return what the target model samples at each one, so
    // the caller can find the accepted prefix. Only rank 0's values are
    // meaningful; workers return a rank-local argmax that must be discarded.
    //
    // Forwards the drafts one at a time, not as a batch -- see the comment on
    // the definition for why the batched form is not numerically interchangeable
    // with plain decode.
    std::vector<int> verify_step(const std::vector<int>& draft_tokens,
                                 int start_position,
                                 const SamplingParams& sp);

    // Small-batch continuation verify. The target forwards the whole block once,
    // samples one successor per input row, commits the input rows through the
    // first mismatch, and rolls back the rejected suffix. The returned vector
    // still contains every row's successor so callers can inspect divergence.
    // This path can differ numerically from sequential verify because its GEMM
    // reduction shapes differ.
    std::vector<int> batch_verify_step(const std::vector<int>& draft_tokens,
                                       int start_position,
                                       const SamplingParams& sp);

    // Snapshot the compressor accumulators (layers 2-42) to host memory so a
    // rejected sequential verify's partial state can be restored. The batched
    // continuation path instead uses a row-tagged device journal covering the
    // sliding ring, pooled KV, and both compressor states. Without either form
    // of rollback, replaying a position that crossed an overlap boundary reads
    // overwritten or zeroed slots and diverges.
    void snapshot_compressor_state();
    void restore_compressor_state();

    // One speculative round: draft from the committed token, verify the block,
    // accept the longest matching prefix, and commit it.
    //
    // `committed_token` is the token to seed the draft with; `position` is its
    // own position (the draft's seed hidden came from position - 1).
    //
    // Sequential verification stops at the first mismatch, so it mutates only
    // committed positions. With POCKETLLM_CPP_BATCHED_VERIFY=1, the whole block is
    // forwarded once and a row-tagged device undo journal removes the rejected
    // suffix while preserving the accepted prefix; no accepted row is replayed.
    //
    // Returns the tokens generated this round (accepted drafts + bonus token).
    // Length is n_accepted + 1, always >= 1. The caller advances position by
    // that count.
    //
    // Must have called load_dspark() first. Rank 0 only; workers stay in
    // run_worker_loop().
    std::vector<int> speculative_step(int committed_token, int position,
                                       const SamplingParams& sp);

    // DSpark main-hidden capture. The draft module's main_proj consumes the
    // main model's block output at a few late layers, mean-pooled over the hc
    // dimension and concatenated on the last axis -- exactly what the
    // reference's forward hooks on model.layers[idx] record.
    //
    // Off until layers are set; capture costs one pooling kernel per target
    // layer per forward and a [positions * n_target * dim] D2H copy.
    //
    // `prefill_window` is how many trailing prompt positions prefill keeps.
    // Priming the draft's attention ring needs every committed position, so
    // load_dspark() sets this to the draft's window_size; 1 is right when only
    // the committed position's hidden is wanted.
    void set_dspark_capture_layers(const std::vector<int>& layers, int prefill_window = 1);
    const std::vector<int>& dspark_capture_layers() const;

    // Hidden captured by the most recent prefill/decode_step, [positions,
    // n_target * dim] in position order. Empty if capture is off. A decode step
    // captures one position; a prefill captures the last window's worth and,
    // when DSpark is loaded, automatically primes those rows into the draft ring.
    const std::vector<float>& last_dspark_hidden() const;

    // Number of positions in last_dspark_hidden().
    int last_dspark_hidden_positions() const;

    // Hiddens captured by the most recent verify_step, one row per draft token:
    // [draft_len, n_target * dim]. Row i is the hidden after consuming draft
    // token i, i.e. what seeds a draft continuing from that token. Empty if
    // capture is off.
    const std::vector<float>& last_verify_dspark_hidden() const;

    // Load the DSpark draft module and set capture to its target layers. Must
    // be called before draft_tokens(); the weights are ~12 GB at TP=1 and
    // 3.8 GB at TP=4, so this is not done at construction.
    void load_dspark(const std::string& ckpt_dir);
    bool dspark_loaded() const;

    // Draft block_size tokens continuing from `input_token`, seeded by the
    // hidden the main model produced at `start_pos` -- the position it consumed
    // to predict `input_token`, not `input_token`'s own position. The committed
    // token occupies start_pos + 1; the draft ropes it there. After a prefill of
    // `n` tokens that makes start_pos = n - 1.
    //
    // `hidden` is one row of [n_target * dim], i.e. last_dspark_hidden() or one
    // row of last_verify_dspark_hidden(). Pass it explicitly rather than
    // reading the engine's last capture: after a verify round the caller drafts
    // from the accepted position, which is not the last one forwarded.
    dspark::DraftOutput draft_tokens(int input_token, int start_pos,
                                     const std::vector<float>& hidden);

    // Prime the draft's attention ring with committed positions not already
    // written by prefill/speculative_step. Prefill automatically writes its
    // captured trailing window when DSpark is loaded, and speculative_step
    // writes each accepted block on all TP ranks. `hidden` is
    // [rows, n_target * dim] in position order.
    void prime_dspark_kv(const std::vector<float>& hidden, int rows, int start_pos);

    // Full-vocab top-k of the most recent prefill/decode/verify selection, in
    // descending logit order. Only populated when POCKETLLM_CPP_TOPK_DIAG > 0, and
    // only on rank 0 at TP > 1 (the only rank holding the whole vocabulary).
    // Read-only diagnostic: it reuses the selection's existing gather and does
    // not add a collective or change the sampled token.
    const std::vector<int>& last_topk_tokens() const;
    const std::vector<float>& last_topk_logits() const;

    int eos_id() const;
    int max_context() const;
    int layer_count() const;
    const Tokenizer& tokenizer() const;
    const ForwardSmokeOptions& options() const;

    // TP rank > 0 entry point. Blocks on a small NCCL int32 broadcast channel
    // driven by rank 0; runs the requested op until SHUTDOWN.
    void run_worker_loop();

    // Trigger NCCL communicator init on all ranks. Must be called by every
    // rank before any of the worker_command_* / run_worker_loop functions.
    // For TP=1 this is a no-op.
    void warmup_tp();

    // Rank 0 utilities to drive the worker loop. No-op for tp_world == 1.
    enum class WorkerCommand : int32_t {
        Prefill = 0,
        DecodeStep = 1,
        Reset = 2,
        Shutdown = 3,
        Verify = 4,
        Draft = 5,
        SpeculativeDecode = 6,
        PrimeDraftKV = 7,
        BatchVerify = 8,
        FinalizeBatchVerify = 9,
    };
    void worker_command_prefill(const std::vector<int>& token_ids);
    void worker_command_decode(int32_t last_token, int32_t position);
    void worker_command_reset();
    void worker_command_shutdown();
    void worker_command_verify(const std::vector<int>& block, int32_t start_position);
    void worker_command_batch_verify(const std::vector<int>& block, int32_t start_position);
    void worker_command_finalize_batch_verify(int32_t committed_rows);
    void worker_command_draft(int32_t input_token, int32_t start_position);
    void worker_command_speculative_decode(int32_t last_token, int32_t position,
                                           bool first);
    void worker_command_prime_draft_kv(int32_t rows, int32_t start_position);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace pocket
