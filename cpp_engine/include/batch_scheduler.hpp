#pragma once

#include "inference_engine.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pocket {

// Result returned to the caller after generation completes
struct SchedulerGenerationResult {
    uint64_t request_id = 0;
    std::vector<int> generated_tokens;
    std::string finish_reason;  // "stop" or "length"
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double total_seconds = 0.0;
    double ttft_seconds = 0.0;  // Time to first token
};

// Internal request wrapper with scheduling metadata
struct SchedulerRequest {
    uint64_t request_id = 0;
    std::vector<int> prompt_tokens;
    BatchSamplingParams sampling;
    int slot_id = -1;
    int seq_len = 0;  // Tokens processed (prompt + generated)
    // Prompt tokens prefilled so far. Chunked prefill advances this a bounded
    // step at a time, so it is what distinguishes "prompt still in progress"
    // from "ready to decode"; seq_len alone cannot, since a half-prefilled
    // request has seq_len > 0 while having produced no token yet.
    int prefilled_tokens = 0;
    bool prefill_complete = false;
    bool finished = false;
    // Whether generation ended on a stop token rather than on max_new_tokens.
    // `finished` is true for both, so it cannot distinguish them on its own.
    bool stopped_on_token = false;
    // Cancelled before finishing. Recorded here because the cancelled set is
    // erased before results are reported.
    bool cancelled = false;
    int last_token = 0;
    std::vector<int> generated_tokens;

    // Timing
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point completion_time;

    // Result notification
    std::function<void(const SchedulerGenerationResult&)> callback;
    bool callback_invoked = false;
};

// Continuous batching scheduler over any InferenceEngine.
//
// This scheduler runs a background thread that:
// 1. Admits waiting requests when slots are available
// 2. Separates requests into prefill vs decode batches
// 3. Calls engine->batch_prefill() for new requests
// 4. Calls engine->batch_decode_step() for running requests
// 5. Handles completions and frees slots
//
// Thread-safe: submit_request/cancel_request can be called concurrently
class BatchScheduler {
public:
    BatchScheduler(InferenceEngine* engine, int max_batch_size);
    ~BatchScheduler();

    // No copy/move
    BatchScheduler(const BatchScheduler&) = delete;
    BatchScheduler& operator=(const BatchScheduler&) = delete;

    // Submit a new generation request
    // Returns request_id (> 0) on success, 0 on failure
    // The callback will be invoked from the scheduler thread when generation completes
    uint64_t submit_request(
        const std::vector<int>& prompt_tokens,
        const BatchSamplingParams& sampling,
        std::function<void(const SchedulerGenerationResult&)> callback = nullptr);

    // Cancel a pending or running request
    // Returns true if the request was found and marked for cancellation
    bool cancel_request(uint64_t request_id);

    // Prompt tokens each request may advance per schedule iteration. Smaller
    // values let decode interleave sooner at some cost in prefill efficiency:
    // measured on TP4 27B, 4096 tokens runs at 1890 tok/s and 2048 at 1780
    // (0.94x), while 512 drops to 1330 (0.71x). 4096 keeps full prefill
    // throughput while capping a 65K prompt's uninterrupted hold at about 2.2s
    // instead of 56s. 0 disables chunking and restores the previous behaviour of
    // running each prompt to completion. An engine that does not declare
    // chunked_prefill pins this at 0: it would run the whole prompt regardless,
    // and holding a budget it ignores would only misreport what is happening.
    void set_prefill_token_budget(int tokens) {
        prefill_token_budget_ =
            (!caps_.chunked_prefill || tokens < 0) ? 0 : tokens;
    }
    int prefill_token_budget() const { return prefill_token_budget_; }

    // Blocking poll for a request result (for sync API)
    // Returns true if result was populated, false on timeout
    bool poll_result(uint64_t request_id, SchedulerGenerationResult* out,
                     int timeout_ms = 30000);

    // What the engine behind this scheduler declared it can do, read once at
    // construction. `max_batch_size` may have been clamped to caps().max_slots,
    // so this is how a caller learns the width it actually got.
    const Capabilities& engine_caps() const { return caps_; }
    int max_batch_size() const { return max_batch_size_; }

    // Get scheduler statistics
    struct Stats {
        int waiting_requests = 0;
        int running_requests = 0;
        int completed_requests = 0;
        int cancelled_requests = 0;
        int free_slots = 0;
        // Paged KV admission state; all 0 under the contiguous arena.
        int reserved_blocks = 0;
        int total_blocks = 0;
        int free_blocks = 0;
    };
    Stats get_stats() const;

    // Check if scheduler is running
    bool is_running() const { return running_.load(); }

    // Stop the scheduler (blocks until thread exits)
    void stop();

private:
    // Main scheduling loop (runs in background thread)
    void schedule_loop();

    // Admit new requests from waiting queue
    void admit_requests();

    // Run prefill batch.  Returns true if a forward pass ran.
    bool run_prefill_batch();

    // Run decode batch.  Returns true if a forward pass ran.
    bool run_decode_batch();

    // Handle completed requests
    void handle_completions();

    // Blocks a request may end up holding: prompt plus its full generation
    // allowance. 0 under the contiguous arena, where slots are the only budget.
    int worst_case_blocks(const SchedulerRequest& req) const;

    // Notify result (invoke callback or store for poll)
    void notify_result(SchedulerRequest* req);

    // Internal state
    InferenceEngine* engine_;
    // Read once in the constructor. Capabilities are fixed by how the engine was
    // built, so re-reading them per admission would be a virtual call answering
    // the same question. Every place the scheduler used to infer a capability --
    // "kv_total_blocks() == 0 means contiguous arena" -- consults this instead.
    Capabilities caps_;
    int max_batch_size_;
    // Default 4096: the largest chunk that still measured full prefill
    // throughput on TP4 27B, so interleaving costs nothing on the prefill side.
    int prefill_token_budget_ = 4096;
    std::atomic<uint64_t> next_request_id_{1};

    // Request queues (protected by queue_mutex_)
    mutable std::mutex queue_mutex_;
    // Signalled when work arrives, so an idle loop wakes on submission instead
    // of on a fixed timer.
    std::condition_variable work_cv_;
    std::queue<std::unique_ptr<SchedulerRequest>> waiting_queue_;
    std::unordered_map<int, std::unique_ptr<SchedulerRequest>> slot_to_request_;
    std::unordered_map<uint64_t, int> request_id_to_slot_;
    // Sum of worst_case_blocks over admitted requests. Admission compares against
    // the pool total minus this, not against the pool's free count: the engine
    // takes blocks only as tokens arrive, so free blocks include capacity a
    // running request will still need for its remaining decode.
    int reserved_blocks_ = 0;

    // Cancelled requests (protected by queue_mutex_)
    std::unordered_set<uint64_t> cancelled_requests_;

    // Completed results for polling (protected by results_mutex_)
    mutable std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::unordered_map<uint64_t, SchedulerGenerationResult> completed_results_;

    // Scheduler thread
    std::atomic<bool> running_{true};
    std::thread schedule_thread_;

    // Statistics
    std::atomic<int> total_completed_{0};
    std::atomic<int> total_cancelled_{0};
};

}  // namespace pocket
