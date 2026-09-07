#include "batch_scheduler.hpp"
#include "device_runtime.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace pocket {

BatchScheduler::BatchScheduler(InferenceEngine* engine, int max_batch_size)
    : engine_(engine), max_batch_size_(max_batch_size) {
    if (!engine_) {
        throw std::invalid_argument("BatchScheduler: engine cannot be null");
    }
    if (max_batch_size_ <= 0) {
        throw std::invalid_argument("BatchScheduler: max_batch_size must be > 0");
    }

    // Allocate batch slots in the engine
    engine_->allocate_batch_slots(max_batch_size_);

    // Start scheduler thread
    schedule_thread_ = std::thread(&BatchScheduler::schedule_loop, this);
}

BatchScheduler::~BatchScheduler() {
    stop();
}

void BatchScheduler::stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false)) {
        return;  // Already stopped
    }

    // An idle loop is parked on work_cv_; without this it only notices the
    // cleared flag after its wait times out.
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
    }
    work_cv_.notify_all();

    // Wait for scheduler thread to exit
    if (schedule_thread_.joinable()) {
        schedule_thread_.join();
    }

    // Clean up remaining requests
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!waiting_queue_.empty()) {
        waiting_queue_.pop();
    }
    for (auto& [slot_id, req] : slot_to_request_) {
        if (req && req->slot_id >= 0) {
            engine_->free_slot(req->request_id);
        }
    }
    slot_to_request_.clear();
    request_id_to_slot_.clear();
    reserved_blocks_ = 0;
}

uint64_t BatchScheduler::submit_request(
    const std::vector<int>& prompt_tokens,
    const BatchSamplingParams& sampling,
    std::function<void(const SchedulerGenerationResult&)> callback) {

    if (prompt_tokens.empty()) {
        return 0;  // Invalid request
    }

    uint64_t request_id = next_request_id_.fetch_add(1);

    auto req = std::make_unique<SchedulerRequest>();
    req->request_id = request_id;
    req->prompt_tokens = prompt_tokens;
    req->sampling = sampling;
    req->callback = std::move(callback);
    req->submit_time = std::chrono::steady_clock::now();

    // A request whose worst case exceeds the entire pool can never be admitted,
    // so reject it here instead of letting it sit at the head of the queue
    // blocking everything behind it forever. Safe outside queue_mutex_: this
    // reads only the pool geometry, fixed once the engine is constructed, and
    // the request's own fields, which no other thread can see yet.
    if (engine_->kv_paged() &&
        worst_case_blocks(*req) > engine_->kv_total_blocks()) {
        return 0;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        waiting_queue_.push(std::move(req));
    }
    // Wake an idle loop now rather than letting it wait out its tick.
    work_cv_.notify_one();

    return request_id;
}

bool BatchScheduler::cancel_request(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    // Check if request is waiting or running
    auto slot_it = request_id_to_slot_.find(request_id);
    if (slot_it == request_id_to_slot_.end()) {
        // Not found in running requests, might be in waiting queue
        // Mark as cancelled so it gets rejected during admission
        cancelled_requests_.insert(request_id);
        return true;
    }

    // Mark running request as cancelled
    cancelled_requests_.insert(request_id);
    work_cv_.notify_one();
    return true;
}

bool BatchScheduler::poll_result(
    uint64_t request_id, SchedulerGenerationResult* out, int timeout_ms) {

    if (!out) {
        return false;
    }

    auto deadline = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(results_mutex_);

    while (true) {
        // Check if result is ready
        auto it = completed_results_.find(request_id);
        if (it != completed_results_.end()) {
            *out = it->second;
            completed_results_.erase(it);
            return true;
        }

        // Wait with timeout
        if (timeout_ms <= 0) {
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;  // Timeout
        }

        results_cv_.wait_until(lock, deadline);
    }
}

BatchScheduler::Stats BatchScheduler::get_stats() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    Stats stats;
    stats.waiting_requests = static_cast<int>(waiting_queue_.size());
    stats.running_requests = static_cast<int>(slot_to_request_.size());
    stats.completed_requests = total_completed_.load();
    stats.cancelled_requests = total_cancelled_.load();
    stats.free_slots = max_batch_size_ - stats.running_requests;
    stats.reserved_blocks = reserved_blocks_;
    stats.total_blocks = engine_->kv_total_blocks();
    stats.free_blocks = engine_->kv_free_blocks();

    return stats;
}

void BatchScheduler::schedule_loop() {
    // The current device is per-thread, and the engine bound it on the thread
    // that constructed it.  This loop runs every forward pass from its own
    // thread, so it has to bind the same device before touching device memory.
    if (!device_set(engine_->device())) {
        std::cerr << "BatchScheduler: failed to select device "
                  << engine_->device()
                  << "; scheduler thread stopping" << std::endl;
        running_.store(false);
        return;
    }

    while (running_.load()) {
        // Whether this iteration ran a forward pass.  Only an iteration that did
        // nothing should park the thread: sleeping unconditionally cost a
        // millisecond per decode step -- one iteration produces one token per
        // running request -- which showed up as a 1.14x single-request latency
        // regression against the serial path at 32 tokens.
        bool progressed = false;
        try {
            admit_requests();
            progressed |= run_prefill_batch();
            progressed |= run_decode_batch();
            handle_completions();
        } catch (const std::exception& e) {
            std::cerr << "BatchScheduler: exception in schedule loop: "
                     << e.what() << std::endl;
        }

        if (progressed) {
            continue;
        }

        // Nothing advanced, so park until a submission arrives.  The timeout is
        // only a backstop for a state change that forgot to notify; both real
        // triggers (submit_request, cancel_request) signal work_cv_.
        std::unique_lock<std::mutex> lock(queue_mutex_);
        work_cv_.wait_for(lock, std::chrono::milliseconds(20), [this] {
            return !running_.load() || !waiting_queue_.empty();
        });
    }
}

int BatchScheduler::worst_case_blocks(const SchedulerRequest& req) const {
    if (!engine_->kv_paged()) return 0;
    // Prompt plus every token the request is still allowed to generate. Charging
    // only the prompt would admit a request whose decode cannot be backed, and a
    // decode step must write its K/V somewhere. Requests therefore hold their
    // full future footprint for their whole life; this trades some admission
    // headroom for never failing a request that was already accepted.
    const int max_context_tokens =
        static_cast<int>(req.prompt_tokens.size()) + req.sampling.max_new_tokens;
    return engine_->kv_blocks_for_tokens(max_context_tokens);
}

void BatchScheduler::admit_requests() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    while (!waiting_queue_.empty() &&
           slot_to_request_.size() < static_cast<size_t>(max_batch_size_)) {

        // Under paging, capacity is blocks rather than slots: a request costs the
        // blocks its context will span, so a queue of short prompts can run at
        // full width while long ones are held back. The contiguous arena gives
        // every slot max_context up front, so there is nothing to weigh and the
        // slot bound above is the whole admission rule.
        if (engine_->kv_paged()) {
            const int needed =
                worst_case_blocks(*waiting_queue_.front());
            // Checked against the pool total minus what admitted requests may
            // still grow into, not against free_blocks. The engine takes blocks
            // as tokens arrive, so free_blocks counts a running request's future
            // decode blocks as available; admitting against it would overcommit
            // and exhaust the pool inside a decode forward, which has nowhere to
            // put the K/V it must write and no way to back out.
            if (needed > engine_->kv_total_blocks() - reserved_blocks_) {
                // Head-of-line blocking is deliberate: taking a later, smaller
                // request first would starve long prompts indefinitely under a
                // steady stream of short ones. The request waits for blocks its
                // predecessors return.
                break;
            }
        }

        auto req = std::move(waiting_queue_.front());
        waiting_queue_.pop();

        // Check if cancelled
        if (cancelled_requests_.count(req->request_id)) {
            cancelled_requests_.erase(req->request_id);
            total_cancelled_.fetch_add(1);
            continue;
        }

        // Allocate slot
        int slot_id = engine_->allocate_slot(req->request_id);
        if (slot_id < 0) {
            // No slots available (shouldn't happen due to size check)
            waiting_queue_.push(std::move(req));
            break;
        }

        req->slot_id = slot_id;
        // Held until the request completes, so later passes see this request's
        // full future footprint rather than only the blocks it has taken so far.
        reserved_blocks_ += worst_case_blocks(*req);
        request_id_to_slot_[req->request_id] = slot_id;
        slot_to_request_[slot_id] = std::move(req);
    }
}

bool BatchScheduler::run_prefill_batch() {
    std::vector<BatchedRequest*> prefill_batch;
    std::vector<SchedulerRequest*> prefill_requests;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        for (auto& [slot_id, req] : slot_to_request_) {
            // Check if cancelled
            if (cancelled_requests_.count(req->request_id)) {
                continue;
            }

            if (!req->prefill_complete) {
                // Create BatchedRequest wrapper. seq_len carries how far the
                // prompt already got so batch_prefill can charge only the new
                // tokens to its token total.
                auto* batch_req = new BatchedRequest();
                batch_req->request_id = req->request_id;
                batch_req->prompt_tokens = req->prompt_tokens;
                batch_req->slot_id = req->slot_id;
                batch_req->sampling = req->sampling;
                batch_req->seq_len = req->prefilled_tokens;

                prefill_batch.push_back(batch_req);
                prefill_requests.push_back(req.get());
            }
        }
    }

    if (prefill_batch.empty()) {
        return false;
    }

    // Advance each prompt by at most prefill_token_budget_ tokens so a long
    // prompt cannot hold the device for its whole length while running decodes
    // stall behind it.
    try {
        auto result = engine_->batch_prefill(prefill_batch, prefill_token_budget_);

        // Update request states
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (size_t i = 0; i < prefill_batch.size(); ++i) {
            auto* req = prefill_requests[i];
            if (i >= result.results.size()) continue;

            req->prefilled_tokens = prefill_batch[i]->seq_len;
            const bool complete =
                i < result.incomplete.size() && !result.incomplete[i];
            if (!complete) {
                // Interior logits predict the next prompt token, which the prompt
                // already supplies. Emitting it would inject a token the caller
                // never asked for, so this row just waits for its next chunk.
                continue;
            }

            req->prefill_complete = true;
            req->last_token = result.results[i].top_token;
            req->generated_tokens.push_back(req->last_token);
            req->seq_len = req->prefilled_tokens;
            // batch_prefill flags a prompt whose very first predicted token is a
            // stop token; such a request must never reach the decode batch.
            if (prefill_batch[i]->finished) {
                req->finished = true;
                req->stopped_on_token = true;
            }

            // Record TTFT
            if (req->first_token_time == std::chrono::steady_clock::time_point{}) {
                req->first_token_time = std::chrono::steady_clock::now();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "BatchScheduler: prefill failed: " << e.what() << std::endl;
    }

    // Clean up temporary batch requests
    for (auto* batch_req : prefill_batch) {
        delete batch_req;
    }

    return true;
}

bool BatchScheduler::run_decode_batch() {
    std::vector<BatchedRequest*> decode_batch;
    std::vector<SchedulerRequest*> decode_requests;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        for (auto& [slot_id, req] : slot_to_request_) {
            // Check if cancelled
            if (cancelled_requests_.count(req->request_id)) {
                continue;
            }

            // Decode only rows whose prompt is fully consumed. A chunked
            // prefill leaves seq_len > 0 partway through the prompt, and
            // decoding there would append a generated token into the middle of
            // the prompt's own KV positions.
            if (req->prefill_complete && !req->finished) {
                // Create BatchedRequest wrapper
                auto* batch_req = new BatchedRequest();
                batch_req->request_id = req->request_id;
                batch_req->slot_id = req->slot_id;
                batch_req->seq_len = req->seq_len;
                batch_req->last_token = req->last_token;
                batch_req->sampling = req->sampling;

                decode_batch.push_back(batch_req);
                decode_requests.push_back(req.get());
            }
        }
    }

    if (decode_batch.empty()) {
        return false;
    }

    // Call engine batch_decode_step
    try {
        auto result = engine_->batch_decode_step(decode_batch);

        // Update request states
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (size_t i = 0; i < decode_batch.size(); ++i) {
            auto* req = decode_requests[i];
            if (i < result.next_tokens.size()) {
                req->last_token = result.next_tokens[i];
                req->generated_tokens.push_back(req->last_token);
                req->seq_len++;

                // Record TTFT for first decode token (if prefill didn't set it)
                if (req->first_token_time == std::chrono::steady_clock::time_point{} &&
                    req->generated_tokens.size() == 1) {
                    req->first_token_time = std::chrono::steady_clock::now();
                }

                // Check if finished
                if (i < result.finished.size() && result.finished[i]) {
                    req->finished = true;
                }
                if (i < result.hit_stop_token.size() && result.hit_stop_token[i]) {
                    req->stopped_on_token = true;
                }

                // Check max_new_tokens
                if (req->generated_tokens.size() >=
                    static_cast<size_t>(req->sampling.max_new_tokens)) {
                    req->finished = true;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "BatchScheduler: decode failed: " << e.what() << std::endl;
    }

    // Clean up temporary batch requests
    for (auto* batch_req : decode_batch) {
        delete batch_req;
    }

    return true;
}

void BatchScheduler::handle_completions() {
    std::vector<std::unique_ptr<SchedulerRequest>> completed;
    std::vector<bool> cancelled_completions;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // Find completed or cancelled requests.  Moving the unique_ptr out of the
        // map leaves the mapped value null, so read every field this loop and the
        // erase pass need *before* the move rather than through the moved-from
        // slot.
        std::vector<int> slots_to_remove;
        for (auto& [slot_id, req] : slot_to_request_) {
            const uint64_t request_id = req->request_id;
            const bool is_cancelled = cancelled_requests_.count(request_id) > 0;

            if (req->finished || is_cancelled) {
                slots_to_remove.push_back(slot_id);
                // The cancelled set is erased below and is guarded by this mutex,
                // so the stats pass outside the lock cannot re-derive this.
                cancelled_completions.push_back(is_cancelled);
                // notify_result runs after the erase, so it cannot consult the
                // cancelled set either; record the reason on the request while
                // the answer is still available.
                if (is_cancelled && !req->finished) req->cancelled = true;

                // Record completion time
                req->completion_time = std::chrono::steady_clock::now();

                // Drop this request's admission reservation here, under the same
                // mutex that guards it, rather than beside the free_slot call
                // below which runs unlocked. engine_->free_slot returns the
                // blocks themselves; this returns the right to count on them.
                reserved_blocks_ -= worst_case_blocks(*req);
                if (reserved_blocks_ < 0) reserved_blocks_ = 0;

                // Move to completed list
                completed.push_back(std::move(req));

                // Clean up cancelled set
                if (is_cancelled) {
                    cancelled_requests_.erase(request_id);
                }

                request_id_to_slot_.erase(request_id);
            }
        }

        // Remove from active maps
        for (int slot_id : slots_to_remove) {
            slot_to_request_.erase(slot_id);
        }
    }

    // Process completed requests (outside lock to avoid deadlock with callbacks)
    for (size_t i = 0; i < completed.size(); ++i) {
        auto& req = completed[i];
        // Free slot
        if (req->slot_id >= 0) {
            engine_->free_slot(req->request_id);
        }

        // Notify result
        notify_result(req.get());

        // Update stats
        if (cancelled_completions[i]) {
            total_cancelled_.fetch_add(1);
        } else {
            total_completed_.fetch_add(1);
        }
    }
}

void BatchScheduler::notify_result(SchedulerRequest* req) {
    if (!req || req->callback_invoked) {
        return;
    }

    SchedulerGenerationResult result;
    result.request_id = req->request_id;
    result.generated_tokens = req->generated_tokens;
    // Previously `finished ? "length" : "stop"`, which was backwards: `finished`
    // was only ever set by the length cap, so a normal completion reported
    // "length" and "stop" was reachable only via cancellation.
    if (req->cancelled) {
        result.finish_reason = "cancelled";
    } else {
        result.finish_reason = req->stopped_on_token ? "stop" : "length";
    }
    result.prompt_tokens = static_cast<int>(req->prompt_tokens.size());
    result.completion_tokens = static_cast<int>(req->generated_tokens.size());

    // Calculate timings
    auto submit = req->submit_time;
    auto first_token = req->first_token_time;
    auto completion = req->completion_time;

    if (completion > submit) {
        result.total_seconds = std::chrono::duration<double>(completion - submit).count();
    }
    if (first_token > submit) {
        result.ttft_seconds = std::chrono::duration<double>(first_token - submit).count();
    }

    req->callback_invoked = true;

    // Invoke callback if provided
    if (req->callback) {
        try {
            req->callback(result);
        } catch (const std::exception& e) {
            std::cerr << "BatchScheduler: callback exception: " << e.what() << std::endl;
        }
    }

    // Store result for polling
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        completed_results_[req->request_id] = result;
    }
    results_cv_.notify_all();
}

}  // namespace pocket
