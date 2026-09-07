#include "persistent_engine_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace pocket {
namespace {

// Matches the server's rule and PersistentEngine's own contract: a temperature
// at or below this is greedy argmax, which is what keeps existing greedy runs
// reproducible when a request leaves temperature unset.
constexpr float kGreedyTemperatureEpsilon = 1.0e-5f;

SamplingParams to_persistent_sampling(const BatchSamplingParams& sampling) {
    SamplingParams sp;
    sp.temperature = sampling.temperature;
    sp.top_p = sampling.top_p;
    sp.greedy = sampling.temperature <= kGreedyTemperatureEpsilon;
    sp.seed = sampling.seed;
    // PersistentEngine's sampler has no top-k stage, so BatchSamplingParams::top_k
    // is dropped here rather than silently reinterpreted as something else.
    return sp;
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    const std::chrono::duration<double> delta =
        std::chrono::steady_clock::now() - start;
    return delta.count();
}

}  // namespace

PersistentEngineAdapter::PersistentEngineAdapter(const std::string& ckpt_dir,
                                                 const ForwardSmokeOptions& opts,
                                                 int layer_count,
                                                 int max_context)
    : owned_(std::make_unique<PersistentEngine>(ckpt_dir, opts, layer_count, max_context)),
      engine_(owned_.get()) {}

PersistentEngineAdapter::PersistentEngineAdapter(PersistentEngine& engine)
    : engine_(&engine) {}

PersistentEngineAdapter::~PersistentEngineAdapter() = default;

Capabilities PersistentEngineAdapter::caps() const {
    Capabilities c;
    c.paged_kv = false;
    c.continuous_batching = false;
    c.chunked_prefill = false;
    c.max_slots = 1;
    return c;
}

int PersistentEngineAdapter::max_context() const { return engine_->max_context(); }

int PersistentEngineAdapter::device() const { return engine_->options().device; }

void PersistentEngineAdapter::allocate_batch_slots(int max_batch_size) {
    if (max_batch_size > 1) {
        throw std::invalid_argument(
            "PersistentEngineAdapter: DeepSeek-V4 runs one session at a time; "
            "requested " + std::to_string(max_batch_size) +
            " slots but caps().max_slots is 1");
    }
    // One slot always exists; there is nothing to allocate.
}

int PersistentEngineAdapter::allocate_slot(uint64_t request_id) {
    if (slot_taken_) return -1;
    slot_taken_ = true;
    slot_request_id_ = request_id;
    position_ = 0;
    // The session carries the previous request's KV cache and position, so a new
    // occupant has to start from a cleared one. Doing it here rather than inside
    // batch_prefill keeps it to exactly one reset per request, which matters
    // because reset is the only thing standing between two requests' caches.
    engine_->worker_command_reset();
    engine_->reset_session();
    return 0;
}

void PersistentEngineAdapter::free_slot(uint64_t request_id) {
    if (!slot_taken_ || slot_request_id_ != request_id) return;
    slot_taken_ = false;
    slot_request_id_ = 0;
    position_ = 0;
}

bool PersistentEngineAdapter::kv_paged() const { return false; }
int PersistentEngineAdapter::kv_free_blocks() const { return 0; }
int PersistentEngineAdapter::kv_total_blocks() const { return 0; }
int PersistentEngineAdapter::kv_blocks_for_tokens(int) const { return 0; }

bool PersistentEngineAdapter::is_stop_token(const BatchSamplingParams& sampling,
                                            int token) const {
    if (sampling.ignore_eos) return false;
    if (!sampling.stop_token_ids.empty()) {
        return std::find(sampling.stop_token_ids.begin(),
                         sampling.stop_token_ids.end(),
                         token) != sampling.stop_token_ids.end();
    }
    return token == engine_->eos_id();
}

BatchPrefillResult PersistentEngineAdapter::batch_prefill(
    const std::vector<BatchedRequest*>& requests, int /*token_budget*/) {
    BatchPrefillResult out;
    if (requests.empty()) return out;
    if (requests.size() > 1) {
        throw std::invalid_argument(
            "PersistentEngineAdapter::batch_prefill: one request at a time "
            "(caps().max_slots is 1), got " + std::to_string(requests.size()));
    }

    BatchedRequest* req = requests[0];
    if (req == nullptr) throw std::invalid_argument(
        "PersistentEngineAdapter::batch_prefill: null request");
    if (req->seq_len != 0) {
        // caps().chunked_prefill is false, so a prompt always completes in one
        // call and the scheduler never resumes one. Arriving here means it did
        // anyway, and replaying the prompt from token 0 into a cache that already
        // holds part of it would duplicate positions.
        throw std::invalid_argument(
            "PersistentEngineAdapter::batch_prefill: cannot resume a partially "
            "prefilled prompt; this engine does not declare chunked_prefill");
    }

    const auto started = std::chrono::steady_clock::now();
    const SamplingParams sp = to_persistent_sampling(req->sampling);
    engine_->worker_command_prefill(req->prompt_tokens);
    const int token = engine_->prefill(req->prompt_tokens, sp);

    const int prompt_tokens = static_cast<int>(req->prompt_tokens.size());
    position_ = prompt_tokens;

    ForwardResult result;
    result.token = token;
    result.top_token = token;
    result.position = position_;

    req->seq_len = prompt_tokens;
    req->last_token = token;
    req->last_result = result;
    req->finished = is_stop_token(req->sampling, token);

    out.results.push_back(result);
    out.incomplete.push_back(false);
    out.total_tokens = prompt_tokens;
    out.seconds = elapsed_seconds(started);
    return out;
}

BatchDecodeResult PersistentEngineAdapter::batch_decode_step(
    const std::vector<BatchedRequest*>& requests) {
    BatchDecodeResult out;
    if (requests.empty()) return out;
    if (requests.size() > 1) {
        throw std::invalid_argument(
            "PersistentEngineAdapter::batch_decode_step: one request at a time "
            "(caps().max_slots is 1), got " + std::to_string(requests.size()));
    }

    BatchedRequest* req = requests[0];
    if (req == nullptr) throw std::invalid_argument(
        "PersistentEngineAdapter::batch_decode_step: null request");

    const auto started = std::chrono::steady_clock::now();
    const SamplingParams sp = to_persistent_sampling(req->sampling);
    // position_ is where `last_token` itself sits, which is the position
    // decode_step wants -- not the position being produced.
    engine_->worker_command_decode(req->last_token, position_);
    const int token = engine_->decode_step(req->last_token, position_, sp);
    ++position_;

    const bool stopped = is_stop_token(req->sampling, token);
    req->last_token = token;
    req->seq_len += 1;
    req->finished = stopped;

    out.next_tokens.push_back(token);
    out.finished.push_back(stopped);
    out.hit_stop_token.push_back(stopped);
    out.seconds = elapsed_seconds(started);
    return out;
}

}  // namespace pocket
