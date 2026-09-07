// Scheduler admission against the paged block pool.
//
// The point of paging is that capacity stops being "one slot per request" and
// becomes "blocks a request actually spans". That only reaches throughput if
// admission counts blocks, so these tests pin the accounting rather than the
// kernels: what gets admitted, what is held back, and what is never allowed to
// overcommit the pool. Correctness of the paged reads themselves is covered by
// test_qwen_paged_kv_kernels and test_qwen_paged_engine_parity.

#include "batch_scheduler.hpp"
#include "qwen_engine.hpp"
#include "qwen_parity_fixture.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr int kMaxContext = 2048;
constexpr int kSlots = 8;
constexpr int kBlockSize = 256;
// Blocks a single max-context sequence spans, which is the floor the engine
// enforces on the pool. 8 here.
constexpr int kBlocksPerSeq = kMaxContext / kBlockSize;

// Pool budget for exactly `blocks` blocks. The fixture has one full-attention
// layer with kv_heads * head_dim = 512 elements per token, K and V, FP16.
uint64_t budget_for_blocks(int blocks) {
    return static_cast<uint64_t>(blocks) * kBlockSize * 512 * 2 * 2;
}

pocket::QwenEngineOptions paged_options() {
    pocket::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.prefill_chunk_tokens = 1024;
    options.kv_cache_dtype = pocket::QwenKvCacheDType::Fp16;
    // Chunked prefill resumes a partially-prefilled row through the prefix-cache
    // path: with it off, each pass rezeroes the slot and restarts from token 0,
    // so a prompt longer than one chunk never finishes.
    options.prefix_cache = true;
    options.max_batch_size = kSlots;
    options.kv_paged = true;
    options.kv_block_size = kBlockSize;
    return options;
}

std::vector<int> prompt_of(int length) {
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] = i % 64;
    }
    return tokens;
}

// Waits for a predicate on scheduler stats. Admission runs on the scheduler
// thread, so every assertion here has to be reached by polling rather than by
// assuming a submit has taken effect yet.
template <typename Predicate>
bool wait_for(const pocket::BatchScheduler& scheduler, Predicate pred,
              int timeout_ms = 30000) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred(scheduler.get_stats())) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

// ============================================================================
// Blocks, not slots, are the budget.
//
// The pool is sized so the slot count is deliberately not the binding
// constraint: 8 slots are free, but two long requests already exhaust the
// blocks. Under the old slot-based rule all three would be admitted; the
// assertion is that admission stops on blocks while slots remain.
// ============================================================================
void test_admission_is_bounded_by_blocks(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    // Room for two max-context sequences. Eight slots stay available, so any
    // admission limit observed below can only have come from blocks.
    options.kv_cache_bytes = budget_for_blocks(2 * kBlocksPerSeq);
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    require(engine.kv_paged(), "engine should report paging enabled");
    const int total_blocks = engine.kv_total_blocks();
    require(total_blocks == 2 * kBlocksPerSeq,
            "pool should hold two max-context sequences, got " +
                std::to_string(total_blocks));

    // Each request charges prompt + max_new_tokens, sized at max_context so
    // exactly two fit and a third cannot.
    const int per_request_tokens = kMaxContext;
    const int prompt_len = per_request_tokens / 2;
    const int gen_len = per_request_tokens - prompt_len;

    pocket::BatchScheduler scheduler(&engine, kSlots);
    scheduler.set_prefill_token_budget(1024);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = gen_len;
    sampling.ignore_eos = true;  // keep the footprint fixed

    const int expected = engine.kv_blocks_for_tokens(per_request_tokens);

    for (int i = 0; i < 3; ++i) {
        require(scheduler.submit_request(prompt_of(prompt_len), sampling) != 0,
                "submit should be accepted");
    }

    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.running_requests == 2 && s.waiting_requests == 1;
            }),
            "exactly two requests should be admitted, the third held on blocks");

    pocket::BatchScheduler::Stats stats = scheduler.get_stats();
    require(stats.free_slots >= 4,
            "slots must still be free, proving blocks and not slots bound "
            "admission (free_slots=" + std::to_string(stats.free_slots) + ")");
    require(stats.reserved_blocks == 2 * expected,
            "reservation should account both admitted requests: expected " +
                std::to_string(2 * expected) + " got " +
                std::to_string(stats.reserved_blocks));
    // The invariant that keeps a decode from running out mid-forward.
    require(stats.reserved_blocks <= stats.total_blocks,
            "reserved blocks must never exceed the pool");

    scheduler.stop();
    std::cout << "  admission bounded by blocks while slots remain free: PASS"
              << std::endl;
}

// ============================================================================
// Completion returns blocks, and the held request then starts.
//
// This is the property that turns paging into throughput: without release the
// third request above would wait forever. Watching the held request start only
// after an earlier one finishes is the only way to see reservation and release
// paired correctly.
// ============================================================================
void test_completion_releases_blocks(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    // Exactly one max-context sequence fits, so every admission after the first
    // has to be paid for by a completion returning its blocks.
    options.kv_cache_bytes = budget_for_blocks(kBlocksPerSeq);
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    require(engine.kv_total_blocks() == kBlocksPerSeq,
            "pool should hold exactly one max-context sequence");

    // prompt + generation rounds up to the whole pool, so two cannot coexist.
    const int gen_len = 8;
    const int prompt_len = kMaxContext - 64;

    pocket::BatchScheduler scheduler(&engine, kSlots);
    scheduler.set_prefill_token_budget(1024);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = gen_len;
    sampling.ignore_eos = true;

    constexpr int kRequests = 3;
    for (int i = 0; i < kRequests; ++i) {
        require(scheduler.submit_request(prompt_of(prompt_len), sampling) != 0,
                "submit should be accepted");
    }

    // Only one fits at a time, so the queue can only drain if blocks come back.
    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.completed_requests == kRequests;
            }),
            "all requests should complete, which requires freed blocks to be "
            "reused; a leak here stalls the queue permanently");

    pocket::BatchScheduler::Stats stats = scheduler.get_stats();
    require(stats.waiting_requests == 0, "queue should be drained");
    require(stats.running_requests == 0, "no request should still be running");
    // Reservation and release must balance exactly; a drift would silently
    // shrink usable capacity over a long run.
    require(stats.reserved_blocks == 0,
            "reservation should return to zero once idle, got " +
                std::to_string(stats.reserved_blocks));
    require(stats.free_blocks == stats.total_blocks,
            "every block should be back in the pool: " +
                std::to_string(stats.free_blocks) + " of " +
                std::to_string(stats.total_blocks));

    scheduler.stop();
    std::cout << "  completion releases blocks and admits the waiting request: "
                 "PASS"
              << std::endl;
}

// ============================================================================
// Short requests still run at full slot width.
//
// The block rule must not cost concurrency when requests are small. This is the
// case paging is supposed to win: with the contiguous arena these 8 requests
// would each have pinned max_context.
// ============================================================================
void test_short_requests_fill_all_slots(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    pocket::BatchScheduler scheduler(&engine, kSlots);
    scheduler.set_prefill_token_budget(1024);

    pocket::BatchSamplingParams sampling;
    sampling.max_new_tokens = 32;
    sampling.ignore_eos = true;

    for (int i = 0; i < kSlots; ++i) {
        require(scheduler.submit_request(prompt_of(64), sampling) != 0,
                "submit should be accepted");
    }

    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.running_requests == kSlots ||
                       s.completed_requests == kSlots;
            }),
            "short requests should reach full slot width, not be throttled by "
            "the block rule");

    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.completed_requests == kSlots;
            }),
            "all short requests should complete");

    scheduler.stop();
    std::cout << "  short requests still fill every slot: PASS" << std::endl;
}

// ============================================================================
// Cancelling a running request returns its reservation.
//
// Cancellation reaches the accounting by its own route, so completion balancing
// does not imply this one balances too. A leak here is invisible until capacity
// has quietly drained away, so the check is that a request needing the whole
// pool is admitted *after* a cancellation freed it.
// ============================================================================
void test_cancellation_releases_blocks(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    // One max-context sequence fits, so the second request can only start if the
    // first one's cancellation gave its blocks back.
    options.kv_cache_bytes = budget_for_blocks(kBlocksPerSeq);
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    pocket::BatchScheduler scheduler(&engine, kSlots);
    scheduler.set_prefill_token_budget(1024);

    pocket::BatchSamplingParams sampling;
    // Long enough to still be running when cancelled, and to reserve the pool.
    sampling.max_new_tokens = 64;
    sampling.ignore_eos = true;
    const int prompt_len = kMaxContext - 128;

    const uint64_t first = scheduler.submit_request(prompt_of(prompt_len), sampling);
    require(first != 0, "first submit should be accepted");
    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.reserved_blocks > 0;
            }),
            "first request should reserve blocks");

    // Queued behind the first: the pool cannot back both at once.
    const uint64_t second = scheduler.submit_request(prompt_of(prompt_len), sampling);
    require(second != 0, "second submit should be queued, not rejected");

    require(scheduler.cancel_request(first), "cancel should find the request");

    // The second request starting at all is the assertion: under a leak the
    // reservation would outlive the cancelled request and nothing could follow.
    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.completed_requests == 1;
            }),
            "the queued request should run once cancellation freed the pool; a "
            "leaked reservation stalls it forever");

    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.running_requests == 0 && s.waiting_requests == 0;
            }),
            "scheduler should drain");

    pocket::BatchScheduler::Stats stats = scheduler.get_stats();
    require(stats.cancelled_requests == 1, "one request should be cancelled");
    require(stats.reserved_blocks == 0,
            "reservation should return to zero after cancel and completion, got " +
                std::to_string(stats.reserved_blocks));
    require(stats.free_blocks == stats.total_blocks,
            "every block should be back in the pool: " +
                std::to_string(stats.free_blocks) + " of " +
                std::to_string(stats.total_blocks));

    scheduler.stop();
    std::cout << "  cancellation releases blocks: PASS" << std::endl;
}

// ============================================================================
// A request larger than the whole pool is rejected at submit.
//
// It could never be admitted, so leaving it queued would block everything
// behind it forever. Rejecting at submit is the only outcome the caller can act
// on.
// ============================================================================
void test_oversized_request_rejected(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    // The engine floors the pool at one max-context sequence, so an unbackable
    // request is one whose prompt plus generation exceeds max_context itself.
    options.kv_cache_bytes = budget_for_blocks(kBlocksPerSeq);
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    pocket::BatchScheduler scheduler(&engine, kSlots);

    pocket::BatchSamplingParams sampling;
    // Worst case is one block beyond what the whole pool can back.
    sampling.max_new_tokens = kMaxContext;
    sampling.ignore_eos = true;

    require(scheduler.submit_request(prompt_of(kBlockSize), sampling) == 0,
            "a request whose worst case exceeds the pool must be rejected at "
            "submit rather than queued forever");

    // A normal request behind it must still be servable.
    pocket::BatchSamplingParams ok;
    ok.max_new_tokens = 16;
    ok.ignore_eos = true;
    require(scheduler.submit_request(prompt_of(64), ok) != 0,
            "a serviceable request should still be accepted");
    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.completed_requests == 1;
            }),
            "the serviceable request should complete, proving the rejected one "
            "left no residue");

    scheduler.stop();
    std::cout << "  oversized request rejected at submit: PASS" << std::endl;
}

// ============================================================================
// The contiguous arena keeps the slot rule.
//
// Paging is opt-in, so the block accounting must be inert when it is off:
// max_batch_size requests should run concurrently regardless of length.
// ============================================================================
void test_contiguous_unaffected(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    options.kv_paged = false;
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    require(!engine.kv_paged(), "paging should be off");
    require(engine.kv_total_blocks() == 0,
            "contiguous arena should report no blocks");

    pocket::BatchScheduler scheduler(&engine, kSlots);
    scheduler.set_prefill_token_budget(1024);

    pocket::BatchSamplingParams sampling;
    // Together these reach max_context, a footprint the block rule would have
    // throttled to two concurrent requests if it applied here.
    sampling.max_new_tokens = 512;
    sampling.ignore_eos = true;

    for (int i = 0; i < kSlots; ++i) {
        require(scheduler.submit_request(prompt_of(kMaxContext - 512),
                                        sampling) != 0,
                "submit should be accepted under the contiguous arena");
    }

    require(wait_for(scheduler, [](const pocket::BatchScheduler::Stats& s) {
                return s.running_requests == kSlots;
            }),
            "all slots should be admitted under the contiguous arena, showing "
            "the block rule is inert when paging is off");

    pocket::BatchScheduler::Stats stats = scheduler.get_stats();
    require(stats.reserved_blocks == 0,
            "no blocks should be reserved when paging is off");

    scheduler.stop();
    std::cout << "  contiguous arena keeps the slot rule: PASS" << std::endl;
}

// ============================================================================
// Capabilities are declared, not inferred.
//
// The scheduler used to read kv_total_blocks() == 0 as "contiguous arena",
// which is an inference from an accounting field: an engine that supports
// neither paging nor batching reports the same zero. These assertions pin what
// QwenEngine declares for each configuration, so a scheduler reading caps()
// gets the truth rather than a coincidence.
// ============================================================================
void test_caps_declare_the_configuration(const std::string& dir) {
    {
        pocket::QwenEngineOptions options = paged_options();
        options.kv_cache_bytes = budget_for_blocks(2 * kBlocksPerSeq);
        pocket::QwenEngine engine(dir, options, 2, kMaxContext);
        const pocket::Capabilities caps = engine.caps();
        require(caps.paged_kv, "paged engine should declare paged_kv");
        require(caps.paged_kv == engine.kv_paged(),
                "caps().paged_kv must agree with kv_paged()");
        require(caps.continuous_batching,
                "an 8-slot engine should declare continuous batching");
        require(caps.chunked_prefill,
                "batch_prefill honours its token budget on every configuration");
        require(caps.max_slots == kSlots,
                "max_slots should be the configured batch size, got " +
                    std::to_string(caps.max_slots));
    }
    {
        pocket::QwenEngineOptions options = paged_options();
        options.kv_paged = false;
        pocket::QwenEngine engine(dir, options, 2, kMaxContext);
        const pocket::Capabilities caps = engine.caps();
        require(!caps.paged_kv,
                "contiguous engine should declare paged_kv false");
        require(caps.max_slots == kSlots, "max_slots should still be the "
                                          "configured batch size");
    }
    {
        // One slot is the shape a non-batching engine reports, and it must be
        // distinguishable from the contiguous 8-slot case above by caps() alone
        // -- both report kv_total_blocks() == 0.
        pocket::QwenEngineOptions options = paged_options();
        options.kv_paged = false;
        options.max_batch_size = 1;
        pocket::QwenEngine engine(dir, options, 2, kMaxContext);
        const pocket::Capabilities caps = engine.caps();
        require(!caps.continuous_batching,
                "a single-slot engine should not declare continuous batching");
        require(caps.max_slots == 1, "single-slot engine should report 1 slot");
        require(engine.kv_total_blocks() == 0,
                "this is the case the old zero-blocks inference could not tell "
                "apart from a wide contiguous engine");
    }

    std::cout << "  capabilities are declared, not inferred: PASS" << std::endl;
}

}  // namespace

int main() {
    try {
        const std::string dir =
            qwen_fixture::fixture_dir("qwen_scheduler_admission_fixture");
        require(qwen_fixture::write_fixture(dir),
                "failed to write the checkpoint fixture");

        std::cout << "Qwen scheduler paged admission tests" << std::endl;
        std::cout << "====================================" << std::endl;

        test_admission_is_bounded_by_blocks(dir);
        test_completion_releases_blocks(dir);
        test_short_requests_fill_all_slots(dir);
        test_cancellation_releases_blocks(dir);
        test_oversized_request_rejected(dir);
        test_contiguous_unaffected(dir);
        test_caps_declare_the_configuration(dir);

        std::cout << "\nAll scheduler paged admission tests PASSED"
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nFAILED: " << e.what() << std::endl;
        return 1;
    }
}
