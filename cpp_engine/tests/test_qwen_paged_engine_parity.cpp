// End-to-end parity: a paged KV cache must produce the same tokens as the
// contiguous arena, through the real engine entry points.
//
// The kernel-level test (test_qwen_paged_kv_kernels) pins the block arithmetic
// against fragmented tables. This test covers what that one cannot: that the
// engine reserves blocks at the right moments, uploads the table before the
// forward that reads it, returns blocks when a request finishes, and that a slot
// reusing another's freed blocks still decodes correctly. Those are ordering
// properties of the engine, invisible to a kernel test.
//
// Both engines run the same prompts in the same slots, so any divergence is
// attributable to paging alone.

#include "qwen_config.hpp"
#include "qwen_engine.hpp"
#include "qwen_parity_fixture.hpp"
#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr int kMaxContext = 8192;
constexpr int kSlots = 8;
constexpr int kBlockSize = 256;

pocket::QwenEngineOptions base_options() {
    pocket::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.prefill_chunk_tokens = 1024;
    options.kv_cache_dtype = pocket::QwenKvCacheDType::Fp16;
    options.prefix_cache = false;
    options.max_batch_size = kSlots;
    return options;
}

pocket::QwenEngineOptions paged_options() {
    pocket::QwenEngineOptions options = base_options();
    options.kv_paged = true;
    options.kv_block_size = kBlockSize;
    return options;
}

std::vector<int> prompt_for(int seq, int length) {
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] = (seq * 7 + i) % 64;
    }
    return tokens;
}

void compare(const pocket::QwenForwardResult& contiguous,
             const pocket::QwenForwardResult& paged, const std::string& label) {
    require(contiguous.top_token == paged.top_token,
            label + " top_token mismatch: contiguous " +
                std::to_string(contiguous.top_token) + " vs paged " +
                std::to_string(paged.top_token));
    require(contiguous.position == paged.position,
            label + " position mismatch");
    // Bit-exact is the right bar: both paths run the same kernels over the same
    // values in the same order, so any difference in the reduction would be a
    // real addressing or ordering bug rather than arithmetic noise.
    require(contiguous.checksum == paged.checksum,
            label + " checksum mismatch: contiguous " +
                std::to_string(contiguous.checksum) + " vs paged " +
                std::to_string(paged.checksum));
    require(contiguous.top_logit == paged.top_logit,
            label + " top_logit mismatch");
}

// ============================================================================
// Prefill then a run of decode steps, on several slots. The decode run is long
// enough to cross block boundaries, which is where a missing reservation or a
// stale table upload shows up.
// ============================================================================
void test_prefill_decode_parity(const std::string& dir) {
    const std::vector<int> prompt_lens = {700, 1024, 2049};
    const std::vector<int> slots = {1, 3, 6};
    const int decode_steps = 600;  // Crosses at least two 256-token blocks.

    pocket::QwenEngine contiguous(dir, base_options(), 2, kMaxContext);
    contiguous.allocate_batch_slots(kSlots);
    pocket::QwenEngine paged(dir, paged_options(), 2, kMaxContext);
    paged.allocate_batch_slots(kSlots);

    require(paged.runtime_telemetry().kv_paged_blocks > 0,
            "paged engine did not report a block pool");
    require(paged.runtime_telemetry().kv_paged_block_size == kBlockSize,
            "paged engine reported the wrong block size");
    require(contiguous.runtime_telemetry().kv_paged_blocks == 0,
            "contiguous engine reported a block pool");

    for (size_t seq = 0; seq < prompt_lens.size(); ++seq) {
        const int slot = slots[seq];
        const std::vector<int> tokens =
            prompt_for(static_cast<int>(seq), prompt_lens[seq]);
        compare(contiguous.prefill(tokens, slot), paged.prefill(tokens, slot),
                "prefill seq=" + std::to_string(seq));
    }

    for (int step = 0; step < decode_steps; ++step) {
        for (size_t seq = 0; seq < slots.size(); ++seq) {
            const int slot = slots[seq];
            const int token = (static_cast<int>(seq) * 13 + step * 3 + 5) % 64;
            compare(contiguous.decode_step(token, slot),
                    paged.decode_step(token, slot),
                    "decode step=" + std::to_string(step) +
                        " seq=" + std::to_string(seq));
        }
    }
    std::cout << "  prefill + " << decode_steps
              << " decode steps on " << slots.size()
              << " slots: token-exact PASS\n";
}

// ============================================================================
// Batched decode. This is the path that reads blocks natively rather than
// through the gather, so it is the one where a wrong block table produces wrong
// attention rather than a crash.
// ============================================================================
void test_batched_decode_parity(const std::string& dir) {
    const std::vector<int> prompt_lens = {4096, 5000, 6144, 7500};
    const std::vector<int> slots = {1, 3, 5, 6};
    const int steps = 40;

    pocket::QwenEngine contiguous(dir, base_options(), 2, kMaxContext);
    contiguous.allocate_batch_slots(kSlots);
    pocket::QwenEngine paged(dir, paged_options(), 2, kMaxContext);
    paged.allocate_batch_slots(kSlots);

    for (size_t seq = 0; seq < prompt_lens.size(); ++seq) {
        const std::vector<int> tokens =
            prompt_for(static_cast<int>(seq), prompt_lens[seq]);
        (void)contiguous.prefill(tokens, slots[seq]);
        (void)paged.prefill(tokens, slots[seq]);
    }

    for (int step = 0; step < steps; ++step) {
        std::vector<int> tokens(slots.size());
        for (size_t seq = 0; seq < slots.size(); ++seq) {
            tokens[seq] = (static_cast<int>(seq) * 13 + step * 7 + 42) % 64;
        }
        const std::vector<pocket::QwenForwardResult> reference =
            contiguous.batch_decode_tokens(tokens, slots);
        const std::vector<pocket::QwenForwardResult> actual =
            paged.batch_decode_tokens(tokens, slots);
        require(reference.size() == actual.size(),
                "batch result count mismatch");
        for (size_t seq = 0; seq < reference.size(); ++seq) {
            compare(reference[seq], actual[seq],
                    "batched step=" + std::to_string(step) +
                        " seq=" + std::to_string(seq));
        }
    }
    std::cout << "  batched decode: " << slots.size() << " rows x " << steps
              << " steps at 4k-7.5k context: token-exact PASS\n";
}

// ============================================================================
// Blocks are returned and reused. A long request runs, finishes, and a second
// request on a different slot then runs on the blocks the first gave back. If
// free_slot did not release them, the pool here is too small for the
// second request and reservation fails.
// ============================================================================
void test_block_reuse(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    // Room for roughly one max-context sequence, so the second request can only
    // succeed by reusing the first's blocks.
    const int elements_per_token = 4 * 128;  // kv_heads * head_dim in the fixture
    options.kv_cache_bytes = static_cast<uint64_t>(kMaxContext + kBlockSize) *
                             elements_per_token * sizeof(uint16_t) * 2;
    pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    engine.allocate_batch_slots(kSlots);

    const int blocks = engine.runtime_telemetry().kv_paged_blocks;
    require(blocks > 0, "reuse fixture has no blocks");

    // First request: most of the pool, on slot 0.
    const int slot_a = engine.allocate_slot(1001);
    require(slot_a >= 0, "could not acquire first slot");
    const std::vector<int> long_prompt = prompt_for(0, 6000);
    const pocket::QwenForwardResult first = engine.prefill(long_prompt, slot_a);
    require(first.position == 6000, "first prefill position");
    engine.free_slot(1001);

    // Second request of the same size on a different slot. Without the release
    // this throws, since 6000 more tokens do not fit alongside the first.
    const int slot_b = engine.allocate_slot(1002);
    require(slot_b >= 0, "could not acquire second slot");
    const pocket::QwenForwardResult second = engine.prefill(long_prompt, slot_b);
    require(second.position == 6000, "second prefill position");
    // Same prompt, same weights, different slot: the result must not depend on
    // which physical blocks happened to back it.
    compare(first, second, "reused-block prefill");
    engine.free_slot(1002);
    std::cout << "  block reuse across requests (" << blocks
              << "-block pool, 2 x 6000 tokens): PASS\n";
}

// ============================================================================
// A pool too small for one max-context sequence is rejected at construction
// rather than at the token that runs out, since no scheduler could recover from
// the latter.
// ============================================================================
void test_undersized_pool_rejected(const std::string& dir) {
    pocket::QwenEngineOptions options = paged_options();
    options.kv_cache_bytes = 64 * 1024;  // Far below one sequence.
    bool threw = false;
    try {
        pocket::QwenEngine engine(dir, options, 2, kMaxContext);
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "undersized block pool was accepted");

    // A quantized cache cannot be paged, and saying so beats silently running
    // contiguous.
    pocket::QwenEngineOptions fp8 = paged_options();
    fp8.kv_cache_dtype = pocket::QwenKvCacheDType::Fp8;
    threw = false;
    try {
        pocket::QwenEngine engine(dir, fp8, 2, kMaxContext);
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "paged FP8 cache was accepted");
    std::cout << "  undersized pool and paged-FP8 both rejected: PASS\n";
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::cout << "[SKIP] test_qwen_paged_engine_parity requires CUDA\n";
        return 0;
    }
    try {
        const std::string dir =
            qwen_fixture::fixture_dir("qwen_paged_engine_parity_fixture");
        require(qwen_fixture::write_fixture(dir),
                "could not create paged parity fixture");
        test_prefill_decode_parity(dir);
        test_batched_decode_parity(dir);
        test_block_reuse(dir);
        test_undersized_pool_rejected(dir);
        std::cout << "[PASS] test_qwen_paged_engine_parity\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_paged_engine_parity " << ex.what()
                  << "\n";
        return 1;
    }
}
