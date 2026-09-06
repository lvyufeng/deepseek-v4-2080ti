// Token-exact parity test: batched decode vs sequential decode steps.
// Validates that batch_decode_tokens produces identical results to N independent
// decode_step calls at the same positions/slots. Uses a zero-weight fixture so
// results are deterministic and the test runs without a 27B checkpoint.

#include "qwen_config.hpp"
#include "qwen_engine.hpp"
#include "qwen_parity_fixture.hpp"
#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The checkpoint fixture lives in a shared header so this test and
// test_qwen_paged_engine_parity cannot drift onto different model geometries.
using qwen_fixture::write_fixture;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_batched_decode_parity(const std::string& dir,
                                pocket::QwenKvCacheDType cache_dtype) {
    pocket::QwenEngineOptions options;
    options.tp_world = 1;
    options.tp_rank = 0;
    options.device = 0;
    options.prefill_chunk_tokens = 1024;
    options.kv_cache_dtype = cache_dtype;
    options.prefix_cache = false;  // Isolate batching from prefix cache
    options.max_batch_size = 8;
    pocket::QwenEngine engine(dir, options, 2, 8192);
    engine.allocate_batch_slots(8);

    // Four sequences at positions 4096, 5000, 6144, 7500 in slots 1, 3, 5, 6
    // (non-identity mapping). All positions >= 4096 so the optimized decode
    // split path runs on the single-row side.
    const std::vector<int> prompt_lens = {4096, 5000, 6144, 7500};
    const std::vector<int> slots = {1, 3, 5, 6};
    const int num_seqs = static_cast<int>(prompt_lens.size());

    // Prefill each sequence to its starting position in its assigned slot
    for (int seq = 0; seq < num_seqs; ++seq) {
        const int len = prompt_lens[static_cast<size_t>(seq)];
        const int slot = slots[static_cast<size_t>(seq)];
        std::vector<int> tokens(static_cast<size_t>(len));
        for (int i = 0; i < len; ++i) {
            tokens[static_cast<size_t>(i)] = (seq * 7 + i) % 64;
        }
        const pocket::QwenForwardResult prefill = engine.prefill(tokens, slot);
        require(prefill.position == len, "prefill position accounting");
    }

    // Sequential baseline: decode one step per sequence independently
    std::vector<pocket::QwenForwardResult> sequential_results(
        static_cast<size_t>(num_seqs));
    std::vector<int> sequential_tokens(static_cast<size_t>(num_seqs));
    for (int seq = 0; seq < num_seqs; ++seq) {
        const int slot = slots[static_cast<size_t>(seq)];
        const int input_token = (seq * 13 + 42) % 64;
        sequential_results[static_cast<size_t>(seq)] =
            engine.decode_step(input_token, slot);
        sequential_tokens[static_cast<size_t>(seq)] =
            sequential_results[static_cast<size_t>(seq)].top_token;
    }

    // Reset engine and re-prefill for the batched path
    engine.reset();
    for (int seq = 0; seq < num_seqs; ++seq) {
        const int len = prompt_lens[static_cast<size_t>(seq)];
        const int slot = slots[static_cast<size_t>(seq)];
        std::vector<int> tokens(static_cast<size_t>(len));
        for (int i = 0; i < len; ++i) {
            tokens[static_cast<size_t>(i)] = (seq * 7 + i) % 64;
        }
        (void)engine.prefill(tokens, slot);
    }

    // Batched path: one forward for all sequences
    std::vector<int> batch_tokens(static_cast<size_t>(num_seqs));
    for (int seq = 0; seq < num_seqs; ++seq) {
        batch_tokens[static_cast<size_t>(seq)] = (seq * 13 + 42) % 128;
    }
    const std::vector<pocket::QwenForwardResult> batched_results =
        engine.batch_decode_tokens(batch_tokens, slots);

    // Compare: every sequence's top_token, top_logit, and checksum must match
    require(batched_results.size() == sequential_results.size(),
           "batch result count mismatch");
    for (int seq = 0; seq < num_seqs; ++seq) {
        const size_t index = static_cast<size_t>(seq);
        const pocket::QwenForwardResult& seq_result = sequential_results[index];
        const pocket::QwenForwardResult& batch_result = batched_results[index];
        const int slot = slots[index];
        const int pos = prompt_lens[index] + 1;
        const std::string label = "seq=" + std::to_string(seq) +
                                 " slot=" + std::to_string(slot) +
                                 " pos=" + std::to_string(pos);
        require(batch_result.top_token == seq_result.top_token,
               label + " top_token mismatch: batched=" +
                   std::to_string(batch_result.top_token) + " sequential=" +
                   std::to_string(seq_result.top_token));
        require(batch_result.top_logit == seq_result.top_logit,
               label + " top_logit mismatch");
        require(batch_result.checksum == seq_result.checksum,
               label + " checksum mismatch");
        require(batch_result.position == seq_result.position,
               label + " position mismatch");
    }

    std::cout << "  cache_dtype="
              << pocket::qwen_kv_cache_dtype_name(cache_dtype)
              << " sequences=" << num_seqs << " token-exact parity PASS\n";
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::cout << "[SKIP] test_qwen_batched_engine_parity requires CUDA\n";
        return 0;
    }
    try {
        const std::string dir = qwen_fixture::fixture_dir(
            "qwen_batched_engine_parity_fixture");
        require(write_fixture(dir), "could not create batched parity fixture");
        test_batched_decode_parity(dir, pocket::QwenKvCacheDType::Fp16);
        // FP8 and TurboQuant batched decode are not yet implemented
        std::cout << "[PASS] test_qwen_batched_engine_parity\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_batched_engine_parity " << ex.what()
                  << "\n";
        return 1;
    }
}
