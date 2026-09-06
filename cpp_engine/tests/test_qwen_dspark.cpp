#include "qwen_dspark.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_qwen_dspark <draft_checkpoint>\n";
        return 2;
    }
    try {
        const std::string checkpoint = argv[1];
        const pocket::QwenDSparkConfig config =
            pocket::QwenDSparkConfig::from_directory(checkpoint);
        config.validate_for_target(5120, 248320, 64);
        require(config.block_size == 7, "unexpected DSpark block size");
        require(config.target_layer_ids ==
                    std::vector<int>({4, 16, 28, 40, 52}),
                "unexpected target taps");

        const pocket::SafeTensorsIndex index =
            pocket::SafeTensorsIndex::from_single_file(checkpoint);
        require(index.tensor_count() == 62, "unexpected DSpark tensor count");
        const pocket::QwenDSparkWeightMap rank0(index, config, 4, 0);
        const pocket::QwenDSparkWeightMap rank3(index, config, 4, 3);
        require(rank0.tensor_count() == 62, "weight map missed tensors");
        require(rank0.markov_w2().local_shape ==
                    std::vector<uint64_t>({62080, 256}),
                "bad rank-local Markov head shape");
        require(rank3.markov_w2().shard_start == 186240,
                "bad Markov rank offset");
        require(rank0.local_device_bytes() < 3ull * 1024 * 1024 * 1024,
                "replicated draft exceeds expected device budget");

        const std::vector<float> frequencies =
            pocket::qwen_dspark_yarn_inv_freqs(config);
        require(frequencies.size() == 64, "bad YaRN frequency count");
        require(std::abs(config.rope.attention_factor -
                         (0.1 * std::log(32.0) + 1.0)) < 1.0e-9,
                "bad default YaRN attention factor");
        require(std::abs(frequencies.front() - 1.0f) < 1.0e-6f,
                "bad leading YaRN frequency");
        require(frequencies.back() > 0.0f &&
                    frequencies.back() < frequencies.front(),
                "bad trailing YaRN frequency");

        std::cout << "[PASS] qwen_dspark tensors=" << index.tensor_count()
                  << " local_device_bytes=" << rank0.local_device_bytes()
                  << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << "\n";
        return 1;
    }
}
