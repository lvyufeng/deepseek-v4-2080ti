#include "qwen_dflash2.hpp"

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
        std::cerr << "usage: test_qwen_dflash2 <draft_checkpoint>\n";
        return 2;
    }
    try {
        const std::string checkpoint = argv[1];
        const pocket::QwenDFlash2Config config =
            pocket::QwenDFlash2Config::from_directory(checkpoint);
        config.validate_for_target(5120, 248320, 64);
        require(config.architecture == "DFlash2DraftModel",
                "unexpected DFlash2 architecture");
        require(config.block_size == 8 && config.draft_tokens() == 7,
                "unexpected DFlash2 block size");
        require(config.target_layer_ids ==
                    std::vector<int>({5, 19, 33, 47, 61}),
                "unexpected DFlash2 target taps");
        require(config.layer_types.size() == 5,
                "unexpected DFlash2 layer type count");

        const pocket::SafeTensorsIndex index =
            pocket::SafeTensorsIndex::from_single_file(checkpoint);
        require(index.tensor_count() == 81,
                "unexpected DFlash2 tensor count");
        const pocket::QwenDFlash2WeightMap rank0(index, config, 4, 0);
        const pocket::QwenDFlash2WeightMap rank3(index, config, 4, 3);
        require(rank0.tensor_count() == 81,
                "rank0 weight map missed tensors");
        require(rank3.tensor_count() == 81,
                "rank3 weight map missed tensors");
        require(rank0.projector().full_shape ==
                    std::vector<uint64_t>({5120, 25600}),
                "bad projector shape");
        require(rank0.layers().size() == 5,
                "bad DFlash2 layer count");
        require(rank0.layers().front().q_proj.weight.full_shape ==
                    std::vector<uint64_t>({4096, 5120}),
                "bad Q projection shape");
        require(rank0.layers().front().attention_conv.kernel_projection.weight.full_shape ==
                    std::vector<uint64_t>({1280, 5120}),
                "bad attention convolution shape");
        // Sharding is on by default, so a TP4 rank owns the replicated projector,
        // codebooks and norms plus a quarter of each layer projection. Both ranks
        // must still agree on the budget, and the sharded map must be well under
        // the replicated 3-4 GiB.
        require(rank0.sharded(), "TP4 DFlash2 weight map should shard by default");
        require(rank0.local_device_bytes() > 1ull * 1024 * 1024 * 1024 &&
                    rank0.local_device_bytes() < 3ull * 1024 * 1024 * 1024,
                "unexpected sharded DFlash2 device budget");
        require(rank0.local_device_bytes() == rank3.local_device_bytes(),
                "sharded DFlash2 ranks disagree on device bytes");
        require(rank0.layers().front().q_proj.weight.local_shape ==
                    std::vector<uint64_t>({1024, 5120}),
                "bad sharded Q projection shape");
        require(rank3.layers().front().q_proj.weight.shard_start == 3072,
                "bad sharded Q projection offset");
        require(rank0.layers().front().o_proj.weight.local_shape ==
                    std::vector<uint64_t>({5120, 1024}),
                "bad sharded O projection shape");
        require(rank0.layers().front().down_proj.weight.local_shape ==
                    std::vector<uint64_t>({5120, 4352}),
                "bad sharded down projection shape");
        // The per-head norms, convolutions and selector stay replicated so the
        // conv channel mixing and selector comparator are unchanged.
        require(rank0.layers().front().q_norm.local_shape ==
                    std::vector<uint64_t>({128}),
                "Q norm must stay replicated");
        require(rank0.layers().front().attention_conv.kernel_projection.weight.local_shape ==
                    std::vector<uint64_t>({1280, 5120}),
                "attention convolution must stay replicated");

        const pocket::QwenDFlash2WeightMap single(index, config, 1, 0);
        require(!single.sharded(), "single-rank DFlash2 must stay replicated");
        require(single.local_device_bytes() > 3ull * 1024 * 1024 * 1024 &&
                    single.local_device_bytes() < 4ull * 1024 * 1024 * 1024,
                "unexpected replicated DFlash2 device budget");
        require(single.local_device_bytes() > rank0.local_device_bytes(),
                "sharded DFlash2 map must be smaller than the replicated map");

        std::cout << "[PASS] qwen_dflash2 tensors=" << index.tensor_count()
                  << " sharded_device_bytes=" << rank0.local_device_bytes()
                  << " replicated_device_bytes=" << single.local_device_bytes()
                  << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << "\n";
        return 1;
    }
}
