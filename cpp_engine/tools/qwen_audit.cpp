// Host-only Qwen checkpoint audit.
//
// Validates a real Qwen3.5/3.8 Safetensors checkpoint against the runtime's
// weight map without touching an accelerator: config parsing, index/shard
// integrity, tensor coverage, dtypes, and per-rank tensor-parallel shard
// metadata. Because it links only the device-agnostic core, it runs on a machine
// with no CUDA toolkit, which is where checkpoint problems are cheapest to find.
//
// Usage:
//   qwen_audit <ckpt_dir> [--tp-world N] [--strict] [--verbose]

#include "qwen_config.hpp"
#include "qwen_weights.hpp"
#include "safetensors_reader.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string ckpt;
    int tp_world = 1;
    bool strict = false;
    bool verbose = false;
};

Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "usage: qwen_audit <ckpt_dir> [--tp-world N] [--strict] [--verbose]");
    }
    Args args;
    args.ckpt = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tp-world" && i + 1 < argc) {
            args.tp_world = std::stoi(argv[++i]);
        } else if (arg == "--strict") {
            args.strict = true;
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (args.tp_world <= 0) throw std::runtime_error("--tp-world must be positive");
    return args;
}

double gib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// Opening every shard header validates the index end to end: each mapped shard
// exists, parses, and reports byte extents that fall inside the file. Headers are
// small, and no tensor payload is read.
uint64_t audit_shards(const pocket::SafeTensorsIndex& index, bool verbose) {
    uint64_t header_tensors = 0;
    for (const std::string& shard_name : index.shards()) {
        const pocket::SafeTensorsShard shard(index.shard_path(shard_name));
        header_tensors += shard.tensors().size();
        if (verbose) {
            std::cout << "shard " << shard_name << " file_bytes=" << shard.file_size()
                      << " header_bytes=" << shard.header_len()
                      << " tensors=" << shard.tensors().size() << '\n';
        }
    }
    return header_tensors;
}

void print_rank(const pocket::SafeTensorsIndex& index, const pocket::QwenConfig& config,
                int tp_world, int tp_rank, bool strict) {
    const pocket::QwenWeightMap map(index, config, tp_world, tp_rank);
    if (strict) map.require_full_coverage();
    const pocket::QwenCoverage cover = map.coverage();
    const pocket::QwenLinearKindCounts kinds = map.checkpoint_linear_kind_counts();
    const uint64_t resident =
        map.local_weight_bytes() + map.local_scale_bytes();

    std::cout << "rank " << tp_rank << '/' << tp_world
              << " mapped=" << cover.mapped_tensors
              << " visual_ignored=" << cover.visual_tensors
              << " unexpected=" << cover.unexpected_tensors
              << " resident_GiB=" << gib(resident)
              << " sharded_GiB=" << gib(cover.sharded_local_bytes)
              << " replicated_GiB=" << gib(cover.replicated_local_bytes)
              << " dense=" << kinds.dense_f16
              << " fp8_block128=" << kinds.fp8_block128
              << " fp8_channel=" << kinds.fp8_channel
              << " nvfp4=" << kinds.nvfp4_group16 << '\n';

    if (tp_rank == 0) {
        const auto& embed = map.embed_tokens();
        const auto& head = map.lm_head();
        const auto& mlp = map.layers().front().mlp;
        std::cout << "  embed_rows=" << embed.local_shape.at(0)
                  << " head_rows=" << head.logical_local_shape.at(0)
                  << " mlp_intermediate_rows=" << mlp.gate_proj.logical_local_shape.at(0)
                  << " down_proj_k=" << mlp.down_proj.logical_local_shape.at(1)
                  << " storage_dtype=" << pocket::safe_dtype_name(embed.dtype)
                  << " device_dtype=" << pocket::safe_dtype_name(embed.device_dtype)
                  << " mtp=" << (map.mtp().found ? 1 : 0) << '\n';
        std::cout << "  checkpoint_text_GiB=" << gib(cover.checkpoint_text_bytes)
                  << " index_tensors=" << cover.index_tensors << '\n';
    }

    if (!cover.unexpected_examples.empty()) {
        for (const std::string& name : cover.unexpected_examples) {
            std::cout << "  unexpected: " << name << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(args.ckpt);
        const pocket::SafeTensorsIndex index(args.ckpt);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "ckpt=" << args.ckpt << '\n'
                  << "index_tensors=" << index.tensor_count()
                  << " shards=" << index.shard_count()
                  << " total_size=" << index.total_size()
                  << " total_GiB=" << gib(index.total_size()) << '\n';
        std::cout << config.to_string();

        const uint64_t header_tensors = audit_shards(index, args.verbose);
        if (header_tensors < index.tensor_count()) {
            throw std::runtime_error(
                "shard headers expose fewer tensors than the index maps");
        }

        for (int rank = 0; rank < args.tp_world; ++rank) {
            print_rank(index, config, args.tp_world, rank, args.strict);
        }
        std::cout << "[PASS] qwen_audit tp_world=" << args.tp_world << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] qwen_audit " << ex.what() << '\n';
        return 1;
    }
}
