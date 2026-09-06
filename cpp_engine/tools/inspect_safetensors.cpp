#include "safetensors_reader.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string ckpt;
    size_t limit = 20;
    std::string tensor;
    std::string grep;
};

Args parse_args(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: inspect_safetensors <ckpt_dir> [--limit N] [--tensor NAME] [--grep NEEDLE]");
    Args args;
    args.ckpt = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--limit" && i + 1 < argc) args.limit = static_cast<size_t>(std::stoul(argv[++i]));
        else if (arg == "--tensor" && i + 1 < argc) args.tensor = argv[++i];
        else if (arg == "--grep" && i + 1 < argc) args.grep = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    return args;
}

void print_tensor(const pocket::SafeTensorInfo& info, const std::string& shard = "") {
    std::cout << info.name << " dtype=" << pocket::safe_dtype_name(info.dtype) << " shape=[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << info.shape[i];
    }
    std::cout << "] begin=" << info.data_begin << " abs=" << info.absolute_begin << " bytes=" << info.nbytes;
    if (!shard.empty()) std::cout << " shard=" << shard;
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        pocket::SafeTensorsIndex index(args.ckpt);
        std::cout << "ckpt=" << index.ckpt_dir() << '\n';
        std::cout << "tensors=" << index.tensor_count() << " shards=" << index.shard_count() << " total_size=" << index.total_size() << '\n';
        if (!args.grep.empty()) {
            for (const auto& name : index.grep_tensors(args.grep, args.limit)) {
                const std::string* shard = index.shard_for_tensor(name);
                std::cout << name << " -> " << (shard ? *shard : "") << '\n';
            }
            return 0;
        }
        if (!args.tensor.empty()) {
            const std::string* shard_name = index.shard_for_tensor(args.tensor);
            if (shard_name == nullptr) throw std::runtime_error("tensor not found in index: " + args.tensor);
            pocket::SafeTensorsShard shard(index.shard_path(*shard_name));
            const auto* info = shard.find_tensor(args.tensor);
            if (info == nullptr) throw std::runtime_error("tensor not found in shard header: " + args.tensor);
            print_tensor(*info, *shard_name);
            return 0;
        }
        size_t shown = 0;
        for (const auto& [name, shard_name] : index.weight_map()) {
            std::cout << name << " -> " << shard_name << '\n';
            if (++shown >= args.limit) break;
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
