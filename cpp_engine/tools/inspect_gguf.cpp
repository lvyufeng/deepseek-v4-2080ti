#include "gguf_reader.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
    std::string path;
    size_t limit = 20;
    std::string tensor;
    bool metadata = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2) {
        throw std::runtime_error("usage: inspect_gguf <model.gguf> [--limit N] [--tensor NAME] [--metadata]");
    }
    args.path = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--limit" && i + 1 < argc) {
            args.limit = static_cast<size_t>(std::stoul(argv[++i]));
        } else if (arg == "--tensor" && i + 1 < argc) {
            args.tensor = argv[++i];
        } else if (arg == "--metadata") {
            args.metadata = true;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    return args;
}

void print_tensor(const pocket::GGUFTensorInfo& info) {
    std::cout << info.name << " type=" << pocket::ggml_type_name(info.ggml_type) << " shape=[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << info.shape[i];
    }
    std::cout << "] offset=" << info.offset << " abs=" << info.absolute_offset << " bytes=" << info.nbytes << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        pocket::GGUFFile file(args.path);
        std::cout << "path=" << file.path() << '\n';
        std::cout << "version=" << file.version() << " tensors=" << file.tensor_count() << " metadata=" << file.metadata_count()
                  << " alignment=" << file.alignment() << " data_start=" << file.data_start() << " file_size=" << file.file_size() << '\n';
        if (args.metadata) {
            size_t shown = 0;
            for (const auto& [key, value] : file.metadata()) {
                if (shown++ >= args.limit) break;
                std::cout << "meta " << key << '=' << pocket::metadata_value_to_string(value) << '\n';
            }
        }
        if (!args.tensor.empty()) {
            const auto* info = file.find_tensor(args.tensor);
            if (info == nullptr) {
                throw std::runtime_error("tensor not found: " + args.tensor);
            }
            print_tensor(*info);
            return 0;
        }
        const size_t n = std::min(args.limit, file.tensors().size());
        for (size_t i = 0; i < n; ++i) {
            print_tensor(file.tensors()[i]);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
