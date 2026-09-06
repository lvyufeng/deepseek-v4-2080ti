#include "gguf_reader.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_reader <model.gguf>\n";
        return 2;
    }
    try {
        pocket::GGUFFile file(argv[1]);
        if (file.version() == 0 || file.tensor_count() == 0 || file.metadata_count() == 0) {
            throw std::runtime_error("invalid parsed GGUF summary");
        }
        if (file.data_start() == 0 || file.data_start() >= file.file_size()) {
            throw std::runtime_error("invalid data_start");
        }
        for (const auto& tensor : file.tensors()) {
            if (tensor.absolute_offset < file.data_start()) {
                throw std::runtime_error("tensor absolute offset before data_start: " + tensor.name);
            }
            if (tensor.nbytes == 0) {
                throw std::runtime_error("unknown or zero tensor bytes: " + tensor.name);
            }
        }
        std::cout << "[PASS] gguf_reader tensors=" << file.tensor_count() << " metadata=" << file.metadata_count() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
