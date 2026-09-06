#pragma once

#include "qwen_dflash2.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pocket_test {

constexpr uint32_t kDFlash2TensorMagic = 0x32464451u;  // "QDF2"
constexpr uint32_t kDFlash2TensorVersion = 1;

struct DFlash2TensorFile {
    int32_t position_offset = 0;
    int32_t anchor_token = 0;
    std::vector<pocket::QwenDFlash2DebugTensor> tensors;
};

inline uint32_t dtype_code(pocket::QwenDFlash2DebugDType dtype) {
    switch (dtype) {
        case pocket::QwenDFlash2DebugDType::F16: return 1;
        case pocket::QwenDFlash2DebugDType::F32: return 2;
        case pocket::QwenDFlash2DebugDType::I32: return 3;
    }
    throw std::runtime_error("unknown DFlash2 debug dtype");
}

inline pocket::QwenDFlash2DebugDType debug_dtype(uint32_t code) {
    switch (code) {
        case 1: return pocket::QwenDFlash2DebugDType::F16;
        case 2: return pocket::QwenDFlash2DebugDType::F32;
        case 3: return pocket::QwenDFlash2DebugDType::I32;
        default: throw std::runtime_error("unknown DFlash2 tensor dtype code");
    }
}

inline size_t dtype_size(pocket::QwenDFlash2DebugDType dtype) {
    switch (dtype) {
        case pocket::QwenDFlash2DebugDType::F16: return 2;
        case pocket::QwenDFlash2DebugDType::F32: return 4;
        case pocket::QwenDFlash2DebugDType::I32: return 4;
    }
    throw std::runtime_error("unknown DFlash2 debug dtype");
}

template <typename T>
inline void write_scalar(std::ofstream& output, T value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!output) throw std::runtime_error("failed writing DFlash2 tensor file");
}

template <typename T>
inline T read_scalar(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("truncated DFlash2 tensor file");
    return value;
}

inline uint64_t element_count(const std::vector<uint64_t>& shape) {
    uint64_t count = 1;
    for (uint64_t dimension : shape) {
        if (dimension == 0 || count > std::numeric_limits<uint64_t>::max() / dimension) {
            throw std::runtime_error("invalid DFlash2 tensor shape");
        }
        count *= dimension;
    }
    return count;
}

inline void write_tensor_file(const std::string& path,
                              const DFlash2TensorFile& file) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create DFlash2 tensor file: " + path);
    write_scalar(output, kDFlash2TensorMagic);
    write_scalar(output, kDFlash2TensorVersion);
    write_scalar(output, file.position_offset);
    write_scalar(output, file.anchor_token);
    write_scalar(output, static_cast<uint32_t>(file.tensors.size()));
    for (const auto& tensor : file.tensors) {
        if (tensor.name.empty() || tensor.name.size() > 4096 || tensor.shape.empty() ||
            tensor.shape.size() > 8) {
            throw std::runtime_error("invalid DFlash2 tensor metadata: " + tensor.name);
        }
        const uint64_t expected = element_count(tensor.shape) * dtype_size(tensor.dtype);
        if (tensor.bytes.size() != expected) {
            throw std::runtime_error("invalid DFlash2 tensor extent: " + tensor.name);
        }
        write_scalar(output, static_cast<uint32_t>(tensor.name.size()));
        write_scalar(output, dtype_code(tensor.dtype));
        write_scalar(output, static_cast<uint32_t>(tensor.shape.size()));
        write_scalar(output, static_cast<uint32_t>(0));
        write_scalar(output, static_cast<uint64_t>(tensor.bytes.size()));
        output.write(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
        for (uint64_t dimension : tensor.shape) write_scalar(output, dimension);
        output.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                     static_cast<std::streamsize>(tensor.bytes.size()));
        if (!output) throw std::runtime_error("failed writing DFlash2 tensor payload");
    }
}

inline DFlash2TensorFile read_tensor_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open DFlash2 tensor file: " + path);
    if (read_scalar<uint32_t>(input) != kDFlash2TensorMagic ||
        read_scalar<uint32_t>(input) != kDFlash2TensorVersion) {
        throw std::runtime_error("unsupported DFlash2 tensor file: " + path);
    }
    DFlash2TensorFile file;
    file.position_offset = read_scalar<int32_t>(input);
    file.anchor_token = read_scalar<int32_t>(input);
    const uint32_t count = read_scalar<uint32_t>(input);
    if (count > 512) throw std::runtime_error("too many tensors in DFlash2 tensor file");
    file.tensors.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t name_size = read_scalar<uint32_t>(input);
        const auto dtype = debug_dtype(read_scalar<uint32_t>(input));
        const uint32_t rank = read_scalar<uint32_t>(input);
        (void)read_scalar<uint32_t>(input);
        const uint64_t byte_size = read_scalar<uint64_t>(input);
        if (name_size == 0 || name_size > 4096 || rank == 0 || rank > 8 ||
            byte_size > (1ull << 34)) {
            throw std::runtime_error("invalid DFlash2 tensor record");
        }
        pocket::QwenDFlash2DebugTensor tensor;
        tensor.name.resize(name_size);
        input.read(tensor.name.data(), name_size);
        tensor.dtype = dtype;
        tensor.shape.resize(rank);
        for (uint64_t& dimension : tensor.shape) {
            dimension = read_scalar<uint64_t>(input);
        }
        const uint64_t expected = element_count(tensor.shape) * dtype_size(dtype);
        if (byte_size != expected) {
            throw std::runtime_error("DFlash2 tensor byte extent mismatch: " + tensor.name);
        }
        tensor.bytes.resize(static_cast<size_t>(byte_size));
        input.read(reinterpret_cast<char*>(tensor.bytes.data()),
                   static_cast<std::streamsize>(tensor.bytes.size()));
        if (!input) throw std::runtime_error("truncated DFlash2 tensor payload");
        file.tensors.push_back(std::move(tensor));
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("trailing data in DFlash2 tensor file");
    }
    return file;
}

inline const pocket::QwenDFlash2DebugTensor& require_tensor(
    const DFlash2TensorFile& file, const std::string& name) {
    const pocket::QwenDFlash2DebugTensor* found = nullptr;
    for (const auto& tensor : file.tensors) {
        if (tensor.name != name) continue;
        if (found != nullptr) throw std::runtime_error("duplicate DFlash2 tensor: " + name);
        found = &tensor;
    }
    if (found == nullptr) throw std::runtime_error("missing DFlash2 tensor: " + name);
    return *found;
}

}  // namespace pocket_test
