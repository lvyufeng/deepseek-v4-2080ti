#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pocket {

enum class DType {
    F32,
    F16,
    BF16,
    F8_E4M3,
    Q8_0,
    Q2_K,
    IQ2_XXS,
    IQ1_M,
    Unknown,
};

struct TensorView {
    std::string name;
    DType dtype = DType::Unknown;
    std::vector<uint64_t> shape;
    uint64_t offset = 0;
    uint64_t absolute_offset = 0;
    uint64_t nbytes = 0;
    const uint8_t* data = nullptr;
};

std::string dtype_name(DType dtype);
uint64_t tensor_element_count(const std::vector<uint64_t>& shape);

}  // namespace pocket
