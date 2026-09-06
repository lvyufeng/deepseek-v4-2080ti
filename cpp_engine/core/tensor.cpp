#include "tensor.hpp"

#include <numeric>

namespace pocket {

std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::F8_E4M3: return "f8_e4m3";
        case DType::Q8_0: return "q8_0";
        case DType::Q2_K: return "q2_k";
        case DType::IQ2_XXS: return "iq2_xxs";
        case DType::IQ1_M: return "iq1_m";
        case DType::Unknown: return "unknown";
    }
    return "unknown";
}

uint64_t tensor_element_count(const std::vector<uint64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    uint64_t total = 1;
    for (uint64_t dim : shape) {
        total *= dim;
    }
    return total;
}

}  // namespace pocket
