#include "gguf_reader.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        check(pocket::ggml_type_name(29) == "iq1_m", "type 29 should be named iq1_m");
        check(pocket::ggml_type_to_dtype(29) == pocket::DType::IQ1_M, "type 29 should map to DType::IQ1_M");
        check(pocket::dtype_name(pocket::DType::IQ1_M) == "iq1_m", "DType::IQ1_M should stringify as iq1_m");

        check(pocket::ggml_tensor_nbytes(29, {256}) == 56, "one IQ1_M block should be 56 bytes");
        check(pocket::ggml_tensor_nbytes(29, {257}) == 112, "IQ1_M byte count should round up to blocks");
        check(pocket::ggml_tensor_nbytes(29, {4096, 2048, 256}) == 469762048ULL,
              "DeepSeek-V4 routed IQ1_M tensor should be 448 MiB");

        std::cout << "[PASS] gguf_iq1m_reader parser-only checks\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
