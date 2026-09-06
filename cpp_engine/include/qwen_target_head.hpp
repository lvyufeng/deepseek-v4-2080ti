#pragma once

#include "qwen_weights.hpp"

#include <cstdint>

namespace pocket {

// Non-owning view of the target model's vocab-parallel head. External drafters
// use this instead of assuming the checkpoint keeps a dense FP16 lm_head.
struct QwenTargetHeadAdapter {
    QwenLinearKind kind = QwenLinearKind::DenseF16;
    const QwenDeviceTensor* weight = nullptr;
    const QwenDeviceTensor* scale = nullptr;
    int local_vocab = 0;
    int hidden_size = 0;
    uint64_t vocab_start = 0;
    bool cublas_fp32 = false;

    bool valid() const;
    bool project_f16_to_f32(const uint16_t* hidden, float* logits, int rows,
                            void* stream = nullptr) const;
};

}  // namespace pocket
