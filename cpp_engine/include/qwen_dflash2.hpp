#pragma once

#include "qwen_target_head.hpp"
#include "qwen_weights.hpp"
#include "safetensors_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pocket {

class QwenDFlash2WeightMap;

struct QwenDFlash2Config {
    int block_size = 0;
    int conv_group_size = 0;
    int conv_kernel_size = 0;
    int mask_token_id = -1;
    int selector_rank = 0;
    int selector_top_k = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int num_target_layers = 0;
    int max_position_embeddings = 0;
    int sliding_window = 0;
    float rms_norm_eps = 1.0e-6f;
    double rope_theta = 10000000.0;
    bool attention_bias = false;
    bool is_causal = false;
    std::string architecture;
    std::string hidden_act;
    std::string rope_type;
    std::vector<int> target_layer_ids;
    std::vector<std::string> layer_types;

    int draft_tokens() const { return block_size - 1; }

    static QwenDFlash2Config from_directory(const std::string& checkpoint_dir);
    void validate_for_target(uint64_t target_hidden_size,
                             uint64_t target_vocab_size,
                             uint64_t target_layers) const;
};

struct QwenDFlash2ConvWeights {
    QwenTensorRef base_kernel;
    QwenLinearRef kernel_projection;
};

struct QwenDFlash2LayerWeights {
    QwenTensorRef input_layernorm;
    QwenTensorRef post_attention_layernorm;
    QwenLinearRef q_proj;
    QwenLinearRef k_proj;
    QwenLinearRef v_proj;
    QwenLinearRef o_proj;
    QwenTensorRef q_norm;
    QwenTensorRef k_norm;
    QwenLinearRef gate_proj;
    QwenLinearRef up_proj;
    QwenLinearRef down_proj;
    QwenDFlash2ConvWeights attention_conv;
    QwenDFlash2ConvWeights mlp_conv;
};

struct QwenDFlash2Proposal {
    std::vector<int> tokens;
    std::vector<int> candidates;
    std::vector<float> candidate_logits;
};

// Debug-only host view. The callback is invoked synchronously immediately after
// each named stage, before its workspace can be reused. Production runtimes leave
// the callback empty and pay no device-to-host or file-I/O cost.
enum class QwenDFlash2DebugDType {
    F16,
    F32,
    I32,
};

struct QwenDFlash2DebugTensor {
    std::string name;
    QwenDFlash2DebugDType dtype = QwenDFlash2DebugDType::F16;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
};

using QwenDFlash2DebugCallback =
    std::function<void(const QwenDFlash2DebugTensor&)>;

// Device-resident replicated DFlash2 draft runtime. The target feature taps are
// appended as committed position-indexed K/V, while propose() evaluates one
// fixed eight-row diffusion block and returns seven selector-linked drafts.
class QwenDFlash2Runtime {
public:
    QwenDFlash2Runtime(const std::string& checkpoint_dir,
                       const QwenDFlash2Config& config,
                       const QwenDFlash2WeightMap& weights,
                       const QwenDeviceTensor& target_embedding,
                       QwenTargetHeadAdapter target_head,
                       int tp_world, int tp_rank, int device,
                       std::string nccl_id_path, int max_context);
    ~QwenDFlash2Runtime();

    QwenDFlash2Runtime(const QwenDFlash2Runtime&) = delete;
    QwenDFlash2Runtime& operator=(const QwenDFlash2Runtime&) = delete;

    void reset();
    int committed_position() const;
    uint64_t resident_weight_bytes() const;
    uint64_t context_cache_bytes() const;
    uint64_t activation_workspace_bytes() const;

    void set_debug_callback(QwenDFlash2DebugCallback callback);
    void debug_load_target_taps(const std::vector<uint16_t>& target_taps,
                                int rows, int position_offset);
    QwenDFlash2Proposal debug_propose_from_host(
        const std::vector<uint16_t>& target_taps, int context_rows,
        int position_offset, int anchor_token);
    void append_target_taps(const uint16_t* target_taps, int rows,
                            int position_offset);
    void crop_context(int position);
    QwenDFlash2Proposal propose(int anchor_token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class QwenDFlash2WeightMap {
public:
    QwenDFlash2WeightMap(const SafeTensorsIndex& index,
                         const QwenDFlash2Config& config,
                         int tp_world = 1, int tp_rank = 0);

    const QwenTensorRef& projector() const { return projector_; }
    const QwenTensorRef& hidden_norm() const { return hidden_norm_; }
    const QwenTensorRef& final_norm() const { return final_norm_; }
    const QwenTensorRef& predecessor_codebook() const {
        return predecessor_codebook_;
    }
    const QwenTensorRef& successor_codebook() const {
        return successor_codebook_;
    }
    const QwenTensorRef& selector_projection() const {
        return selector_projection_;
    }
    const std::vector<QwenDFlash2LayerWeights>& layers() const {
        return layers_;
    }
    uint64_t local_device_bytes() const { return local_device_bytes_; }
    size_t tensor_count() const { return tensor_count_; }
    // True when the drafter projections are split across the TP ranks instead of
    // replicated. The runtime derives its local head/intermediate counts from the
    // uploaded shapes, so this only reports which layout was chosen.
    bool sharded() const { return shard_; }

private:
    QwenTensorRef require_tensor(const std::string& name,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule = QwenShardRule::Replicated,
                                 int shard_dim = -1) const;
    QwenLinearRef require_linear(const std::string& name,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule = QwenShardRule::Replicated,
                                 int shard_dim = -1) const;
    void record(const QwenTensorRef& ref);

    const SafeTensorsIndex& index_;
    QwenDFlash2Config config_;
    int tp_world_ = 1;
    int tp_rank_ = 0;
    bool shard_ = false;
    QwenTensorRef projector_;
    QwenTensorRef hidden_norm_;
    QwenTensorRef final_norm_;
    QwenTensorRef predecessor_codebook_;
    QwenTensorRef successor_codebook_;
    QwenTensorRef selector_projection_;
    std::vector<QwenDFlash2LayerWeights> layers_;
    uint64_t local_device_bytes_ = 0;
    size_t tensor_count_ = 0;
};

}  // namespace pocket
