#pragma once

#include "qwen_target_head.hpp"
#include "qwen_weights.hpp"
#include "safetensors_reader.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pocket {

class QwenDSparkWeightMap;

struct QwenDSparkRopeConfig {
    double theta = 10000000.0;
    double factor = 32.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
    int original_max_positions = 8192;
    double attention_factor = 0.0;
};

struct QwenDSparkConfig {
    int block_size = 0;
    int mask_token_id = -1;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int num_target_layers = 0;
    int max_position_embeddings = 0;
    int markov_rank = 0;
    float rms_norm_eps = 1.0e-6f;
    bool attention_bias = false;
    bool confidence_head_with_markov = false;
    bool enable_confidence_head = false;
    std::string hidden_act;
    std::string markov_head_type;
    std::string attention_mode;
    std::string projector_type;
    std::vector<int> target_layer_ids;
    std::vector<std::string> layer_types;
    QwenDSparkRopeConfig rope;

    static QwenDSparkConfig from_directory(const std::string& checkpoint_dir);
    void validate_for_target(uint64_t target_hidden_size,
                             uint64_t target_vocab_size,
                             uint64_t target_layers) const;
};

struct QwenDSparkLayerWeights {
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
};

struct QwenDSparkProposal {
    std::vector<int> tokens;
    std::vector<float> confidences;
    std::vector<float> top_logits;
};

// Device-resident replicated 5-layer DSpark drafter. Target feature taps are
// appended as committed rows, then propose() generates the fixed seven-token
// block without mutating the committed context.
class QwenDSparkRuntime {
public:
    QwenDSparkRuntime(const std::string& checkpoint_dir,
                      const QwenDSparkConfig& config,
                      const QwenDSparkWeightMap& weights,
                      const QwenDeviceTensor& target_embedding,
                      QwenTargetHeadAdapter target_head,
                      int tp_world, int tp_rank, int device,
                      std::string nccl_id_path, int max_context);
    ~QwenDSparkRuntime();

    QwenDSparkRuntime(const QwenDSparkRuntime&) = delete;
    QwenDSparkRuntime& operator=(const QwenDSparkRuntime&) = delete;

    void reset();
    int committed_position() const;
    uint64_t resident_weight_bytes() const;
    uint64_t context_cache_bytes() const;
    uint64_t activation_workspace_bytes() const;

    // target_taps is [rows, target_layer_ids.size() * hidden_size], containing
    // raw target post-layer hidden states in target_layer_ids order.
    void append_target_taps(const uint16_t* target_taps, int rows,
                            int position_offset);
    void crop_context(int position);
    QwenDSparkProposal propose(int anchor_token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class QwenDSparkWeightMap {
public:
    QwenDSparkWeightMap(const SafeTensorsIndex& index,
                        const QwenDSparkConfig& config,
                        int tp_world = 1, int tp_rank = 0);

    const QwenTensorRef& projector() const { return projector_; }
    const QwenTensorRef& hidden_norm() const { return hidden_norm_; }
    const QwenTensorRef& final_norm() const { return final_norm_; }
    const QwenTensorRef& markov_w1() const { return markov_w1_; }
    const QwenTensorRef& markov_w2() const { return markov_w2_; }
    const QwenTensorRef& confidence_weight() const { return confidence_weight_; }
    const QwenTensorRef& confidence_bias() const { return confidence_bias_; }
    const std::vector<QwenDSparkLayerWeights>& layers() const { return layers_; }
    uint64_t local_device_bytes() const { return local_device_bytes_; }
    size_t tensor_count() const { return tensor_count_; }

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
    QwenDSparkConfig config_;
    int tp_world_ = 1;
    int tp_rank_ = 0;
    QwenTensorRef projector_;
    QwenTensorRef hidden_norm_;
    QwenTensorRef final_norm_;
    QwenTensorRef markov_w1_;
    QwenTensorRef markov_w2_;
    QwenTensorRef confidence_weight_;
    QwenTensorRef confidence_bias_;
    std::vector<QwenDSparkLayerWeights> layers_;
    uint64_t local_device_bytes_ = 0;
    size_t tensor_count_ = 0;
};

std::vector<float> qwen_dspark_yarn_inv_freqs(
    const QwenDSparkConfig& config);

}  // namespace pocket
