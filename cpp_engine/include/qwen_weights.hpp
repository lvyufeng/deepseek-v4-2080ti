#pragma once

#include "qwen_config.hpp"
#include "safetensors_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace pocket {

enum class QwenShardRule {
    Replicated,
    ColumnParallel,
    RowParallel,
    ParallelEmbedding,
    ParallelHead,
    PackedQkvColumnParallel,
    PackedConvChannelParallel,
};

enum class QwenLinearKind {
    DenseF16,
    Fp8Block128,
    Fp8Channel,
    NvFp4Group16,
};

struct QwenTensorRef {
    std::string name;
    std::string shard_name;
    // dtype is the checkpoint/storage dtype. device_dtype is the dtype the
    // Qwen loader must materialize on Turing GPUs.
    SafeDType dtype = SafeDType::Unknown;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> full_shape;
    std::vector<uint64_t> local_shape;
    QwenShardRule rule = QwenShardRule::Replicated;
    int shard_dim = -1;
    uint64_t shard_start = 0;
    uint64_t shard_size = 0;
    uint64_t nbytes = 0;
    uint64_t device_nbytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> segments;
    bool found = false;
};

struct QwenHostTensor {
    SafeDType storage_dtype = SafeDType::Unknown;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
};

// Lossless device record for 64 logical compressed-tensors NVFP4 weights.
// Four E4M3 group-16 scales precede the original low-nibble-first E2M1 codes.
struct QwenNvfp4Block64 {
    uint8_t d[4];
    uint8_t qs[32];
};
static_assert(sizeof(QwenNvfp4Block64) == 36,
              "Qwen NVFP4 block64 layout changed");

struct QwenNvfp4HostLinear {
    std::vector<uint64_t> logical_shape;
    std::vector<QwenNvfp4Block64> blocks;
    float weight_global_factor = 0.0f;
    float input_global_scale = 0.0f;
};

// Which checkpoint tensors the text runtime actually reads. Official Qwen3.8
// checkpoints bundle a vision tower in the same shards, so a complete audit has
// to account for every index entry as either mapped, deliberately ignored
// vision weights, or unexpected.
struct QwenCoverage {
    size_t index_tensors = 0;
    size_t mapped_tensors = 0;
    size_t visual_tensors = 0;
    size_t unexpected_tensors = 0;
    // Full (unsharded) bytes of every mapped tensor, i.e. the text share of the
    // checkpoint. Independent of TP world size.
    uint64_t checkpoint_text_bytes = 0;
    // Local resident bytes split by whether a rank holds a full copy or a shard.
    // Replicated bytes are present on every rank, so per-rank totals must not be
    // expected to sum to checkpoint_text_bytes.
    uint64_t replicated_local_bytes = 0;
    uint64_t sharded_local_bytes = 0;
    std::vector<std::string> unexpected_examples;
};

struct QwenLinearKindCounts {
    uint64_t dense_f16 = 0;
    uint64_t fp8_block128 = 0;
    uint64_t fp8_channel = 0;
    uint64_t nvfp4_group16 = 0;
};

struct QwenDeviceTensor {
    void* data = nullptr;
    SafeDType device_dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    // nbytes is the logical extent currently exposed to an operator. capacity
    // is the allocation size, so workspaces can reuse a larger buffer.
    uint64_t nbytes = 0;
    uint64_t capacity = 0;

    ~QwenDeviceTensor();
    QwenDeviceTensor() = default;
    QwenDeviceTensor(const QwenDeviceTensor&) = delete;
    QwenDeviceTensor& operator=(const QwenDeviceTensor&) = delete;
    QwenDeviceTensor(QwenDeviceTensor&& other) noexcept;
    QwenDeviceTensor& operator=(QwenDeviceTensor&& other) noexcept;

    float* f32_data();
    const float* f32_data() const;
    uint16_t* f16_data();
    const uint16_t* f16_data() const;
    uint8_t* fp8_data();
    const uint8_t* fp8_data() const;
    int8_t* int8_data();
    const int8_t* int8_data() const;
    // Raw byte storage for packed caches whose slot mixes several element types,
    // so they cannot claim a single arithmetic dtype. Accepts I8 only.
    uint8_t* byte_data();
    const uint8_t* byte_data() const;
    // Packed NVFP4 block records, which are U8-typed rather than I8.
    uint8_t* u8_data();
    const uint8_t* u8_data() const;
};

struct QwenLinearRef {
    QwenLinearKind kind = QwenLinearKind::DenseF16;
    std::vector<uint64_t> logical_full_shape;
    std::vector<uint64_t> logical_local_shape;
    QwenShardRule rule = QwenShardRule::Replicated;
    int shard_dim = -1;
    QwenTensorRef weight;
    QwenTensorRef scale;
    QwenTensorRef weight_global_scale;
    QwenTensorRef input_global_scale;
    bool has_scale = false;
    bool has_weight_global_scale = false;
    bool has_input_global_scale = false;
};

struct QwenLinearAttentionWeights {
    QwenLinearRef in_proj_qkv;
    QwenLinearRef in_proj_z;
    QwenLinearRef out_proj;
    QwenLinearRef in_proj_a;
    QwenLinearRef in_proj_b;
    QwenTensorRef conv1d;
    QwenTensorRef a_log;
    QwenTensorRef dt_bias;
    QwenTensorRef norm;
};

struct QwenFullAttentionWeights {
    QwenLinearRef q_proj;
    QwenLinearRef k_proj;
    QwenLinearRef v_proj;
    QwenLinearRef o_proj;
    QwenTensorRef q_norm;
    QwenTensorRef k_norm;
};

struct QwenMlpWeights {
    QwenLinearRef gate_proj;
    QwenLinearRef up_proj;
    QwenLinearRef down_proj;
};

struct QwenLayerWeights {
    QwenTensorRef input_layernorm;
    QwenTensorRef post_attention_layernorm;
    QwenLinearAttentionWeights linear_attention;
    QwenFullAttentionWeights full_attention;
    QwenMlpWeights mlp;
};

struct QwenMtpWeights {
    QwenTensorRef pre_fc_norm_embedding;
    QwenTensorRef pre_fc_norm_hidden;
    QwenLinearRef fc;
    QwenLayerWeights layer;
    QwenTensorRef norm;
    bool found = false;
};

class QwenWeightMap {
public:
    QwenWeightMap(const SafeTensorsIndex& index, const QwenConfig& config,
                  int tp_world = 1, int tp_rank = 0);

    const QwenTensorRef& embed_tokens() const { return embed_tokens_; }
    const QwenTensorRef& final_norm() const { return final_norm_; }
    const QwenLinearRef& lm_head() const { return lm_head_; }
    const std::vector<QwenLayerWeights>& layers() const { return layers_; }
    const QwenMtpWeights& mtp() const { return mtp_; }
    const QwenConfig& config() const { return config_; }
    int tp_world() const { return tp_world_; }
    int tp_rank() const { return tp_rank_; }

    uint64_t local_weight_bytes() const { return local_weight_bytes_; }
    uint64_t local_scale_bytes() const { return local_scale_bytes_; }
    uint64_t host_global_metadata_bytes() const {
        return host_global_metadata_bytes_;
    }
    // Counts every linear descriptor present in the checkpoint map, including
    // optional MTP weights whether or not the runtime enables MTP residency.
    const QwenLinearKindCounts& checkpoint_linear_kind_counts() const {
        return checkpoint_linear_kind_counts_;
    }
    size_t tensor_count() const { return tensor_count_; }

    // Classify every checkpoint index entry against what this map claims.
    QwenCoverage coverage() const;
    // Throw unless every index entry is either mapped or a vision tensor.
    void require_full_coverage() const;

private:
    QwenTensorRef require_tensor(const std::string& name,
                                 SafeDType dtype,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule = QwenShardRule::Replicated,
                                 int shard_dim = -1) const;
    QwenLinearRef require_linear(const std::string& name,
                                 const std::vector<uint64_t>& shape,
                                 QwenShardRule rule,
                                 int shard_dim) const;
    void record(const QwenTensorRef& ref, bool scale);
    void record_linear(const QwenLinearRef& ref);
    void claim(const QwenTensorRef& ref);

    const SafeTensorsIndex& index_;
    QwenConfig config_;
    int tp_world_ = 1;
    int tp_rank_ = 0;
    QwenTensorRef embed_tokens_;
    QwenTensorRef final_norm_;
    QwenLinearRef lm_head_;
    std::vector<QwenLayerWeights> layers_;
    QwenMtpWeights mtp_;
    uint64_t local_weight_bytes_ = 0;
    uint64_t local_scale_bytes_ = 0;
    uint64_t host_global_metadata_bytes_ = 0;
    QwenLinearKindCounts checkpoint_linear_kind_counts_;
    size_t tensor_count_ = 0;
    std::set<std::string> claimed_tensors_;
    uint64_t checkpoint_text_bytes_ = 0;
    uint64_t replicated_local_bytes_ = 0;
    uint64_t sharded_local_bytes_ = 0;
};

const char* qwen_shard_rule_name(QwenShardRule rule);
const char* qwen_linear_kind_name(QwenLinearKind kind);
// True for vision-tower tensors bundled into an official multimodal checkpoint.
bool qwen_is_visual_tensor(const std::string& name);

// Qwen checkpoint tensors retain their source dtype for validation. Storage
// dtype is what the checkpoint holds; device dtype is what the backend keeps
// resident. This is the CUDA/SM75 policy: Turing has no native BF16 arithmetic,
// so every BF16 tensor -- whether it comes from the official BF16 checkpoint or
// from the BF16 scale metadata of an FP8 checkpoint -- is converted to IEEE FP16
// at the upload boundary, while FP8 and NVFP4 codes stay compressed. A native
// BF16 backend must supply its own policy here rather than inherit this one.
SafeDType qwen_device_dtype(SafeDType storage_dtype);
uint16_t qwen_bf16_to_fp16_bits(uint16_t bits);
void qwen_convert_bf16_to_fp16(const uint16_t* src, uint16_t* dst, size_t count);

// Materialize a local tensor from its mmap'd source shard. The output is ready
// for upload: BF16 storage is converted to FP16 and packed/row slices are copied
// without expanding FP8 weights. Both supported backends want FP16 here, for
// unrelated reasons; see qwen_device_dtype in core/qwen_weight_map.cpp.
QwenHostTensor qwen_materialize_host_tensor(const SafeTensorsIndex& index,
                                            const QwenTensorRef& ref);

// True for the RMSNorm affine weights that the Qwen3.5 runtime applies as
// (1 + weight): the layer input/post norms, the final norm, and the per-head
// q/k norms. False for linear_attn.norm.weight, which the gated norm applies
// directly, and for everything that is not a norm weight.
bool qwen_is_one_plus_norm_gamma(const std::string& name);

// Backend policy hook applied between materialization and upload.
//
// On CUDA this is a no-op: the kernels carry the (1 + weight) convention in the
// arithmetic. On Ascend the normalization comes from aclnnRmsNorm, which applies
// gamma directly and rejects an FP32 gamma against FP16 activations, so the +1 is
// folded into the FP16 weight here instead. Doing it once at load time rather than
// as a per-layer fixup op keeps 64 layers x 3 norms off the hot path.
//
// Folding in FP16 loses precision relative to the CUDA path, which adds 1.0 in
// FP32 at use time. Gamma values sit near zero where FP16 has ~2^-24 resolution
// but 1+gamma sits near one where it has 2^-11, so the folded value is the FP16
// neighbour of the exact sum. See test_qwen_ascend_norm_gamma for the bound.
void qwen_apply_norm_gamma_policy(const QwenTensorRef& ref, QwenHostTensor& host);
QwenNvfp4HostLinear qwen_materialize_nvfp4_host_linear(
    const SafeTensorsIndex& index, const QwenLinearRef& ref);
QwenDeviceTensor qwen_upload_tensor(const SafeTensorsIndex& index,
                                         const QwenTensorRef& ref,
                                         void* stream = nullptr);
QwenDeviceTensor qwen_upload_nvfp4_linear_cuda(
    const SafeTensorsIndex& index, const QwenLinearRef& ref,
    float* weight_global_factor, float* input_global_scale,
    void* stream = nullptr);

}  // namespace pocket
