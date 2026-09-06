#pragma once

#include "gguf_reader.hpp"
#include "safetensors_reader.hpp"
#include "tensor.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pocket {

// Read-only view of one tensor as raw bytes plus typed metadata. Borrowed from
// either the safetensors shard mmap or the GGUF file mmap; the caller must not
// outlive the owning WeightSource.
struct WeightView {
    std::string name;          // internal canonical name, e.g. "layers.0.attn.wq_a.weight"
    DType dtype = DType::Unknown;
    std::vector<uint64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t nbytes = 0;
    bool found = false;
};

// Abstracts the difference between (multi-shard FP4 safetensors) and (single
// GGUF Q2/Q8/BF16) checkpoints. Lifetime: same as SafeForwardContext.
class WeightSource {
public:
    virtual ~WeightSource() = default;

    // Returns true if the source has a tensor with the given internal name.
    virtual bool has(const std::string& name) const = 0;

    // Get the tensor by internal name. Returns a view with found=false on miss.
    virtual WeightView get(const std::string& name) const = 0;

    // Same as get() but the tensor is required; throws on miss.
    WeightView require(const std::string& name) const;

    // For 3D routed-expert tensors that pack n_experts experts into one tensor
    // (GGUF format), return just one expert's slice. For per-expert safetensors
    // tensors the default impl just calls get() with the per-expert name.
    virtual WeightView get_expert(const std::string& routed_name_3d,
                                  const std::string& per_expert_name,
                                  int expert_id) const = 0;

    // Returns the quantization "family" the source uses. Drives format-specific
    // weight loading paths in SafeForwardContext.
    enum class Format { Safetensors_FP4, GGUF_Q2 };
    virtual Format format() const = 0;
};

// Wraps the existing SafeTensorsIndex + lazy SafeTensorsShard opens.
class SafetensorsWeightSource : public WeightSource {
public:
    explicit SafetensorsWeightSource(const std::string& ckpt_dir);

    bool has(const std::string& name) const override;
    WeightView get(const std::string& name) const override;
    WeightView get_expert(const std::string& routed_name_3d,
                          const std::string& per_expert_name,
                          int expert_id) const override;
    Format format() const override { return Format::Safetensors_FP4; }

    const SafeTensorsIndex& index() const { return index_; }

private:
    SafeTensorsShard& shard_for(const std::string& tensor_name) const;

    SafeTensorsIndex index_;
    mutable std::map<std::string, std::unique_ptr<SafeTensorsShard>> shards_;
};

// Wraps a GGUFFile and translates internal canonical names to GGUF tensor names
// via the DeepSeek-V4 mapping (mirrors src/gguf/ds4_mapping.py).
class GGUFWeightSource : public WeightSource {
public:
    explicit GGUFWeightSource(const std::string& path);

    bool has(const std::string& name) const override;
    WeightView get(const std::string& name) const override;
    WeightView get_expert(const std::string& routed_name_3d,
                          const std::string& per_expert_name,
                          int expert_id) const override;
    Format format() const override { return Format::GGUF_Q2; }

    const GGUFFile& file() const { return file_; }

private:
    enum class Transform {
        Direct,
        Transpose2D,          // [a, b] -> [b, a]
        RoutedExpertTranspose // [a, b, n_experts] -> per-expert [b, a]
    };

    struct Mapping {
        std::string gguf_name;
        Transform transform = Transform::Direct;
    };

    void build_mapping(int n_layers, int n_hash_layers, int n_experts);

    GGUFFile file_;
    std::map<std::string, Mapping> by_internal_name_;
};

// Returns "ckpt_path looks like a GGUF file" (suffix-based).
bool is_gguf_path(const std::string& path);

// Convenience factory: open a SafetensorsWeightSource or GGUFWeightSource based
// on whether the path ends in ".gguf".
std::unique_ptr<WeightSource> open_weight_source(const std::string& path);

}  // namespace pocket
