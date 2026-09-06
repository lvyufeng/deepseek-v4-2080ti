#pragma once

#include "tensor.hpp"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace pocket {

enum class SafeDType {
    BF16,
    F16,
    F32,
    I64,
    I8,
    U8,
    F8_E4M3,
    F8_E8M0,
    Unknown,
};

struct SafeTensorInfo {
    std::string name;
    SafeDType dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
    uint64_t absolute_begin = 0;
    uint64_t nbytes = 0;
};

struct SafeFp4TensorPair {
    std::string weight_name;
    std::string scale_name;
    std::string shard_name;
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t packed_cols = 0;
    uint64_t scale_cols = 0;
};

class SafeTensorsIndex {
public:
    explicit SafeTensorsIndex(const std::string& ckpt_dir);
    // Build an index for a checkpoint exported as one unsharded file. This is
    // deliberately explicit so existing sharded-checkpoint error behavior stays
    // unchanged for callers that require model.safetensors.index.json.
    static SafeTensorsIndex from_single_file(const std::string& ckpt_dir,
                                             const std::string& filename =
                                                 "model.safetensors");

    const std::string& ckpt_dir() const { return ckpt_dir_; }
    uint64_t total_size() const { return total_size_; }
    size_t tensor_count() const { return weight_map_.size(); }
    size_t shard_count() const { return shards_.size(); }
    const std::map<std::string, std::string>& weight_map() const { return weight_map_; }
    const std::set<std::string>& shards() const { return shards_; }

    const std::string* shard_for_tensor(const std::string& tensor) const;
    std::string shard_path(const std::string& shard) const;
    std::vector<std::string> grep_tensors(const std::string& needle, size_t limit = 50) const;

private:
    SafeTensorsIndex() = default;

    std::string ckpt_dir_;
    uint64_t total_size_ = 0;
    std::map<std::string, std::string> weight_map_;
    std::set<std::string> shards_;
};

class SafeTensorsShard {
public:
    explicit SafeTensorsShard(const std::string& path);
    ~SafeTensorsShard();
    SafeTensorsShard(const SafeTensorsShard&) = delete;
    SafeTensorsShard& operator=(const SafeTensorsShard&) = delete;
    SafeTensorsShard(SafeTensorsShard&& other) noexcept;
    SafeTensorsShard& operator=(SafeTensorsShard&& other) noexcept;

    const std::string& path() const { return path_; }
    uint64_t file_size() const { return size_; }
    uint64_t header_len() const { return header_len_; }
    uint64_t data_start() const { return data_start_; }
    const std::map<std::string, SafeTensorInfo>& tensors() const { return tensors_; }
    const SafeTensorInfo* find_tensor(const std::string& name) const;
    const uint8_t* tensor_data(const SafeTensorInfo& info) const;

private:
    void open(const std::string& path);
    void close();
    void parse();
    const uint8_t* bytes() const { return static_cast<const uint8_t*>(data_); }

    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    uint64_t size_ = 0;
    uint64_t header_len_ = 0;
    uint64_t data_start_ = 0;
    std::map<std::string, SafeTensorInfo> tensors_;
};

std::string safe_dtype_name(SafeDType dtype);
SafeDType safe_dtype_from_string(const std::string& dtype);
uint64_t safe_dtype_size(SafeDType dtype);
uint64_t safe_tensor_numel(const std::vector<uint64_t>& shape);
SafeFp4TensorPair resolve_fp4_tensor_pair(const SafeTensorsIndex& index, const SafeTensorsShard& shard, const std::string& weight_name);

}  // namespace pocket
