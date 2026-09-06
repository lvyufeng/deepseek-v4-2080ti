#pragma once

#include "tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace pocket {

using MetadataValue = std::variant<int64_t, uint64_t, double, bool, std::string, std::vector<int64_t>, std::vector<uint64_t>, std::vector<double>, std::vector<bool>, std::vector<std::string>>;

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t ggml_type = 0;
    DType dtype = DType::Unknown;
    uint64_t offset = 0;
    uint64_t absolute_offset = 0;
    uint64_t nbytes = 0;
};

class GGUFFile {
public:
    GGUFFile() = default;
    explicit GGUFFile(const std::string& path);
    ~GGUFFile();

    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;
    GGUFFile(GGUFFile&& other) noexcept;
    GGUFFile& operator=(GGUFFile&& other) noexcept;

    void open(const std::string& path);
    void close();

    const std::string& path() const { return path_; }
    uint32_t version() const { return version_; }
    uint64_t tensor_count() const { return tensors_.size(); }
    uint64_t metadata_count() const { return metadata_.size(); }
    uint64_t alignment() const { return alignment_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t file_size() const { return size_; }

    const std::map<std::string, MetadataValue>& metadata() const { return metadata_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }
    const GGUFTensorInfo* find_tensor(const std::string& name) const;
    TensorView tensor_view(const GGUFTensorInfo& info) const;

    std::optional<uint64_t> metadata_u64(const std::string& key) const;
    std::optional<int64_t> metadata_i64(const std::string& key) const;
    std::optional<double> metadata_f64(const std::string& key) const;
    std::optional<bool> metadata_bool(const std::string& key) const;
    std::optional<std::string> metadata_string(const std::string& key) const;
    std::vector<uint64_t> metadata_u64_array(const std::string& key) const;
    std::vector<double> metadata_f64_array(const std::string& key) const;

    // Raw mmap base, for CPU-side reads of tensor bytes only.
    // DO NOT cudaHostRegister this whole file-backed region: pinning an
    // 80GB+ mmap poisons Linux dirty-page accounting and stalls writeback
    // system-wide (balance_dirty_pages). Expert H2D must copy from these
    // pageable pointers directly, or via a small bounded pinned staging buffer.
    const uint8_t* bytes() const { return static_cast<const uint8_t*>(data_); }

private:
    void parse();
    uint8_t read_u8(size_t& cursor) const;
    uint16_t read_u16(size_t& cursor) const;
    uint32_t read_u32(size_t& cursor) const;
    uint64_t read_u64(size_t& cursor) const;
    int8_t read_i8(size_t& cursor) const;
    int16_t read_i16(size_t& cursor) const;
    int32_t read_i32(size_t& cursor) const;
    int64_t read_i64(size_t& cursor) const;
    float read_f32(size_t& cursor) const;
    double read_f64(size_t& cursor) const;
    std::string read_string(size_t& cursor) const;
    MetadataValue read_metadata_value(size_t& cursor, uint32_t value_type) const;

    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    uint64_t size_ = 0;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    uint64_t data_start_ = 0;
    std::map<std::string, MetadataValue> metadata_;
    std::vector<GGUFTensorInfo> tensors_;
};

std::string ggml_type_name(uint32_t ggml_type);
DType ggml_type_to_dtype(uint32_t ggml_type);
uint64_t ggml_tensor_nbytes(uint32_t ggml_type, const std::vector<uint64_t>& shape);
std::string metadata_value_to_string(const MetadataValue& value, size_t max_items = 8);

}  // namespace pocket
