#include "safetensors_reader.hpp"

#include "json_lite.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pocket {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t read_le_u64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(p[i]) << (8 * i);
    return value;
}

}  // namespace

SafeTensorsIndex::SafeTensorsIndex(const std::string& ckpt_dir) : ckpt_dir_(ckpt_dir) {
    const std::string path = ckpt_dir + "/model.safetensors.index.json";
    JsonValue root_value = parse_json(read_file(path));
    const JsonObject& obj = root_value.object();
    if (const JsonValue* metadata = object_get(obj, "metadata")) {
        const JsonObject& meta = metadata->object();
        if (const JsonValue* total = object_get(meta, "total_size")) {
            total_size_ = static_cast<uint64_t>(total->number());
        }
    }
    const JsonObject& weight_map = object_get(obj, "weight_map")->object();
    for (const auto& [name, shard_value] : weight_map) {
        const std::string shard = shard_value.string();
        weight_map_[name] = shard;
        shards_.insert(shard);
    }
}

SafeTensorsIndex SafeTensorsIndex::from_single_file(
    const std::string& ckpt_dir, const std::string& filename) {
    if (filename.empty()) {
        throw std::runtime_error("safetensors filename must not be empty");
    }
    SafeTensorsIndex index;
    index.ckpt_dir_ = ckpt_dir;
    SafeTensorsShard shard(ckpt_dir + "/" + filename);
    index.total_size_ = shard.file_size();
    index.shards_.insert(filename);
    for (const auto& [name, _info] : shard.tensors()) {
        index.weight_map_[name] = filename;
    }
    return index;
}

const std::string* SafeTensorsIndex::shard_for_tensor(const std::string& tensor) const {
    auto it = weight_map_.find(tensor);
    return it == weight_map_.end() ? nullptr : &it->second;
}

std::string SafeTensorsIndex::shard_path(const std::string& shard) const {
    return ckpt_dir_ + "/" + shard;
}

std::vector<std::string> SafeTensorsIndex::grep_tensors(const std::string& needle, size_t limit) const {
    std::vector<std::string> out;
    for (const auto& [name, _shard] : weight_map_) {
        if (name.find(needle) != std::string::npos) {
            out.push_back(name);
            if (out.size() >= limit) break;
        }
    }
    return out;
}

SafeTensorsShard::SafeTensorsShard(const std::string& path) { open(path); }
SafeTensorsShard::~SafeTensorsShard() { close(); }
SafeTensorsShard::SafeTensorsShard(SafeTensorsShard&& other) noexcept { *this = std::move(other); }
SafeTensorsShard& SafeTensorsShard::operator=(SafeTensorsShard&& other) noexcept {
    if (this == &other) return *this;
    close();
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    header_len_ = other.header_len_;
    data_start_ = other.data_start_;
    tensors_ = std::move(other.tensors_);
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void SafeTensorsShard::open(const std::string& path) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("failed to open safetensors shard: " + path);
    struct stat st;
    if (fstat(fd_, &st) != 0) throw std::runtime_error("failed to stat safetensors shard: " + path);
    size_ = static_cast<uint64_t>(st.st_size);
    if (size_ < 8) throw std::runtime_error("safetensors shard too small: " + path);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        throw std::runtime_error("failed to mmap safetensors shard: " + path);
    }
    parse();
}

void SafeTensorsShard::close() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    header_len_ = 0;
    data_start_ = 0;
    tensors_.clear();
}

void SafeTensorsShard::parse() {
    header_len_ = read_le_u64(bytes());
    data_start_ = 8 + header_len_;
    if (data_start_ > size_) throw std::runtime_error("safetensors header past end: " + path_);
    std::string header(reinterpret_cast<const char*>(bytes() + 8), static_cast<size_t>(header_len_));
    JsonValue root_value = parse_json(header);
    const JsonObject& root = root_value.object();
    for (const auto& [name, value] : root) {
        if (name == "__metadata__") continue;
        const JsonObject& obj = value.object();
        SafeTensorInfo info;
        info.name = name;
        info.dtype = safe_dtype_from_string(json_required_string(obj, "dtype"));
        info.shape = json_required_u64_array(obj, "shape");
        std::vector<uint64_t> offsets = json_required_u64_array(obj, "data_offsets");
        if (offsets.size() != 2 || offsets[1] < offsets[0]) throw std::runtime_error("bad data_offsets for " + name);
        info.data_begin = offsets[0];
        info.data_end = offsets[1];
        info.absolute_begin = data_start_ + info.data_begin;
        info.nbytes = info.data_end - info.data_begin;
        if (info.absolute_begin + info.nbytes > size_) throw std::runtime_error("tensor past end: " + name);
        const uint64_t expected = safe_tensor_numel(info.shape) * safe_dtype_size(info.dtype);
        if (expected != info.nbytes) {
            std::string shape;
            for (size_t i = 0; i < info.shape.size(); ++i) {
                if (i) shape += ",";
                shape += std::to_string(info.shape[i]);
            }
            throw std::runtime_error(
                "tensor byte mismatch for " + name + " dtype=" + safe_dtype_name(info.dtype) + " shape=[" + shape + "] expected=" +
                std::to_string(expected) + " actual=" + std::to_string(info.nbytes));
        }
        tensors_[name] = std::move(info);
    }
}

const SafeTensorInfo* SafeTensorsShard::find_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const uint8_t* SafeTensorsShard::tensor_data(const SafeTensorInfo& info) const {
    return bytes() + info.absolute_begin;
}

std::string safe_dtype_name(SafeDType dtype) {
    switch (dtype) {
        case SafeDType::BF16: return "BF16";
        case SafeDType::F16: return "F16";
        case SafeDType::F32: return "F32";
        case SafeDType::I64: return "I64";
        case SafeDType::I8: return "I8";
        case SafeDType::U8: return "U8";
        case SafeDType::F8_E4M3: return "F8_E4M3";
        case SafeDType::F8_E8M0: return "F8_E8M0";
        case SafeDType::Unknown: return "Unknown";
    }
    return "Unknown";
}

SafeDType safe_dtype_from_string(const std::string& dtype) {
    if (dtype == "BF16") return SafeDType::BF16;
    if (dtype == "F16") return SafeDType::F16;
    if (dtype == "F32") return SafeDType::F32;
    if (dtype == "I64") return SafeDType::I64;
    if (dtype == "I8") return SafeDType::I8;
    if (dtype == "U8") return SafeDType::U8;
    if (dtype == "F8_E4M3") return SafeDType::F8_E4M3;
    if (dtype == "F8_E8M0") return SafeDType::F8_E8M0;
    return SafeDType::Unknown;
}

uint64_t safe_dtype_size(SafeDType dtype) {
    switch (dtype) {
        case SafeDType::BF16: return 2;
        case SafeDType::F16: return 2;
        case SafeDType::F32: return 4;
        case SafeDType::I64: return 8;
        case SafeDType::I8: return 1;
        case SafeDType::U8: return 1;
        case SafeDType::F8_E4M3: return 1;
        case SafeDType::F8_E8M0: return 1;
        case SafeDType::Unknown: return 0;
    }
    return 0;
}

uint64_t safe_tensor_numel(const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (uint64_t dim : shape) total *= dim;
    return total;
}

SafeFp4TensorPair resolve_fp4_tensor_pair(const SafeTensorsIndex& index, const SafeTensorsShard& shard, const std::string& weight_name) {
    const std::string suffix = ".weight";
    const std::string scale_name =
        weight_name.size() >= suffix.size() && weight_name.compare(weight_name.size() - suffix.size(), suffix.size(), suffix) == 0
            ? weight_name.substr(0, weight_name.size() - suffix.size()) + ".scale"
            : weight_name + ".scale";

    const std::string* weight_shard = index.shard_for_tensor(weight_name);
    if (weight_shard == nullptr) throw std::runtime_error("FP4 weight not found: " + weight_name);
    const std::string* scale_shard = index.shard_for_tensor(scale_name);
    if (scale_shard == nullptr) throw std::runtime_error("FP4 scale not found: " + scale_name);
    if (*weight_shard != *scale_shard) throw std::runtime_error("FP4 weight and scale are in different shards: " + weight_name);

    const SafeTensorInfo* weight = shard.find_tensor(weight_name);
    const SafeTensorInfo* scale = shard.find_tensor(scale_name);
    if (weight == nullptr) throw std::runtime_error("FP4 weight missing from shard header: " + weight_name);
    if (scale == nullptr) throw std::runtime_error("FP4 scale missing from shard header: " + scale_name);
    if (weight->dtype != SafeDType::I8 || weight->shape.size() != 2) throw std::runtime_error("FP4 weight must be 2D I8 packed bytes: " + weight_name);
    if (scale->dtype != SafeDType::F8_E8M0 || scale->shape.size() != 2) throw std::runtime_error("FP4 scale must be 2D F8_E8M0: " + scale_name);

    SafeFp4TensorPair pair;
    pair.weight_name = weight_name;
    pair.scale_name = scale_name;
    pair.shard_name = *weight_shard;
    pair.rows = weight->shape[0];
    pair.packed_cols = weight->shape[1];
    pair.cols = pair.packed_cols * 2;
    pair.scale_cols = pair.cols / 32;
    if ((pair.cols % 32) != 0) throw std::runtime_error("FP4 cols must be divisible by 32: " + weight_name);
    if (scale->shape[0] != pair.rows || scale->shape[1] != pair.scale_cols) {
        throw std::runtime_error("FP4 weight/scale shape mismatch: " + weight_name + " / " + scale_name);
    }
    return pair;
}

}  // namespace pocket
