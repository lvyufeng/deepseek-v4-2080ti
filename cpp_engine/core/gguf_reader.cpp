#include "gguf_reader.hpp"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pocket {
namespace {

constexpr uint32_t kGGUFMagic = 0x46554747;  // GGUF little endian

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

template <typename T>
T read_scalar(const uint8_t* data, uint64_t size, size_t& cursor) {
    if (cursor + sizeof(T) > size) {
        throw std::runtime_error("GGUF parse past end of file");
    }
    T out;
    std::memcpy(&out, data + cursor, sizeof(T));
    cursor += sizeof(T);
    return out;
}

uint64_t block_count(uint64_t elems, uint64_t block_size) {
    return (elems + block_size - 1) / block_size;
}

}  // namespace

GGUFFile::GGUFFile(const std::string& path) { open(path); }

GGUFFile::~GGUFFile() { close(); }

GGUFFile::GGUFFile(GGUFFile&& other) noexcept { *this = std::move(other); }

GGUFFile& GGUFFile::operator=(GGUFFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    close();
    path_ = std::move(other.path_);
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    version_ = other.version_;
    alignment_ = other.alignment_;
    data_start_ = other.data_start_;
    metadata_ = std::move(other.metadata_);
    tensors_ = std::move(other.tensors_);
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void GGUFFile::open(const std::string& path) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("failed to open GGUF file: " + path);
    }
    struct stat st;
    if (fstat(fd_, &st) != 0) {
        throw std::runtime_error("failed to stat GGUF file: " + path);
    }
    size_ = static_cast<uint64_t>(st.st_size);
    if (size_ == 0) {
        throw std::runtime_error("empty GGUF file: " + path);
    }
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        throw std::runtime_error("failed to mmap GGUF file: " + path);
    }
    parse();
}

void GGUFFile::close() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    version_ = 0;
    alignment_ = 32;
    data_start_ = 0;
    metadata_.clear();
    tensors_.clear();
}

void GGUFFile::parse() {
    size_t cursor = 0;
    const uint32_t magic = read_u32(cursor);
    if (magic != kGGUFMagic) {
        throw std::runtime_error("not a GGUF file: bad magic");
    }
    version_ = read_u32(cursor);
    if (version_ < 2 || version_ > 3) {
        throw std::runtime_error("unsupported GGUF version: " + std::to_string(version_));
    }
    const uint64_t tensor_count = read_u64(cursor);
    const uint64_t metadata_count = read_u64(cursor);

    metadata_.clear();
    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key = read_string(cursor);
        uint32_t value_type = read_u32(cursor);
        metadata_[std::move(key)] = read_metadata_value(cursor, value_type);
    }
    if (auto alignment = metadata_u64("general.alignment")) {
        alignment_ = *alignment;
    }

    tensors_.clear();
    tensors_.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo info;
        info.name = read_string(cursor);
        const uint32_t n_dims = read_u32(cursor);
        info.shape.reserve(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.shape.push_back(read_u64(cursor));
        }
        info.ggml_type = read_u32(cursor);
        info.dtype = ggml_type_to_dtype(info.ggml_type);
        info.offset = read_u64(cursor);
        info.nbytes = ggml_tensor_nbytes(info.ggml_type, info.shape);
        tensors_.push_back(std::move(info));
    }

    data_start_ = align_up(static_cast<uint64_t>(cursor), alignment_);
    for (auto& tensor : tensors_) {
        tensor.absolute_offset = data_start_ + tensor.offset;
        if (tensor.absolute_offset + tensor.nbytes > size_) {
            throw std::runtime_error("tensor extends past end of file: " + tensor.name);
        }
    }
}

uint8_t GGUFFile::read_u8(size_t& cursor) const { return read_scalar<uint8_t>(bytes(), size_, cursor); }
uint16_t GGUFFile::read_u16(size_t& cursor) const { return read_scalar<uint16_t>(bytes(), size_, cursor); }
uint32_t GGUFFile::read_u32(size_t& cursor) const { return read_scalar<uint32_t>(bytes(), size_, cursor); }
uint64_t GGUFFile::read_u64(size_t& cursor) const { return read_scalar<uint64_t>(bytes(), size_, cursor); }
int8_t GGUFFile::read_i8(size_t& cursor) const { return read_scalar<int8_t>(bytes(), size_, cursor); }
int16_t GGUFFile::read_i16(size_t& cursor) const { return read_scalar<int16_t>(bytes(), size_, cursor); }
int32_t GGUFFile::read_i32(size_t& cursor) const { return read_scalar<int32_t>(bytes(), size_, cursor); }
int64_t GGUFFile::read_i64(size_t& cursor) const { return read_scalar<int64_t>(bytes(), size_, cursor); }
float GGUFFile::read_f32(size_t& cursor) const { return read_scalar<float>(bytes(), size_, cursor); }
double GGUFFile::read_f64(size_t& cursor) const { return read_scalar<double>(bytes(), size_, cursor); }

std::string GGUFFile::read_string(size_t& cursor) const {
    const uint64_t len = read_u64(cursor);
    if (cursor + len > size_) {
        throw std::runtime_error("GGUF string past end of file");
    }
    std::string out(reinterpret_cast<const char*>(bytes() + cursor), static_cast<size_t>(len));
    cursor += static_cast<size_t>(len);
    return out;
}

MetadataValue GGUFFile::read_metadata_value(size_t& cursor, uint32_t value_type) const {
    switch (value_type) {
        case 0: return static_cast<uint64_t>(read_u8(cursor));
        case 1: return static_cast<int64_t>(read_i8(cursor));
        case 2: return static_cast<uint64_t>(read_u16(cursor));
        case 3: return static_cast<int64_t>(read_i16(cursor));
        case 4: return static_cast<uint64_t>(read_u32(cursor));
        case 5: return static_cast<int64_t>(read_i32(cursor));
        case 6: return static_cast<double>(read_f32(cursor));
        case 7: return static_cast<bool>(read_u8(cursor));
        case 8: return read_string(cursor);
        case 10: return static_cast<uint64_t>(read_u64(cursor));
        case 11: return static_cast<int64_t>(read_i64(cursor));
        case 12: return read_f64(cursor);
        case 9: {
            const uint32_t elem_type = read_u32(cursor);
            const uint64_t count = read_u64(cursor);
            switch (elem_type) {
                case 0: {
                    std::vector<uint64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_u8(cursor));
                    return v;
                }
                case 1: {
                    std::vector<int64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_i8(cursor));
                    return v;
                }
                case 2: {
                    std::vector<uint64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_u16(cursor));
                    return v;
                }
                case 3: {
                    std::vector<int64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_i16(cursor));
                    return v;
                }
                case 4: {
                    std::vector<uint64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_u32(cursor));
                    return v;
                }
                case 5: {
                    std::vector<int64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_i32(cursor));
                    return v;
                }
                case 6: {
                    std::vector<double> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_f32(cursor));
                    return v;
                }
                case 7: {
                    std::vector<bool> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_u8(cursor) != 0);
                    return v;
                }
                case 8: {
                    std::vector<std::string> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_string(cursor));
                    return v;
                }
                case 10: {
                    std::vector<uint64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_u64(cursor));
                    return v;
                }
                case 11: {
                    std::vector<int64_t> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_i64(cursor));
                    return v;
                }
                case 12: {
                    std::vector<double> v; v.reserve(count);
                    for (uint64_t i = 0; i < count; ++i) v.push_back(read_f64(cursor));
                    return v;
                }
                default:
                    throw std::runtime_error("unsupported GGUF array element type: " + std::to_string(elem_type));
            }
        }
        default:
            throw std::runtime_error("unsupported GGUF metadata value type: " + std::to_string(value_type));
    }
}

const GGUFTensorInfo* GGUFFile::find_tensor(const std::string& name) const {
    auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const GGUFTensorInfo& info) { return info.name == name; });
    return it == tensors_.end() ? nullptr : &*it;
}

TensorView GGUFFile::tensor_view(const GGUFTensorInfo& info) const {
    TensorView view;
    view.name = info.name;
    view.dtype = info.dtype;
    view.shape = info.shape;
    view.offset = info.offset;
    view.absolute_offset = info.absolute_offset;
    view.nbytes = info.nbytes;
    view.data = bytes() + info.absolute_offset;
    return view;
}

std::optional<uint64_t> GGUFFile::metadata_u64(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return std::nullopt;
    if (auto* v = std::get_if<uint64_t>(&it->second)) return *v;
    if (auto* v = std::get_if<int64_t>(&it->second); v && *v >= 0) return static_cast<uint64_t>(*v);
    return std::nullopt;
}

std::optional<int64_t> GGUFFile::metadata_i64(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return std::nullopt;
    if (auto* v = std::get_if<int64_t>(&it->second)) return *v;
    if (auto* v = std::get_if<uint64_t>(&it->second)) return static_cast<int64_t>(*v);
    return std::nullopt;
}

std::optional<double> GGUFFile::metadata_f64(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return std::nullopt;
    if (auto* v = std::get_if<double>(&it->second)) return *v;
    if (auto* v = std::get_if<uint64_t>(&it->second)) return static_cast<double>(*v);
    if (auto* v = std::get_if<int64_t>(&it->second)) return static_cast<double>(*v);
    return std::nullopt;
}

std::optional<bool> GGUFFile::metadata_bool(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return std::nullopt;
    if (auto* v = std::get_if<bool>(&it->second)) return *v;
    return std::nullopt;
}

std::optional<std::string> GGUFFile::metadata_string(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return std::nullopt;
    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
    return std::nullopt;
}

std::vector<uint64_t> GGUFFile::metadata_u64_array(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return {};
    if (auto* v = std::get_if<std::vector<uint64_t>>(&it->second)) return *v;
    if (auto* v = std::get_if<std::vector<int64_t>>(&it->second)) {
        std::vector<uint64_t> out;
        out.reserve(v->size());
        for (int64_t x : *v) {
            out.push_back(x < 0 ? 0 : static_cast<uint64_t>(x));
        }
        return out;
    }
    return {};
}

std::vector<double> GGUFFile::metadata_f64_array(const std::string& key) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return {};
    if (auto* v = std::get_if<std::vector<double>>(&it->second)) return *v;
    if (auto* v = std::get_if<std::vector<uint64_t>>(&it->second)) {
        std::vector<double> out;
        out.reserve(v->size());
        for (uint64_t x : *v) out.push_back(static_cast<double>(x));
        return out;
    }
    if (auto* v = std::get_if<std::vector<int64_t>>(&it->second)) {
        std::vector<double> out;
        out.reserve(v->size());
        for (int64_t x : *v) out.push_back(static_cast<double>(x));
        return out;
    }
    return {};
}

std::string ggml_type_name(uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: return "f32";
        case 1: return "f16";
        case 32: return "bf16";
        case 8: return "q8_0";
        case 10: return "q2_k";
        case 16: return "iq2_xxs";
        case 29: return "iq1_m";
        case 19: return "iq3_s";
        case 26: return "i32";
        default: return "ggml_type_" + std::to_string(ggml_type);
    }
}

DType ggml_type_to_dtype(uint32_t ggml_type) {
    switch (ggml_type) {
        case 0: return DType::F32;
        case 1: return DType::F16;
        case 32: return DType::BF16;
        case 8: return DType::Q8_0;
        case 10: return DType::Q2_K;
        case 16: return DType::IQ2_XXS;
        case 29: return DType::IQ1_M;
        default: return DType::Unknown;
    }
}

uint64_t ggml_tensor_nbytes(uint32_t ggml_type, const std::vector<uint64_t>& shape) {
    const uint64_t elems = tensor_element_count(shape);
    switch (ggml_type) {
        case 0: return elems * 4;
        case 1: return elems * 2;
        case 32: return elems * 2;
        case 8: return block_count(elems, 32) * 34;
        case 10: return block_count(elems, 256) * 84;
        case 16: return block_count(elems, 256) * 66;
        case 29: return block_count(elems, 256) * 56;
        case 26: return elems * 4;
        default: return 0;
    }
}

std::string metadata_value_to_string(const MetadataValue& value, size_t max_items) {
    std::ostringstream oss;
    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
            oss << '"' << v << '"';
        } else if constexpr (std::is_same_v<T, bool>) {
            oss << (v ? "true" : "false");
        } else if constexpr (std::is_arithmetic_v<T>) {
            oss << v;
        } else {
            oss << '[';
            const size_t n = std::min(max_items, v.size());
            for (size_t i = 0; i < n; ++i) {
                if (i) oss << ", ";
                if constexpr (std::is_same_v<typename T::value_type, std::string>) {
                    oss << '"' << v[i] << '"';
                } else if constexpr (std::is_same_v<typename T::value_type, bool>) {
                    oss << (v[i] ? "true" : "false");
                } else {
                    oss << v[i];
                }
            }
            if (v.size() > n) oss << ", ... (" << v.size() << ")";
            oss << ']';
        }
    }, value);
    return oss.str();
}

}  // namespace pocket
