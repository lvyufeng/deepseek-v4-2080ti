// Device side of the Qwen weight loader: device residency and host-to-device
// upload, expressed against the vendor-neutral device runtime so the same code
// serves CUDA and Ascend. The checkpoint mapping, TP sharding and host
// materialization live in core/qwen_weight_map.cpp and stay free of any vendor
// SDK, so a checkpoint can be audited on a machine that has no CUDA toolkit at
// all.

#include "qwen_weights.hpp"

#include "device_runtime.hpp"

#include <cstring>
#include <stdexcept>

namespace pocket {

namespace {

float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1Fu;
    const uint32_t mantissa = bits & 0x3FFu;
    uint32_t widened = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            widened = sign;
        } else {
            // Subnormal: normalize by shifting the mantissa up until the implicit
            // bit appears, decrementing the exponent to match.
            uint32_t shifted = mantissa;
            int shift = 0;
            while ((shifted & 0x400u) == 0) {
                shifted <<= 1;
                ++shift;
            }
            shifted &= 0x3FFu;
            const uint32_t biased =
                static_cast<uint32_t>(127 - 15 - shift + 1);
            widened = sign | (biased << 23) | (shifted << 13);
        }
    } else if (exponent == 31) {
        widened = sign | 0x7F800000u | (mantissa << 13);
    } else {
        widened = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

uint16_t float_to_fp16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x007FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        // Inf or NaN: keep a non-zero mantissa non-zero so a NaN stays a NaN.
        return static_cast<uint16_t>(sign | 0x7C00u |
                                     (mantissa != 0 ? 0x200u : 0u));
    }
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    uint32_t half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
            return static_cast<uint16_t>(
                sign | (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | half);
}

}  // namespace

void qwen_apply_norm_gamma_policy(const QwenTensorRef& ref,
                                  QwenHostTensor& host) {
#ifdef POCKET_BACKEND_ASCEND
    if (host.device_dtype != SafeDType::F16) return;
    if (!qwen_is_one_plus_norm_gamma(ref.name)) return;
    const size_t count = host.bytes.size() / sizeof(uint16_t);
    if (count == 0) return;
    auto* values = reinterpret_cast<uint16_t*>(host.bytes.data());
    for (size_t i = 0; i < count; ++i) {
        values[i] = float_to_fp16_bits(1.0f + fp16_bits_to_float(values[i]));
    }
#else
    // CUDA kernels apply (1 + gamma) themselves; the stored weight is untouched.
    (void)ref;
    (void)host;
#endif
}

QwenDeviceTensor::~QwenDeviceTensor() {
    if (data != nullptr) device_free(data);
}

QwenDeviceTensor::QwenDeviceTensor(QwenDeviceTensor&& other) noexcept
    : data(other.data), device_dtype(other.device_dtype), shape(std::move(other.shape)),
      nbytes(other.nbytes), capacity(other.capacity) {
    other.data = nullptr;
    other.nbytes = 0;
    other.capacity = 0;
    other.device_dtype = SafeDType::Unknown;
}

QwenDeviceTensor& QwenDeviceTensor::operator=(QwenDeviceTensor&& other) noexcept {
    if (this == &other) return *this;
    if (data != nullptr) device_free(data);
    data = other.data;
    device_dtype = other.device_dtype;
    shape = std::move(other.shape);
    nbytes = other.nbytes;
    capacity = other.capacity;
    other.data = nullptr;
    other.nbytes = 0;
    other.capacity = 0;
    other.device_dtype = SafeDType::Unknown;
    return *this;
}

float* QwenDeviceTensor::f32_data() {
    if (device_dtype != SafeDType::F32) throw std::runtime_error("Qwen tensor is not F32");
    return static_cast<float*>(data);
}

const float* QwenDeviceTensor::f32_data() const {
    if (device_dtype != SafeDType::F32) throw std::runtime_error("Qwen tensor is not F32");
    return static_cast<const float*>(data);
}

uint16_t* QwenDeviceTensor::f16_data() {
    if (device_dtype != SafeDType::F16) throw std::runtime_error("Qwen tensor is not F16");
    return static_cast<uint16_t*>(data);
}

const uint16_t* QwenDeviceTensor::f16_data() const {
    if (device_dtype != SafeDType::F16) throw std::runtime_error("Qwen tensor is not F16");
    return static_cast<const uint16_t*>(data);
}

uint8_t* QwenDeviceTensor::fp8_data() {
    if (device_dtype != SafeDType::F8_E4M3) throw std::runtime_error("Qwen tensor is not FP8 E4M3");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::fp8_data() const {
    if (device_dtype != SafeDType::F8_E4M3) throw std::runtime_error("Qwen tensor is not FP8 E4M3");
    return static_cast<const uint8_t*>(data);
}

uint8_t* QwenDeviceTensor::byte_data() {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not raw bytes");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::byte_data() const {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not raw bytes");
    return static_cast<const uint8_t*>(data);
}

int8_t* QwenDeviceTensor::int8_data() {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not INT8");
    return static_cast<int8_t*>(data);
}

const int8_t* QwenDeviceTensor::int8_data() const {
    if (device_dtype != SafeDType::I8) throw std::runtime_error("Qwen tensor is not INT8");
    return static_cast<const int8_t*>(data);
}

uint8_t* QwenDeviceTensor::u8_data() {
    if (device_dtype != SafeDType::U8) throw std::runtime_error("Qwen tensor is not U8");
    return static_cast<uint8_t*>(data);
}

const uint8_t* QwenDeviceTensor::u8_data() const {
    if (device_dtype != SafeDType::U8) throw std::runtime_error("Qwen tensor is not U8");
    return static_cast<const uint8_t*>(data);
}

QwenDeviceTensor qwen_upload_tensor(const SafeTensorsIndex& index,
                                        const QwenTensorRef& ref,
                                        void* stream) {
    QwenHostTensor host = qwen_materialize_host_tensor(index, ref);
    qwen_apply_norm_gamma_policy(ref, host);
    QwenDeviceTensor device;
    device.device_dtype = host.device_dtype;
    device.shape = host.shape;
    device.nbytes = host.bytes.size();
    device.capacity = device.nbytes;
    if (device.nbytes == 0) {
        throw std::runtime_error("failed to allocate Qwen device tensor: " + ref.name);
    }
    device.data = device_malloc(device.nbytes);
    if (device.data == nullptr) {
        throw std::runtime_error("failed to allocate Qwen device tensor: " + ref.name);
    }
    if (!memcpy_h2d_async(device.data, host.bytes.data(), device.nbytes, stream)) {
        device_free(device.data);
        device.data = nullptr;
        device.nbytes = 0;
        device.capacity = 0;
        throw std::runtime_error("failed to upload Qwen device tensor: " + ref.name);
    }
    return device;
}

QwenDeviceTensor qwen_upload_nvfp4_linear_cuda(
    const SafeTensorsIndex& index, const QwenLinearRef& ref,
    float* weight_global_factor, float* input_global_scale, void* stream) {
    if (weight_global_factor == nullptr || input_global_scale == nullptr) {
        throw std::invalid_argument("null Qwen NVFP4 global metadata output");
    }
    QwenNvfp4HostLinear host = qwen_materialize_nvfp4_host_linear(index, ref);
    QwenDeviceTensor device;
    device.device_dtype = SafeDType::U8;
    device.shape = host.logical_shape;
    device.shape.push_back(sizeof(QwenNvfp4Block64));
    device.nbytes = host.blocks.size() * sizeof(QwenNvfp4Block64);
    device.capacity = device.nbytes;
    if (device.nbytes == 0) {
        throw std::runtime_error("failed to allocate Qwen NVFP4 device linear");
    }
    device.data = device_malloc(device.nbytes);
    if (device.data == nullptr) {
        throw std::runtime_error("failed to allocate Qwen NVFP4 device linear");
    }
    if (!memcpy_h2d_async(device.data, host.blocks.data(), device.nbytes, stream)) {
        device_free(device.data);
        device.data = nullptr;
        device.nbytes = 0;
        device.capacity = 0;
        throw std::runtime_error("failed to upload Qwen NVFP4 device linear");
    }
    *weight_global_factor = host.weight_global_factor;
    *input_global_scale = host.input_global_scale;
    return device;
}

}  // namespace pocket
