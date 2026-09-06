// CUDA implementation of the vendor-neutral device runtime.
//
// Every function here is a thin pass-through to the CUDA runtime. That is
// deliberate: the engine must pay nothing for portability, so there is no vtable,
// no handle table and no bookkeeping beyond what CUDA already does. Streams and
// events cast straight through between void* and cudaStream_t / cudaEvent_t.

#include "device_runtime.hpp"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace pocket {
namespace {

cudaStream_t as_stream(void* stream) {
    return static_cast<cudaStream_t>(stream);
}

cudaEvent_t as_event(void* event) {
    return static_cast<cudaEvent_t>(event);
}

}  // namespace

DeviceBackend device_backend() { return DeviceBackend::Cuda; }

const char* device_backend_name() { return "cuda"; }

bool device_runtime_available() {
    int count = 0;
    // cudaGetDeviceCount is the cheapest probe that also clears a sticky
    // "no driver" error, so a host-only tool can call this without aborting.
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

bool device_set(int device) {
    return cudaSetDevice(device) == cudaSuccess;
}

bool device_reset(int device) {
    if (cudaSetDevice(device) != cudaSuccess) return false;
    return cudaDeviceReset() == cudaSuccess;
}

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count;
}

std::string device_name(int device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return std::string();
    return std::string(prop.name);
}

bool device_mem_info(size_t* free_bytes, size_t* total_bytes) {
    size_t free_value = 0;
    size_t total_value = 0;
    if (cudaMemGetInfo(&free_value, &total_value) != cudaSuccess) return false;
    if (free_bytes != nullptr) *free_bytes = free_value;
    if (total_bytes != nullptr) *total_bytes = total_value;
    return true;
}

void* device_malloc(size_t bytes) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return ptr;
}

void device_free(void* ptr) {
    if (ptr == nullptr) return;
    cudaFree(ptr);
}

void* host_alloc_pinned(size_t bytes) {
    void* ptr = nullptr;
    if (cudaMallocHost(&ptr, bytes) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return ptr;
}

void host_free_pinned(void* ptr) {
    if (ptr == nullptr) return;
    cudaFreeHost(ptr);
}

bool memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool memcpy_d2d(void* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
}

bool memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                           as_stream(stream)) == cudaSuccess;
}

bool memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                           as_stream(stream)) == cudaSuccess;
}

bool memcpy_d2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                           as_stream(stream)) == cudaSuccess;
}

bool memcpy_2d_d2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height) {
    if (width == 0 || height == 0) return true;
    return cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height,
                        cudaMemcpyDeviceToDevice) == cudaSuccess;
}

bool memcpy_2d_h2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height) {
    if (width == 0 || height == 0) return true;
    return cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height,
                        cudaMemcpyHostToDevice) == cudaSuccess;
}

bool memcpy_2d_d2h(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height) {
    if (width == 0 || height == 0) return true;
    return cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, height,
                        cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool device_memset(void* dst, int value, size_t bytes) {
    return cudaMemset(dst, value, bytes) == cudaSuccess;
}

bool device_memset_async(void* dst, int value, size_t bytes, void* stream) {
    return cudaMemsetAsync(dst, value, bytes, as_stream(stream)) == cudaSuccess;
}

void* stream_create() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return static_cast<void*>(stream);
}

void stream_destroy(void* stream) {
    if (stream == nullptr) return;
    cudaStreamDestroy(as_stream(stream));
}

bool stream_synchronize(void* stream) {
    return cudaStreamSynchronize(as_stream(stream)) == cudaSuccess;
}

void* event_create(bool with_timing) {
    cudaEvent_t event = nullptr;
    const unsigned int flags = with_timing ? cudaEventDefault : cudaEventDisableTiming;
    if (cudaEventCreateWithFlags(&event, flags) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return static_cast<void*>(event);
}

void event_destroy(void* event) {
    if (event == nullptr) return;
    cudaEventDestroy(as_event(event));
}

bool event_record(void* event, void* stream) {
    return cudaEventRecord(as_event(event), as_stream(stream)) == cudaSuccess;
}

bool event_synchronize(void* event) {
    return cudaEventSynchronize(as_event(event)) == cudaSuccess;
}

bool stream_wait_event(void* stream, void* event) {
    return cudaStreamWaitEvent(as_stream(stream), as_event(event), 0) == cudaSuccess;
}

bool event_elapsed_ms(void* start, void* end, float* out_ms) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, as_event(start), as_event(end)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (out_ms != nullptr) *out_ms = ms;
    return true;
}

bool device_synchronize() {
    return cudaDeviceSynchronize() == cudaSuccess;
}

std::string device_last_error() {
    const cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) return std::string();
    return std::string(cudaGetErrorString(status));
}

void device_range_push(const char* label) {
    nvtxRangePushA(label != nullptr ? label : "");
}

void device_range_pop() { nvtxRangePop(); }

}  // namespace pocket
