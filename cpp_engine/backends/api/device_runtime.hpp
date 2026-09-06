#pragma once

// Vendor-neutral device runtime. This is the seam that lets one engine drive
// either CUDA or Ascend: allocation, transfers, streams and events, expressed
// with no vendor type in the signature.
//
// Streams and events are opaque void*. Each backend casts them to its own handle
// (cudaStream_t, aclrtStream). There is no vtable and no runtime dispatch: the
// selected backend is the only implementation linked into the binary, so a call
// here costs the same as calling the vendor API directly.
//
// This header must not include any vendor SDK header; check_layering enforces
// that. Engine code that includes only this stays portable.

#include <cstddef>
#include <cstdint>
#include <string>

namespace pocket {

// Which vendor implementation is linked. Useful for logs and for tests that
// need to skip a vendor-specific expectation.
enum class DeviceBackend {
    Cuda,
    Ascend,
};

DeviceBackend device_backend();
const char* device_backend_name();

// True when a usable accelerator runtime is present. Never throws, so a host-only
// tool can branch on it.
bool device_runtime_available();

// Bind the calling thread to a device. On Ascend this also creates and sets the
// thread's ACL context, which CUDA has no equivalent of but which is mandatory
// there; hiding it here keeps the engine free of that asymmetry.
bool device_set(int device);
bool device_reset(int device);
int device_count();

// Human-readable device identity, for logs. "" when unavailable.
std::string device_name(int device);

// Free and total device memory in bytes. Returns false and leaves the outputs
// untouched when the query fails.
bool device_mem_info(size_t* free_bytes, size_t* total_bytes);

// Device allocation. Returns nullptr on failure rather than throwing, matching
// how the engine already checks cudaMalloc results.
void* device_malloc(size_t bytes);
void device_free(void* ptr);

// Pinned/page-locked host allocation for overlapped transfers.
void* host_alloc_pinned(size_t bytes);
void host_free_pinned(void* ptr);

// Allocate into an already-typed pointer, so a call site does not have to name
// its own element type. This is what cudaMalloc's void** out-parameter bought,
// without the cast: `device_malloc_into(d_logits, bytes)`. Returns false when
// the allocation failed, matching how the engine tests allocation results.
template <typename T>
bool device_malloc_into(T*& ptr, size_t bytes) {
    ptr = static_cast<T*>(device_malloc(bytes));
    return ptr != nullptr;
}

template <typename T>
bool host_alloc_pinned_into(T*& ptr, size_t bytes) {
    ptr = static_cast<T*>(host_alloc_pinned(bytes));
    return ptr != nullptr;
}

// Synchronous transfers. `bytes` is the copy length for both source and
// destination; the backend supplies whatever capacity argument its API wants.
bool memcpy_h2d(void* dst, const void* src, size_t bytes);
bool memcpy_d2h(void* dst, const void* src, size_t bytes);
bool memcpy_d2d(void* dst, const void* src, size_t bytes);

// Asynchronous transfers on `stream`. A null stream means the backend default.
bool memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream);
bool memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream);
bool memcpy_d2d_async(void* dst, const void* src, size_t bytes, void* stream);

// Strided 2D transfer: `height` rows of `width` bytes, advancing `dst_pitch` and
// `src_pitch` bytes per row. Both backends expose this natively (cudaMemcpy2D,
// aclrtMemcpy2d), so the engine keeps using it instead of a row loop.
bool memcpy_2d_d2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height);
bool memcpy_2d_h2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height);
bool memcpy_2d_d2h(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height);

bool device_memset(void* dst, int value, size_t bytes);
bool device_memset_async(void* dst, int value, size_t bytes, void* stream);

// Streams. `create_stream` returns nullptr on failure. Non-blocking is the only
// mode the engine uses, so it is not a parameter.
void* stream_create();
void stream_destroy(void* stream);
bool stream_synchronize(void* stream);

// Events. Created with timing disabled unless `with_timing` is set, because the
// engine's hot path only needs ordering.
void* event_create(bool with_timing = false);
void event_destroy(void* event);
bool event_record(void* event, void* stream);
bool event_synchronize(void* event);
bool stream_wait_event(void* stream, void* event);
// Elapsed milliseconds between two timing-enabled events. Returns false when
// either event was created without timing.
bool event_elapsed_ms(void* start, void* end, float* out_ms);

bool device_synchronize();

// Last asynchronous error on this thread, cleared by the call. Empty when clean.
std::string device_last_error();

// Profiler range markers. No-ops on a backend without a profiler.
void device_range_push(const char* label);
void device_range_pop();

}  // namespace pocket
