// Ascend (CANN / ACL) implementation of the vendor-neutral device runtime.
//
// Two things differ structurally from CUDA and are absorbed here so that no
// engine code has to know about them:
//
//   1. ACL needs an explicit one-time aclInit() before any other call. CUDA
//      initializes lazily. We do it once, on first use, under a call_once.
//   2. ACL binds work to a *context* that is per-thread state, not just a device
//      id. A thread that has not called aclrtSetCurrentContext cannot launch or
//      copy at all. device_set() therefore creates one context per device on
//      first touch and installs it into the calling thread; every later
//      device_set from another thread reuses the same context, which is what
//      lets several host threads share one device's memory.
//
// Everything else is a pass-through. The extra `destMax` / `maxCount` capacity
// arguments that ACL wants get the copy length, since that is the only bound the
// CUDA-shaped API can honestly promise.

#include "device_runtime.hpp"

#include <acl/acl.h>

#include <array>
#include <mutex>
#include <vector>

namespace pocket {
namespace {

constexpr int kMaxDevices = 64;

// aclInit is process-global and must not be called twice; aclFinalize is
// deliberately never called, because tearing the runtime down while another
// thread still holds device memory is a hard crash and the process is about to
// exit anyway.
bool acl_initialize() {
    static bool ok = false;
    static std::once_flag once;
    std::call_once(once, [] {
        const aclError status = aclInit(nullptr);
        // ACL_ERROR_REPEAT_INITIALIZE means someone else (a Python sidecar, a
        // test harness) already initialized in this process, which is fine.
        ok = (status == ACL_SUCCESS || status == ACL_ERROR_REPEAT_INITIALIZE);
    });
    return ok;
}

struct ContextTable {
    std::mutex mutex;
    std::array<aclrtContext, kMaxDevices> contexts{};
};

ContextTable& context_table() {
    static ContextTable table;
    return table;
}

// One context per device, shared by every thread that binds to that device.
// Returns nullptr when the device cannot be opened.
aclrtContext context_for(int device) {
    if (device < 0 || device >= kMaxDevices) return nullptr;
    ContextTable& table = context_table();
    std::lock_guard<std::mutex> guard(table.mutex);
    if (table.contexts[device] != nullptr) return table.contexts[device];
    if (aclrtSetDevice(device) != ACL_SUCCESS) return nullptr;
    aclrtContext context = nullptr;
    if (aclrtCreateContext(&context, device) != ACL_SUCCESS) return nullptr;
    table.contexts[device] = context;
    return context;
}

aclrtStream as_stream(void* stream) {
    return static_cast<aclrtStream>(stream);
}

aclrtEvent as_event(void* event) {
    return static_cast<aclrtEvent>(event);
}

// The blocking copy, and the third structural difference from CUDA. Nothing
// reports an error when this one is missed, which is what makes it expensive.
//
// cudaMemcpy is ordered against the null stream: it waits for previously issued
// work before it touches memory. aclrtMemcpy blocks only the *host* -- it does
// not join the default stream -- so a read issued straight after a kernel launch
// returns whatever the buffer held before the kernel ran. Engine code is written
// to the CUDA contract (launch, then copy the result out) throughout, so the
// ordering is restored at this one point rather than by auditing every call site.
// The _async variants keep ACL's semantics untouched, so a caller that manages
// its own stream ordering pays nothing for this.
bool copy(void* dst, const void* src, size_t bytes, aclrtMemcpyKind kind) {
    if (bytes == 0) return true;
    if (aclrtSynchronizeStream(nullptr) != ACL_SUCCESS) return false;
    return aclrtMemcpy(dst, bytes, src, bytes, kind) == ACL_SUCCESS;
}

bool copy_async(void* dst, const void* src, size_t bytes, aclrtMemcpyKind kind,
                void* stream) {
    if (bytes == 0) return true;
    return aclrtMemcpyAsync(dst, bytes, src, bytes, kind, as_stream(stream)) ==
           ACL_SUCCESS;
}

}  // namespace

DeviceBackend device_backend() { return DeviceBackend::Ascend; }

const char* device_backend_name() { return "ascend"; }

bool device_runtime_available() {
    if (!acl_initialize()) return false;
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS) return false;
    return count > 0;
}

bool device_set(int device) {
    if (!acl_initialize()) return false;
    if (aclrtSetDevice(device) != ACL_SUCCESS) return false;
    aclrtContext context = context_for(device);
    if (context == nullptr) return false;
    // Idempotent and cheap: ACL keeps the current context in thread-local
    // storage, so re-installing the same handle on every device_set costs
    // nothing and removes any ordering requirement from the caller.
    return aclrtSetCurrentContext(context) == ACL_SUCCESS;
}

bool device_reset(int device) {
    if (!acl_initialize()) return false;
    if (device >= 0 && device < kMaxDevices) {
        ContextTable& table = context_table();
        std::lock_guard<std::mutex> guard(table.mutex);
        if (table.contexts[device] != nullptr) {
            aclrtDestroyContext(table.contexts[device]);
            table.contexts[device] = nullptr;
        }
    }
    return aclrtResetDevice(device) == ACL_SUCCESS;
}

int device_count() {
    if (!acl_initialize()) return 0;
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS) return 0;
    return static_cast<int>(count);
}

std::string device_name(int /*device*/) {
    if (!acl_initialize()) return std::string();
    // ACL reports the SoC, not a per-card product string. Note this returns the
    // platform name ("Ascend910B"), which for the 910 series is not the same as
    // the SoC generation; see the naming rule in CLAUDE.md.
    const char* name = aclrtGetSocName();
    return name != nullptr ? std::string(name) : std::string();
}

bool device_mem_info(size_t* free_bytes, size_t* total_bytes) {
    if (!acl_initialize()) return false;
    size_t free_value = 0;
    size_t total_value = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &free_value, &total_value) != ACL_SUCCESS) {
        return false;
    }
    if (free_bytes != nullptr) *free_bytes = free_value;
    if (total_bytes != nullptr) *total_bytes = total_value;
    return true;
}

void* device_malloc(size_t bytes) {
    if (!acl_initialize()) return nullptr;
    void* ptr = nullptr;
    // HUGE_FIRST asks for 2 MB pages and falls back to normal pages. Weights and
    // KV blocks are large and long-lived, which is exactly the case it exists for.
    if (aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return nullptr;
    }
    return ptr;
}

void device_free(void* ptr) {
    if (ptr == nullptr) return;
    aclrtFree(ptr);
}

void* host_alloc_pinned(size_t bytes) {
    if (!acl_initialize()) return nullptr;
    void* ptr = nullptr;
    if (aclrtMallocHost(&ptr, bytes) != ACL_SUCCESS) return nullptr;
    return ptr;
}

void host_free_pinned(void* ptr) {
    if (ptr == nullptr) return;
    aclrtFreeHost(ptr);
}

bool memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return copy(dst, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

bool memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return copy(dst, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

bool memcpy_d2d(void* dst, const void* src, size_t bytes) {
    return copy(dst, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

bool memcpy_h2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    return copy_async(dst, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
}

bool memcpy_d2h_async(void* dst, const void* src, size_t bytes, void* stream) {
    return copy_async(dst, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
}

bool memcpy_d2d_async(void* dst, const void* src, size_t bytes, void* stream) {
    return copy_async(dst, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
}

namespace {

bool copy_2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
             size_t width, size_t height, aclrtMemcpyKind kind) {
    if (width == 0 || height == 0) return true;
    // Stream-ordered for the same reason `copy` above is; see the note there.
    if (aclrtSynchronizeStream(nullptr) != ACL_SUCCESS) return false;
    return aclrtMemcpy2d(dst, dst_pitch, src, src_pitch, width, height, kind) ==
           ACL_SUCCESS;
}

}  // namespace

// aclrtMemcpy2d rejects ACL_MEMCPY_DEVICE_TO_DEVICE on this CANN release (rtMemcpy2d
// reports "the feature is not supported", runtime result 207000), even though the
// host-to-device and device-to-host directions work. Fall back to one linear copy
// per row, which is what the strided form would have done anyway.
bool memcpy_2d_d2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height) {
    if (width == 0 || height == 0) return true;
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    for (size_t row = 0; row < height; ++row) {
        if (!copy(dst_bytes + row * dst_pitch, src_bytes + row * src_pitch, width,
                  ACL_MEMCPY_DEVICE_TO_DEVICE)) {
            return false;
        }
    }
    return true;
}

bool memcpy_2d_h2d(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height) {
    return copy_2d(dst, dst_pitch, src, src_pitch, width, height,
                   ACL_MEMCPY_HOST_TO_DEVICE);
}

bool memcpy_2d_d2h(void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
                   size_t width, size_t height) {
    return copy_2d(dst, dst_pitch, src, src_pitch, width, height,
                   ACL_MEMCPY_DEVICE_TO_HOST);
}

bool device_memset(void* dst, int value, size_t bytes) {
    if (bytes == 0) return true;
    // cudaMemset is ordered against the null stream, so a caller may legitimately
    // clear a buffer that queued work still reads. Same fence as `copy`.
    if (aclrtSynchronizeStream(nullptr) != ACL_SUCCESS) return false;
    return aclrtMemset(dst, bytes, value, bytes) == ACL_SUCCESS;
}

bool device_memset_async(void* dst, int value, size_t bytes, void* stream) {
    if (bytes == 0) return true;
    return aclrtMemsetAsync(dst, bytes, value, bytes, as_stream(stream)) ==
           ACL_SUCCESS;
}

void* stream_create() {
    if (!acl_initialize()) return nullptr;
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) return nullptr;
    return static_cast<void*>(stream);
}

void stream_destroy(void* stream) {
    if (stream == nullptr) return;
    aclrtDestroyStream(as_stream(stream));
}

bool stream_synchronize(void* stream) {
    return aclrtSynchronizeStream(as_stream(stream)) == ACL_SUCCESS;
}

void* event_create(bool with_timing) {
    if (!acl_initialize()) return nullptr;
    aclrtEvent event = nullptr;
    // ACL_EVENT_SYNC is the ordering-only event and is the cheaper one; timing
    // needs ACL_EVENT_TIME_LINE, which the hot path never asks for.
    const uint32_t flag = with_timing ? ACL_EVENT_TIME_LINE : ACL_EVENT_SYNC;
    if (aclrtCreateEventWithFlag(&event, flag) != ACL_SUCCESS) return nullptr;
    return static_cast<void*>(event);
}

void event_destroy(void* event) {
    if (event == nullptr) return;
    aclrtDestroyEvent(as_event(event));
}

bool event_record(void* event, void* stream) {
    return aclrtRecordEvent(as_event(event), as_stream(stream)) == ACL_SUCCESS;
}

bool event_synchronize(void* event) {
    return aclrtSynchronizeEvent(as_event(event)) == ACL_SUCCESS;
}

bool stream_wait_event(void* stream, void* event) {
    return aclrtStreamWaitEvent(as_stream(stream), as_event(event)) == ACL_SUCCESS;
}

bool event_elapsed_ms(void* start, void* end, float* out_ms) {
    float ms = 0.0f;
    if (aclrtEventElapsedTime(&ms, as_event(start), as_event(end)) != ACL_SUCCESS) {
        return false;
    }
    if (out_ms != nullptr) *out_ms = ms;
    return true;
}

bool device_synchronize() {
    return aclrtSynchronizeDevice() == ACL_SUCCESS;
}

std::string device_last_error() {
    // ACL has no "get and clear last error" the way CUDA does; the closest thing
    // is the recent-error text buffer, which is per-thread and sticky. Returning
    // it and letting the next failure overwrite it matches how the engine uses
    // this: for a message, not for control flow.
    const char* message = aclGetRecentErrMsg();
    if (message == nullptr || message[0] == '\0') return std::string();
    return std::string(message);
}

void device_range_push(const char* /*label*/) {
    // mstx is the Ascend profiler marker API, but it is range-id based
    // (mstxRangeStartA / mstxRangeEnd) rather than a stack, and it ships as a
    // bare mstx.so that has to be found by absolute path. Wiring it up is worth
    // doing when profiling starts in earnest; until then a no-op keeps the engine
    // portable at zero cost.
}

void device_range_pop() {}

}  // namespace pocket
