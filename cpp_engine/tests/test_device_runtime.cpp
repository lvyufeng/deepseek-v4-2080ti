// Device runtime conformance test. Deliberately backend-agnostic: the same binary
// is the acceptance gate for the CUDA implementation and for the Ascend one, so
// any behavioural divergence between them shows up here rather than deep inside
// the engine.
//
// Skips cleanly with exit 0 when no accelerator is present, so it can stay in the
// default test set on a host with no device.

#include "device_runtime.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_checks = 0;

void require(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        std::string detail = pocket::device_last_error();
        if (!detail.empty()) detail = " (" + detail + ")";
        throw std::runtime_error("check failed: " + what + detail);
    }
}

// Scoped device buffer so a failing check cannot leak HBM across the test.
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : ptr_(pocket::device_malloc(bytes)) {}
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
};

void check_device_query() {
    const int count = pocket::device_count();
    require(count > 0, "device_count reports at least one device");

    const std::string name = pocket::device_name(0);
    require(!name.empty(), "device_name returns a non-empty identity");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    require(pocket::device_mem_info(&free_bytes, &total_bytes),
            "device_mem_info succeeds");
    require(total_bytes > 0, "total device memory is positive");
    require(free_bytes <= total_bytes, "free memory does not exceed total");

    std::printf("runtime backend=%s devices=%d name=%s total_GiB=%.3f\n",
                pocket::device_backend_name(), count, name.c_str(),
                static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
}

void check_sync_transfers() {
    constexpr size_t kCount = 4096;
    constexpr size_t kBytes = kCount * sizeof(uint32_t);

    std::vector<uint32_t> host_in(kCount);
    std::iota(host_in.begin(), host_in.end(), 1u);
    std::vector<uint32_t> host_out(kCount, 0u);

    DeviceBuffer a(kBytes);
    DeviceBuffer b(kBytes);
    require(a.get() != nullptr && b.get() != nullptr, "device_malloc succeeds");

    require(pocket::memcpy_h2d(a.get(), host_in.data(), kBytes), "H2D copy");
    require(pocket::memcpy_d2d(b.get(), a.get(), kBytes), "D2D copy");
    require(pocket::memcpy_d2h(host_out.data(), b.get(), kBytes), "D2H copy");
    require(host_out == host_in, "H2D -> D2D -> D2H round-trip is bit-exact");

    // memset writes a byte pattern, so every byte of every word must match.
    require(pocket::device_memset(a.get(), 0, kBytes), "device_memset to zero");
    require(pocket::memcpy_d2h(host_out.data(), a.get(), kBytes),
            "D2H after memset");
    bool all_zero = true;
    for (uint32_t value : host_out) all_zero = all_zero && (value == 0u);
    require(all_zero, "device_memset zeroes the whole buffer");

    require(pocket::device_memset(a.get(), 0xAB, kBytes), "device_memset to 0xAB");
    require(pocket::memcpy_d2h(host_out.data(), a.get(), kBytes),
            "D2H after byte-pattern memset");
    require(host_out[0] == 0xABABABABu, "memset replicates the byte pattern");

    // A zero-length copy must be a no-op, not an error: the engine issues these
    // for empty batches and should not have to special-case them.
    require(pocket::memcpy_h2d(a.get(), host_in.data(), 0), "zero-length H2D");
    require(pocket::memcpy_d2h(host_out.data(), a.get(), 0), "zero-length D2H");
}

// The typed allocation wrappers exist so call sites can drop the void** cast.
// Verify the pointer they hand back is usable and that the failure path reports
// false rather than a stale pointer.
void check_typed_allocation() {
    constexpr size_t kCount = 256;

    float* device_ptr = nullptr;
    require(pocket::device_malloc_into(device_ptr, kCount * sizeof(float)),
            "device_malloc_into reports success");
    require(device_ptr != nullptr, "device_malloc_into fills the typed pointer");

    std::vector<float> host_in(kCount, 2.5f);
    std::vector<float> host_out(kCount, 0.0f);
    require(pocket::memcpy_h2d(device_ptr, host_in.data(), kCount * sizeof(float)),
            "H2D into device_malloc_into buffer");
    require(pocket::memcpy_d2h(host_out.data(), device_ptr, kCount * sizeof(float)),
            "D2H from device_malloc_into buffer");
    require(host_out == host_in, "device_malloc_into buffer round-trips");
    pocket::device_free(device_ptr);

    uint32_t* pinned = nullptr;
    require(pocket::host_alloc_pinned_into(pinned, kCount * sizeof(uint32_t)),
            "host_alloc_pinned_into reports success");
    require(pinned != nullptr, "host_alloc_pinned_into fills the typed pointer");
    pinned[0] = 7u;
    pinned[kCount - 1] = 9u;
    require(pinned[0] == 7u && pinned[kCount - 1] == 9u,
            "pinned host memory is directly writable");
    pocket::host_free_pinned(pinned);
}

// Strided 2D copies back the engine's confidence-head concat, where a narrow
// source is written into a wider destination row. Check that the untouched tail
// of each destination row survives.
void check_strided_2d_copy() {
    constexpr size_t kRows = 8;
    constexpr size_t kSrcCols = 4;
    constexpr size_t kDstCols = 6;
    const size_t src_pitch = kSrcCols * sizeof(uint32_t);
    const size_t dst_pitch = kDstCols * sizeof(uint32_t);

    std::vector<uint32_t> host_src(kRows * kSrcCols);
    std::iota(host_src.begin(), host_src.end(), 100u);
    std::vector<uint32_t> host_dst(kRows * kDstCols, 0xFFFFFFFFu);

    DeviceBuffer src(host_src.size() * sizeof(uint32_t));
    DeviceBuffer dst(host_dst.size() * sizeof(uint32_t));
    require(pocket::memcpy_h2d(src.get(), host_src.data(),
                             host_src.size() * sizeof(uint32_t)),
            "H2D 2D source");
    require(pocket::memcpy_h2d(dst.get(), host_dst.data(),
                             host_dst.size() * sizeof(uint32_t)),
            "H2D 2D destination sentinel");

    require(pocket::memcpy_2d_d2d(dst.get(), dst_pitch, src.get(), src_pitch,
                                src_pitch, kRows),
            "memcpy_2d_d2d succeeds");
    require(pocket::memcpy_d2h(host_dst.data(), dst.get(),
                             host_dst.size() * sizeof(uint32_t)),
            "D2H after 2D copy");

    bool rows_ok = true;
    for (size_t row = 0; row < kRows; ++row) {
        for (size_t col = 0; col < kSrcCols; ++col) {
            rows_ok = rows_ok && host_dst[row * kDstCols + col] ==
                                     host_src[row * kSrcCols + col];
        }
        // The columns past the copied width must keep the sentinel.
        for (size_t col = kSrcCols; col < kDstCols; ++col) {
            rows_ok = rows_ok && host_dst[row * kDstCols + col] == 0xFFFFFFFFu;
        }
    }
    require(rows_ok, "2D copy places each row at its pitch and leaves the tail");

    // Host-to-device and device-to-host 2D forms, using the same strides.
    std::vector<uint32_t> wide(kRows * kDstCols, 0u);
    require(pocket::memcpy_2d_h2d(dst.get(), dst_pitch, host_src.data(), src_pitch,
                                src_pitch, kRows),
            "memcpy_2d_h2d succeeds");
    require(pocket::memcpy_2d_d2h(wide.data(), dst_pitch, dst.get(), dst_pitch,
                                src_pitch, kRows),
            "memcpy_2d_d2h succeeds");
    bool h2d_ok = true;
    for (size_t row = 0; row < kRows; ++row) {
        for (size_t col = 0; col < kSrcCols; ++col) {
            h2d_ok = h2d_ok && wide[row * kDstCols + col] ==
                                   host_src[row * kSrcCols + col];
        }
    }
    require(h2d_ok, "2D H2D then D2H round-trips the strided region");

    require(pocket::memcpy_2d_d2d(dst.get(), dst_pitch, src.get(), src_pitch, 0, 0),
            "zero-extent 2D copy is a no-op");
}

void check_streams_and_pinned() {
    constexpr size_t kCount = 8192;
    constexpr size_t kBytes = kCount * sizeof(uint32_t);

    void* stream = pocket::stream_create();
    require(stream != nullptr, "stream_create succeeds");

    auto* pinned_in = static_cast<uint32_t*>(pocket::host_alloc_pinned(kBytes));
    auto* pinned_out = static_cast<uint32_t*>(pocket::host_alloc_pinned(kBytes));
    require(pinned_in != nullptr && pinned_out != nullptr,
            "host_alloc_pinned succeeds");

    for (size_t i = 0; i < kCount; ++i) pinned_in[i] = static_cast<uint32_t>(i * 7u + 3u);
    for (size_t i = 0; i < kCount; ++i) pinned_out[i] = 0u;

    {
        DeviceBuffer a(kBytes);
        DeviceBuffer b(kBytes);
        require(a.get() != nullptr && b.get() != nullptr,
                "device_malloc for the async path");

        require(pocket::device_memset_async(b.get(), 0, kBytes, stream),
                "device_memset_async");
        require(pocket::memcpy_h2d_async(a.get(), pinned_in, kBytes, stream),
                "async H2D");
        require(pocket::memcpy_d2d_async(b.get(), a.get(), kBytes, stream),
                "async D2D");
        require(pocket::memcpy_d2h_async(pinned_out, b.get(), kBytes, stream),
                "async D2H");
        require(pocket::stream_synchronize(stream), "stream_synchronize");

        bool equal = true;
        for (size_t i = 0; i < kCount; ++i) {
            equal = equal && (pinned_out[i] == pinned_in[i]);
        }
        require(equal, "async round-trip over pinned memory is bit-exact");
    }

    pocket::host_free_pinned(pinned_in);
    pocket::host_free_pinned(pinned_out);
    pocket::stream_destroy(stream);
}

void check_events() {
    constexpr size_t kCount = 16384;
    constexpr size_t kBytes = kCount * sizeof(uint32_t);

    void* producer = pocket::stream_create();
    void* consumer = pocket::stream_create();
    require(producer != nullptr && consumer != nullptr,
            "two streams for the ordering test");

    void* event = pocket::event_create(false);
    require(event != nullptr, "event_create without timing");

    std::vector<uint32_t> host_in(kCount);
    std::iota(host_in.begin(), host_in.end(), 11u);
    std::vector<uint32_t> host_out(kCount, 0u);

    {
        DeviceBuffer a(kBytes);
        DeviceBuffer b(kBytes);
        require(a.get() != nullptr && b.get() != nullptr,
                "device_malloc for the ordering test");

        // Cross-stream ordering: the consumer's read of `a` must not start before
        // the producer's write to `a` finished. Without stream_wait_event the D2D
        // below would be free to race.
        require(pocket::memcpy_h2d_async(a.get(), host_in.data(), kBytes, producer),
                "producer H2D");
        require(pocket::event_record(event, producer), "event_record on producer");
        require(pocket::stream_wait_event(consumer, event),
                "consumer waits on the producer event");
        require(pocket::memcpy_d2d_async(b.get(), a.get(), kBytes, consumer),
                "consumer D2D after the wait");
        require(pocket::stream_synchronize(consumer), "consumer synchronize");
        require(pocket::event_synchronize(event), "event_synchronize");
        require(pocket::memcpy_d2h(host_out.data(), b.get(), kBytes),
                "D2H of the consumer result");
        require(host_out == host_in, "cross-stream ordering held");
    }

    pocket::event_destroy(event);

    // Timing events measure a real interval. The bound is loose on purpose: this
    // asserts the API works, not how fast the device is.
    void* start = pocket::event_create(true);
    void* end = pocket::event_create(true);
    require(start != nullptr && end != nullptr, "event_create with timing");
    {
        DeviceBuffer a(kBytes);
        require(a.get() != nullptr, "device_malloc for the timing test");
        require(pocket::event_record(start, producer), "record start event");
        for (int i = 0; i < 32; ++i) {
            require(pocket::memcpy_h2d_async(a.get(), host_in.data(), kBytes, producer),
                    "timed H2D");
        }
        require(pocket::event_record(end, producer), "record end event");
        require(pocket::stream_synchronize(producer), "synchronize the timed stream");
        float ms = -1.0f;
        require(pocket::event_elapsed_ms(start, end, &ms), "event_elapsed_ms");
        require(ms >= 0.0f, "elapsed time is not negative");
        std::printf("runtime timed_32x%zuB elapsed_ms=%.4f\n", kBytes, ms);
    }
    pocket::event_destroy(start);
    pocket::event_destroy(end);

    pocket::stream_destroy(producer);
    pocket::stream_destroy(consumer);

    require(pocket::device_synchronize(), "device_synchronize");
}

// Binding a second device must work when one is available, and must not disturb
// the first. This is the case where the ACL per-thread context handling would
// break if device_set did not reinstall it.
void check_second_device() {
    if (pocket::device_count() < 2) {
        std::printf("runtime second_device=skipped (only one device)\n");
        return;
    }
    constexpr size_t kBytes = 1024;
    std::vector<uint8_t> host_in(kBytes, 0x5Au);
    std::vector<uint8_t> host_out(kBytes, 0u);

    require(pocket::device_set(1), "device_set(1)");
    {
        DeviceBuffer a(kBytes);
        require(a.get() != nullptr, "device_malloc on device 1");
        require(pocket::memcpy_h2d(a.get(), host_in.data(), kBytes),
                "H2D on device 1");
        require(pocket::memcpy_d2h(host_out.data(), a.get(), kBytes),
                "D2H on device 1");
        require(host_out == host_in, "round-trip on device 1 is bit-exact");
    }
    require(pocket::device_set(0), "device_set back to 0");
    {
        DeviceBuffer a(kBytes);
        require(a.get() != nullptr, "device_malloc after switching back");
        require(pocket::memcpy_h2d(a.get(), host_in.data(), kBytes),
                "H2D after switching back");
    }
}

}  // namespace

int main() {
    // Never throws, so a host with no accelerator toolkit reports a skip instead
    // of a failure.
    if (!pocket::device_runtime_available()) {
        std::printf("test_device_runtime SKIP: no %s device available\n",
                    pocket::device_backend_name());
        return 0;
    }

    try {
        if (!pocket::device_set(0)) {
            std::printf("test_device_runtime SKIP: cannot bind device 0 (%s)\n",
                        pocket::device_last_error().c_str());
            return 0;
        }
        check_device_query();
        check_sync_transfers();
        check_typed_allocation();
        check_strided_2d_copy();
        check_streams_and_pinned();
        check_events();
        check_second_device();
    } catch (const std::exception& error) {
        std::printf("test_device_runtime FAIL: %s\n", error.what());
        return 1;
    }

    std::printf("test_device_runtime PASS (%d checks, backend=%s)\n", g_checks,
                pocket::device_backend_name());
    return 0;
}
