// Tensor-parallel collective smoke test, backend independent.
//
// One process per rank. Rank 0 publishes the rendezvous id to --id-path and the
// others poll for it, which is the same handshake the engine uses.
//
//   for r in 0 1 2 3; do
//     ./tests/test_tp_comm_smoke --world 4 --rank $r --device $r \
//         --id-path /tmp/pocket_tp_id.bin &
//   done; wait
//
// Rank 0 should delete a stale --id-path first; a leftover id from a previous run
// makes every rank hang in comm init waiting for peers that no longer exist.
//
// This is deliberately the smallest program that can prove the collective works:
// no engine, no weights, no kernels. On Ascend it is the regression test for
// HcclCommInitAll being unusable (it segfaults once a device is set), which is
// why the engine goes through the root-info path instead.

#include "device_runtime.hpp"
#include "tp_comm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "  FAIL " << what << "\n";
    }
}

// IEEE half decode, enough for the magnitudes this test produces (no subnormals,
// no infinities). Kept local so the test does not depend on a device header.
float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const int32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ffu;
    if (exponent == 0) return sign ? -0.0f : 0.0f;
    const uint32_t bits =
        sign | (static_cast<uint32_t>(exponent - 15 + 127) << 23) | (mantissa << 13);
    float out = 0.0f;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

}  // namespace

namespace {

// Device scratch that frees itself, so an expect() failure below cannot leak the
// allocation on the way out.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (!pocket::device_malloc_into(ptr_, count * sizeof(T))) {
            throw std::runtime_error("device_malloc failed in tp_comm smoke test");
        }
    }
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

    void upload(const T* host) {
        if (!pocket::memcpy_h2d(ptr_, host, count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_h2d failed in tp_comm smoke test");
        }
    }
    void download(T* host) {
        if (!pocket::memcpy_d2h(host, ptr_, count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_d2h failed in tp_comm smoke test");
        }
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    int world = 1;
    int rank = 0;
    int device = 0;
    std::string id_path = "/tmp/pocket_tp_id.bin";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--world" && i + 1 < argc) world = std::stoi(argv[++i]);
        else if (arg == "--rank" && i + 1 < argc) rank = std::stoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
        else if (arg == "--id-path" && i + 1 < argc) id_path = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    if (!pocket::tp_comm_available()) {
        std::cout << "[SKIP] no collective library in this build\n";
        return 0;
    }
#ifdef POCKET_HAVE_TP_COMM
    std::cout << "rank " << rank << "/" << world << " device " << device << "\n";

    pocket::run_tp_float_sum_smoke(world, rank, device, id_path.c_str(),
                                 static_cast<float>(rank + 1));

    // FP16 all-reduce over one hidden row, the shape the engine's TP reduce
    // actually uses. Every rank contributes a different value per element so a
    // no-op reduce (or one that only reaches a subset of ranks) cannot pass.
    const int count = 5120;
    std::vector<uint16_t> host(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        host[static_cast<size_t>(i)] =
            float_to_half(static_cast<float>(rank + 1) * 0.5f +
                          static_cast<float>(i % 8));
    }

    if (!pocket::device_set(device)) {
        throw std::runtime_error("device_set failed in tp_comm smoke test");
    }
    DeviceBuffer<uint16_t> d_values(static_cast<size_t>(count));

    // Repeat, because a communicator cached across calls is a different code
    // path from a freshly created one and only the first iteration exercises
    // creation.
    for (int iter = 0; iter < 3; ++iter) {
        std::vector<uint16_t> values = host;
        d_values.upload(values.data());
        pocket::tp_all_reduce_sum_f16_inplace(world, rank, device, id_path.c_str(),
                                            d_values.get(), count);
        if (!pocket::device_synchronize()) {
            throw std::runtime_error("device_synchronize failed after all-reduce");
        }
        d_values.download(values.data());
        int mismatches = 0;
        for (int i = 0; i < count; ++i) {
            float want = 0.0f;
            for (int r = 0; r < world; ++r) {
                want += static_cast<float>(r + 1) * 0.5f + static_cast<float>(i % 8);
            }
            const float got = half_to_float(values[static_cast<size_t>(i)]);
            // FP16 has ~3 decimal digits and the reduce order is not specified,
            // so compare relatively rather than bit-exactly.
            if (std::fabs(got - want) > 0.01f * std::fabs(want) + 1e-3f) ++mismatches;
        }
        expect(mismatches == 0, "fp16 all-reduce iteration " + std::to_string(iter) +
                                    ": " + std::to_string(mismatches) +
                                    " mismatched elements");
    }

    // Global top-1 across ranks. Rank `world - 1` holds the winning logit, so a
    // collective that returned the local value would disagree here on every
    // other rank.
    const int rows = 4;
    std::vector<int> local_tokens(static_cast<size_t>(rows));
    std::vector<float> local_logits(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        local_tokens[static_cast<size_t>(row)] = rank * 1000 + row;
        local_logits[static_cast<size_t>(row)] =
            static_cast<float>(rank) + 0.25f * static_cast<float>(row);
    }
    std::vector<int> global_tokens(static_cast<size_t>(rows), -1);
    std::vector<float> global_logits(static_cast<size_t>(rows), 0.0f);
    DeviceBuffer<int> d_local_tokens(static_cast<size_t>(rows));
    DeviceBuffer<float> d_local_logits(static_cast<size_t>(rows));
    d_local_tokens.upload(local_tokens.data());
    d_local_logits.upload(local_logits.data());
    // Inputs are device-resident, outputs are host: that is the signature the
    // engine's sampler uses on the non-device-resident path.
    pocket::tp_global_top1_rows(world, rank, device, id_path.c_str(),
                              d_local_tokens.get(), d_local_logits.get(), rows,
                              global_tokens.data(), global_logits.data());
    for (int row = 0; row < rows; ++row) {
        const int want_token = (world - 1) * 1000 + row;
        const float want_logit =
            static_cast<float>(world - 1) + 0.25f * static_cast<float>(row);
        expect(global_tokens[static_cast<size_t>(row)] == want_token,
               "top1 row " + std::to_string(row) + " token " +
                   std::to_string(global_tokens[static_cast<size_t>(row)]) +
                   " != " + std::to_string(want_token));
        expect(std::fabs(global_logits[static_cast<size_t>(row)] - want_logit) < 1e-5f,
               "top1 row " + std::to_string(row) + " logit");
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " rank " << rank << " ("
              << failures << " failures)\n";
    return failures == 0 ? 0 : 1;
#else
    return 0;
#endif
}
