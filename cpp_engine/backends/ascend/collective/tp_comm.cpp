// HCCL implementation of the tensor-parallel collective contract.
//
// Two things about HCCL on CANN 9.0.0 / first-generation Ascend 910 shape this
// file, and both are traps:
//
//   1. HcclCommInitAll (one process, several devices) is unusable. With no device
//      set it returns HCCL_E_RUNTIME reporting a null context; once any
//      aclrtSetDevice has run it segfaults, with or without an explicit
//      aclrtCreateContext, and resetting the device first does not help. So this
//      file only ever uses the root-info path: one process per rank, rank 0
//      publishes a 4108-byte HcclRootInfo through a shared file.
//
//   2. Every init entry point in hccl_comm.h is declared __attribute__((weak)).
//      Forgetting to link libhccl therefore does not fail the link -- the call
//      jumps to address zero and crashes at run time, looking exactly like the
//      HcclCommInitAll bug above. CMake treats a missing libhccl as fatal, and
//      comm_or_throw below checks the pointer anyway so the failure names itself.
//
// The top-k merge helpers the CUDA backend runs as device kernels are done on the
// host here. That is a deliberate correctness-first choice: the merge touches
// world * rows * top_k elements, which is kilobytes per step, not a hot path
// worth an AscendC kernel before profiling says so.

#include "tp_comm.hpp"

#ifdef POCKET_HAVE_TP_COMM
#include "device_runtime.hpp"

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#endif

namespace pocket {

bool tp_comm_available() {
#ifdef POCKET_HAVE_TP_COMM
    return true;
#else
    return false;
#endif
}

#ifdef POCKET_HAVE_TP_COMM
namespace {

void check_acl(aclError err, const char* what) {
    if (err != ACL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": ACL error " +
                                 std::to_string(static_cast<int>(err)));
    }
}

void check_hccl(HcclResult err, const char* what) {
    if (err != HCCL_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": HCCL result " +
                                 std::to_string(static_cast<int>(err)));
    }
}

void validate(int world, int rank, const char* what) {
    if (world <= 0 || rank < 0 || rank >= world) {
        throw std::runtime_error(std::string(what) + ": invalid world/rank");
    }
}

// Bind this thread to the device through the shared runtime rather than calling
// aclrtSetDevice here. The runtime installs an explicit ACL context per device;
// a bare aclrtSetDevice gets a different (implicit) context, and a communicator
// created under one context cannot operate on buffers allocated under another.
// That mismatch surfaces as HCCL_E_RUNTIME from the first collective, or as the
// "ctx is NULL" report from HcclCommInitAll -- never as an error at init.
void bind_device(int device) {
    if (!device_set(device)) {
        throw std::runtime_error("device_set failed before HCCL operation on device " +
                                 std::to_string(device));
    }
}

HcclRootInfo load_or_create_root_info(int rank, const char* path) {
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("TP collectives need a rendezvous id path");
    }
    HcclRootInfo info{};
    if (rank == 0) {
        // Reuse an id this process already published, so repeated calls in one
        // run rendezvous with the peers that are already waiting on it.
        std::ifstream existing(path, std::ios::binary);
        if (existing) {
            existing.read(reinterpret_cast<char*>(&info), sizeof(info));
            if (existing.gcount() == static_cast<std::streamsize>(sizeof(info))) {
                return info;
            }
        }
        check_hccl(HcclGetRootInfo(&info), "HcclGetRootInfo");
        // Write to a temporary and rename, so a peer never reads a partially
        // written 4108-byte blob and then blocks forever on a malformed id.
        const std::string tmp = std::string(path) + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("failed to open HCCL id temp file");
            out.write(reinterpret_cast<const char*>(&info), sizeof(info));
            if (!out) throw std::runtime_error("failed to write HCCL id bytes");
        }
        if (std::rename(tmp.c_str(), path) != 0) {
            throw std::runtime_error("failed to publish HCCL id file");
        }
        return info;
    }

    // Same generous bound as the CUDA path: ranks reach their first collective
    // only after loading weights, and four processes reading one checkpoint off
    // the same disk skew by minutes. A short timeout makes a slow loader look
    // like a communicator failure.
    int attempts = 6000;  // 10 minutes at 100ms
    if (const char* env = std::getenv("POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS")) {
        const int value = std::atoi(env);
        if (value > 0) attempts = value;
    }
    for (int attempt = 0; attempt < attempts; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.read(reinterpret_cast<char*>(&info), sizeof(info));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(info))) return info;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("timed out waiting for HCCL id file");
}

HcclComm cached_comm(int world, int rank, int device, const char* id_path) {
    static std::unordered_map<std::string, HcclComm> comms;
    const std::string key = std::to_string(world) + ":" + std::to_string(rank) +
                            ":" + std::to_string(device) + ":" +
                            (id_path == nullptr ? "" : id_path);
    auto it = comms.find(key);
    if (it != comms.end()) {
        // Re-bind on every call: a cached communicator says nothing about which
        // context this thread currently has installed.
        bind_device(device);
        return it->second;
    }

    bind_device(device);
    const HcclRootInfo info = load_or_create_root_info(rank, id_path);
    HcclComm comm = nullptr;
    check_hccl(HcclCommInitRootInfo(static_cast<uint32_t>(world), &info,
                                    static_cast<uint32_t>(rank), &comm),
               "HcclCommInitRootInfo");
    if (comm == nullptr) {
        // Reachable if libhccl resolved as a weak stub: the call "succeeds"
        // without producing a communicator.
        throw std::runtime_error(
            "HcclCommInitRootInfo returned a null communicator; is libhccl linked?");
    }
    comms.emplace(key, comm);
    return comm;
}

// HCCL rejects a null stream with HCCL_E_PTR, where every aclrt* entry point
// reads nullptr as "the default stream". So a caller passing nullptr -- which in
// this API means "synchronous, result visible on return" -- needs a real stream
// substituted. One per device, created once.
aclrtStream internal_stream(int device) {
    static std::unordered_map<int, aclrtStream> streams;
    auto it = streams.find(device);
    if (it != streams.end()) return it->second;
    aclrtStream stream = nullptr;
    check_acl(aclrtCreateStream(&stream), "aclrtCreateStream for collective");
    streams.emplace(device, stream);
    return stream;
}

// Resolve the stream a collective should run on, and report whether the caller
// asked for synchronous semantics. Substituting a stream loses ordering against
// work already queued on the default stream, so drain that first; the nullptr
// paths are synchronous by contract, so this costs no overlap that existed.
aclrtStream resolve_stream(void* stream, int device, bool& synchronous) {
    synchronous = (stream == nullptr);
    if (!synchronous) return static_cast<aclrtStream>(stream);
    check_acl(aclrtSynchronizeStream(nullptr),
              "drain default stream before collective");
    return internal_stream(device);
}

// Scratch for the all-gather that backs the top-1 and top-k merges. Sized once
// per device and grown on demand, because the rows count varies with the batch.
struct GatherWorkspace {
    void* d_tokens = nullptr;
    void* d_logits = nullptr;
    size_t capacity = 0;
};

GatherWorkspace& gather_workspace(int device, size_t count) {
    static std::unordered_map<int, GatherWorkspace> workspaces;
    GatherWorkspace& workspace = workspaces[device];
    if (workspace.capacity >= count) return workspace;
    if (workspace.d_tokens != nullptr) aclrtFree(workspace.d_tokens);
    if (workspace.d_logits != nullptr) aclrtFree(workspace.d_logits);
    workspace.d_tokens = nullptr;
    workspace.d_logits = nullptr;
    workspace.capacity = 0;
    check_acl(aclrtMalloc(&workspace.d_tokens, count * sizeof(int32_t),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc gather tokens");
    check_acl(aclrtMalloc(&workspace.d_logits, count * sizeof(float),
                          ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc gather logits");
    workspace.capacity = count;
    return workspace;
}

// All-gather both candidate arrays and bring them back to the host in the
// world-major [rank][row * top_k] layout the merge helpers below expect.
void gather_candidates(int world, int rank, int device, const char* id_path,
                       const int* d_local_tokens, const float* d_local_logits,
                       size_t local_count, std::vector<int>& all_tokens,
                       std::vector<float>& all_logits, void* stream) {
    HcclComm comm = cached_comm(world, rank, device, id_path);
    const size_t gathered = static_cast<size_t>(world) * local_count;
    GatherWorkspace& workspace = gather_workspace(device, gathered);
    bool synchronous = false;
    aclrtStream acl_stream = resolve_stream(stream, device, synchronous);

    check_hccl(HcclAllGather(const_cast<void*>(static_cast<const void*>(d_local_tokens)),
                             workspace.d_tokens, local_count, HCCL_DATA_TYPE_INT32,
                             comm, acl_stream),
               "HcclAllGather candidate tokens");
    check_hccl(HcclAllGather(const_cast<void*>(static_cast<const void*>(d_local_logits)),
                             workspace.d_logits, local_count, HCCL_DATA_TYPE_FP32,
                             comm, acl_stream),
               "HcclAllGather candidate logits");
    // Unconditional: the merge below reads the gathered buffers on the host, so
    // it must wait for the transfer even when the caller supplied a stream.
    check_acl(aclrtSynchronizeStream(acl_stream), "sync gathered candidates");

    all_tokens.resize(gathered);
    all_logits.resize(gathered);
    check_acl(aclrtMemcpy(all_tokens.data(), gathered * sizeof(int32_t),
                          workspace.d_tokens, gathered * sizeof(int32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "copy gathered tokens");
    check_acl(aclrtMemcpy(all_logits.data(), gathered * sizeof(float),
                          workspace.d_logits, gathered * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "copy gathered logits");
}

// The comparator has to match the CUDA backend's bit for bit, or two ranks pick
// different tokens and the sequences diverge: larger logit wins, and an exact tie
// picks the smaller token id.
bool beats(float candidate_logit, int candidate_token, float best_logit,
           int best_token) {
    if (candidate_logit > best_logit) return true;
    if (candidate_logit < best_logit) return false;
    return candidate_token < best_token;
}

}  // namespace

void run_tp_float_sum_smoke(int world, int rank, int device, const char* id_path,
                            float value) {
    validate(world, rank, "TP float sum smoke");
    HcclComm comm = cached_comm(world, rank, device, id_path);
    void* d_value = nullptr;
    void* d_sum = nullptr;
    check_acl(aclrtMalloc(&d_value, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc smoke value");
    check_acl(aclrtMalloc(&d_sum, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc smoke sum");
    check_acl(aclrtMemcpy(d_value, sizeof(float), &value, sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy smoke input");
    bool synchronous = false;
    aclrtStream stream = resolve_stream(nullptr, device, synchronous);
    check_hccl(HcclAllReduce(d_value, d_sum, 1, HCCL_DATA_TYPE_FP32,
                             HCCL_REDUCE_SUM, comm, stream),
               "HcclAllReduce smoke");
    check_acl(aclrtSynchronizeStream(stream), "sync smoke all-reduce");
    float sum = 0.0f;
    check_acl(aclrtMemcpy(&sum, sizeof(float), d_sum, sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "copy smoke sum");
    std::cout << "tp_comm_smoke rank=" << rank << " value=" << value << " sum="
              << sum << "\n";
    aclrtFree(d_value);
    aclrtFree(d_sum);
}

void tp_all_reduce_sum_float_inplace(int world, int rank, int device,
                                     const char* id_path, float* d_values,
                                     int count, void* stream) {
    validate(world, rank, "TP fp32 all-reduce");
    if (d_values == nullptr || count <= 0) {
        throw std::runtime_error("TP fp32 all-reduce: invalid buffer");
    }
    HcclComm comm = cached_comm(world, rank, device, id_path);
    bool synchronous = false;
    aclrtStream acl_stream = resolve_stream(stream, device, synchronous);
    check_hccl(HcclAllReduce(d_values, d_values, static_cast<uint64_t>(count),
                             HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, acl_stream),
               "HcclAllReduce fp32");
    if (synchronous) {
        check_acl(aclrtSynchronizeStream(acl_stream), "sync fp32 all-reduce");
    }
}

void tp_all_reduce_sum_f16_inplace(int world, int rank, int device,
                                   const char* id_path, uint16_t* d_values,
                                   int count, void* stream) {
    validate(world, rank, "TP fp16 all-reduce");
    if (d_values == nullptr || count <= 0) {
        throw std::runtime_error("TP fp16 all-reduce: invalid buffer");
    }
    HcclComm comm = cached_comm(world, rank, device, id_path);
    bool synchronous = false;
    aclrtStream acl_stream = resolve_stream(stream, device, synchronous);
    check_hccl(HcclAllReduce(d_values, d_values, static_cast<uint64_t>(count),
                             HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, comm, acl_stream),
               "HcclAllReduce fp16");
    if (synchronous) {
        check_acl(aclrtSynchronizeStream(acl_stream), "sync fp16 all-reduce");
    }
}

void tp_all_reduce_sum_bf16_inplace(int world, int rank, int device,
                                    const char* id_path, uint16_t* d_values,
                                    int count, void* stream) {
    // HCCL itself accepts HCCL_DATA_TYPE_BFP16, but first-generation 910
    // (Short_SoC_version=Ascend910) has no BF16 arithmetic at all, so a reduce in
    // that dtype has no defined behaviour on this hardware and every device
    // buffer in this build is FP16 anyway. Refuse loudly instead of reducing
    // bytes that cannot have been produced correctly.
    (void)world;
    (void)rank;
    (void)device;
    (void)id_path;
    (void)d_values;
    (void)count;
    (void)stream;
    throw std::runtime_error(
        "BF16 all-reduce is not available on first-generation Ascend 910; "
        "device tensors are FP16 on this backend");
}

void tp_global_top1_rows(int world, int rank, int device, const char* id_path,
                         const int* d_local_tokens, const float* d_local_logits,
                         int rows, int* global_tokens, float* global_logits,
                         void* stream) {
    validate(world, rank, "TP top-1 rows");
    if (rows <= 0 || d_local_tokens == nullptr || d_local_logits == nullptr ||
        global_tokens == nullptr || global_logits == nullptr) {
        throw std::runtime_error("TP top-1 rows: invalid arguments");
    }
    std::vector<int> all_tokens;
    std::vector<float> all_logits;
    gather_candidates(world, rank, device, id_path, d_local_tokens, d_local_logits,
                      static_cast<size_t>(rows), all_tokens, all_logits, stream);
    for (int row = 0; row < rows; ++row) {
        int best_token = all_tokens[static_cast<size_t>(row)];
        float best_logit = all_logits[static_cast<size_t>(row)];
        for (int r = 1; r < world; ++r) {
            const size_t index = static_cast<size_t>(r) * static_cast<size_t>(rows) +
                                 static_cast<size_t>(row);
            if (beats(all_logits[index], all_tokens[index], best_logit, best_token)) {
                best_logit = all_logits[index];
                best_token = all_tokens[index];
            }
        }
        global_tokens[row] = best_token;
        global_logits[row] = best_logit;
    }
}

void tp_global_top1_rows_device(int world, int rank, int device,
                                const char* id_path, const int* d_local_tokens,
                                const float* d_local_logits, int rows,
                                int* d_global_tokens, float* d_global_logits,
                                void* stream) {
    validate(world, rank, "TP device top-1 rows");
    if (rows <= 0 || d_local_tokens == nullptr || d_local_logits == nullptr ||
        d_global_tokens == nullptr || d_global_logits == nullptr) {
        throw std::runtime_error("TP device top-1 rows: invalid arguments");
    }
    // The CUDA backend keeps the merge on device to avoid a sync in the decode
    // loop. Here it round-trips through the host: correct and identical in
    // result, but it does serialize the stream. Revisit with an AscendC merge
    // kernel once profiling shows this in the decode critical path.
    std::vector<int> host_tokens(static_cast<size_t>(rows));
    std::vector<float> host_logits(static_cast<size_t>(rows));
    tp_global_top1_rows(world, rank, device, id_path, d_local_tokens,
                        d_local_logits, rows, host_tokens.data(),
                        host_logits.data(), stream);
    check_acl(aclrtMemcpy(d_global_tokens, host_tokens.size() * sizeof(int),
                          host_tokens.data(), host_tokens.size() * sizeof(int),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy merged top-1 tokens");
    check_acl(aclrtMemcpy(d_global_logits, host_logits.size() * sizeof(float),
                          host_logits.data(), host_logits.size() * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy merged top-1 logits");
}

void tp_global_top1_rows_packed_device(int world, int rank, int device,
                                       const char* id_path,
                                       const int* d_local_tokens,
                                       const float* d_local_logits, int rows,
                                       int* d_global_tokens,
                                       float* d_global_logits, void* stream) {
    // The packed uint64 Max variant exists on CUDA purely to halve the collective
    // count. Same result, so forward rather than maintain a second merge.
    tp_global_top1_rows_device(world, rank, device, id_path, d_local_tokens,
                               d_local_logits, rows, d_global_tokens,
                               d_global_logits, stream);
}

void tp_global_topk_rows(int world, int rank, int device, const char* id_path,
                         const int* d_local_tokens, const float* d_local_logits,
                         int rows, int top_k, int* global_tokens,
                         float* global_logits, void* stream) {
    validate(world, rank, "TP top-k rows");
    if (rows <= 0 || top_k <= 0 || d_local_tokens == nullptr ||
        d_local_logits == nullptr || global_tokens == nullptr ||
        global_logits == nullptr) {
        throw std::runtime_error("TP top-k rows: invalid arguments");
    }
    const size_t local_count = static_cast<size_t>(rows) * static_cast<size_t>(top_k);
    std::vector<int> all_tokens;
    std::vector<float> all_logits;
    gather_candidates(world, rank, device, id_path, d_local_tokens, d_local_logits,
                      local_count, all_tokens, all_logits, stream);

    // Per row, merge the world * top_k candidates down to the best top_k. Sorting
    // by the same comparator as the top-1 path keeps the tie rule consistent, so
    // a top_k of 1 agrees with tp_global_top1_rows exactly.
    std::vector<size_t> order;
    for (int row = 0; row < rows; ++row) {
        order.clear();
        for (int r = 0; r < world; ++r) {
            for (int k = 0; k < top_k; ++k) {
                order.push_back(static_cast<size_t>(r) * local_count +
                                static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                                static_cast<size_t>(k));
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return beats(all_logits[a], all_tokens[a], all_logits[b], all_tokens[b]);
        });
        for (int k = 0; k < top_k; ++k) {
            const size_t out = static_cast<size_t>(row) * static_cast<size_t>(top_k) +
                               static_cast<size_t>(k);
            const size_t src = order[static_cast<size_t>(k)];
            global_tokens[out] = all_tokens[src];
            global_logits[out] = all_logits[src];
        }
    }
}

void tp_global_topk_rows_device(int world, int rank, int device,
                                const char* id_path, const int* d_local_tokens,
                                const float* d_local_logits, int rows, int top_k,
                                int* d_global_tokens, float* d_global_logits,
                                void* stream) {
    validate(world, rank, "TP device top-k rows");
    if (rows <= 0 || top_k <= 0 || d_global_tokens == nullptr ||
        d_global_logits == nullptr) {
        throw std::runtime_error("TP device top-k rows: invalid arguments");
    }
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(top_k);
    std::vector<int> host_tokens(count);
    std::vector<float> host_logits(count);
    tp_global_topk_rows(world, rank, device, id_path, d_local_tokens,
                        d_local_logits, rows, top_k, host_tokens.data(),
                        host_logits.data(), stream);
    check_acl(aclrtMemcpy(d_global_tokens, count * sizeof(int), host_tokens.data(),
                          count * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE),
              "copy merged top-k tokens");
    check_acl(aclrtMemcpy(d_global_logits, count * sizeof(float),
                          host_logits.data(), count * sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy merged top-k logits");
}

void tp_broadcast_int32(int world, int rank, int device, const char* id_path,
                        int32_t* buf, int count, int root) {
    validate(world, rank, "TP int32 broadcast");
    if (buf == nullptr || count <= 0) {
        throw std::runtime_error("TP int32 broadcast: invalid buffer");
    }
    if (root < 0 || root >= world) {
        throw std::runtime_error("TP int32 broadcast: invalid root");
    }
    HcclComm comm = cached_comm(world, rank, device, id_path);
    const size_t bytes = static_cast<size_t>(count) * sizeof(int32_t);
    void* d_buf = nullptr;
    check_acl(aclrtMalloc(&d_buf, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc broadcast buffer");
    if (rank == root) {
        check_acl(aclrtMemcpy(d_buf, bytes, buf, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "copy broadcast input");
    }
    bool synchronous = false;
    aclrtStream stream = resolve_stream(nullptr, device, synchronous);
    const HcclResult result =
        HcclBroadcast(d_buf, static_cast<uint64_t>(count), HCCL_DATA_TYPE_INT32,
                      static_cast<uint32_t>(root), comm, stream);
    if (result == HCCL_SUCCESS) {
        const aclError sync = aclrtSynchronizeStream(stream);
        if (sync == ACL_SUCCESS) {
            const aclError copy = aclrtMemcpy(buf, bytes, d_buf, bytes,
                                              ACL_MEMCPY_DEVICE_TO_HOST);
            aclrtFree(d_buf);
            check_acl(copy, "copy broadcast result");
            return;
        }
        aclrtFree(d_buf);
        check_acl(sync, "sync broadcast");
        return;
    }
    aclrtFree(d_buf);
    check_hccl(result, "HcclBroadcast int32");
}

void tp_gather_floats_to_root(int world, int rank, int device,
                              const char* id_path, const float* h_local,
                              int local_count, float* h_root_out, int root) {
    validate(world, rank, "TP float gather");
    if (h_local == nullptr || local_count <= 0) {
        throw std::runtime_error("TP float gather: invalid buffer");
    }
    if (root < 0 || root >= world) {
        throw std::runtime_error("TP float gather: invalid root");
    }
    if (rank == root && h_root_out == nullptr) {
        throw std::runtime_error("TP float gather: root needs an output buffer");
    }
    HcclComm comm = cached_comm(world, rank, device, id_path);
    const size_t local_bytes = static_cast<size_t>(local_count) * sizeof(float);
    const size_t gathered_bytes = local_bytes * static_cast<size_t>(world);
    void* d_local = nullptr;
    void* d_gathered = nullptr;
    check_acl(aclrtMalloc(&d_local, local_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc gather local");
    check_acl(aclrtMalloc(&d_gathered, gathered_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc gather output");
    check_acl(aclrtMemcpy(d_local, local_bytes, h_local, local_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy gather input");
    // HCCL has no Gather, so all-gather and let non-root ranks discard. The
    // payload is a handful of floats, so the extra traffic does not matter.
    bool synchronous = false;
    aclrtStream stream = resolve_stream(nullptr, device, synchronous);
    const HcclResult result =
        HcclAllGather(d_local, d_gathered, static_cast<uint64_t>(local_count),
                      HCCL_DATA_TYPE_FP32, comm, stream);
    if (result != HCCL_SUCCESS) {
        aclrtFree(d_local);
        aclrtFree(d_gathered);
        check_hccl(result, "HcclAllGather floats");
        return;
    }
    const aclError sync = aclrtSynchronizeStream(stream);
    aclError copy = ACL_SUCCESS;
    if (sync == ACL_SUCCESS && rank == root) {
        copy = aclrtMemcpy(h_root_out, gathered_bytes, d_gathered, gathered_bytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
    }
    aclrtFree(d_local);
    aclrtFree(d_gathered);
    check_acl(sync, "sync gather");
    check_acl(copy, "copy gather result");
}

TpTopResult tp_global_top1(int world, int rank, int device, const char* id_path,
                           int local_token, float local_logit) {
    validate(world, rank, "TP scalar top-1");
    void* d_token = nullptr;
    void* d_logit = nullptr;
    check_acl(aclrtMalloc(&d_token, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc scalar top-1 token");
    check_acl(aclrtMalloc(&d_logit, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST),
              "aclrtMalloc scalar top-1 logit");
    const int32_t token = local_token;
    check_acl(aclrtMemcpy(d_token, sizeof(int32_t), &token, sizeof(int32_t),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy scalar top-1 token");
    check_acl(aclrtMemcpy(d_logit, sizeof(float), &local_logit, sizeof(float),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "copy scalar top-1 logit");
    TpTopResult result{};
    int merged_token = 0;
    float merged_logit = 0.0f;
    tp_global_top1_rows(world, rank, device, id_path,
                        static_cast<const int*>(d_token),
                        static_cast<const float*>(d_logit), 1, &merged_token,
                        &merged_logit, nullptr);
    aclrtFree(d_token);
    aclrtFree(d_logit);
    result.token = merged_token;
    result.logit = merged_logit;
    return result;
}
#endif  // POCKET_HAVE_TP_COMM

}  // namespace pocket
