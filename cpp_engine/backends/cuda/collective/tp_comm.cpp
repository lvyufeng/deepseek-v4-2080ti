#include "tp_comm.hpp"

#ifdef POCKET_HAVE_TP_COMM
// Both are required: sampler_ops.hpp for the sampler uniforms and
// qwen_cuda_ops.hpp for the DFlash2 top-k pack/merge kernels used by the
// global-top1 collective below. The latter is only visible in NCCL builds,
// which is why dropping it still compiled with NCCL off.
#include "qwen_cuda_ops.hpp"
#include "sampler_ops.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

void check_nccl(ncclResult_t err, const char* what) {
    if (err != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(err));
}

ncclUniqueId load_or_create_id(int rank, const char* path) {
    ncclUniqueId id;
    if (rank == 0) {
        std::ifstream existing(path, std::ios::binary);
        if (existing) {
            existing.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (existing.gcount() == static_cast<std::streamsize>(sizeof(id))) return id;
        }
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to write NCCL id file");
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        if (!out) throw std::runtime_error("failed to write NCCL id bytes");
        return id;
    }
    // Wait for rank 0 to publish the id. The bound is generous because the
    // ranks reach their first collective only after loading weights, and that
    // skew is minutes when four processes read a 167 GB checkpoint off the same
    // disk -- a 30s bound made a slow loader look like a communicator failure.
    int attempts = 6000;  // 10 minutes at 100ms
    if (const char* env = std::getenv("POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS")) {
        const int v = std::atoi(env);
        if (v > 0) attempts = v;
    }
    for (int attempt = 0; attempt < attempts; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (in.gcount() == static_cast<std::streamsize>(sizeof(id))) return id;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("timed out waiting for NCCL id file");
}

struct CachedComm {
    ncclComm_t comm = nullptr;
};

struct Top1RowsWorkspace {
    int* d_tokens = nullptr;
    float* d_logits = nullptr;
    size_t capacity = 0;
};

struct PackedTop1RowsWorkspace {
    uint64_t* d_keys = nullptr;
    size_t capacity = 0;
};

struct TopKRowsWorkspace {
    int* d_tokens = nullptr;
    float* d_logits = nullptr;
    size_t capacity = 0;
};

TopKRowsWorkspace& topk_rows_workspace(int device, size_t count) {
    static std::unordered_map<int, TopKRowsWorkspace> workspaces;
    TopKRowsWorkspace& workspace = workspaces[device];
    if (workspace.capacity >= count) return workspace;
    check_cuda(cudaSetDevice(device), "cudaSetDevice top-k merge workspace");
    if (workspace.d_tokens != nullptr) {
        check_cuda(cudaFree(workspace.d_tokens), "cudaFree gathered top-k tokens");
        workspace.d_tokens = nullptr;
    }
    if (workspace.d_logits != nullptr) {
        check_cuda(cudaFree(workspace.d_logits), "cudaFree gathered top-k logits");
        workspace.d_logits = nullptr;
    }
    check_cuda(cudaMalloc(&workspace.d_tokens, count * sizeof(int)),
               "cudaMalloc gathered top-k tokens");
    check_cuda(cudaMalloc(&workspace.d_logits, count * sizeof(float)),
               "cudaMalloc gathered top-k logits");
    workspace.capacity = count;
    return workspace;
}

Top1RowsWorkspace& top1_rows_workspace(int device, size_t count) {
    static std::unordered_map<int, Top1RowsWorkspace> workspaces;
    Top1RowsWorkspace& workspace = workspaces[device];
    if (workspace.capacity >= count) return workspace;
    check_cuda(cudaSetDevice(device), "cudaSetDevice top1 workspace");
    if (workspace.d_tokens != nullptr) {
        check_cuda(cudaFree(workspace.d_tokens), "cudaFree gathered top tokens");
        workspace.d_tokens = nullptr;
    }
    if (workspace.d_logits != nullptr) {
        check_cuda(cudaFree(workspace.d_logits), "cudaFree gathered top logits");
        workspace.d_logits = nullptr;
    }
    check_cuda(cudaMalloc(&workspace.d_tokens, count * sizeof(int)),
               "cudaMalloc gathered batched top tokens");
    check_cuda(cudaMalloc(&workspace.d_logits, count * sizeof(float)),
               "cudaMalloc gathered batched top logits");
    workspace.capacity = count;
    return workspace;
}

PackedTop1RowsWorkspace& packed_top1_rows_workspace(int device, size_t count) {
    static std::unordered_map<int, PackedTop1RowsWorkspace> workspaces;
    PackedTop1RowsWorkspace& workspace = workspaces[device];
    if (workspace.capacity >= count) return workspace;
    check_cuda(cudaSetDevice(device), "cudaSetDevice packed top1 workspace");
    if (workspace.d_keys != nullptr) {
        check_cuda(cudaFree(workspace.d_keys), "cudaFree packed top1 keys");
        workspace.d_keys = nullptr;
    }
    check_cuda(cudaMalloc(&workspace.d_keys, count * sizeof(uint64_t)),
               "cudaMalloc packed top1 keys");
    workspace.capacity = count;
    return workspace;
}

void nccl_all_reduce_max_u64(ncclComm_t comm, uint64_t* d_values, int count,
                             cudaStream_t stream) {
    check_nccl(ncclAllReduce(d_values, d_values, count, ncclUint64, ncclMax,
                             comm, stream), "ncclAllReduce packed top1");
}

void unpack_packed_top1(uint64_t* d_keys, int* d_tokens, float* d_logits,
                        int rows, cudaStream_t stream) {
    if (!qwen_dflash2_unpack_top1_key_f32_cuda(
            d_keys, d_tokens, d_logits, rows, stream)) {
        throw std::runtime_error("Qwen packed top-1 unpack launch failed");
    }
}

void pack_top1(const int* d_tokens, const float* d_logits, uint64_t* d_keys,
               int rows, cudaStream_t stream) {
    if (!qwen_dflash2_pack_top1_key_f32_cuda(
            d_tokens, d_logits, d_keys, rows, stream)) {
        throw std::runtime_error("Qwen packed top-1 pack launch failed");
    }
}


ncclComm_t cached_comm(int world, int rank, int device, const char* id_path) {
    static std::unordered_map<std::string, CachedComm> comms;
    const std::string key = std::to_string(world) + ":" + std::to_string(rank) + ":" + std::to_string(device) + ":" + id_path;
    auto it = comms.find(key);
    if (it != comms.end()) return it->second.comm;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    ncclUniqueId id = load_or_create_id(rank, id_path);
    ncclComm_t comm;
    check_nccl(ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    comms.emplace(key, CachedComm{comm});
    return comm;
}

}  // namespace

void run_tp_float_sum_smoke(int world, int rank, int device, const char* id_path, float value) {
    if (world <= 0 || rank < 0 || rank >= world) throw std::runtime_error("invalid NCCL world/rank");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    ncclUniqueId id = load_or_create_id(rank, id_path);
    ncclComm_t comm;
    check_nccl(ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    float* d_value = nullptr;
    float* d_sum = nullptr;
    check_cuda(cudaMalloc(&d_value, sizeof(float)), "cudaMalloc nccl value");
    check_cuda(cudaMalloc(&d_sum, sizeof(float)), "cudaMalloc nccl sum");
    check_cuda(cudaMemcpy(d_value, &value, sizeof(float), cudaMemcpyHostToDevice), "copy nccl input");
    check_nccl(ncclAllReduce(d_value, d_sum, 1, ncclFloat, ncclSum, comm, nullptr), "ncclAllReduce");
    check_cuda(cudaDeviceSynchronize(), "sync nccl smoke");
    float sum = 0.0f;
    check_cuda(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost), "copy nccl sum");
    std::cout << "nccl_smoke rank=" << rank << " value=" << value << " sum=" << sum << "\n";
    cudaFree(d_value);
    cudaFree(d_sum);
    ncclCommDestroy(comm);
}

void tp_global_topk_rows(int world, int rank, int device, const char* id_path,
                           const int* d_local_tokens,
                           const float* d_local_logits, int rows, int top_k,
                           int* global_tokens, float* global_logits,
                           void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || rows <= 0 || top_k <= 0 ||
        d_local_tokens == nullptr || d_local_logits == nullptr ||
        global_tokens == nullptr || global_logits == nullptr) {
        throw std::runtime_error("invalid NCCL batched top-k args");
    }
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    const size_t local_count = static_cast<size_t>(rows) * top_k;
    const size_t gathered_count = static_cast<size_t>(world) * local_count;
    TopKRowsWorkspace& workspace = topk_rows_workspace(device, gathered_count);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    check_nccl(ncclGroupStart(), "ncclGroupStart batched top-k");
    check_nccl(ncclAllGather(d_local_tokens, workspace.d_tokens, local_count,
                             ncclInt32, comm, cuda_stream),
               "ncclAllGather batched top-k tokens");
    check_nccl(ncclAllGather(d_local_logits, workspace.d_logits, local_count,
                             ncclFloat, comm, cuda_stream),
               "ncclAllGather batched top-k logits");
    check_nccl(ncclGroupEnd(), "ncclGroupEnd batched top-k");
    check_cuda(cudaStreamSynchronize(cuda_stream),
               "sync gathered batched top-k");
    std::vector<int> all_tokens(gathered_count);
    std::vector<float> all_logits(gathered_count);
    check_cuda(cudaMemcpyAsync(all_tokens.data(), workspace.d_tokens,
                               gathered_count * sizeof(int),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "copy gathered batched top-k tokens");
    check_cuda(cudaMemcpyAsync(all_logits.data(), workspace.d_logits,
                               gathered_count * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "copy gathered batched top-k logits");
    check_cuda(cudaStreamSynchronize(cuda_stream),
               "sync copied batched top-k");
    for (int row = 0; row < rows; ++row) {
        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(static_cast<size_t>(world) * top_k);
        for (int r = 0; r < world; ++r) {
            for (int k = 0; k < top_k; ++k) {
                const size_t at = static_cast<size_t>(r) * local_count +
                                  static_cast<size_t>(row) * top_k + k;
                candidates.emplace_back(all_logits[at], all_tokens[at]);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& left, const auto& right) {
                      if (left.first != right.first) return left.first > right.first;
                      return left.second < right.second;
                  });
        for (int k = 0; k < top_k; ++k) {
            global_tokens[static_cast<size_t>(row) * top_k + k] =
                candidates[static_cast<size_t>(k)].second;
            global_logits[static_cast<size_t>(row) * top_k + k] =
                candidates[static_cast<size_t>(k)].first;
        }
    }
}

void tp_global_topk_rows_device(int world, int rank, int device,
                                  const char* id_path,
                                  const int* d_local_tokens,
                                  const float* d_local_logits, int rows,
                                  int top_k, int* d_global_tokens,
                                  float* d_global_logits, void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || rows <= 0 || top_k <= 0 ||
        d_local_tokens == nullptr || d_local_logits == nullptr ||
        d_global_tokens == nullptr || d_global_logits == nullptr) {
        throw std::runtime_error("invalid NCCL device top-k args");
    }
    check_cuda(cudaSetDevice(device), "cudaSetDevice device top-k");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const size_t local_count = static_cast<size_t>(rows) * top_k;
    const size_t gathered_count = static_cast<size_t>(world) * local_count;
    TopKRowsWorkspace& workspace = topk_rows_workspace(device, gathered_count);
    check_nccl(ncclGroupStart(), "ncclGroupStart device top-k");
    check_nccl(ncclAllGather(d_local_tokens, workspace.d_tokens, local_count,
                             ncclInt32, comm, cuda_stream),
               "ncclAllGather device top-k tokens");
    check_nccl(ncclAllGather(d_local_logits, workspace.d_logits, local_count,
                             ncclFloat, comm, cuda_stream),
               "ncclAllGather device top-k logits");
    check_nccl(ncclGroupEnd(), "ncclGroupEnd device top-k");
    if (!merge_topk_candidates(
            workspace.d_tokens, workspace.d_logits, d_global_tokens,
            d_global_logits, world, rows, top_k, cuda_stream)) {
        throw std::runtime_error("sampling device top-k merge launch failed");
    }
}

void tp_global_top1_rows_device(int world, int rank, int device,
                                  const char* id_path,
                                  const int* d_local_tokens,
                                  const float* d_local_logits, int rows,
                                  int* d_global_tokens,
                                  float* d_global_logits, void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || rows <= 0 ||
        d_local_tokens == nullptr || d_local_logits == nullptr ||
        d_global_tokens == nullptr || d_global_logits == nullptr) {
        throw std::runtime_error("invalid NCCL device top args");
    }
    check_cuda(cudaSetDevice(device), "cudaSetDevice device top");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const size_t gathered_count = static_cast<size_t>(world) * rows;
    Top1RowsWorkspace& workspace = top1_rows_workspace(device, gathered_count);
    check_nccl(ncclGroupStart(), "ncclGroupStart device top");
    check_nccl(ncclAllGather(d_local_tokens, workspace.d_tokens, rows,
                             ncclInt32, comm, cuda_stream),
               "ncclAllGather device top tokens");
    check_nccl(ncclAllGather(d_local_logits, workspace.d_logits, rows,
                             ncclFloat, comm, cuda_stream),
               "ncclAllGather device top logits");
    check_nccl(ncclGroupEnd(), "ncclGroupEnd device top");
    if (!qwen_dflash2_merge_topk_f32_cuda(
            workspace.d_tokens, workspace.d_logits, d_global_tokens,
            d_global_logits, world, rows, 1, cuda_stream)) {
        throw std::runtime_error("Qwen device top-1 merge launch failed");
    }
}

void tp_global_top1_rows_packed_device(
    int world, int rank, int device, const char* id_path,
    const int* d_local_tokens, const float* d_local_logits, int rows,
    int* d_global_tokens, float* d_global_logits, void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || rows <= 0 ||
        d_local_tokens == nullptr || d_local_logits == nullptr ||
        d_global_tokens == nullptr || d_global_logits == nullptr) {
        throw std::runtime_error("invalid NCCL packed device top args");
    }
    check_cuda(cudaSetDevice(device), "cudaSetDevice packed device top");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    PackedTop1RowsWorkspace& workspace =
        packed_top1_rows_workspace(device, static_cast<size_t>(rows));
    pack_top1(d_local_tokens, d_local_logits, workspace.d_keys, rows,
              cuda_stream);
    nccl_all_reduce_max_u64(comm, workspace.d_keys, rows, cuda_stream);
    unpack_packed_top1(workspace.d_keys, d_global_tokens, d_global_logits,
                       rows, cuda_stream);
}

void tp_global_top1_rows(int world, int rank, int device, const char* id_path,
                           const int* d_local_tokens,
                           const float* d_local_logits, int rows,
                           int* global_tokens, float* global_logits,
                           void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || rows <= 0 ||
        d_local_tokens == nullptr || d_local_logits == nullptr ||
        global_tokens == nullptr || global_logits == nullptr) {
        throw std::runtime_error("invalid NCCL batched top args");
    }
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const size_t gathered_count = static_cast<size_t>(world) * rows;
    Top1RowsWorkspace& workspace = top1_rows_workspace(device, gathered_count);
    check_nccl(ncclGroupStart(), "ncclGroupStart batched top");
    check_nccl(ncclAllGather(d_local_tokens, workspace.d_tokens, rows, ncclInt32,
                             comm, cuda_stream),
               "ncclAllGather batched top tokens");
    check_nccl(ncclAllGather(d_local_logits, workspace.d_logits, rows, ncclFloat,
                             comm, cuda_stream),
               "ncclAllGather batched top logits");
    check_nccl(ncclGroupEnd(), "ncclGroupEnd batched top");
    check_cuda(cudaStreamSynchronize(cuda_stream), "sync gathered batched top");
    std::vector<int> all_tokens(gathered_count);
    std::vector<float> all_logits(gathered_count);
    check_cuda(cudaMemcpyAsync(all_tokens.data(), workspace.d_tokens,
                               gathered_count * sizeof(int),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "copy gathered batched top tokens");
    check_cuda(cudaMemcpyAsync(all_logits.data(), workspace.d_logits,
                               gathered_count * sizeof(float),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "copy gathered batched top logits");
    check_cuda(cudaStreamSynchronize(cuda_stream), "sync copied batched top");

    // The device-resident sibling below uses the same gathered layout and
    // comparator. Keep this synchronous API's CPU merge for compatibility.
    if (rows <= 0) return;
    for (int row = 0; row < rows; ++row) {
        float best_logit = -INFINITY;
        int best_token = INT_MAX;
        for (int r = 0; r < world; ++r) {
            const size_t at = static_cast<size_t>(r) * rows + row;
            const float logit = all_logits[at];
            const int token = all_tokens[at];
            if (std::isnan(logit)) continue;
            if (logit > best_logit ||
                (logit == best_logit && token < best_token)) {
                best_logit = logit;
                best_token = token;
            }
        }
        global_tokens[row] = best_token;
        global_logits[row] = best_logit;
    }
}

void tp_all_reduce_sum_float_inplace(int world, int rank, int device, const char* id_path, float* d_values, int count, void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || d_values == nullptr || count <= 0) throw std::runtime_error("invalid NCCL all-reduce args");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    check_nccl(ncclAllReduce(d_values, d_values, count, ncclFloat, ncclSum, comm,
                             static_cast<cudaStream_t>(stream)), "ncclAllReduce inplace");
}

void tp_all_reduce_sum_f16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count, void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || d_values == nullptr || count <= 0) throw std::runtime_error("invalid NCCL fp16 all-reduce args");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    check_nccl(ncclAllReduce(d_values, d_values, count, ncclHalf, ncclSum, comm,
                             static_cast<cudaStream_t>(stream)), "ncclAllReduce fp16 inplace");
}

void tp_all_reduce_sum_bf16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count, void* stream) {
    if (world <= 0 || rank < 0 || rank >= world || d_values == nullptr || count <= 0) throw std::runtime_error("invalid NCCL bf16 all-reduce args");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    check_nccl(ncclAllReduce(d_values, d_values, count, ncclBfloat16, ncclSum, comm,
                             static_cast<cudaStream_t>(stream)), "ncclAllReduce bf16 inplace");
}

void tp_broadcast_int32(int world, int rank, int device, const char* id_path, int32_t* buf, int count, int root) {
    if (world <= 0 || rank < 0 || rank >= world || buf == nullptr || count <= 0) throw std::runtime_error("invalid NCCL bcast args");
    if (root < 0 || root >= world) throw std::runtime_error("invalid NCCL bcast root");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    int32_t* d_buf = nullptr;
    check_cuda(cudaMalloc(&d_buf, static_cast<size_t>(count) * sizeof(int32_t)), "cudaMalloc nccl bcast");
    if (rank == root) {
        check_cuda(cudaMemcpy(d_buf, buf, static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyHostToDevice), "copy bcast input");
    }
    check_nccl(ncclBroadcast(d_buf, d_buf, count, ncclInt32, root, comm, nullptr), "ncclBroadcast int32");
    check_cuda(cudaDeviceSynchronize(), "sync nccl bcast");
    if (rank != root) {
        check_cuda(cudaMemcpy(buf, d_buf, static_cast<size_t>(count) * sizeof(int32_t), cudaMemcpyDeviceToHost), "copy bcast output");
    }
    cudaFree(d_buf);
}

void tp_gather_floats_to_root(int world, int rank, int device, const char* id_path, const float* h_local, int local_count, float* h_root_out, int root) {
    if (world <= 0 || rank < 0 || rank >= world || h_local == nullptr || local_count <= 0) throw std::runtime_error("invalid NCCL gather args");
    if (root < 0 || root >= world) throw std::runtime_error("invalid NCCL gather root");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    float* d_local = nullptr;
    float* d_all = nullptr;
    check_cuda(cudaMalloc(&d_local, static_cast<size_t>(local_count) * sizeof(float)), "cudaMalloc gather local");
    check_cuda(cudaMalloc(&d_all, static_cast<size_t>(world) * static_cast<size_t>(local_count) * sizeof(float)), "cudaMalloc gather all");
    check_cuda(cudaMemcpy(d_local, h_local, static_cast<size_t>(local_count) * sizeof(float), cudaMemcpyHostToDevice), "copy gather input");
    // Use AllGather so we don't need a separate Gather symbol. The receive
    // buffer is meaningful on all ranks; non-root callers ignore it.
    check_nccl(ncclAllGather(d_local, d_all, local_count, ncclFloat, comm, nullptr), "ncclAllGather logits");
    check_cuda(cudaDeviceSynchronize(), "sync gather");
    if (rank == root && h_root_out != nullptr) {
        check_cuda(cudaMemcpy(h_root_out, d_all, static_cast<size_t>(world) * static_cast<size_t>(local_count) * sizeof(float), cudaMemcpyDeviceToHost), "copy gather output");
    }
    cudaFree(d_local);
    cudaFree(d_all);
}

TpTopResult tp_global_top1(int world, int rank, int device, const char* id_path, int local_token, float local_logit) {
    if (world <= 0 || rank < 0 || rank >= world) throw std::runtime_error("invalid NCCL world/rank");
    ncclComm_t comm = cached_comm(world, rank, device, id_path);
    float local[2] = {local_logit, static_cast<float>(local_token)};
    float* d_local = nullptr;
    float* d_all = nullptr;
    check_cuda(cudaMalloc(&d_local, 2 * sizeof(float)), "cudaMalloc local top");
    check_cuda(cudaMalloc(&d_all, static_cast<size_t>(world) * 2 * sizeof(float)), "cudaMalloc gathered top");
    check_cuda(cudaMemcpy(d_local, local, 2 * sizeof(float), cudaMemcpyHostToDevice), "copy local top");
    check_nccl(ncclAllGather(d_local, d_all, 2, ncclFloat, comm, nullptr), "ncclAllGather top");
    check_cuda(cudaDeviceSynchronize(), "sync top gather");
    std::vector<float> all(static_cast<size_t>(world) * 2);
    check_cuda(cudaMemcpy(all.data(), d_all, all.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy gathered top");
    TpTopResult result;
    result.logit = -INFINITY;
    for (int r = 0; r < world; ++r) {
        const float logit = all[static_cast<size_t>(r) * 2];
        const int token = static_cast<int>(all[static_cast<size_t>(r) * 2 + 1]);
        if (logit > result.logit) {
            result.logit = logit;
            result.token = token;
        }
    }
    cudaFree(d_local);
    cudaFree(d_all);
    return result;
}
#endif

}  // namespace pocket
