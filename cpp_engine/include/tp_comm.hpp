#pragma once

// Tensor-parallel collective contract, one implementation per backend: NCCL under
// backends/cuda/collective, HCCL under backends/ascend/collective. POCKET_HAVE_TP_COMM
// means "a collective library is linked", whichever one that is.
//
// `id_path` names a file used for rendezvous: rank 0 publishes an opaque id there
// and the other ranks poll for it. One process per rank, always -- the Ascend
// backend cannot use the single-process multi-device form at all.
//
// Every entry point throws on failure rather than returning a status, because a
// half-completed collective leaves the ranks out of step with no way to recover.

#include <cstdint>

namespace pocket {

struct TpTopResult {
    int token = 0;
    float logit = 0.0f;
};

bool tp_comm_available();

#ifdef POCKET_HAVE_TP_COMM
void run_tp_float_sum_smoke(int world, int rank, int device, const char* id_path, float value);
TpTopResult tp_global_top1(int world, int rank, int device, const char* id_path, int local_token, float local_logit);
void tp_global_top1_rows(int world, int rank, int device, const char* id_path,
                         const int* d_local_tokens,
                         const float* d_local_logits, int rows,
                         int* global_tokens, float* global_logits,
                         void* stream = nullptr);
// Device-resident top-1 merge. On CUDA the all-gather and the deterministic
// comparator both run on `stream` with no host round-trip. The Ascend backend
// merges on the host for now, which is the same result but does synchronize.
void tp_global_top1_rows_device(int world, int rank, int device,
                                const char* id_path,
                                const int* d_local_tokens,
                                const float* d_local_logits, int rows,
                                int* d_global_tokens,
                                float* d_global_logits,
                                void* stream = nullptr);
// Device-resident top-1 using one packed uint64 Max reduction per row instead of
// all-gathering both arrays. The packed order is deterministic and matches the
// host/device comparator: larger logit wins, ties pick the smaller token id.
void tp_global_top1_rows_packed_device(
    int world, int rank, int device, const char* id_path,
    const int* d_local_tokens, const float* d_local_logits, int rows,
    int* d_global_tokens, float* d_global_logits, void* stream = nullptr);
void tp_global_topk_rows(int world, int rank, int device, const char* id_path,
                         const int* d_local_tokens,
                         const float* d_local_logits, int rows, int top_k,
                         int* global_tokens, float* global_logits,
                         void* stream = nullptr);
// All-gather and merge batched top-k candidates. The output buffers stay
// device-resident for the next device stage.
void tp_global_topk_rows_device(int world, int rank, int device,
                                const char* id_path,
                                const int* d_local_tokens,
                                const float* d_local_logits, int rows,
                                int top_k, int* d_global_tokens,
                                float* d_global_logits,
                                void* stream = nullptr);
void tp_all_reduce_sum_float_inplace(int world, int rank, int device, const char* id_path, float* d_values, int count, void* stream = nullptr);
void tp_all_reduce_sum_f16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count, void* stream = nullptr);
// BF16 reduce. Available on CUDA (Ampere and later); the Ascend backend throws,
// because first-generation 910 has no BF16 at all and every device tensor there
// is FP16. See qwen_device_dtype in core/qwen_weight_map.cpp.
void tp_all_reduce_sum_bf16_inplace(int world, int rank, int device, const char* id_path, uint16_t* d_values, int count, void* stream = nullptr);
// Broadcast a small int32 buffer from rank `root` to all ranks. Synchronous.
void tp_broadcast_int32(int world, int rank, int device, const char* id_path, int32_t* buf, int count, int root);
// Gather equal-sized float32 chunks from every rank to `root`. `h_local` is read on
// each rank; on rank == root, `h_root_out` (size world * local_count) receives the
// concatenated result. May pass nullptr for h_root_out on non-root ranks.
void tp_gather_floats_to_root(int world, int rank, int device, const char* id_path, const float* h_local, int local_count, float* h_root_out, int root);
#endif

}  // namespace pocket
