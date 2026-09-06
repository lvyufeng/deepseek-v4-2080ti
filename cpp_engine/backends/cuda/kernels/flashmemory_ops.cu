// FlashMemory DS-V4 retriever scoring kernels (paper-reproduction plugin).
//
// Reproduces retriever.py forward_and_score on-device:
//   hidden [4096] -> wq_a -> RMSNorm(q_norm) -> wq_b -> reshape[H,D]
//                 -> RoPE(YaRN, bf16) -> Hadamard -> q [H,D]
//   hidden [4096] -> weights_proj -> * weight_scale -> fused_w [H]
//   compressed_k [N,132] bytes -> dequant fp8e4m3 * f32 scale -> k [N,D]
//   score[n] = sigmoid( sum_h( relu(sum_d q[h,d]*k[n,d]) * fused_w[h] ) )
//
// Precision notes (must match the trained/deployed path):
//   - RoPE applies to the last ROPE_DIM dims, in bf16, then cast back to fp32.
//   - q is rounded to bf16 *before* RoPE (retriever.py does q.to(bfloat16)).
//   - Hadamard is the normalized Walsh-Hadamard transform over HEAD_DIM (/sqrt(D)).
//   - weight_scale = head_dim^-0.5 * n_heads^-0.5.
//
// All weights are fp32 (the shipped checkpoint is fp32). This is a correctness-
// first implementation; it is only invoked behind the default-OFF FlashMemory
// plugin and never touches the default DeepSeek inference path.

#include "flashmemory_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pocket {
namespace {

// fp32 -> bf16 round-to-nearest-even, returned as fp32 (round-trip through bf16).
__device__ float round_bf16_device(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7FFFu + lsb;
    const uint32_t hi = (rounded >> 16) << 16;
    return __uint_as_float(hi);
}

// float8_e4m3fn (OCP / torch.float8_e4m3fn) byte -> float.
// 1 sign, 4 exponent (bias 7), 3 mantissa. No inf; 0xFF/0x7F == NaN.
__device__ float fp8_e4m3_to_float_device(uint8_t b) {
    const uint32_t sign = (b >> 7) & 0x1u;
    const uint32_t exp = (b >> 3) & 0xFu;
    const uint32_t man = b & 0x7u;
    float value;
    if (exp == 0u) {
        // subnormal: value = man/8 * 2^(1-7)
        value = static_cast<float>(man) * (1.0f / 8.0f) * exp2f(-6.0f);
    } else if (exp == 0xFu && man == 0x7u) {
        // e4m3fn NaN encoding (S.1111.111)
        value = nanf("");
    } else {
        value = (1.0f + static_cast<float>(man) * (1.0f / 8.0f)) * exp2f(static_cast<float>(exp) - 7.0f);
    }
    return sign ? -value : value;
}

bool valid_hadamard_dim(int dim) {
    return dim > 0 && dim <= 1024 && (dim & (dim - 1)) == 0;
}

// Normalized Walsh-Hadamard transform in shared memory over HEAD_DIM (power of 2).
// vec has HEAD_DIM entries; tid in [0, HEAD_DIM).
__device__ void hadamard_inplace_device(float* vec, int tid, int dim) {
    for (int stride = 1; stride < dim; stride <<= 1) {
        if (tid < dim && (tid & stride) == 0) {
            const float a = vec[tid];
            const float b = vec[tid + stride];
            vec[tid] = a + b;
            vec[tid + stride] = a - b;
        }
        __syncthreads();
    }
    const float inv = rsqrtf(static_cast<float>(dim));
    if (tid < dim) vec[tid] *= inv;
    __syncthreads();
}

// Build q [n_heads, head_dim] and fused_w [n_heads] from hidden [hidden_dim].
// q_lora_scratch must be [q_lora_rank] fp32. One block per head; head 0 also
// writes fused_w. Requires q_lora already computed (see flashmemory_build_q).
//
// We split the q-lora projection (a [q_lora_rank, hidden_dim] matvec) and the
// q projection (a [n_heads*head_dim, q_lora_rank] matvec) into their own simple
// matvec kernels for clarity; this kernel does RoPE + Hadamard per head.
__global__ void flashmemory_rope_hadamard_kernel(
    float* q,                  // [n_heads, head_dim] in/out
    const float* inv_freqs,    // [rope_dim/2]
    int n_heads,
    int head_dim,
    int rope_dim,
    long long position) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= n_heads) return;
    extern __shared__ float vec[];  // [head_dim]
    float* row = q + static_cast<size_t>(head) * head_dim;

    // Load + round to bf16 (retriever casts q to bf16 before RoPE).
    if (tid < head_dim) vec[tid] = round_bf16_device(row[tid]);
    __syncthreads();

    // RoPE on the last rope_dim dims, computed in bf16 then rounded back.
    // Pairs are interleaved (view_as_complex over [.., rope_dim/2, 2]).
    const int rope_start = head_dim - rope_dim;
    for (int pair = tid * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int off = rope_start + pair;
        const float angle = static_cast<float>(position) * inv_freqs[pair >> 1];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float a = vec[off];
        const float b = vec[off + 1];
        // bf16 arithmetic path: round inputs/outputs to bf16 like the trained path.
        vec[off] = round_bf16_device(a * c - b * s);
        vec[off + 1] = round_bf16_device(a * s + b * c);
    }
    __syncthreads();

    // Hadamard (in fp32 — retriever.hadamard_transform upcasts to float).
    hadamard_inplace_device(vec, tid, head_dim);
    if (tid < head_dim) row[tid] = vec[tid];
}

// Per-chunk scoring: one block per chunk. Dequant fp8 K [head_dim], then for
// each head dot with q[head] (head_dim), relu, * fused_w[head], accumulate,
// finally sigmoid. compressed_k layout per chunk: head_dim fp8 bytes + 4-byte
// f32 scale = (head_dim + 4) bytes.
__global__ void flashmemory_score_chunks_kernel(
    const float* q,            // [n_heads, head_dim]
    const float* fused_w,      // [n_heads]
    const uint8_t* compressed_k, // [n_chunks, head_dim + 4]
    float* scores,             // [n_chunks]  (sigmoid)
    int n_chunks,
    int n_heads,
    int head_dim,
    bool apply_sigmoid) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    if (chunk >= n_chunks) return;

    extern __shared__ float smem[];
    float* k = smem;                  // [head_dim]
    float* reduce = smem + head_dim;  // [blockDim.x]

    const int stride_bytes = head_dim + 4;
    const uint8_t* row = compressed_k + static_cast<size_t>(chunk) * stride_bytes;

    // Decode per-chunk f32 scale from the 4 trailing bytes.
    float scale;
    {
        uint32_t sb = static_cast<uint32_t>(row[head_dim]) |
                      (static_cast<uint32_t>(row[head_dim + 1]) << 8) |
                      (static_cast<uint32_t>(row[head_dim + 2]) << 16) |
                      (static_cast<uint32_t>(row[head_dim + 3]) << 24);
        scale = __uint_as_float(sb);
    }

    // Dequant fp8 key into shared memory.
    for (int d = tid; d < head_dim; d += blockDim.x) {
        k[d] = fp8_e4m3_to_float_device(row[d]) * scale;
    }
    __syncthreads();

    // Sum over heads of relu(q[h]·k) * fused_w[h]. Each thread handles a subset
    // of heads; dot products are computed serially over head_dim per head.
    float local = 0.0f;
    for (int h = tid; h < n_heads; h += blockDim.x) {
        const float* qh = q + static_cast<size_t>(h) * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * k[d];
        local += fmaxf(dot, 0.0f) * fused_w[h];
    }
    reduce[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const float logit = reduce[0];
        scores[chunk] = apply_sigmoid ? (1.0f / (1.0f + expf(-logit))) : logit;
    }
}

// Per-chunk scoring for pre-dequantized float keys. This is used when the GGUF
// engine reuses its native indexer_kv_cache as FlashMemory K^IComp: the keys are
// already 128-dim, Hadamard-rotated, and fp4-fake-quantized but stored as float.
// One block per chunk; the score math is identical to the fp8 path above.
__global__ void flashmemory_score_float_keys_kernel(
    const float* q,            // [n_heads, head_dim]
    const float* fused_w,      // [n_heads]
    const float* keys,         // [n_chunks, head_dim]
    float* scores,             // [n_chunks]
    int n_chunks,
    int n_heads,
    int head_dim,
    bool apply_sigmoid) {
    const int chunk = blockIdx.x;
    const int tid = threadIdx.x;
    if (chunk >= n_chunks) return;

    extern __shared__ float smem[];
    float* k = smem;                  // [head_dim]
    float* reduce = smem + head_dim;  // [blockDim.x]

    const float* row = keys + static_cast<size_t>(chunk) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) k[d] = row[d];
    __syncthreads();

    float local = 0.0f;
    for (int h = tid; h < n_heads; h += blockDim.x) {
        const float* qh = q + static_cast<size_t>(h) * head_dim;
        float dot = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * k[d];
        local += fmaxf(dot, 0.0f) * fused_w[h];
    }
    reduce[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        const float logit = reduce[0];
        scores[chunk] = apply_sigmoid ? (1.0f / (1.0f + expf(-logit))) : logit;
    }
}

__global__ void flashmemory_ensemble_kernel(
    const float* layer_scores,  // [n_layers, score_stride]
    float* out,                 // [n_chunks]
    int n_layers,
    int n_chunks,
    int score_stride,
    bool use_max) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;
    if (use_max) {
        float best = -INFINITY;
        for (int l = 0; l < n_layers; ++l) {
            const float v = layer_scores[static_cast<size_t>(l) * score_stride + i];
            best = fmaxf(best, v);
        }
        out[i] = best;
    } else {
        float sum = 0.0f;
        for (int l = 0; l < n_layers; ++l) sum += layer_scores[static_cast<size_t>(l) * score_stride + i];
        out[i] = sum / static_cast<float>(n_layers);
    }
}

__global__ void flashmemory_topk_kernel(
    const float* scores,
    int n_chunks,
    int keep,
    int* out_indices,
    float* out_scores) {
    const int tid = threadIdx.x;
    extern __shared__ char smem_raw[];
    float* best_vals = reinterpret_cast<float*>(smem_raw);
    int* best_idxs = reinterpret_cast<int*>(best_vals + blockDim.x);

    for (int k = 0; k < keep; ++k) {
        float local_best = -INFINITY;
        int local_idx = n_chunks;
        for (int i = tid; i < n_chunks; i += blockDim.x) {
            bool already_selected = false;
            for (int prev = 0; prev < k; ++prev) {
                if (out_indices[prev] == i) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) continue;
            const float v = scores[i];
            if (v > local_best || (v == local_best && i < local_idx)) {
                local_best = v;
                local_idx = i;
            }
        }
        best_vals[tid] = local_best;
        best_idxs[tid] = local_idx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_v = best_vals[tid + stride];
                const int other_i = best_idxs[tid + stride];
                if (other_v > best_vals[tid] || (other_v == best_vals[tid] && other_i < best_idxs[tid])) {
                    best_vals[tid] = other_v;
                    best_idxs[tid] = other_i;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int idx = best_idxs[0] < 0 ? 0 : best_idxs[0];
            out_indices[k] = idx;
            if (out_scores != nullptr) out_scores[k] = best_vals[0];
        }
        __syncthreads();
    }
}

__global__ void flashmemory_write_global_indices_kernel(
    const int* logical,
    int keep,
    int offset,
    int* out_indices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < keep) out_indices[i] = offset + logical[i];
}

// hidden [hidden_dim] @ W [out_dim, hidden_dim]^T -> y [out_dim]. One block per
// output row, blockDim threads reduce over hidden_dim. fp32.
__global__ void flashmemory_matvec_kernel(
    const float* x,
    const float* w,
    float* y,
    int out_dim,
    int in_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= out_dim) return;
    extern __shared__ float reduce[];
    const float* wr = w + static_cast<size_t>(row) * in_dim;
    float local = 0.0f;
    for (int i = tid; i < in_dim; i += blockDim.x) local += wr[i] * x[i];
    reduce[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = reduce[0];
}

// RMSNorm over q_lora [rank]: x / sqrt(mean(x^2)+eps) * gamma. Single block.
__global__ void flashmemory_rmsnorm_kernel(
    float* x,
    const float* gamma,
    int rank,
    float eps) {
    const int tid = threadIdx.x;
    extern __shared__ float reduce[];
    float local = 0.0f;
    for (int i = tid; i < rank; i += blockDim.x) {
        const float v = x[i];
        local += v * v;
    }
    reduce[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    __shared__ float inv_norm;
    if (tid == 0) inv_norm = rsqrtf(reduce[0] / static_cast<float>(rank) + eps);
    __syncthreads();
    for (int i = tid; i < rank; i += blockDim.x) x[i] = x[i] * inv_norm * gamma[i];
}

// Scale fused_w in place: per_head_w * weight_scale. Single block.
__global__ void flashmemory_scale_kernel(float* w, float scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] *= scale;
}

}  // namespace

bool flashmemory_score_layer_cuda(
    const float* d_hidden,
    const float* d_wq_a,
    const float* d_wq_b,
    const float* d_q_norm,
    const float* d_weights_proj,
    const float* d_inv_freqs,
    const uint8_t* d_compressed_k,
    float* d_q_scratch,
    float* d_qlora_scratch,
    float* d_fused_w,
    float* d_scores,
    int n_chunks,
    int n_heads,
    int head_dim,
    int q_lora_rank,
    int hidden_dim,
    int rope_dim,
    float rms_norm_eps,
    long long position,
    bool apply_sigmoid,
    void* stream) {
    if (d_hidden == nullptr || d_wq_a == nullptr || d_wq_b == nullptr || d_q_norm == nullptr ||
        d_weights_proj == nullptr || d_inv_freqs == nullptr || d_compressed_k == nullptr ||
        d_q_scratch == nullptr || d_qlora_scratch == nullptr || d_fused_w == nullptr || d_scores == nullptr)
        return false;
    if (n_chunks <= 0 || n_heads <= 0 || !valid_hadamard_dim(head_dim) || q_lora_rank <= 0 || hidden_dim <= 0 ||
        rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0)
        return false;
    auto cs = reinterpret_cast<cudaStream_t>(stream);
    const int q_out = n_heads * head_dim;

    // q_lora = wq_a @ hidden  -> [q_lora_rank]
    flashmemory_matvec_kernel<<<q_lora_rank, 256, 256 * sizeof(float), cs>>>(
        d_hidden, d_wq_a, d_qlora_scratch, q_lora_rank, hidden_dim);
    // RMSNorm(q_lora, q_norm)
    flashmemory_rmsnorm_kernel<<<1, 256, 256 * sizeof(float), cs>>>(
        d_qlora_scratch, d_q_norm, q_lora_rank, rms_norm_eps);
    // q = wq_b @ q_lora -> [n_heads*head_dim]
    flashmemory_matvec_kernel<<<q_out, 256, 256 * sizeof(float), cs>>>(
        d_qlora_scratch, d_wq_b, d_q_scratch, q_out, q_lora_rank);
    // RoPE (bf16) + Hadamard per head
    flashmemory_rope_hadamard_kernel<<<n_heads, head_dim, head_dim * sizeof(float), cs>>>(
        d_q_scratch, d_inv_freqs, n_heads, head_dim, rope_dim, position);

    // fused_w = (weights_proj @ hidden) * weight_scale
    flashmemory_matvec_kernel<<<n_heads, 256, 256 * sizeof(float), cs>>>(
        d_hidden, d_weights_proj, d_fused_w, n_heads, hidden_dim);
    const float weight_scale = rsqrtf(static_cast<float>(head_dim)) * rsqrtf(static_cast<float>(n_heads));
    flashmemory_scale_kernel<<<(n_heads + 255) / 256, 256, 0, cs>>>(d_fused_w, weight_scale, n_heads);

    // Per-chunk scores
    const int score_threads = 128;
    const size_t score_smem = (static_cast<size_t>(head_dim) + score_threads) * sizeof(float);
    flashmemory_score_chunks_kernel<<<n_chunks, score_threads, score_smem, cs>>>(
        d_q_scratch, d_fused_w, d_compressed_k, d_scores, n_chunks, n_heads, head_dim, apply_sigmoid);

    return cudaGetLastError() == cudaSuccess;
}

bool flashmemory_score_layer_floatk_cuda(
    const float* d_hidden,
    const float* d_wq_a,
    const float* d_wq_b,
    const float* d_q_norm,
    const float* d_weights_proj,
    const float* d_inv_freqs,
    const float* d_float_keys,
    float* d_q_scratch,
    float* d_qlora_scratch,
    float* d_fused_w,
    float* d_scores,
    int n_chunks,
    int n_heads,
    int head_dim,
    int q_lora_rank,
    int hidden_dim,
    int rope_dim,
    float rms_norm_eps,
    long long position,
    bool apply_sigmoid,
    void* stream) {
    if (d_hidden == nullptr || d_wq_a == nullptr || d_wq_b == nullptr || d_q_norm == nullptr ||
        d_weights_proj == nullptr || d_inv_freqs == nullptr || d_float_keys == nullptr ||
        d_q_scratch == nullptr || d_qlora_scratch == nullptr || d_fused_w == nullptr || d_scores == nullptr)
        return false;
    if (n_chunks <= 0 || n_heads <= 0 || !valid_hadamard_dim(head_dim) || q_lora_rank <= 0 || hidden_dim <= 0 ||
        rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0)
        return false;
    auto cs = reinterpret_cast<cudaStream_t>(stream);
    const int q_out = n_heads * head_dim;

    // q_lora = wq_a @ hidden  -> [q_lora_rank]
    flashmemory_matvec_kernel<<<q_lora_rank, 256, 256 * sizeof(float), cs>>>(
        d_hidden, d_wq_a, d_qlora_scratch, q_lora_rank, hidden_dim);
    // RMSNorm(q_lora, q_norm)
    flashmemory_rmsnorm_kernel<<<1, 256, 256 * sizeof(float), cs>>>(
        d_qlora_scratch, d_q_norm, q_lora_rank, rms_norm_eps);
    // q = wq_b @ q_lora -> [n_heads*head_dim]
    flashmemory_matvec_kernel<<<q_out, 256, 256 * sizeof(float), cs>>>(
        d_qlora_scratch, d_wq_b, d_q_scratch, q_out, q_lora_rank);
    // RoPE (bf16) + Hadamard per head
    flashmemory_rope_hadamard_kernel<<<n_heads, head_dim, head_dim * sizeof(float), cs>>>(
        d_q_scratch, d_inv_freqs, n_heads, head_dim, rope_dim, position);

    // fused_w = (weights_proj @ hidden) * weight_scale
    flashmemory_matvec_kernel<<<n_heads, 256, 256 * sizeof(float), cs>>>(
        d_hidden, d_weights_proj, d_fused_w, n_heads, hidden_dim);
    const float weight_scale = rsqrtf(static_cast<float>(head_dim)) * rsqrtf(static_cast<float>(n_heads));
    flashmemory_scale_kernel<<<(n_heads + 255) / 256, 256, 0, cs>>>(d_fused_w, weight_scale, n_heads);

    // Per-chunk scores against pre-dequantized float keys.
    const int score_threads = 128;
    const size_t score_smem = (static_cast<size_t>(head_dim) + score_threads) * sizeof(float);
    flashmemory_score_float_keys_kernel<<<n_chunks, score_threads, score_smem, cs>>>(
        d_q_scratch, d_fused_w, d_float_keys, d_scores, n_chunks, n_heads, head_dim, apply_sigmoid);

    return cudaGetLastError() == cudaSuccess;
}

bool flashmemory_ensemble_cuda(
    const float* d_layer_scores,
    int n_layers,
    int n_chunks,
    int score_stride,
    bool use_max,
    float* d_ensemble_scores,
    void* stream) {
    if (d_layer_scores == nullptr || d_ensemble_scores == nullptr) return false;
    if (n_layers <= 0 || n_chunks <= 0 || score_stride < n_chunks) return false;
    auto cs = reinterpret_cast<cudaStream_t>(stream);
    flashmemory_ensemble_kernel<<<(n_chunks + 255) / 256, 256, 0, cs>>>(
        d_layer_scores, d_ensemble_scores, n_layers, n_chunks, score_stride, use_max);
    return cudaGetLastError() == cudaSuccess;
}

bool flashmemory_topk_cuda(
    const float* d_scores,
    int n_chunks,
    int keep,
    int* d_out_indices,
    float* d_out_scores,
    void* stream) {
    if (d_scores == nullptr || d_out_indices == nullptr) return false;
    if (n_chunks <= 0 || keep <= 0 || keep > n_chunks) return false;
    auto cs = reinterpret_cast<cudaStream_t>(stream);
    flashmemory_topk_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), cs>>>(
        d_scores, n_chunks, keep, d_out_indices, d_out_scores);
    return cudaGetLastError() == cudaSuccess;
}

bool flashmemory_write_global_indices_cuda(
    const int* d_logical,
    int keep,
    int offset,
    int* d_out_indices,
    void* stream) {
    if (d_logical == nullptr || d_out_indices == nullptr) return false;
    if (keep < 0 || offset < 0) return false;
    if (keep == 0) return true;
    auto cs = reinterpret_cast<cudaStream_t>(stream);
    flashmemory_write_global_indices_kernel<<<(keep + 255) / 256, 256, 0, cs>>>(
        d_logical, keep, offset, d_out_indices);
    return cudaGetLastError() == cudaSuccess;
}

// Host-side YaRN inverse-frequency precompute. Mirrors retriever.py
// precompute_freqs_cis: builds the per-pair "mixed" angular frequency used to
// rotate the trailing rope_dim dims. The full freqs_cis table is just
// outer(positions, mixed); we keep only `mixed` and apply the position at
// scoring time, which is exactly equivalent for our single-position use.
std::vector<float> flashmemory_yarn_inv_freqs(
    int rope_dim,
    double base,
    double factor,
    int original_seq_len,
    double beta_fast,
    double beta_slow) {
    const int half = rope_dim / 2;
    std::vector<float> mixed(half, 0.0f);
    if (half <= 0) return mixed;

    auto correction_dim = [&](double n_rot) {
        return (rope_dim * std::log(original_seq_len / (n_rot * 2.0 * M_PI))) / (2.0 * std::log(base));
    };
    double low_d = std::floor(correction_dim(beta_fast));
    double high_d = std::ceil(correction_dim(beta_slow));
    int low = static_cast<int>(low_d < 0 ? 0 : low_d);
    int high = static_cast<int>(high_d > (half - 1) ? (half - 1) : high_d);

    for (int i = 0; i < half; ++i) {
        const double freq = 1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(rope_dim));
        double ramp;
        if (i < low) ramp = 0.0;
        else if (i >= high) ramp = 1.0;
        else ramp = static_cast<double>(i - low) / static_cast<double>(std::max(high - low, 1));
        const double m = freq * (1.0 - ramp) + (freq / factor) * ramp;
        mixed[i] = static_cast<float>(m);
    }
    return mixed;
}

}  // namespace pocket
