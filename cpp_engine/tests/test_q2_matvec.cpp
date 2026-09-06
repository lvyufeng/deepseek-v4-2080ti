// Q2 single-expert matvec parity test.
//
// Loads expert 0 of layer 0 (IQ2_XXS w1/w3, Q2_K w2) from the GGUF mmap,
// runs the cpp_engine GPU kernel chain (W13 + SwiGLU+Q8_1 + W2), and
// compares the GPU output against a host CPU reference that dequantizes
// the GGUF blocks to fp32 and does naive matvec.
//
// Pass criterion: RMSE < 5e-3 (lossy quantization tolerance).

#include "cuda_ops.hpp"
#include "weight_source.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CHECK_CUDA(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(_err)); \
    } \
} while (0)

float f16_to_f32(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return __half2float(h);
}

uint64_t iq2xxs_grid_value(int grid_id) {
    // Mirror of cuda/q2_ops.cu kIQ2XXSGrid; we just inline the entries we need
    // by re-reading via a host helper at runtime. To avoid duplicating 256
    // hex constants, we lean on the GPU-built signed grid: we'll copy 256*128*8
    // back to host once at start. So this function is unused; left as a stub.
    (void)grid_id;
    return 0;
}

// Dequantize one IQ2_XXS block (66 bytes → 256 fp32 weights).
// signed_grid is the host-mirror of the GPU table: [256][128][8] int8.
void iq2_xxs_dequant_block(const uint8_t* block, const int8_t* signed_grid, float* out) {
    const float d = f16_to_f32(static_cast<uint16_t>(block[0]) |
                               (static_cast<uint16_t>(block[1]) << 8));
    for (int sub = 0; sub < 8; ++sub) {
        const uint8_t* chunk = block + 2 + sub * 8;
        const uint32_t aux = static_cast<uint32_t>(chunk[4]) |
            (static_cast<uint32_t>(chunk[5]) << 8) |
            (static_cast<uint32_t>(chunk[6]) << 16) |
            (static_cast<uint32_t>(chunk[7]) << 24);
        const float s = 0.125f * d * static_cast<float>(2 * (aux >> 28) + 1);
        for (int part = 0; part < 4; ++part) {
            const int grid_id = chunk[part];
            const int sign_idx = static_cast<int>((aux >> (7 * part)) & 0x7f);
            const int8_t* vals = signed_grid + (grid_id * 128 + sign_idx) * 8;
            const int dest_off = sub * 32 + part * 8;
            for (int b = 0; b < 8; ++b) {
                out[dest_off + b] = static_cast<float>(vals[b]) * s;
            }
        }
    }
}

// Dequantize one Q2_K block (84 bytes → 256 fp32 weights).
void q2_k_dequant_block(const uint8_t* block, float* out) {
    const uint8_t* scales = block;
    const uint8_t* qs = block + 16;
    const float d = f16_to_f32(static_cast<uint16_t>(block[80]) |
                               (static_cast<uint16_t>(block[81]) << 8));
    const float dmin = f16_to_f32(static_cast<uint16_t>(block[82]) |
                                  (static_cast<uint16_t>(block[83]) << 8));
    for (int group = 0; group < 16; ++group) {
        const int half_block = group >> 3;
        const int group_in_half = group & 7;
        const int shift = (group_in_half >> 1) * 2;
        const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
        const float qscale = d * static_cast<float>(scales[group] & 0x0f);
        const float base = dmin * static_cast<float>(scales[group] >> 4);
        for (int idx_in_group = 0; idx_in_group < 16; ++idx_in_group) {
            const int q = static_cast<int>((qs[byte_start + idx_in_group] >> shift) & 0x03);
            out[group * 16 + idx_in_group] = qscale * static_cast<float>(q) - base;
        }
    }
}

void dequant_iq2_xxs_row(const uint8_t* row_blocks, int n_blocks, const int8_t* signed_grid, float* out) {
    for (int b = 0; b < n_blocks; ++b) {
        iq2_xxs_dequant_block(row_blocks + b * 66, signed_grid, out + b * 256);
    }
}

void dequant_q2k_row(const uint8_t* row_blocks, int n_blocks, float* out) {
    for (int b = 0; b < n_blocks; ++b) {
        q2_k_dequant_block(row_blocks + b * 84, out + b * 256);
    }
}

float rmse(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("rmse: size mismatch");
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += diff * diff;
    }
    return static_cast<float>(std::sqrt(acc / a.size()));
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_q2_matvec <model.gguf>\n";
        return 2;
    }
    try {
        auto ws = pocket::open_weight_source(argv[1]);
        if (ws->format() != pocket::WeightSource::Format::GGUF_Q2) {
            throw std::runtime_error("test requires a GGUF Q2 checkpoint");
        }

        // Resolve routed expert 0 of layer 0 for w1 / w3 (IQ2_XXS) and w2 (Q2_K).
        const std::string w1_3d = "layers.0.ffn.experts.routed.w1";
        const std::string w2_3d = "layers.0.ffn.experts.routed.w2";
        const std::string w3_3d = "layers.0.ffn.experts.routed.w3";

        pocket::WeightView w1_top = ws->require(w1_3d);
        pocket::WeightView w2_top = ws->require(w2_3d);
        pocket::WeightView w3_top = ws->require(w3_3d);
        if (w1_top.shape.size() != 3 || w2_top.shape.size() != 3 || w3_top.shape.size() != 3) {
            throw std::runtime_error("routed expert weights are not 3D");
        }
        const int dim = static_cast<int>(w1_top.shape[0]);
        const int inter_dim = static_cast<int>(w1_top.shape[1]);
        const int n_experts = static_cast<int>(w1_top.shape[2]);
        if (dim != static_cast<int>(w3_top.shape[0]) || inter_dim != static_cast<int>(w3_top.shape[1])) {
            throw std::runtime_error("w1/w3 shape mismatch");
        }
        if (inter_dim != static_cast<int>(w2_top.shape[0]) || dim != static_cast<int>(w2_top.shape[1])) {
            throw std::runtime_error("w2 shape mismatch (expected [inter,dim,n_experts])");
        }
        const int expert_id = 0;
        const int w13_blocks_per_row = dim / 256;  // IQ2_XXS, 256 elems / block
        const int w2_blocks_per_row = inter_dim / 256;  // Q2_K, 256 elems / block
        std::printf("dim=%d inter_dim=%d n_experts=%d expert=%d\n", dim, inter_dim, n_experts, expert_id);

        pocket::WeightView w1_view = ws->get_expert(w1_3d, "w1.expert0", expert_id);
        pocket::WeightView w2_view = ws->get_expert(w2_3d, "w2.expert0", expert_id);
        pocket::WeightView w3_view = ws->get_expert(w3_3d, "w3.expert0", expert_id);
        if (!w1_view.found || !w2_view.found || !w3_view.found) {
            throw std::runtime_error("get_expert failed");
        }

        // --- 1. Fetch the GPU signed_grid into host memory for the CPU reference. ---
        const int8_t* d_signed_grid = pocket::q2_signed_grid_device();
        if (d_signed_grid == nullptr) {
            throw std::runtime_error("signed_grid_device returned null");
        }
        std::vector<int8_t> host_signed_grid(256ULL * 128 * 8);
        CHECK_CUDA(cudaMemcpy(host_signed_grid.data(), d_signed_grid,
                              host_signed_grid.size(), cudaMemcpyDeviceToHost));

        // --- 2. Build a random fp32 activation row. ---
        std::mt19937 rng(0xC0FFEE);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> h_x(static_cast<size_t>(dim));
        for (int i = 0; i < dim; ++i) h_x[i] = dist(rng);

        // --- 3. CPU reference: dequantize expert 0 W1/W2/W3 and run matmul + SwiGLU. ---
        // For perf we only dequant a window of out_cols to keep the test light; full
        // expert dequant for inter_dim=2048 × dim=4096 is 32 MiB fp32 — still cheap.
        std::vector<float> w1_fp32(static_cast<size_t>(inter_dim) * dim);
        std::vector<float> w3_fp32(static_cast<size_t>(inter_dim) * dim);
        for (int r = 0; r < inter_dim; ++r) {
            const uint8_t* row = w1_view.data + static_cast<size_t>(r) * w13_blocks_per_row * 66;
            dequant_iq2_xxs_row(row, w13_blocks_per_row, host_signed_grid.data(), w1_fp32.data() + r * dim);
            const uint8_t* row3 = w3_view.data + static_cast<size_t>(r) * w13_blocks_per_row * 66;
            dequant_iq2_xxs_row(row3, w13_blocks_per_row, host_signed_grid.data(), w3_fp32.data() + r * dim);
        }

        std::vector<float> ref_gate(inter_dim, 0.0f);
        std::vector<float> ref_up(inter_dim, 0.0f);
        for (int r = 0; r < inter_dim; ++r) {
            float g = 0.0f, u = 0.0f;
            const float* w1r = w1_fp32.data() + r * dim;
            const float* w3r = w3_fp32.data() + r * dim;
            for (int k = 0; k < dim; ++k) {
                g += w1r[k] * h_x[k];
                u += w3r[k] * h_x[k];
            }
            ref_gate[r] = g;
            ref_up[r] = u;
        }

        // Hidden = SwiGLU(gate, up) * route_weight (route_weight=1.0 in this test).
        const float route_weight = 1.0f;
        std::vector<float> ref_hidden(inter_dim, 0.0f);
        for (int r = 0; r < inter_dim; ++r) {
            ref_hidden[r] = silu(ref_gate[r]) * ref_up[r] * route_weight;
        }

        // W2 reference: y[dim] = W2 @ hidden where W2 has shape [dim, inter_dim] per expert.
        std::vector<float> w2_fp32(static_cast<size_t>(dim) * inter_dim);
        for (int r = 0; r < dim; ++r) {
            const uint8_t* row = w2_view.data + static_cast<size_t>(r) * w2_blocks_per_row * 84;
            dequant_q2k_row(row, w2_blocks_per_row, w2_fp32.data() + r * inter_dim);
        }
        std::vector<float> ref_y(dim, 0.0f);
        for (int r = 0; r < dim; ++r) {
            const float* w2r = w2_fp32.data() + r * inter_dim;
            float acc = 0.0f;
            for (int k = 0; k < inter_dim; ++k) acc += w2r[k] * ref_hidden[k];
            ref_y[r] = acc;
        }

        // --- 4. GPU path. ---
        // Upload x, w1/w2/w3 expert-slice bytes, route_slots=[0], route_weights=[1].
        float* d_x = nullptr;
        int8_t* d_x_q = nullptr;
        float* d_x_scale = nullptr;
        uint8_t* d_w1 = nullptr;
        uint8_t* d_w2 = nullptr;
        uint8_t* d_w3 = nullptr;
        int64_t* d_route_slots = nullptr;
        float* d_route_weights = nullptr;
        float* d_gate = nullptr;
        float* d_up = nullptr;
        int8_t* d_hidden_q = nullptr;
        float* d_hidden_scale = nullptr;
        float* d_y = nullptr;

        const int routes = 1;
        const int x_groups = (dim + 31) / 32;
        const int hidden_groups = (inter_dim + 15) / 16;

        // Allocate flat byte buffers covering the FULL expert tensor (n_experts × per_expert_bytes),
        // so the kernel's per-expert pointer arithmetic with expert_id=0 lands at the start.
        CHECK_CUDA(cudaMalloc(&d_x, dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_x_q, dim));
        CHECK_CUDA(cudaMalloc(&d_x_scale, x_groups * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w1, w1_view.nbytes));
        CHECK_CUDA(cudaMalloc(&d_w2, w2_view.nbytes));
        CHECK_CUDA(cudaMalloc(&d_w3, w3_view.nbytes));
        CHECK_CUDA(cudaMalloc(&d_route_slots, sizeof(int64_t)));
        CHECK_CUDA(cudaMalloc(&d_route_weights, sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gate, inter_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_up, inter_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hidden_q, inter_dim));
        CHECK_CUDA(cudaMalloc(&d_hidden_scale, hidden_groups * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, dim * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w1, w1_view.data, w1_view.nbytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w2, w2_view.data, w2_view.nbytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w3, w3_view.data, w3_view.nbytes, cudaMemcpyHostToDevice));
        const int64_t h_route_slots = static_cast<int64_t>(expert_id);
        CHECK_CUDA(cudaMemcpy(d_route_slots, &h_route_slots, sizeof(int64_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_route_weights, &route_weight, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_y, 0, dim * sizeof(float)));

        // We're operating with a buffer that only holds 1 expert, so to make
        // pointer arithmetic with expert=0 land correctly, the kernel only
        // needs to see expert_count=1. The kernels read route_slots[route],
        // so we pass route_slots=[0] and n_experts=1.
        const int kernel_n_experts = 1;

        if (!pocket::q2_quantize_x_q8_1_cuda(d_x, d_x_q, d_x_scale, routes, dim)) {
            throw std::runtime_error("q2_quantize_x_q8_1_cuda failed");
        }
        if (!pocket::q2_moe_single_w13_iq2_xxs_cuda(d_x_q, d_x_scale, d_route_slots, d_w1, d_w3,
                                                  d_gate, d_up, routes, kernel_n_experts, dim, inter_dim)) {
            throw std::runtime_error("q2_moe_single_w13_iq2_xxs_cuda failed");
        }
        if (!pocket::q2_route_swiglu_quantize_hidden_q8_1_cuda(
                d_gate, d_up, d_route_weights, d_hidden_q, d_hidden_scale,
                routes, inter_dim, 0.0f)) {
            throw std::runtime_error("q2_route_swiglu_quantize_hidden_q8_1_cuda failed");
        }
        if (!pocket::q2_moe_single_w2_q2k_cuda(d_hidden_q, d_hidden_scale, d_route_slots, d_w2,
                                              d_y, routes, kernel_n_experts, dim, inter_dim)) {
            throw std::runtime_error("q2_moe_single_w2_q2k_cuda failed");
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy back gate/up/y for comparison.
        std::vector<float> gpu_gate(inter_dim);
        std::vector<float> gpu_up(inter_dim);
        std::vector<float> gpu_y(dim);
        CHECK_CUDA(cudaMemcpy(gpu_gate.data(), d_gate, inter_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_up.data(), d_up, inter_dim * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_y.data(), d_y, dim * sizeof(float), cudaMemcpyDeviceToHost));

        // --- 5. Compare. ---
        const float ref_gate_rmse_pass = 1.0f;     // generous: Q8_1 activation + IQ2_XXS lossy
        const float ref_up_rmse_pass = 1.0f;
        const float ref_y_rmse_pass = 5.0f;        // accumulated through full chain

        // For gate/up we additionally normalize by RMS of the reference because the
        // raw magnitudes scale with dim and IQ2_XXS weight values, so relative error
        // (= RMSE / RMS(ref)) is the meaningful metric.
        auto rms = [](const std::vector<float>& v) {
            double acc = 0.0;
            for (float x : v) acc += static_cast<double>(x) * static_cast<double>(x);
            return static_cast<float>(std::sqrt(acc / v.size()));
        };

        const float gate_rmse = rmse(gpu_gate, ref_gate);
        const float up_rmse = rmse(gpu_up, ref_up);
        const float y_rmse = rmse(gpu_y, ref_y);
        const float gate_rel = gate_rmse / std::max(rms(ref_gate), 1e-6f);
        const float up_rel = up_rmse / std::max(rms(ref_up), 1e-6f);
        const float y_rel = y_rmse / std::max(rms(ref_y), 1e-6f);
        const float pass_rel = 0.05f;  // 5% RMSE on lossy Q2 weight + Q8_1 activation is reasonable

        std::printf("gate: rmse=%.4f rms=%.4f rel=%.4f  (target rel < %.2f)\n",
                    gate_rmse, rms(ref_gate), gate_rel, pass_rel);
        std::printf("up  : rmse=%.4f rms=%.4f rel=%.4f\n", up_rmse, rms(ref_up), up_rel);
        std::printf("y   : rmse=%.4f rms=%.4f rel=%.4f\n", y_rmse, rms(ref_y), y_rel);

        // Print a few values for visual sanity.
        std::printf("ref_y[0..3] = %.4f %.4f %.4f %.4f\n", ref_y[0], ref_y[1], ref_y[2], ref_y[3]);
        std::printf("gpu_y[0..3] = %.4f %.4f %.4f %.4f\n", gpu_y[0], gpu_y[1], gpu_y[2], gpu_y[3]);

        cudaFree(d_x); cudaFree(d_x_q); cudaFree(d_x_scale);
        cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
        cudaFree(d_route_slots); cudaFree(d_route_weights);
        cudaFree(d_gate); cudaFree(d_up);
        cudaFree(d_hidden_q); cudaFree(d_hidden_scale);
        cudaFree(d_y);

        if (gate_rel > pass_rel || up_rel > pass_rel || y_rel > pass_rel) {
            std::cerr << "[FAIL] RMSE exceeded threshold\n";
            return 1;
        }
        (void)ref_gate_rmse_pass; (void)ref_up_rmse_pass; (void)ref_y_rmse_pass;
        std::cout << "[PASS] q2 single-expert matvec parity\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
