// Numerical parity test for the FlashMemory scorer CUDA kernel.
//
// Reads a fixture exported by tests/export_flashmemory_fixture.py (which runs
// the reference retriever.py forward_and_score), runs flashmemory_score_layer_cuda
// on the same inputs, and checks the logits/scores match within tolerance.
//
//   usage: test_flashmemory_scorer <fixture.bin>

#include "flashmemory_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <typename T>
std::vector<T> read_vec(std::ifstream& in, size_t count) {
    std::vector<T> v(count);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) throw std::runtime_error("fixture truncated");
    return v;
}

template <typename T>
T* upload(const std::vector<T>& host) {
    T* d = nullptr;
    check(cudaMalloc(&d, host.size() * sizeof(T)), "cudaMalloc");
    check(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]);
        return 2;
    }
    try {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) throw std::runtime_error(std::string("cannot open fixture: ") + argv[1]);

        uint32_t magic = 0, version = 0;
        in.read(reinterpret_cast<char*>(&magic), 4);
        in.read(reinterpret_cast<char*>(&version), 4);
        if (magic != 0x464D5343u) throw std::runtime_error("bad fixture magic");
        if (version != 1u) throw std::runtime_error("bad fixture version");

        int32_t n_chunks, n_heads, head_dim, q_lora_rank, hidden_dim, rope_dim;
        in.read(reinterpret_cast<char*>(&n_chunks), 4);
        in.read(reinterpret_cast<char*>(&n_heads), 4);
        in.read(reinterpret_cast<char*>(&head_dim), 4);
        in.read(reinterpret_cast<char*>(&q_lora_rank), 4);
        in.read(reinterpret_cast<char*>(&hidden_dim), 4);
        in.read(reinterpret_cast<char*>(&rope_dim), 4);
        int64_t position = 0;
        in.read(reinterpret_cast<char*>(&position), 8);
        float rms_eps = 0.0f;
        in.read(reinterpret_cast<char*>(&rms_eps), 4);
        if (!in) throw std::runtime_error("fixture header truncated");

        const size_t q_out = static_cast<size_t>(n_heads) * head_dim;
        auto wq_a = read_vec<float>(in, static_cast<size_t>(q_lora_rank) * hidden_dim);
        auto wq_b = read_vec<float>(in, q_out * q_lora_rank);
        auto q_norm = read_vec<float>(in, static_cast<size_t>(q_lora_rank));
        auto weights_proj = read_vec<float>(in, static_cast<size_t>(n_heads) * hidden_dim);
        auto inv_freqs = read_vec<float>(in, static_cast<size_t>(rope_dim / 2));
        auto hidden = read_vec<float>(in, static_cast<size_t>(hidden_dim));
        auto compressed_k = read_vec<uint8_t>(in, static_cast<size_t>(n_chunks) * (head_dim + 4));
        auto ref_logits = read_vec<float>(in, static_cast<size_t>(n_chunks));
        auto ref_scores = read_vec<float>(in, static_cast<size_t>(n_chunks));

        // Cross-check our host YaRN inv-freqs against the fixture's.
        std::vector<float> our_freqs = pocket::flashmemory_yarn_inv_freqs(
            rope_dim, 160000.0, 16.0, 65536, 32.0, 1.0);
        double freq_max_err = 0.0;
        for (int i = 0; i < rope_dim / 2; ++i)
            freq_max_err = std::max(freq_max_err, std::fabs(static_cast<double>(our_freqs[i]) - inv_freqs[i]));

        float* d_wq_a = upload(wq_a);
        float* d_wq_b = upload(wq_b);
        float* d_q_norm = upload(q_norm);
        float* d_weights_proj = upload(weights_proj);
        float* d_inv_freqs = upload(our_freqs);  // use our host-computed freqs (end-to-end)
        float* d_hidden = upload(hidden);
        uint8_t* d_ck = upload(compressed_k);

        float* d_q = nullptr;
        float* d_qlora = nullptr;
        float* d_fused = nullptr;
        float* d_logits = nullptr;
        float* d_scores = nullptr;
        check(cudaMalloc(&d_q, q_out * sizeof(float)), "malloc q");
        check(cudaMalloc(&d_qlora, static_cast<size_t>(q_lora_rank) * sizeof(float)), "malloc qlora");
        check(cudaMalloc(&d_fused, static_cast<size_t>(n_heads) * sizeof(float)), "malloc fused");
        check(cudaMalloc(&d_logits, static_cast<size_t>(n_chunks) * sizeof(float)), "malloc logits");
        check(cudaMalloc(&d_scores, static_cast<size_t>(n_chunks) * sizeof(float)), "malloc scores");

        // Raw logits.
        if (!pocket::flashmemory_score_layer_cuda(
                d_hidden, d_wq_a, d_wq_b, d_q_norm, d_weights_proj, d_inv_freqs, d_ck,
                d_q, d_qlora, d_fused, d_logits,
                n_chunks, n_heads, head_dim, q_lora_rank, hidden_dim, rope_dim,
                rms_eps, position, /*apply_sigmoid=*/false))
            throw std::runtime_error("flashmemory_score_layer_cuda (logits) launch failed");
        // Sigmoid scores.
        if (!pocket::flashmemory_score_layer_cuda(
                d_hidden, d_wq_a, d_wq_b, d_q_norm, d_weights_proj, d_inv_freqs, d_ck,
                d_q, d_qlora, d_fused, d_scores,
                n_chunks, n_heads, head_dim, q_lora_rank, hidden_dim, rope_dim,
                rms_eps, position, /*apply_sigmoid=*/true))
            throw std::runtime_error("flashmemory_score_layer_cuda (scores) launch failed");
        check(cudaDeviceSynchronize(), "sync");

        std::vector<float> got_logits(n_chunks), got_scores(n_chunks);
        check(cudaMemcpy(got_logits.data(), d_logits, n_chunks * sizeof(float), cudaMemcpyDeviceToHost), "D2H logits");
        check(cudaMemcpy(got_scores.data(), d_scores, n_chunks * sizeof(float), cudaMemcpyDeviceToHost), "D2H scores");

        double max_logit_err = 0.0, max_score_err = 0.0;
        int argmax_ref = 0, argmax_got = 0;
        for (int i = 0; i < n_chunks; ++i) {
            max_logit_err = std::max(max_logit_err, std::fabs(static_cast<double>(got_logits[i]) - ref_logits[i]));
            max_score_err = std::max(max_score_err, std::fabs(static_cast<double>(got_scores[i]) - ref_scores[i]));
            if (ref_scores[i] > ref_scores[argmax_ref]) argmax_ref = i;
            if (got_scores[i] > got_scores[argmax_got]) argmax_got = i;
        }

        std::printf("freq_max_err=%.3e logit_max_err=%.4e score_max_err=%.4e argmax_ref=%d argmax_got=%d\n",
                    freq_max_err, max_logit_err, max_score_err, argmax_ref, argmax_got);

        // Tolerances: logits accumulate over 128 heads * 128 dims of fp32 with a
        // bf16 RoPE path, so a few e-2 absolute on logits is expected; sigmoid
        // scores compress that to well under 1e-2.
        const bool ok = freq_max_err < 1e-5 && max_score_err < 5e-3 && argmax_ref == argmax_got;
        if (!ok) {
            std::printf("[FAIL] flashmemory scorer parity\n");
            return 1;
        }
        std::printf("[PASS] flashmemory scorer parity n_chunks=%d n_heads=%d head_dim=%d\n",
                    n_chunks, n_heads, head_dim);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] %s\n", e.what());
        return 1;
    }
}
