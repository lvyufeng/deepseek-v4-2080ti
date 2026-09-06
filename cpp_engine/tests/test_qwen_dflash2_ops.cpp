#include "qwen_ops.hpp"
#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <random>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const std::string& message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(status));
    }
}

uint16_t half_bits(float value) {
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    return pocket::qwen_bf16_to_fp16_bits(static_cast<uint16_t>(bits >> 16));
}

float half_value(uint16_t value) {
    const uint16_t exponent = static_cast<uint16_t>((value >> 10) & 0x1f);
    const uint16_t mantissa = static_cast<uint16_t>(value & 0x03ff);
    const float sign = (value & 0x8000) ? -1.0f : 1.0f;
    if (exponent == 0) return sign * std::ldexp(static_cast<float>(mantissa), -24);
    if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                             static_cast<int>(exponent) - 15);
}

template <typename T>
T* upload(const std::vector<T>& values) {
    T* device = nullptr;
    check_cuda(cudaMalloc(&device, values.size() * sizeof(T)), "cudaMalloc");
    check_cuda(cudaMemcpy(device, values.data(), values.size() * sizeof(T),
                          cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device;
}

template <typename T>
std::vector<T> download(const T* device, size_t count) {
    std::vector<T> output(count);
    check_cuda(cudaMemcpy(output.data(), device, count * sizeof(T),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return output;
}

void test_dynamic_conv() {
    const int rows = 2;
    const int hidden = 4;
    const int groups = 2;
    const int group_size = 2;
    const int kernel = 2;
    const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> dynamic = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f};
    const std::vector<float> base = {1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<uint16_t> h_input, h_dynamic, h_base;
    for (float value : input) h_input.push_back(half_bits(value));
    for (float value : dynamic) h_dynamic.push_back(half_bits(value));
    for (float value : base) h_base.push_back(half_bits(value));
    uint16_t* d_input = upload(h_input);
    uint16_t* d_dynamic = upload(h_dynamic);
    uint16_t* d_base = upload(h_base);
    uint16_t* d_output = nullptr;
    check_cuda(cudaMalloc(&d_output, h_input.size() * sizeof(uint16_t)), "conv output");
    require(pocket::qwen_dflash2_grouped_dynamic_conv_f16_cuda(
                d_input, d_dynamic, d_base, d_output, rows, hidden, groups,
                group_size, kernel), "dynamic convolution launch failed");
    const auto actual = download(d_output, h_input.size());
    for (int row = 0; row < rows; ++row) {
        for (int channel = 0; channel < hidden; ++channel) {
            const int group = channel / group_size;
            float expected = 0.0f;
            for (int tap = 0; tap < kernel; ++tap) {
                const int source = row - tap;
                const float value = source < 0 ? 0.0f : input[source * hidden + channel];
                expected += value * (base[tap * hidden + channel] +
                    dynamic[(row * kernel + tap) * groups + group]);
            }
            if (std::abs(half_value(actual[row * hidden + channel]) - expected) >= 0.05f) {
                throw std::runtime_error(
                    "dynamic convolution mismatch row=" + std::to_string(row) +
                    " channel=" + std::to_string(channel) +
                    " expected=" + std::to_string(expected) +
                    " actual=" + std::to_string(half_value(actual[row * hidden + channel])));
            }
        }
    }
    cudaFree(d_input); cudaFree(d_dynamic); cudaFree(d_base); cudaFree(d_output);
}

void test_strided_dynamic_conv() {
    const int rows = 2;
    const int hidden = 4;
    const int groups = 2;
    const int group_size = 2;
    const int kernel = 2;
    const int dynamic_taps = kernel * groups;
    const int dynamic_stride = dynamic_taps * 2;
    const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> compact_dynamic = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f};
    const std::vector<float> base = {1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<uint16_t> h_input, h_compact, h_strided, h_base;
    for (float value : input) h_input.push_back(half_bits(value));
    for (float value : compact_dynamic) h_compact.push_back(half_bits(value));
    h_strided.resize(static_cast<size_t>(rows) * dynamic_stride);
    for (int row = 0; row < rows; ++row) {
        for (int tap = 0; tap < dynamic_taps; ++tap) {
            h_strided[static_cast<size_t>(row) * dynamic_stride + tap] =
                h_compact[static_cast<size_t>(row) * dynamic_taps + tap];
        }
    }
    for (float value : base) h_base.push_back(half_bits(value));
    uint16_t* d_input = upload(h_input);
    uint16_t* d_compact = upload(h_compact);
    uint16_t* d_strided = upload(h_strided);
    uint16_t* d_base = upload(h_base);
    uint16_t* d_compact_output = nullptr;
    uint16_t* d_strided_output = nullptr;
    check_cuda(cudaMalloc(&d_compact_output, h_input.size() * sizeof(uint16_t)),
               "compact conv output");
    check_cuda(cudaMalloc(&d_strided_output, h_input.size() * sizeof(uint16_t)),
               "strided conv output");
    require(pocket::qwen_dflash2_grouped_dynamic_conv_f16_cuda(
                d_input, d_compact, d_base, d_compact_output, rows, hidden,
                groups, group_size, kernel), "compact convolution launch failed");
    require(pocket::qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
                d_input, d_strided, d_base, d_strided_output, rows, hidden,
                groups, group_size, kernel, dynamic_stride, 0),
            "strided convolution launch failed");
    const auto compact = download(d_compact_output, h_input.size());
    const auto strided = download(d_strided_output, h_input.size());
    for (size_t i = 0; i < compact.size(); ++i) {
        require(std::abs(half_value(compact[i]) - half_value(strided[i])) < 1e-3f,
                "strided convolution differs from compact convolution");
    }
    cudaFree(d_input); cudaFree(d_compact); cudaFree(d_strided); cudaFree(d_base);
    cudaFree(d_compact_output); cudaFree(d_strided_output);
}

void test_local_topk() {
    const int rows = 2;
    const int vocab = 11;
    const int vocab_start = 100;
    const int top_k = 4;
    const std::vector<float> logits = {
        0.5f, 2.0f, 2.0f, -1.0f, 0.1f, 4.0f, 3.0f, 4.0f, 0.0f, 1.0f, -2.0f,
        9.0f, 9.0f, 1.0f, 8.0f, 7.0f, 9.0f, -1.0f, 3.0f, 2.0f, 0.0f, 6.0f};
    float* d_logits = upload(logits);
    int* d_tokens = nullptr;
    float* d_values = nullptr;
    check_cuda(cudaMalloc(&d_tokens, static_cast<size_t>(rows) * top_k * sizeof(int)),
               "top-k tokens");
    check_cuda(cudaMalloc(&d_values, static_cast<size_t>(rows) * top_k * sizeof(float)),
               "top-k values");
    require(pocket::qwen_dflash2_local_topk_f32_cuda(
                d_logits, d_tokens, d_values, rows, vocab, vocab_start, top_k),
            "local top-k launch failed");
    const auto actual_tokens = download(d_tokens, static_cast<size_t>(rows) * top_k);
    const auto actual_values = download(d_values, static_cast<size_t>(rows) * top_k);
    for (int row = 0; row < rows; ++row) {
        std::vector<int> order(vocab);
        for (int token = 0; token < vocab; ++token) order[token] = token;
        std::sort(order.begin(), order.end(), [&](int left, int right) {
            return logits[static_cast<size_t>(row) * vocab + left] >
                       logits[static_cast<size_t>(row) * vocab + right] ||
                   (logits[static_cast<size_t>(row) * vocab + left] ==
                        logits[static_cast<size_t>(row) * vocab + right] &&
                    left < right);
        });
        for (int i = 0; i < top_k; ++i) {
            const int at = row * top_k + i;
            require(actual_tokens[at] == vocab_start + order[i],
                    "local top-k token mismatch");
            require(actual_values[at] == logits[static_cast<size_t>(row) * vocab + order[i]],
                    "local top-k value mismatch");
        }
    }
    cudaFree(d_logits); cudaFree(d_tokens); cudaFree(d_values);
}

// The split top-k must be bit-identical to the single-block kernel, including
// duplicate logits (tie-break on lower token) and NaN entries (skipped). Uses the
// real DFlash2 shape: 7 rows, top_k 16, TP4 vocab shard, plus an uneven split
// count so the last chunk is short.
void test_local_topk_split_matches_reference() {
    const int rows = 7;
    const int vocab = 62080;
    const int vocab_start = 62080;
    const int top_k = 16;
    std::mt19937 rng(20260824);
    std::uniform_real_distribution<float> dist(-12.0f, 12.0f);
    std::vector<float> logits(static_cast<size_t>(rows) * vocab);
    for (float& value : logits) value = dist(rng);
    // Force exact ties across distant token ids so tie-breaking is exercised in
    // different splits, and inject NaNs that both paths must skip.
    for (int row = 0; row < rows; ++row) {
        float* row_logits = logits.data() + static_cast<size_t>(row) * vocab;
        row_logits[5] = 11.5f;
        row_logits[vocab / 2 + 3] = 11.5f;
        row_logits[vocab - 17] = 11.5f;
        row_logits[9] = std::numeric_limits<float>::quiet_NaN();
        row_logits[vocab / 3] = std::numeric_limits<float>::quiet_NaN();
    }
    float* d_logits = upload(logits);
    const size_t out_elements = static_cast<size_t>(rows) * top_k;
    int* d_ref_tokens = nullptr;
    float* d_ref_values = nullptr;
    int* d_split_tokens = nullptr;
    float* d_split_values = nullptr;
    check_cuda(cudaMalloc(&d_ref_tokens, out_elements * sizeof(int)), "ref tokens");
    check_cuda(cudaMalloc(&d_ref_values, out_elements * sizeof(float)), "ref values");
    check_cuda(cudaMalloc(&d_split_tokens, out_elements * sizeof(int)), "split tokens");
    check_cuda(cudaMalloc(&d_split_values, out_elements * sizeof(float)), "split values");
    require(pocket::qwen_dflash2_local_topk_f32_cuda(
                d_logits, d_ref_tokens, d_ref_values, rows, vocab, vocab_start,
                top_k),
            "reference local top-k launch failed");
    const auto ref_tokens = download(d_ref_tokens, out_elements);
    const auto ref_values = download(d_ref_values, out_elements);

    for (int splits : {1, 8, 24, 64, 100}) {
        int* d_partial_tokens = nullptr;
        float* d_partial_values = nullptr;
        const size_t partial_elements =
            static_cast<size_t>(rows) * splits * top_k;
        check_cuda(cudaMalloc(&d_partial_tokens, partial_elements * sizeof(int)),
                   "partial tokens");
        check_cuda(cudaMalloc(&d_partial_values, partial_elements * sizeof(float)),
                   "partial values");
        require(pocket::qwen_dflash2_local_topk_split_f32_cuda(
                    d_logits, d_partial_tokens, d_partial_values, d_split_tokens,
                    d_split_values, rows, vocab, vocab_start, top_k, splits),
                "split local top-k launch failed");
        const auto split_tokens = download(d_split_tokens, out_elements);
        const auto split_values = download(d_split_values, out_elements);
        for (size_t i = 0; i < out_elements; ++i) {
            require(split_tokens[i] == ref_tokens[i],
                    "split top-k token mismatch splits=" + std::to_string(splits));
            // Bit-identical: these are selected, never accumulated.
            require(split_values[i] == ref_values[i],
                    "split top-k value mismatch splits=" + std::to_string(splits));
        }
        cudaFree(d_partial_tokens);
        cudaFree(d_partial_values);
    }
    cudaFree(d_logits); cudaFree(d_ref_tokens); cudaFree(d_ref_values);
    cudaFree(d_split_tokens); cudaFree(d_split_values);
}

// Viterbi must (a) reproduce greedy exactly when there is no choice (top_k=1),
// and (b) never return a lower total path score than greedy, since it maximises
// the same objective. Scores are recomputed on the host from the same inputs.
void test_selector_viterbi_beats_greedy() {
    const int rows = 8;
    const int rank = 256;
    const int vocab = 4096;
    std::mt19937 rng(99001);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> token_dist(0, vocab - 1);

    std::vector<uint16_t> predecessor(static_cast<size_t>(vocab) * rank);
    std::vector<uint16_t> successor(static_cast<size_t>(vocab) * rank);
    for (size_t i = 0; i < predecessor.size(); ++i) {
        predecessor[i] = half_bits(dist(rng) * 0.2f);
        successor[i] = half_bits(dist(rng) * 0.2f);
    }
    uint16_t* d_predecessor = upload(predecessor);
    uint16_t* d_successor = upload(successor);

    for (int top_k : {1, 4, 16}) {
        std::vector<float> projected(static_cast<size_t>(rows) * rank);
        for (float& value : projected) value = dist(rng);
        std::vector<int> candidates(static_cast<size_t>(rows) * top_k);
        for (int& token : candidates) token = token_dist(rng);
        std::vector<float> unary(static_cast<size_t>(rows) * top_k);
        for (float& value : unary) value = dist(rng) * 3.0f;
        const int anchor = token_dist(rng);

        float* d_projected = upload(projected);
        int* d_candidates = upload(candidates);
        float* d_unary = upload(unary);
        int* d_greedy = nullptr;
        int* d_viterbi = nullptr;
        check_cuda(cudaMalloc(&d_greedy, rows * sizeof(int)), "greedy path");
        check_cuda(cudaMalloc(&d_viterbi, rows * sizeof(int)), "viterbi path");

        require(pocket::qwen_dflash2_selector_path_projected_f16_cuda(
                    d_projected, d_candidates, d_unary, d_predecessor,
                    d_successor, d_greedy, rows, vocab, rank, top_k, anchor),
                "greedy selector launch failed");
        require(pocket::qwen_dflash2_selector_path_viterbi_f16_cuda(
                    d_projected, d_candidates, d_unary, d_predecessor,
                    d_successor, d_viterbi, rows, vocab, rank, top_k, anchor),
                "viterbi selector launch failed");
        const auto greedy_path = download(d_greedy, rows);
        const auto viterbi_path = download(d_viterbi, rows);

        // Host scorer over the same first-order chain.
        auto score_path = [&](const std::vector<int>& path) {
            double total = 0.0;
            int previous = anchor;
            for (int position = 0; position < rows; ++position) {
                const int token = path[position];
                int slot = -1;
                for (int i = 0; i < top_k; ++i) {
                    if (candidates[position * top_k + i] == token) { slot = i; break; }
                }
                require(slot >= 0, "path token not among candidates");
                float score = unary[position * top_k + slot];
                for (int r = 0; r < rank; ++r) {
                    score += half_value(
                                 predecessor[static_cast<size_t>(previous) * rank + r]) *
                             projected[static_cast<size_t>(position) * rank + r] *
                             half_value(
                                 successor[static_cast<size_t>(token) * rank + r]);
                }
                total += score;
                previous = token;
            }
            return total;
        };

        const double greedy_score = score_path(greedy_path);
        const double viterbi_score = score_path(viterbi_path);
        if (top_k == 1) {
            for (int position = 0; position < rows; ++position) {
                require(greedy_path[position] == viterbi_path[position],
                        "top_k=1 viterbi must match greedy exactly");
            }
        }
        require(viterbi_score >= greedy_score - 1e-3,
                "viterbi path scored below greedy for top_k=" +
                    std::to_string(top_k));
        std::printf("  selector top_k=%2d greedy=%.5f viterbi=%.5f gain=%+.5f\n",
                    top_k, greedy_score, viterbi_score,
                    viterbi_score - greedy_score);
        cudaFree(d_projected); cudaFree(d_candidates); cudaFree(d_unary);
        cudaFree(d_greedy); cudaFree(d_viterbi);
    }
    cudaFree(d_predecessor); cudaFree(d_successor);
}

void test_device_global_top1_merge() {
    // This is the exact world-major layout produced by the NCCL all-gather
    // wrapper: [rank][row]. Keep the test independent of NCCL so it also checks
    // the merge kernel on a single GPU, including stream ordering.
    const int world = 4;
    const int rows = 5;
    const int top_k = 1;
    const int invalid_token = std::numeric_limits<int>::max();
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    const std::vector<int> gathered_tokens = {
        40, 700, 2, 99, 3,
        41, 100, 1, 98, 4,
        42, 50, 7, 97, 5,
        43, 200, 8, 96, 6,
    };
    const std::vector<float> gathered_logits = {
        1.0f, 5.0f, NAN, 4.0f, -1.0f,
        3.0f, 5.0f, NAN, 4.0f, -1.0f,
        2.0f, 4.9f, NAN, 4.0f, -1.0f,
        NAN, 5.0f, NAN, NAN, -2.0f,
    };
    const std::vector<int> expected_tokens = {41, 100, invalid_token, 97, 3};
    const std::vector<float> expected_logits = {3.0f, 5.0f, negative_infinity,
                                                4.0f, -1.0f};

    int* d_gathered_tokens = upload(gathered_tokens);
    float* d_gathered_logits = upload(gathered_logits);
    int* d_output_tokens = nullptr;
    float* d_output_logits = nullptr;
    check_cuda(cudaMalloc(&d_output_tokens, rows * sizeof(int)),
               "global top-1 output tokens");
    check_cuda(cudaMalloc(&d_output_logits, rows * sizeof(float)),
               "global top-1 output logits");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "global top-1 stream");
    require(pocket::qwen_dflash2_merge_topk_f32_cuda(
                d_gathered_tokens, d_gathered_logits, d_output_tokens,
                d_output_logits, world, rows, top_k, stream),
            "device global top-1 merge launch failed");
    std::vector<int> output_tokens(rows);
    std::vector<float> output_logits(rows);
    check_cuda(cudaMemcpyAsync(output_tokens.data(), d_output_tokens,
                               rows * sizeof(int), cudaMemcpyDeviceToHost,
                               stream),
               "copy global top-1 tokens");
    check_cuda(cudaMemcpyAsync(output_logits.data(), d_output_logits,
                               rows * sizeof(float), cudaMemcpyDeviceToHost,
                               stream),
               "copy global top-1 logits");
    check_cuda(cudaStreamSynchronize(stream), "sync global top-1 merge");
    for (int row = 0; row < rows; ++row) {
        require(output_tokens[static_cast<size_t>(row)] == expected_tokens[row],
                "device global top-1 token mismatch row=" + std::to_string(row));
        const float actual = output_logits[static_cast<size_t>(row)];
        const float expected = expected_logits[row];
        if (std::isinf(expected)) {
            require(std::isinf(actual) && actual < 0.0f,
                    "device global top-1 empty logit mismatch row=" +
                        std::to_string(row));
        } else {
            require(actual == expected,
                    "device global top-1 logit mismatch row=" +
                        std::to_string(row));
        }
    }
    check_cuda(cudaStreamDestroy(stream), "destroy global top-1 stream");
    cudaFree(d_gathered_tokens);
    cudaFree(d_gathered_logits);
    cudaFree(d_output_tokens);
    cudaFree(d_output_logits);
}

void test_packed_top1_key() {
    // Simulate the NCCL uint64 Max reduction on one GPU. Each rank packs one
    // local candidate per row; the host-side max below is exactly the ordering
    // used by ncclMax for the production packed path.
    const int world = 4;
    const int rows = 8;
    const int invalid_token = std::numeric_limits<int>::max();
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    const std::vector<std::vector<int>> tokens = {
        {40, 700, 2, 99, 3, 8, 10, 20},
        {41, 100, 1, 98, 4, 7, 11, 21},
        {42, 50, 7, 97, 5, 6, 12, 22},
        {43, 200, 8, 96, 6, 5, 13, 23},
    };
    const std::vector<std::vector<float>> logits = {
        {1.0f, 5.0f, NAN, 4.0f, -1.0f, -0.0f, negative_infinity, NAN},
        {3.0f, 5.0f, NAN, 4.0f, -1.0f, +0.0f, negative_infinity, NAN},
        {2.0f, 4.9f, NAN, 4.0f, -1.0f, -0.0f, negative_infinity, NAN},
        {NAN, 5.0f, NAN, NAN, -2.0f, +0.0f, negative_infinity, NAN},
    };
    const std::vector<int> expected_tokens = {
        41, 100, invalid_token, 97, 3, 5, 10, invalid_token};
    const std::vector<float> expected_logits = {
        3.0f, 5.0f, negative_infinity, 4.0f, -1.0f, 0.0f,
        negative_infinity, negative_infinity};

    std::vector<int*> d_tokens(world, nullptr);
    std::vector<float*> d_logits(world, nullptr);
    std::vector<uint64_t*> d_keys(world, nullptr);
    std::vector<std::vector<uint64_t>> packed(
        world, std::vector<uint64_t>(rows));
    for (int rank = 0; rank < world; ++rank) {
        d_tokens[rank] = upload(tokens[rank]);
        d_logits[rank] = upload(logits[rank]);
        check_cuda(cudaMalloc(&d_keys[rank], rows * sizeof(uint64_t)),
                   "packed top-1 keys");
    }

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "packed top-1 stream");
    for (int rank = 0; rank < world; ++rank) {
        require(pocket::qwen_dflash2_pack_top1_key_f32_cuda(
                    d_tokens[rank], d_logits[rank], d_keys[rank], rows, stream),
                "packed top-1 pack launch failed");
        check_cuda(cudaMemcpyAsync(packed[rank].data(), d_keys[rank],
                                   rows * sizeof(uint64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "copy packed top-1 keys");
    }
    check_cuda(cudaStreamSynchronize(stream), "sync packed top-1 pack");

    std::vector<uint64_t> reduced(rows, 0ull);
    for (int row = 0; row < rows; ++row) {
        for (int rank = 0; rank < world; ++rank) {
            reduced[row] = std::max(reduced[row], packed[rank][row]);
        }
    }
    uint64_t* d_reduced = upload(reduced);
    int* d_output_tokens = nullptr;
    float* d_output_logits = nullptr;
    check_cuda(cudaMalloc(&d_output_tokens, rows * sizeof(int)),
               "packed top-1 output tokens");
    check_cuda(cudaMalloc(&d_output_logits, rows * sizeof(float)),
               "packed top-1 output logits");
    require(pocket::qwen_dflash2_unpack_top1_key_f32_cuda(
                d_reduced, d_output_tokens, d_output_logits, rows, stream),
            "packed top-1 unpack launch failed");
    std::vector<int> output_tokens(rows);
    std::vector<float> output_logits(rows);
    check_cuda(cudaMemcpyAsync(output_tokens.data(), d_output_tokens,
                               rows * sizeof(int), cudaMemcpyDeviceToHost,
                               stream),
               "copy unpacked top-1 tokens");
    check_cuda(cudaMemcpyAsync(output_logits.data(), d_output_logits,
                               rows * sizeof(float), cudaMemcpyDeviceToHost,
                               stream),
               "copy unpacked top-1 logits");
    check_cuda(cudaStreamSynchronize(stream), "sync packed top-1 unpack");

    for (int row = 0; row < rows; ++row) {
        require(output_tokens[row] == expected_tokens[row],
                "packed top-1 token mismatch row=" + std::to_string(row));
        if (std::isinf(expected_logits[row])) {
            require(std::isinf(output_logits[row]) && output_logits[row] < 0.0f,
                    "packed top-1 -inf mismatch row=" + std::to_string(row));
        } else {
            require(output_logits[row] == expected_logits[row],
                    "packed top-1 logit mismatch row=" + std::to_string(row));
        }
    }
    // Signed zero is canonicalized before packing so it has the same ordering
    // as the host comparator and decodes to the canonical +0 representation.
    require(!std::signbit(output_logits[5]),
            "packed top-1 failed to canonicalize signed zero");

    check_cuda(cudaStreamDestroy(stream), "destroy packed top-1 stream");
    for (int rank = 0; rank < world; ++rank) {
        cudaFree(d_tokens[rank]);
        cudaFree(d_logits[rank]);
        cudaFree(d_keys[rank]);
    }
    cudaFree(d_reduced);
    cudaFree(d_output_tokens);
    cudaFree(d_output_logits);
}

void test_selector_path() {
    const int rows = 3;
    const int vocab = 8;
    const int hidden_size = 2;
    const int rank = 2;
    const int top_k = 2;
    const int anchor = 0;
    const std::vector<float> hidden = {1, 0, 0, 1, 1, 1};
    std::vector<uint16_t> h_hidden, h_predecessor, h_successor, h_projection;
    for (float value : hidden) h_hidden.push_back(half_bits(value));
    for (int token = 0; token < vocab; ++token) {
        h_predecessor.push_back(half_bits(token == 0 ? 1.0f : token == 1 ? 0.0f : 1.0f));
        h_predecessor.push_back(half_bits(token == 1 ? 1.0f : 0.0f));
        h_successor.push_back(half_bits(token == 1 || token == 4 || token == 5 ? 1.0f : 0.0f));
        h_successor.push_back(half_bits(token == 2 || token == 3 ? 1.0f : 0.0f));
    }
    h_successor[5 * rank] = half_bits(1.0f);
    h_successor[5 * rank + 1] = half_bits(0.0f);
    h_projection = {half_bits(1.0f), half_bits(0.0f),
                    half_bits(0.0f), half_bits(1.0f)};
    const std::vector<int> candidates = {1, 2, 2, 3, 4, 5};
    const std::vector<float> unary(candidates.size(), 0.0f);
    uint16_t* d_hidden = upload(h_hidden);
    int* d_candidates = upload(candidates);
    float* d_unary = upload(unary);
    uint16_t* d_predecessor = upload(h_predecessor);
    uint16_t* d_successor = upload(h_successor);
    uint16_t* d_projection = upload(h_projection);
    int* d_output = nullptr;
    check_cuda(cudaMalloc(&d_output, rows * sizeof(int)), "selector output");
    require(pocket::qwen_dflash2_selector_path_f16_cuda(
                d_hidden, d_candidates, d_unary, d_predecessor, d_successor,
                d_projection, d_output, rows, vocab, hidden_size, rank, top_k,
                anchor), "selector launch failed");
    const auto output = download(d_output, rows);
    require(output == std::vector<int>({1, 2, 4}),
            "selector predecessor update or tie-breaking mismatch");
    cudaFree(d_hidden); cudaFree(d_candidates); cudaFree(d_unary);
    cudaFree(d_predecessor); cudaFree(d_successor); cudaFree(d_projection);
    cudaFree(d_output);
}

void test_grouped_attention() {
    const int rows = 3;
    const int q_heads = 4;
    const int kv_heads = 1;
    const int head_dim = 128;
    const int context_len = 5;
    const int max_context = 12;
    const int sliding_window = 4;
    auto make_values = [](size_t count, float phase) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = 0.02f * std::sin(phase + static_cast<float>(i % 37));
        }
        return values;
    };
    const auto q_values = make_values(
        static_cast<size_t>(rows) * q_heads * head_dim, 0.11f);
    const auto context_k_values = make_values(
        static_cast<size_t>(context_len) * kv_heads * head_dim, 0.23f);
    const auto context_v_values = make_values(
        static_cast<size_t>(context_len) * kv_heads * head_dim, 0.37f);
    const auto block_k_values = make_values(
        static_cast<size_t>(rows) * kv_heads * head_dim, 0.41f);
    const auto block_v_values = make_values(
        static_cast<size_t>(rows) * kv_heads * head_dim, 0.53f);
    auto convert = [](const std::vector<float>& values) {
        std::vector<uint16_t> output;
        output.reserve(values.size());
        for (float value : values) output.push_back(half_bits(value));
        return output;
    };
    const auto h_q = convert(q_values);
    const auto h_context_k = convert(context_k_values);
    const auto h_context_v = convert(context_v_values);
    const auto h_block_k = convert(block_k_values);
    const auto h_block_v = convert(block_v_values);
    uint16_t* d_q = upload(h_q);
    uint16_t* d_context_k = upload(h_context_k);
    uint16_t* d_context_v = upload(h_context_v);
    uint16_t* d_block_k = upload(h_block_k);
    uint16_t* d_block_v = upload(h_block_v);
    uint16_t* d_reference = nullptr;
    uint16_t* d_grouped = nullptr;
    const size_t output_count = static_cast<size_t>(rows) * q_heads * head_dim;
    check_cuda(cudaMalloc(&d_reference, output_count * sizeof(uint16_t)),
               "reference attention output");
    check_cuda(cudaMalloc(&d_grouped, output_count * sizeof(uint16_t)),
               "grouped attention output");
    require(pocket::qwen_dflash2_attention_f16_cuda(
                d_q, d_context_k, d_context_v, d_block_k, d_block_v,
                d_reference, rows, q_heads, kv_heads, head_dim, context_len,
                max_context, sliding_window),
            "reference attention launch failed");
    require(pocket::qwen_dflash2_attention_grouped_f16_cuda(
                d_q, d_context_k, d_context_v, d_block_k, d_block_v,
                d_grouped, rows, q_heads, kv_heads, head_dim, context_len,
                max_context, sliding_window),
            "grouped attention launch failed");
    check_cuda(cudaDeviceSynchronize(), "synchronize grouped attention");
    const auto reference = download(d_reference, output_count);
    const auto grouped = download(d_grouped, output_count);
    float worst = 0.0f;
    for (size_t i = 0; i < output_count; ++i) {
        require(std::isfinite(half_value(reference[i])) &&
                    std::isfinite(half_value(grouped[i])),
                "grouped attention produced non-finite output");
        worst = std::max(worst, std::abs(half_value(reference[i]) -
                                         half_value(grouped[i])));
    }
    require(worst <= 2.0e-3f,
            "grouped attention differs from reference: " + std::to_string(worst));
    cudaFree(d_q); cudaFree(d_context_k); cudaFree(d_context_v);
    cudaFree(d_block_k); cudaFree(d_block_v);
    cudaFree(d_reference); cudaFree(d_grouped);
}

void test_rope_and_attention() {
    const int rows = 2;
    const int q_heads = 2;
    const int kv_heads = 1;
    const int head_dim = 4;
    const std::vector<float> q_values = {
        1, 2, 3, 4, 2, -1, 0.5f, 3,
        0.5f, 1, -2, 2, 3, 0, 1, -1};
    const std::vector<float> k_values = {
        1, 0, 2, -1, 0.5f, 2, -1, 1};
    const std::vector<float> context_k = {1, 0, 0, 1, 0, 1, 1, 0};
    const std::vector<float> context_v = {1, 2, 3, 4, 2, 0, 1, -1};
    const std::vector<float> block_k = {0, 0, 1, 0, 0, 1, 0, 0};
    const std::vector<float> block_v = {-1, 1, 2, 0, 3, -2, 0, 1};
    auto convert = [](const std::vector<float>& values) {
        std::vector<uint16_t> output;
        for (float value : values) output.push_back(half_bits(value));
        return output;
    };
    auto h_q = convert(q_values);
    auto h_k = convert(k_values);
    auto h_context_k = convert(context_k);
    auto h_context_v = convert(context_v);
    auto h_block_k = convert(block_k);
    auto h_block_v = convert(block_v);
    uint16_t* d_q = upload(h_q);
    uint16_t* d_k = upload(h_k);
    uint16_t* d_k_only = upload(h_k);
    uint16_t* d_context_k = upload(h_context_k);
    uint16_t* d_context_v = upload(h_context_v);
    uint16_t* d_block_k = upload(h_block_k);
    uint16_t* d_block_v = upload(h_block_v);
    uint16_t* d_output = nullptr;
    check_cuda(cudaMalloc(&d_output, h_q.size() * sizeof(uint16_t)), "attention output");
    require(pocket::qwen_dflash2_rope_rows_f16_cuda(
                d_q, d_k, rows, q_heads, kv_heads, head_dim, 3, 10000000.0f),
            "DFlash2 RoPE launch failed");
    require(pocket::qwen_dflash2_rope_k_rows_f16_cuda(
                d_k_only, rows, kv_heads, head_dim, 3, 10000000.0f),
            "DFlash2 K-only RoPE launch failed");
    const auto k_only = download(d_k_only, h_k.size());
    const auto paired_k = download(d_k, h_k.size());
    require(k_only == paired_k, "K-only RoPE differs from paired K RoPE");
    require(pocket::qwen_dflash2_attention_f16_cuda(
                d_q, d_context_k, d_context_v, d_block_k, d_block_v, d_output,
                rows, q_heads, kv_heads, head_dim, 2, 8, 2048),
            "DFlash2 attention launch failed");
    const auto output = download(d_output, h_q.size());
    for (float value : output) require(std::isfinite(half_value(value)), "attention produced non-finite");
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_k_only); cudaFree(d_context_k); cudaFree(d_context_v);
    cudaFree(d_block_k); cudaFree(d_block_v); cudaFree(d_output);
}

}  // namespace

int main() {
    try {
        test_dynamic_conv();
        test_strided_dynamic_conv();
        test_local_topk();
        test_local_topk_split_matches_reference();
        test_device_global_top1_merge();
        test_packed_top1_key();
        test_selector_viterbi_beats_greedy();
        test_selector_path();
        test_grouped_attention();
        test_rope_and_attention();
        std::cout << "[PASS] qwen_dflash2_ops\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << "\n";
        return 1;
    }
}
