// Bit-exact parity between the paged KV kernels and their contiguous
// equivalents: append, batched append, batched decode attention, and the gather
// that feeds the single-sequence read path.
//
// The block tables here are deliberately fragmented and shuffled, so a slot's
// logical positions map to non-monotonic physical blocks interleaved with other
// slots'. A kernel that quietly kept contiguous arithmetic, or that translated
// only the first position of a block, produces different numbers under such a
// table; one tested against an identity mapping would not notice either bug.
//
// Random K/V and Q are used rather than the near-uniform weights of the engine
// fixtures, so that a misaddressed token changes the softmax result instead of
// being averaged away.

#include "qwen_block_pool.hpp"
#include "qwen_block_table.hpp"
#include "qwen_cuda_ops.hpp"
#include "cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 cudaGetErrorString(error));
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

uint16_t to_half(float value) {
    uint16_t bits = 0;
    const __half h = __float2half(value);
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float from_half(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

// Device buffer that frees itself, so a failing require() does not leak.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t elements) : elements_(elements) {
        check(cudaMalloc(&data_, elements * sizeof(T)), "cudaMalloc");
        check(cudaMemset(data_, 0, elements * sizeof(T)), "cudaMemset");
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return static_cast<T*>(data_); }
    const T* get() const { return static_cast<const T*>(data_); }

    void upload(const std::vector<T>& host) {
        check(cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice), "upload");
    }

    std::vector<T> download() const {
        std::vector<T> host(elements_);
        check(cudaMemcpy(host.data(), data_, elements_ * sizeof(T),
                         cudaMemcpyDeviceToHost), "download");
        return host;
    }

private:
    void* data_ = nullptr;
    size_t elements_;
};

struct Geometry {
    int q_heads = 8;
    int kv_heads = 2;
    int head_dim = 128;
    int block_size = 16;

    int kv_stride() const { return kv_heads * head_dim; }
};

// A table whose rows interleave: slot 0 takes blocks 0, 2, 4..., slot 1 takes
// 1, 3, 5..., then each row is shuffled. Nothing about a slot's physical layout
// is then contiguous or ordered, which is the case the arena arithmetic cannot
// express.
std::vector<int32_t> fragmented_table(const std::vector<int>& context_lens,
                                      int block_size, int max_blocks_per_seq,
                                      unsigned seed) {
    const size_t slots = context_lens.size();
    std::vector<int32_t> image(slots * static_cast<size_t>(max_blocks_per_seq),
                              pocket::QwenBlockPool::kInvalidBlock);
    std::mt19937 rng(seed);
    int next = 0;
    std::vector<std::vector<int>> rows(slots);
    // Round-robin one block at a time across slots, so consecutive physical
    // blocks belong to different sequences.
    bool progress = true;
    std::vector<int> needed(slots);
    for (size_t slot = 0; slot < slots; ++slot) {
        needed[slot] = (context_lens[slot] + block_size - 1) / block_size;
    }
    while (progress) {
        progress = false;
        for (size_t slot = 0; slot < slots; ++slot) {
            if (static_cast<int>(rows[slot].size()) < needed[slot]) {
                rows[slot].push_back(next++);
                progress = true;
            }
        }
    }
    for (size_t slot = 0; slot < slots; ++slot) {
        std::shuffle(rows[slot].begin(), rows[slot].end(), rng);
        for (size_t index = 0; index < rows[slot].size(); ++index) {
            image[slot * static_cast<size_t>(max_blocks_per_seq) + index] =
                static_cast<int32_t>(rows[slot][index]);
        }
    }
    return image;
}

int total_blocks(const std::vector<int32_t>& image) {
    int highest = -1;
    for (const int32_t block : image) highest = std::max(highest, block);
    return highest + 1;
}

std::vector<uint16_t> random_half(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = to_half(dist(rng));
    return out;
}

// ============================================================================
// The host-side reference for a paged read. Deliberately written from the block
// table rather than from the kernel's arithmetic, so it fails if the two agree
// on a wrong answer.
// ============================================================================
size_t paged_offset(const std::vector<int32_t>& image, int slot, int position,
                    int max_blocks_per_seq, int block_size, int kv_stride) {
    const int32_t block = image[static_cast<size_t>(slot) * max_blocks_per_seq +
                                position / block_size];
    return (static_cast<size_t>(block) * block_size + position % block_size) *
           static_cast<size_t>(kv_stride);
}

// ============================================================================
// Test 1: single-sequence paged append lands every token where the table says.
// ============================================================================
void test_paged_append(const Geometry& geometry) {
    const int context = 300;  // Not a block multiple: exercises the partial tail.
    const int block_size = geometry.block_size;
    const int max_blocks = (context + block_size - 1) / block_size;
    const std::vector<int32_t> image =
        fragmented_table({context}, block_size, max_blocks, 11u);
    const int blocks = total_blocks(image);

    const size_t arena = static_cast<size_t>(blocks) * block_size *
                         geometry.kv_stride();
    const std::vector<uint16_t> k_host =
        random_half(static_cast<size_t>(context) * geometry.kv_stride(), 12u);
    const std::vector<uint16_t> v_host =
        random_half(static_cast<size_t>(context) * geometry.kv_stride(), 13u);

    DeviceBuffer<uint16_t> k_rows(k_host.size());
    DeviceBuffer<uint16_t> v_rows(v_host.size());
    DeviceBuffer<uint16_t> k_cache(arena);
    DeviceBuffer<uint16_t> v_cache(arena);
    DeviceBuffer<int32_t> table(image.size());
    k_rows.upload(k_host);
    v_rows.upload(v_host);
    table.upload(image);

    // Two appends, so the second starts at a non-zero and non-block-aligned
    // offset the way a decode step after a prefill does.
    const int first = 291;
    require(pocket::qwen_append_kv_cache_f16_paged_cuda(
                k_rows.get(), v_rows.get(), k_cache.get(), v_cache.get(),
                first, geometry.kv_heads, geometry.head_dim, 0, table.get(),
                block_size),
            "paged append (prefill) launch");
    require(pocket::qwen_append_kv_cache_f16_paged_cuda(
                k_rows.get() + static_cast<size_t>(first) * geometry.kv_stride(),
                v_rows.get() + static_cast<size_t>(first) * geometry.kv_stride(),
                k_cache.get(), v_cache.get(), context - first,
                geometry.kv_heads, geometry.head_dim, first, table.get(),
                block_size),
            "paged append (decode) launch");
    check(cudaDeviceSynchronize(), "paged append sync");

    const std::vector<uint16_t> k_arena = k_cache.download();
    const std::vector<uint16_t> v_arena = v_cache.download();
    for (int position = 0; position < context; ++position) {
        const size_t at = paged_offset(image, 0, position, max_blocks,
                                       block_size, geometry.kv_stride());
        for (int element = 0; element < geometry.kv_stride(); ++element) {
            const size_t source =
                static_cast<size_t>(position) * geometry.kv_stride() + element;
            require(k_arena[at + element] == k_host[source],
                    "paged append K mismatch at position " +
                        std::to_string(position));
            require(v_arena[at + element] == v_host[source],
                    "paged append V mismatch at position " +
                        std::to_string(position));
        }
    }
    std::cout << "  paged append: " << context
              << " tokens across a shuffled table PASS\n";
}

// ============================================================================
// Test 2: the gather reproduces the dense layout the read kernels index, so the
// single-sequence prefill and decode paths see exactly what a contiguous cache
// would have given them.
// ============================================================================
void test_paged_gather(const Geometry& geometry) {
    const int context = 517;
    const int block_size = geometry.block_size;
    const int max_blocks = (context + block_size - 1) / block_size;
    const std::vector<int32_t> image =
        fragmented_table({context}, block_size, max_blocks, 21u);
    const int blocks = total_blocks(image);

    const size_t arena = static_cast<size_t>(blocks) * block_size *
                         geometry.kv_stride();
    const size_t dense_elements =
        static_cast<size_t>(context) * geometry.kv_stride();
    const std::vector<uint16_t> k_host = random_half(dense_elements, 22u);
    const std::vector<uint16_t> v_host = random_half(dense_elements, 23u);

    // Scatter through the append kernel, gather back, and require the round trip
    // to be the identity on the dense layout.
    DeviceBuffer<uint16_t> k_rows(dense_elements);
    DeviceBuffer<uint16_t> v_rows(dense_elements);
    DeviceBuffer<uint16_t> k_cache(arena);
    DeviceBuffer<uint16_t> v_cache(arena);
    DeviceBuffer<uint16_t> k_dense(dense_elements);
    DeviceBuffer<uint16_t> v_dense(dense_elements);
    DeviceBuffer<int32_t> table(image.size());
    k_rows.upload(k_host);
    v_rows.upload(v_host);
    table.upload(image);

    require(pocket::qwen_append_kv_cache_f16_paged_cuda(
                k_rows.get(), v_rows.get(), k_cache.get(), v_cache.get(),
                context, geometry.kv_heads, geometry.head_dim, 0, table.get(),
                block_size),
            "gather test append launch");
    require(pocket::qwen_gather_kv_cache_f16_paged_cuda(
                k_cache.get(), v_cache.get(), k_dense.get(), v_dense.get(),
                context, geometry.kv_heads, geometry.head_dim, table.get(),
                block_size),
            "paged gather launch");
    check(cudaDeviceSynchronize(), "paged gather sync");

    const std::vector<uint16_t> k_out = k_dense.download();
    const std::vector<uint16_t> v_out = v_dense.download();
    for (size_t element = 0; element < dense_elements; ++element) {
        require(k_out[element] == k_host[element],
                "gather K round-trip mismatch at " + std::to_string(element));
        require(v_out[element] == v_host[element],
                "gather V round-trip mismatch at " + std::to_string(element));
    }
    std::cout << "  paged gather: " << context
              << " tokens round-trip bit-exact PASS\n";
}

// ============================================================================
// Test 3: batched append, where each row has its own slot and its own position,
// and the slots' blocks are interleaved in the arena.
// ============================================================================
void test_paged_batched_append(const Geometry& geometry) {
    const std::vector<int> positions = {0, 15, 16, 4095, 137};
    const std::vector<int> slots = {3, 0, 4, 1, 2};
    const int rows = static_cast<int>(positions.size());
    const int max_context = 4096;
    const int block_size = geometry.block_size;
    const int max_blocks = (max_context + block_size - 1) / block_size;

    // Every slot is backed for the whole context, so position 4095 is reachable.
    std::vector<int> lens(5, max_context);
    const std::vector<int32_t> image =
        fragmented_table(lens, block_size, max_blocks, 31u);
    const int blocks = total_blocks(image);

    const size_t arena = static_cast<size_t>(blocks) * block_size *
                         geometry.kv_stride();
    const size_t row_elements =
        static_cast<size_t>(rows) * geometry.kv_stride();
    const std::vector<uint16_t> k_host = random_half(row_elements, 32u);
    const std::vector<uint16_t> v_host = random_half(row_elements, 33u);

    DeviceBuffer<uint16_t> k_rows(row_elements);
    DeviceBuffer<uint16_t> v_rows(row_elements);
    DeviceBuffer<uint16_t> k_cache(arena);
    DeviceBuffer<uint16_t> v_cache(arena);
    DeviceBuffer<int32_t> table(image.size());
    DeviceBuffer<int32_t> d_positions(positions.size());
    DeviceBuffer<int32_t> d_slots(slots.size());
    k_rows.upload(k_host);
    v_rows.upload(v_host);
    table.upload(image);
    d_positions.upload(std::vector<int32_t>(positions.begin(), positions.end()));
    d_slots.upload(std::vector<int32_t>(slots.begin(), slots.end()));

    require(pocket::qwen_append_kv_cache_f16_batched_cuda(
                k_rows.get(), v_rows.get(), k_cache.get(), v_cache.get(), rows,
                geometry.kv_heads, geometry.head_dim, d_positions.get(),
                d_slots.get(), max_context, /*kv_slot_stride=*/0, table.get(),
                block_size, max_blocks),
            "paged batched append launch");
    check(cudaDeviceSynchronize(), "paged batched append sync");

    const std::vector<uint16_t> k_arena = k_cache.download();
    const std::vector<uint16_t> v_arena = v_cache.download();
    for (int row = 0; row < rows; ++row) {
        const size_t at = paged_offset(image, slots[static_cast<size_t>(row)],
                                       positions[static_cast<size_t>(row)],
                                       max_blocks, block_size,
                                       geometry.kv_stride());
        for (int element = 0; element < geometry.kv_stride(); ++element) {
            const size_t source =
                static_cast<size_t>(row) * geometry.kv_stride() + element;
            require(k_arena[at + element] == k_host[source],
                    "batched paged append K mismatch on row " +
                        std::to_string(row));
            require(v_arena[at + element] == v_host[source],
                    "batched paged append V mismatch on row " +
                        std::to_string(row));
        }
    }
    std::cout << "  paged batched append: " << rows
              << " rows in interleaved slots PASS\n";
}

// ============================================================================
// Test 4: batched decode attention. The same K/V history and the same queries
// are laid out both ways, and the two launches must agree to the bit. This is
// the load-bearing case: batched decode reads the blocks natively rather than
// going through the gather.
// ============================================================================
void test_paged_batched_decode(const Geometry& geometry, int attention_window,
                               int sink_tokens, const char* label) {
    // Context lengths that straddle the split-kernel threshold and land both on
    // and off block boundaries.
    const std::vector<int> context_lens = {4096, 63, 5000, 16, 1};
    const std::vector<int> slots = {2, 4, 0, 3, 1};
    const int rows = static_cast<int>(context_lens.size());
    const int max_context = 8192;
    const int block_size = geometry.block_size;
    const int max_blocks = (max_context + block_size - 1) / block_size;
    const int max_context_len =
        *std::max_element(context_lens.begin(), context_lens.end());

    // Contiguous reference: slot s owns [s * max_context, (s+1) * max_context).
    const size_t slot_stride =
        static_cast<size_t>(max_context) * geometry.kv_stride();
    const int slot_count = 5;
    const size_t contiguous_elements =
        static_cast<size_t>(slot_count) * slot_stride;

    std::vector<int> lens(static_cast<size_t>(slot_count), 0);
    for (int row = 0; row < rows; ++row) {
        lens[static_cast<size_t>(slots[static_cast<size_t>(row)])] =
            context_lens[static_cast<size_t>(row)];
    }
    const std::vector<int32_t> image =
        fragmented_table(lens, block_size, max_blocks, 41u);
    const int blocks = total_blocks(image);
    const size_t arena = static_cast<size_t>(blocks) * block_size *
                         geometry.kv_stride();

    // One K/V history per slot, written into both layouts from the same host
    // data so the only difference is the addressing.
    std::vector<uint16_t> k_contiguous(contiguous_elements, 0);
    std::vector<uint16_t> v_contiguous(contiguous_elements, 0);
    std::vector<uint16_t> k_paged(arena, 0);
    std::vector<uint16_t> v_paged(arena, 0);
    std::mt19937 rng(42u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int slot = 0; slot < slot_count; ++slot) {
        for (int position = 0; position < lens[static_cast<size_t>(slot)];
             ++position) {
            const size_t dense = static_cast<size_t>(slot) * slot_stride +
                static_cast<size_t>(position) * geometry.kv_stride();
            const size_t paged = paged_offset(image, slot, position, max_blocks,
                                              block_size, geometry.kv_stride());
            for (int element = 0; element < geometry.kv_stride(); ++element) {
                const uint16_t k_value = to_half(dist(rng));
                const uint16_t v_value = to_half(dist(rng));
                k_contiguous[dense + element] = k_value;
                v_contiguous[dense + element] = v_value;
                k_paged[paged + element] = k_value;
                v_paged[paged + element] = v_value;
            }
        }
    }

    const size_t q_elements =
        static_cast<size_t>(rows) * geometry.q_heads * geometry.head_dim;
    const std::vector<uint16_t> q_host = random_half(q_elements, 43u);

    DeviceBuffer<uint16_t> d_q(q_elements);
    DeviceBuffer<uint16_t> d_k_contiguous(contiguous_elements);
    DeviceBuffer<uint16_t> d_v_contiguous(contiguous_elements);
    DeviceBuffer<uint16_t> d_k_paged(arena);
    DeviceBuffer<uint16_t> d_v_paged(arena);
    DeviceBuffer<uint16_t> d_out_contiguous(q_elements);
    DeviceBuffer<uint16_t> d_out_paged(q_elements);
    DeviceBuffer<int32_t> d_context_lens(context_lens.size());
    DeviceBuffer<int32_t> d_slots(slots.size());
    DeviceBuffer<int32_t> d_table(image.size());

    d_q.upload(q_host);
    d_k_contiguous.upload(k_contiguous);
    d_v_contiguous.upload(v_contiguous);
    d_k_paged.upload(k_paged);
    d_v_paged.upload(v_paged);
    d_context_lens.upload(
        std::vector<int32_t>(context_lens.begin(), context_lens.end()));
    d_slots.upload(std::vector<int32_t>(slots.begin(), slots.end()));
    d_table.upload(image);

    const int splits = pocket::qwen_gqa_decode_batched_split_count(
        max_context_len, geometry.kv_heads, attention_window, sink_tokens);
    require(splits > 0, "split geometry");
    const size_t partial_elements = static_cast<size_t>(rows) *
        geometry.q_heads * splits * (geometry.head_dim + 2);
    DeviceBuffer<float> d_partials_contiguous(partial_elements);
    DeviceBuffer<float> d_partials_paged(partial_elements);

    require(pocket::qwen_gqa_decode_attention_f16_batched_cuda(
                d_q.get(), d_k_contiguous.get(), d_v_contiguous.get(),
                d_out_contiguous.get(), d_partials_contiguous.get(),
                d_context_lens.get(), d_slots.get(), rows, max_context_len,
                slot_stride, geometry.q_heads, geometry.kv_heads,
                geometry.head_dim, max_context, attention_window, sink_tokens),
            "contiguous batched decode launch");
    require(pocket::qwen_gqa_decode_attention_f16_batched_cuda(
                d_q.get(), d_k_paged.get(), d_v_paged.get(), d_out_paged.get(),
                d_partials_paged.get(), d_context_lens.get(), d_slots.get(),
                rows, max_context_len, /*kv_slot_stride=*/0, geometry.q_heads,
                geometry.kv_heads, geometry.head_dim, max_context,
                attention_window, sink_tokens, d_table.get(), block_size,
                max_blocks),
            "paged batched decode launch");
    check(cudaDeviceSynchronize(), "batched decode sync");

    const std::vector<uint16_t> out_contiguous = d_out_contiguous.download();
    const std::vector<uint16_t> out_paged = d_out_paged.download();
    // Both launches sum the same values in the same order, so they must agree
    // exactly; a tolerance here would hide a wrong-token read whose contribution
    // happens to be small.
    size_t mismatches = 0;
    float worst = 0.0f;
    for (size_t element = 0; element < q_elements; ++element) {
        if (out_contiguous[element] != out_paged[element]) {
            ++mismatches;
            worst = std::max(worst, std::fabs(from_half(out_contiguous[element]) -
                                              from_half(out_paged[element])));
        }
    }
    require(mismatches == 0,
            std::string(label) + ": " + std::to_string(mismatches) + " of " +
                std::to_string(q_elements) +
                " outputs differ from the contiguous cache, worst delta " +
                std::to_string(worst));

    // Guard against both paths being trivially zero, which would make the
    // comparison meaningless.
    bool nonzero = false;
    for (size_t element = 0; element < q_elements && !nonzero; ++element) {
        nonzero = from_half(out_paged[element]) != 0.0f;
    }
    require(nonzero, std::string(label) + ": output is entirely zero");

    std::cout << "  paged batched decode (" << label << "): " << rows
              << " rows, contexts up to " << max_context_len
              << ", bit-exact vs contiguous PASS\n";
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::cout << "[SKIP] test_qwen_paged_kv_kernels requires CUDA\n";
        return 0;
    }
    try {
        const Geometry geometry;
        test_paged_append(geometry);
        test_paged_gather(geometry);
        test_paged_batched_append(geometry);
        test_paged_batched_decode(geometry, 0, 0, "dense");
        // Sparse attention keeps its own window and sink ranges; the block
        // translation has to hold for the positions those ranges actually visit.
        test_paged_batched_decode(geometry, 1024, 64, "windowed+sink");
        std::cout << "[PASS] test_qwen_paged_kv_kernels\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_paged_kv_kernels " << ex.what() << "\n";
        return 1;
    }
}
