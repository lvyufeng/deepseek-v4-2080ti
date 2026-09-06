#include "dspark.hpp"
#include "cuda_ops.hpp"
#include "device_runtime.hpp"
#include "json_lite.hpp"
#include "safetensors_reader.hpp"
#include "tp_comm.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <map>
#include <memory>

namespace dspark {

// The device runtime lives in namespace pocket; pull its names in so the migrated
// allocation and transfer call sites read the same as they do in the DeepSeek-V4 engine.
using namespace pocket;  // NOLINT(build/namespaces)

namespace {

void check_device(bool ok, const char* what) {
    if (!ok) {
        const std::string detail = device_last_error();
        throw std::runtime_error(std::string("device error in ") + what +
                                 (detail.empty() ? "" : ": " + detail));
    }
}

// Optional scalar lookups: config.json for this checkpoint carries every field
// we need, but falling back to the struct default keeps older configs loadable.
int json_int_or(const pocket::JsonObject& obj, const std::string& key, int fallback) {
    const pocket::JsonValue* v = pocket::object_get(obj, key);
    if (v == nullptr || !v->is_number()) return fallback;
    return static_cast<int>(v->number());
}

float json_float_or(const pocket::JsonObject& obj, const std::string& key, float fallback) {
    const pocket::JsonValue* v = pocket::object_get(obj, key);
    if (v == nullptr || !v->is_number()) return fallback;
    return static_cast<float>(v->number());
}

std::vector<int> json_int_array_or(const pocket::JsonObject& obj, const std::string& key,
                                   const std::vector<int>& fallback) {
    const pocket::JsonValue* v = pocket::object_get(obj, key);
    if (v == nullptr || !v->is_array()) return fallback;
    std::vector<int> out;
    for (const auto& item : v->array()) {
        if (!item.is_number()) return fallback;
        out.push_back(static_cast<int>(item.number()));
    }
    return out.empty() ? fallback : out;
}

// Row-major [rows, cols] slice of `rows_take` rows starting at `row_start`.
std::vector<uint8_t> slice_rows_u8(const uint8_t* src, int row_start, int rows_take, int cols) {
    std::vector<uint8_t> out(static_cast<size_t>(rows_take) * cols);
    std::memcpy(out.data(),
                src + static_cast<size_t>(row_start) * cols,
                out.size());
    return out;
}

// Row-major [rows, cols] slice of `cols_take` columns starting at `col_start`.
std::vector<uint8_t> slice_cols_u8(const uint8_t* src, int rows, int cols,
                                   int col_start, int cols_take) {
    std::vector<uint8_t> out(static_cast<size_t>(rows) * cols_take);
    for (int r = 0; r < rows; ++r) {
        std::memcpy(out.data() + static_cast<size_t>(r) * cols_take,
                    src + static_cast<size_t>(r) * cols + col_start,
                    static_cast<size_t>(cols_take));
    }
    return out;
}

}  // namespace

// ============================================================================
// Config
// ============================================================================

Config Config::from_json(const char* config_path) {
    Config cfg;

    std::ifstream f(config_path);
    if (!f) {
        throw std::runtime_error(std::string("DSpark: cannot open config ") + config_path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const pocket::JsonValue root = pocket::parse_json(ss.str());
    if (!root.is_object()) {
        throw std::runtime_error(std::string("DSpark: config is not a JSON object: ") + config_path);
    }
    const pocket::JsonObject& o = root.object();

    cfg.block_size = json_int_or(o, "dspark_block_size", cfg.block_size);
    cfg.noise_token_id = json_int_or(o, "dspark_noise_token_id", cfg.noise_token_id);
    cfg.target_layer_ids = json_int_array_or(o, "dspark_target_layer_ids", cfg.target_layer_ids);
    cfg.markov_rank = json_int_or(o, "dspark_markov_rank", cfg.markov_rank);
    cfg.window_size = json_int_or(o, "sliding_window", cfg.window_size);
    cfg.dim = json_int_or(o, "hidden_size", cfg.dim);
    cfg.vocab_size = json_int_or(o, "vocab_size", cfg.vocab_size);
    cfg.hc_mult = json_int_or(o, "hc_mult", cfg.hc_mult);
    cfg.norm_eps = json_float_or(o, "rms_norm_eps", cfg.norm_eps);

    cfg.n_heads = json_int_or(o, "num_attention_heads", cfg.n_heads);
    cfg.head_dim = json_int_or(o, "head_dim", cfg.head_dim);
    cfg.q_lora_rank = json_int_or(o, "q_lora_rank", cfg.q_lora_rank);
    cfg.o_lora_rank = json_int_or(o, "o_lora_rank", cfg.o_lora_rank);
    cfg.o_groups = json_int_or(o, "o_groups", cfg.o_groups);
    cfg.rope_dim = json_int_or(o, "qk_rope_head_dim", cfg.rope_dim);
    cfg.rope_theta = json_float_or(o, "rope_theta", cfg.rope_theta);

    cfg.n_experts = json_int_or(o, "n_routed_experts", cfg.n_experts);
    cfg.topk = json_int_or(o, "num_experts_per_tok", cfg.topk);
    cfg.moe_inter = json_int_or(o, "moe_intermediate_size", cfg.moe_inter);
    cfg.route_scale = json_float_or(o, "routed_scaling_factor", cfg.route_scale);
    cfg.swiglu_limit = json_float_or(o, "swiglu_limit", cfg.swiglu_limit);

    // The checkpoint stores one DSpark stage per `mtp.N.` prefix. config.json
    // has no field for it (num_nextn_predict_layers counts predicted tokens,
    // not stages), so this default is refined by probing the index at load.
    cfg.n_stages = json_int_or(o, "dspark_n_stages", cfg.n_stages);

    return cfg;
}

// ============================================================================
// DSparkEngine::Impl
// ============================================================================

struct DSparkEngine::Impl {
    Config config;
    int tp_rank;
    int tp_world_size;
    std::string checkpoint_dir;
    // NCCL channel for the TP all-reduces. Empty means none was supplied, which
    // is fine at tp_world_size == 1 and fatal above it.
    int tp_device = 0;
    std::string nccl_id_path;
    // bf16 staging for the all-reduce, sized on first use and kept.
    uint16_t* d_reduce_bf16 = nullptr;
    int reduce_capacity = 0;

    // All-reduce d_values across ranks, in place. Matches the main model's
    // reduction: fp32 packed down to bf16 for the wire, which halves the
    // traffic and is what the main model's activations already round to at
    // every layer boundary anyway.
    void tp_all_reduce(float* d_values, int count) {
        if (tp_world_size <= 1) return;
        if (nccl_id_path.empty()) {
            throw std::runtime_error(
                "DSpark TP>1 needs an NCCL id path; construct with the 5-argument form");
        }
#ifdef POCKET_HAVE_TP_COMM
        if (count > reduce_capacity) {
            if (d_reduce_bf16 != nullptr) device_free(d_reduce_bf16);
            d_reduce_bf16 = nullptr;
            check_device(device_malloc_into(d_reduce_bf16, static_cast<size_t>(count) * sizeof(uint16_t)),
                         "device_malloc dspark reduce scratch");
            reduce_capacity = count;
        }
        if (!pocket::fp32_to_bf16_cuda(d_values, d_reduce_bf16, count))
            throw std::runtime_error("dspark reduce pack failed");
        pocket::tp_all_reduce_sum_bf16_inplace(tp_world_size, tp_rank, tp_device,
                                               nccl_id_path.c_str(), d_reduce_bf16, count);
        if (!pocket::bf16_to_fp32_cuda(d_reduce_bf16, d_values, count))
            throw std::runtime_error("dspark reduce unpack failed");
#else
        throw std::runtime_error("DSpark TP>1 requires a NCCL-enabled build");
#endif
    }

    // Stage 0: main_proj + main_norm
    struct Stage0 {
        // main_proj: [4096, 12288] FP8 linear (transposed storage)
        // weight: F8_E4M3, scale: F8_E8M0
        uint8_t* main_proj_weight = nullptr;  // [out_dim, in_dim]
        uint8_t* main_proj_scale = nullptr;   // [out_dim/128, in_dim/128]
        uint16_t* main_norm_weight = nullptr; // [dim] BF16

        // Embedding (reuse main model's embed.weight)
        uint16_t* embed_weight = nullptr;     // [vocab, dim] BF16
    } stage0;

    // Routed-expert sharding, derived from config + tp_world_size. Each rank
    // owns a contiguous slice [experts_start, experts_start + experts_per_rank);
    // the grouped MoE kernel indexes experts by their local id within it.
    int experts_per_rank = 0;
    int experts_start = 0;

    // Per-rank attention dimensions, derived from config + tp_world_size.
    struct AttnDims {
        int dim = 0;
        int q_a_dim = 0;      // q_lora_rank
        int heads = 0;        // local heads = 64 / tp
        int head_dim = 0;
        int q_dim = 0;        // heads * head_dim
        int kv_dim = 0;       // 1 KV head * head_dim
        int groups = 0;       // local o groups = 8 / tp
        int group_rank = 0;   // o_lora_rank
        int group_dim = 0;    // q_dim / groups
        int attn_mid = 0;     // groups * group_rank
        int rope_dim = 0;
        int window_size = 0;
    } adims;

    // One DSpark draft stage. Same tensor layout as a main-model layer minus
    // the compressor/indexer, so the checkpoint dtypes match 1:1:
    //   attn.w*: F8_E4M3 weight + F8_E8M0 scale, norms BF16, hc_* F32.
    struct StageBlock {
        // hc_pre parameters (F32)
        float* hc_attn_fn = nullptr;      // [3*hc, hc*dim]
        float* hc_attn_scale = nullptr;   // [3]
        float* hc_attn_base = nullptr;    // [3*hc]
        float* hc_ffn_fn = nullptr;
        float* hc_ffn_scale = nullptr;
        float* hc_ffn_base = nullptr;

        // Attention (FP8 weight + FP8 scale, BF16 norms)
        struct Attention {
            uint8_t* wq_a_weight = nullptr;   // [q_a_dim, dim]
            uint8_t* wq_a_scale = nullptr;
            uint8_t* wq_b_weight = nullptr;   // [q_dim, q_a_dim] (TP row-sliced)
            uint8_t* wq_b_scale = nullptr;
            uint16_t* q_norm_weight = nullptr;  // [q_a_dim] BF16
            uint8_t* wkv_weight = nullptr;    // [kv_dim, dim]
            uint8_t* wkv_scale = nullptr;
            uint16_t* kv_norm_weight = nullptr; // [kv_dim] BF16
            uint8_t* wo_a_weight = nullptr;   // [attn_mid, dim] (TP row-sliced)
            uint8_t* wo_a_scale = nullptr;
            uint8_t* wo_b_weight = nullptr;   // [dim, attn_mid] (TP col-sliced)
            uint8_t* wo_b_scale = nullptr;
            float* attn_sink = nullptr;       // [heads] (TP sliced)

            // KV ring cache: [window_size, head_dim] float
            float* kv_cache = nullptr;
        } attn;

        uint16_t* attn_norm_weight = nullptr;  // [dim] BF16

        // FFN (MoE) — identical layout to a main-model layer's ffn.
        struct FFN {
            // Gate
            uint16_t* gate_weight = nullptr;  // [n_experts, dim] BF16
            float* gate_bias = nullptr;       // [n_experts] F32

            // Shared experts: FP8 weight + FP8 scale
            uint8_t* shared_w1_weight = nullptr;  // [moe_inter, dim]
            uint8_t* shared_w1_scale = nullptr;
            uint8_t* shared_w3_weight = nullptr;  // [moe_inter, dim]
            uint8_t* shared_w3_scale = nullptr;
            uint8_t* shared_w2_weight = nullptr;  // [dim, moe_inter]
            uint8_t* shared_w2_scale = nullptr;

            // Routed experts. Unlike the main model these stay resident on the
            // device: a rank owns n_experts/tp_world of them, and at 13.4 MB per
            // expert that is 10.3 GB at TP=1 but only 2.6 GB at TP=4. Staging
            // them per draft instead would put a PCIe transfer on the critical
            // path of every draft round, which is the one thing speculative
            // decoding cannot afford -- the draft has to be cheap relative to
            // the verify it is trying to skip.
            //
            // Laid out as one contiguous arena per matrix so the grouped MoE
            // kernel can index expert `local` at a fixed stride, exactly like
            // the main model's DeviceFp4ActiveArena.
            uint8_t* routed_w1q = nullptr;
            uint8_t* routed_w1s = nullptr;
            uint8_t* routed_w2q = nullptr;
            uint8_t* routed_w2s = nullptr;
            uint8_t* routed_w3q = nullptr;
            uint8_t* routed_w3s = nullptr;
            size_t routed_w1q_stride = 0;
            size_t routed_w1s_stride = 0;
            size_t routed_w2q_stride = 0;
            size_t routed_w2s_stride = 0;
            size_t routed_w3q_stride = 0;
            size_t routed_w3s_stride = 0;
        } ffn;

        uint16_t* ffn_norm_weight = nullptr;  // [dim] BF16
    };

    // 3 stages: mtp.0 / mtp.1 / mtp.2. mtp.0 also owns main_proj/main_norm,
    // mtp.2 also owns norm/hc_head/markov_head/confidence_head.
    static constexpr int kNumStages = 3;
    StageBlock stages[kNumStages];

    // Stage 2 special heads
    struct Stage2Heads {
        uint16_t* norm_weight = nullptr;  // [dim] BF16

        // hc_head parameters (F32)
        float* hc_head_fn = nullptr;      // [hc, hc*dim]
        float* hc_head_scale = nullptr;   // [1]
        float* hc_head_base = nullptr;    // [hc]

        // Markov head. Both tensors are stored [vocab, markov_rank] BF16:
        // w1 is the bigram embedding table, w2 is the output projection whose
        // rows are vocab entries (so it is used transposed).
        uint16_t* markov_w1_weight = nullptr;  // [vocab, rank] BF16
        uint16_t* markov_w2_weight = nullptr;  // [vocab, rank] BF16

        // Confidence head: [1, dim + markov_rank] BF16, no bias in checkpoint.
        uint16_t* confidence_proj_weight = nullptr;
    } stage2_heads;

    // Shared embedding and head (tied to main model, not owned here).
    uint16_t* embed_weight = nullptr;
    uint16_t* head_weight = nullptr;

    // Device buffers for forward pass
    struct DeviceBuffers {
        // Stage 0 buffers
        int* d_draft_input_ids = nullptr;    // [block_size] token IDs
        float* d_main_concat = nullptr;      // [1, 1, dim*3]
        float* d_main_proj_out = nullptr;    // [1, 1, dim]
        float* d_main_normed = nullptr;      // [1, 1, dim]
        float* d_draft_input = nullptr;      // [1, block_size, dim]
        float* d_draft_x = nullptr;          // [1, block_size, hc_mult, dim]

        // Stage 1-2 buffers
        float* d_hidden = nullptr;           // [1, block_size, hc_mult, dim]
        float* d_logits = nullptr;           // [block_size, vocab_size]

        // Stage 2 head buffers
        float* d_head_x = nullptr;           // [block_size, dim] hc_head output
        float* d_head_normed = nullptr;      // [block_size, dim]
        // Confidence input, [block_size, dim + markov_rank]: the hc_head output
        // followed by that position's markov embedding. Laid out as one buffer
        // so the markov lookup can write straight into its half instead of
        // being concatenated afterwards.
        float* d_conf_in = nullptr;
        float* d_markov_bias = nullptr;      // [vocab_size] per-position bigram bias
        float* d_confidence = nullptr;       // [block_size]
        // [block_size + 1]: the committed token, then each drafted token. Lives
        // on device so the argmax of position i can feed position i+1's markov
        // lookup without a round trip to the host.
        int* d_out_token_ids = nullptr;
        float* d_argmax_logit = nullptr;     // [1], argmax's discarded value output

        // Attention path buffers
        float* d_attn_x = nullptr;           // [block_size, dim]
        float* d_attn_post = nullptr;        // [block_size, hc_mult]
        float* d_attn_comb = nullptr;        // [block_size, hc_mult, hc_mult]
        float* d_attn_normed = nullptr;      // [block_size, dim]
        float* d_attn_out = nullptr;         // [block_size, dim]

        // DSparkAttention internals
        float* d_q_a = nullptr;              // [block_size, q_a_dim]
        float* d_q_normed = nullptr;         // [block_size, q_a_dim]
        float* d_q = nullptr;                // [block_size, q_dim]
        float* d_kv_a = nullptr;             // [block_size, kv_dim]
        float* d_draft_kv = nullptr;         // [block_size, head_dim]
        float* d_main_kv_a = nullptr;        // [1, kv_dim]
        float* d_main_kv = nullptr;          // [1, head_dim]
        // Concatenated keys the draft attends to: the whole ring window
        // followed by this block's own draft keys.
        float* d_kv_concat = nullptr;        // [window_size + block_size, head_dim]
        int32_t* d_topk_indices = nullptr;   // [block_size, window_size + block_size]
        float* d_attn_value = nullptr;       // [block_size, q_dim]
        float* d_attn_mid = nullptr;         // [block_size, attn_mid]

        // FFN path buffers
        float* d_ffn_x = nullptr;            // [block_size, dim]
        float* d_ffn_post = nullptr;         // [block_size, hc_mult]
        float* d_ffn_comb = nullptr;         // [block_size, hc_mult, hc_mult]
        float* d_ffn_normed = nullptr;       // [block_size, dim]
        float* d_ffn_out = nullptr;          // [block_size, dim]

        // MoE routing + grouped-expert scratch. Sized for block_size tokens, so
        // every allocation here is small and can live for the engine's lifetime.
        int64_t* d_route_indices = nullptr;  // [block_size, topk]
        float* d_route_weights = nullptr;    // [block_size, topk]
        int64_t* d_group_route_tokens = nullptr;  // [block_size * topk]
        float* d_group_route_weights = nullptr;   // [block_size * topk]
        int32_t* d_seg_starts = nullptr;     // [experts_per_rank + 1]
        int32_t* d_counts = nullptr;         // [experts_per_rank]
        int32_t* d_offsets = nullptr;        // [experts_per_rank]
        int32_t* d_total_routes = nullptr;   // [1]
        int32_t* d_token_slot_routes = nullptr;  // [block_size, topk]
        // Shared-expert intermediates.
        float* d_shared_gate = nullptr;      // [block_size, moe_inter]
        float* d_shared_up = nullptr;        // [block_size, moe_inter]
        float* d_shared_hidden = nullptr;    // [block_size, moe_inter]
        float* d_shared_out = nullptr;       // [block_size, dim]
        // Grouped-MoE workspace, kept across calls: a device malloc/free pair
        // per call synchronizes the device and costs far more than the kernel.
        pocket::MoePrefillFp4GroupedWorkspace moe_ws;

        void allocate(const Config& cfg, const AttnDims& ad, int experts_per_rank) {
            const int dim = cfg.dim;
            const int bsz = cfg.block_size;
            const int hc = cfg.hc_mult;
            const int vocab = cfg.vocab_size;
            const int n_target = static_cast<int>(cfg.target_layer_ids.size());
            const int kv_len = ad.window_size + bsz;

            device_malloc_into(d_draft_input_ids, bsz * sizeof(int));
            device_malloc_into(d_main_concat, static_cast<size_t>(dim) * n_target * sizeof(float));
            device_malloc_into(d_main_proj_out, 1 * 1 * dim * sizeof(float));
            device_malloc_into(d_main_normed, 1 * 1 * dim * sizeof(float));
            device_malloc_into(d_draft_input, 1 * bsz * dim * sizeof(float));
            device_malloc_into(d_draft_x, 1 * bsz * hc * dim * sizeof(float));
            device_malloc_into(d_hidden, 1 * bsz * hc * dim * sizeof(float));
            device_malloc_into(d_logits, static_cast<size_t>(bsz) * vocab * sizeof(float));

            // Stage 2 head buffers
            const int mrank = cfg.markov_rank;
            device_malloc_into(d_head_x, static_cast<size_t>(bsz) * dim * sizeof(float));
            device_malloc_into(d_head_normed, static_cast<size_t>(bsz) * dim * sizeof(float));
            device_malloc_into(d_conf_in, static_cast<size_t>(bsz) * (dim + mrank) * sizeof(float));
            device_malloc_into(d_markov_bias, static_cast<size_t>(vocab) * sizeof(float));
            device_malloc_into(d_confidence, static_cast<size_t>(bsz) * sizeof(float));
            device_malloc_into(d_out_token_ids, (static_cast<size_t>(bsz) + 1) * sizeof(int));
            device_malloc_into(d_argmax_logit, sizeof(float));

            // Attention path buffers
            device_malloc_into(d_attn_x, bsz * dim * sizeof(float));
            device_malloc_into(d_attn_post, bsz * hc * sizeof(float));
            device_malloc_into(d_attn_comb, bsz * hc * hc * sizeof(float));
            device_malloc_into(d_attn_normed, bsz * dim * sizeof(float));
            device_malloc_into(d_attn_out, bsz * dim * sizeof(float));

            // DSparkAttention internals
            device_malloc_into(d_q_a, static_cast<size_t>(bsz) * ad.q_a_dim * sizeof(float));
            device_malloc_into(d_q_normed, static_cast<size_t>(bsz) * ad.q_a_dim * sizeof(float));
            device_malloc_into(d_q, static_cast<size_t>(bsz) * ad.q_dim * sizeof(float));
            device_malloc_into(d_kv_a, static_cast<size_t>(bsz) * ad.kv_dim * sizeof(float));
            device_malloc_into(d_draft_kv, static_cast<size_t>(bsz) * ad.head_dim * sizeof(float));
            device_malloc_into(d_main_kv_a, static_cast<size_t>(ad.kv_dim) * sizeof(float));
            device_malloc_into(d_main_kv, static_cast<size_t>(ad.head_dim) * sizeof(float));
            device_malloc_into(d_kv_concat, static_cast<size_t>(kv_len) * ad.head_dim * sizeof(float));
            device_malloc_into(d_topk_indices, static_cast<size_t>(bsz) * kv_len * sizeof(int32_t));
            device_malloc_into(d_attn_value, static_cast<size_t>(bsz) * ad.q_dim * sizeof(float));
            device_malloc_into(d_attn_mid, static_cast<size_t>(bsz) * ad.attn_mid * sizeof(float));

            // FFN path buffers
            device_malloc_into(d_ffn_x, bsz * dim * sizeof(float));
            device_malloc_into(d_ffn_post, bsz * hc * sizeof(float));
            device_malloc_into(d_ffn_comb, bsz * hc * hc * sizeof(float));
            device_malloc_into(d_ffn_normed, bsz * dim * sizeof(float));
            device_malloc_into(d_ffn_out, bsz * dim * sizeof(float));

            // MoE routing scratch
            const int topk = cfg.topk;
            const int moe_inter = cfg.moe_inter;
            const size_t routes_cap = static_cast<size_t>(bsz) * topk;
            device_malloc_into(d_route_indices, routes_cap * sizeof(int64_t));
            device_malloc_into(d_route_weights, routes_cap * sizeof(float));
            device_malloc_into(d_group_route_tokens, routes_cap * sizeof(int64_t));
            device_malloc_into(d_group_route_weights, routes_cap * sizeof(float));
            device_malloc_into(d_seg_starts, (static_cast<size_t>(experts_per_rank) + 1) * sizeof(int32_t));
            device_malloc_into(d_counts, static_cast<size_t>(experts_per_rank) * sizeof(int32_t));
            device_malloc_into(d_offsets, static_cast<size_t>(experts_per_rank) * sizeof(int32_t));
            device_malloc_into(d_total_routes, sizeof(int32_t));
            device_malloc_into(d_token_slot_routes, routes_cap * sizeof(int32_t));
            device_malloc_into(d_shared_gate, static_cast<size_t>(bsz) * moe_inter * sizeof(float));
            device_malloc_into(d_shared_up, static_cast<size_t>(bsz) * moe_inter * sizeof(float));
            device_malloc_into(d_shared_hidden, static_cast<size_t>(bsz) * moe_inter * sizeof(float));
            device_malloc_into(d_shared_out, bsz * dim * sizeof(float));

            // Grouped-MoE workspace. Every route belongs to one of block_size
            // tokens, so routes <= block_size * topk and the padded path needs
            // at most experts_per_rank * block_size rows.
            moe_ws.dim = dim;
            moe_ws.inter_dim = moe_inter;
            moe_ws.routes_cap = static_cast<int>(routes_cap);
            moe_ws.padded_rows_cap = experts_per_rank * bsz;
            const size_t rd = routes_cap * dim;
            const size_t pd = static_cast<size_t>(moe_ws.padded_rows_cap) * dim;
            const size_t pi = static_cast<size_t>(moe_ws.padded_rows_cap) * moe_inter;
            device_malloc_into(moe_ws.d_x_sorted, rd * sizeof(float));
            device_malloc_into(moe_ws.d_partials, rd * sizeof(float));
            device_malloc_into(moe_ws.d_x_q, rd);
            device_malloc_into(moe_ws.d_x_scale, routes_cap * sizeof(float));
            device_malloc_into(moe_ws.d_x_pad, pd);
            device_malloc_into(moe_ws.d_x_scale_pad, static_cast<size_t>(moe_ws.padded_rows_cap) * sizeof(float));
            device_malloc_into(moe_ws.d_gate, pi * sizeof(float));
            device_malloc_into(moe_ws.d_up, pi * sizeof(float));
            device_malloc_into(moe_ws.d_hidden_q, pi);
            device_malloc_into(moe_ws.d_hidden_scale, static_cast<size_t>(moe_ws.padded_rows_cap) * sizeof(float));
        }

        void free_all() {
            if (d_draft_input_ids) device_free(d_draft_input_ids);
            if (d_main_concat) device_free(d_main_concat);
            if (d_main_proj_out) device_free(d_main_proj_out);
            if (d_main_normed) device_free(d_main_normed);
            if (d_draft_input) device_free(d_draft_input);
            if (d_draft_x) device_free(d_draft_x);
            if (d_hidden) device_free(d_hidden);
            if (d_logits) device_free(d_logits);
            if (d_head_x) device_free(d_head_x);
            if (d_head_normed) device_free(d_head_normed);
            if (d_conf_in) device_free(d_conf_in);
            if (d_markov_bias) device_free(d_markov_bias);
            if (d_confidence) device_free(d_confidence);
            if (d_out_token_ids) device_free(d_out_token_ids);
            if (d_argmax_logit) device_free(d_argmax_logit);

            if (d_attn_x) device_free(d_attn_x);
            if (d_attn_post) device_free(d_attn_post);
            if (d_attn_comb) device_free(d_attn_comb);
            if (d_attn_normed) device_free(d_attn_normed);
            if (d_attn_out) device_free(d_attn_out);

            if (d_q_a) device_free(d_q_a);
            if (d_q_normed) device_free(d_q_normed);
            if (d_q) device_free(d_q);
            if (d_kv_a) device_free(d_kv_a);
            if (d_draft_kv) device_free(d_draft_kv);
            if (d_main_kv_a) device_free(d_main_kv_a);
            if (d_main_kv) device_free(d_main_kv);
            if (d_kv_concat) device_free(d_kv_concat);
            if (d_topk_indices) device_free(d_topk_indices);
            if (d_attn_value) device_free(d_attn_value);
            if (d_attn_mid) device_free(d_attn_mid);

            if (d_ffn_x) device_free(d_ffn_x);
            if (d_ffn_post) device_free(d_ffn_post);
            if (d_ffn_comb) device_free(d_ffn_comb);
            if (d_ffn_normed) device_free(d_ffn_normed);
            if (d_ffn_out) device_free(d_ffn_out);

            for (void* p : {(void*)d_route_indices, (void*)d_route_weights,
                            (void*)d_group_route_tokens, (void*)d_group_route_weights,
                            (void*)d_seg_starts, (void*)d_counts, (void*)d_offsets,
                            (void*)d_total_routes, (void*)d_token_slot_routes,
                            (void*)d_shared_gate, (void*)d_shared_up,
                            (void*)d_shared_hidden, (void*)d_shared_out,
                            (void*)moe_ws.d_x_sorted, (void*)moe_ws.d_partials,
                            (void*)moe_ws.d_x_q, (void*)moe_ws.d_x_scale,
                            (void*)moe_ws.d_x_pad, (void*)moe_ws.d_x_scale_pad,
                            (void*)moe_ws.d_gate, (void*)moe_ws.d_up,
                            (void*)moe_ws.d_hidden_q, (void*)moe_ws.d_hidden_scale}) {
                if (p != nullptr) device_free(p);
            }
        }
    } buffers;

    Impl(const char* ckpt_dir, int rank, int world_size)
        : tp_rank(rank), tp_world_size(world_size), checkpoint_dir(ckpt_dir) {
        config = Config::from_json((std::string(ckpt_dir) + "/config.json").c_str());
        init_dims();
        buffers.allocate(config, adims, experts_per_rank);
    }

    Impl(const char* ckpt_dir, int rank, int world_size, int dev, const char* id_path)
        : Impl(ckpt_dir, rank, world_size) {
        tp_device = dev;
        if (id_path != nullptr) nccl_id_path = id_path;
    }

    ~Impl() {
        buffers.free_all();
        free_weights();
        if (d_reduce_bf16 != nullptr) device_free(d_reduce_bf16);
    }

    void init_dims() {
        if (config.n_heads % tp_world_size != 0) {
            throw std::runtime_error("DSpark: num_attention_heads not divisible by tp_world_size");
        }
        if (config.o_groups % tp_world_size != 0) {
            throw std::runtime_error("DSpark: o_groups not divisible by tp_world_size");
        }
        if (config.n_experts % tp_world_size != 0) {
            throw std::runtime_error("DSpark: n_routed_experts not divisible by tp_world_size");
        }
        experts_per_rank = config.n_experts / tp_world_size;
        experts_start = tp_rank * experts_per_rank;
        adims.dim = config.dim;
        adims.q_a_dim = config.q_lora_rank;
        adims.heads = config.n_heads / tp_world_size;
        adims.head_dim = config.head_dim;
        adims.q_dim = adims.heads * adims.head_dim;
        adims.kv_dim = adims.head_dim;  // single KV head
        adims.groups = config.o_groups / tp_world_size;
        adims.group_rank = config.o_lora_rank;
        adims.group_dim = adims.q_dim / adims.groups;
        adims.attn_mid = adims.groups * adims.group_rank;
        adims.rope_dim = config.rope_dim;
        adims.window_size = config.window_size;
    }

    // ------------------------------------------------------------------
    // Weight loading
    // ------------------------------------------------------------------

    // Every device allocation made by load_weights, so the destructor can free
    // them without each field needing its own device_free call.
    std::vector<void*> owned_device_buffers;

    void* device_alloc(size_t bytes, const char* what) {
        void* p = nullptr;
        check_device(device_malloc_into(p, bytes), what);
        owned_device_buffers.push_back(p);
        return p;
    }

    void free_weights() {
        for (void* p : owned_device_buffers) {
            if (p != nullptr) device_free(p);
        }
        owned_device_buffers.clear();
    }

    void* upload(const void* host, size_t bytes, const char* what) {
        void* d = device_alloc(bytes, what);
        check_device(memcpy_h2d(d, host, bytes), what);
        return d;
    }

    void load_weights() {
        using namespace pocket;

        SafeTensorsIndex index(checkpoint_dir);

        // Determine the real stage count from the index rather than trusting a
        // config default: a checkpoint with a different number of `mtp.N.`
        // prefixes would otherwise be silently half-loaded.
        {
            int found = 0;
            for (int s = 0; s < kNumStages; ++s) {
                if (index.shard_for_tensor("mtp." + std::to_string(s) + ".attn_norm.weight") != nullptr) {
                    found = s + 1;
                } else {
                    break;
                }
            }
            if (found == 0) {
                throw std::runtime_error("DSpark: no mtp.* stages found in checkpoint " +
                                         checkpoint_dir);
            }
            if (index.shard_for_tensor("mtp." + std::to_string(kNumStages) + ".attn_norm.weight") != nullptr) {
                throw std::runtime_error("DSpark: checkpoint has more than " +
                                         std::to_string(kNumStages) + " mtp stages");
            }
            config.n_stages = found;
        }

        // Shards are mmap'd, so keeping every one we touch open for the whole
        // load is cheap and avoids re-opening per tensor.
        std::map<std::string, std::unique_ptr<SafeTensorsShard>> open_shards;

        auto shard_for = [&](const std::string& name) -> SafeTensorsShard& {
            const std::string* s = index.shard_for_tensor(name);
            if (s == nullptr) {
                throw std::runtime_error("DSpark: tensor not found in index: " + name);
            }
            auto it = open_shards.find(*s);
            if (it == open_shards.end()) {
                it = open_shards.emplace(*s, std::make_unique<SafeTensorsShard>(
                                                 index.shard_path(*s)))
                         .first;
            }
            return *it->second;
        };

        // Fetch a tensor and assert its dtype and shape, so a checkpoint whose
        // layout differs fails here rather than as silent numerical garbage.
        struct Loaded {
            const uint8_t* data;
            const SafeTensorInfo* info;
        };
        auto get = [&](const std::string& name, SafeDType want,
                       std::vector<uint64_t> want_shape) -> Loaded {
            SafeTensorsShard& sh = shard_for(name);
            const SafeTensorInfo* info = sh.find_tensor(name);
            if (info == nullptr) {
                throw std::runtime_error("DSpark: tensor missing from shard: " + name);
            }
            if (info->dtype != want) {
                throw std::runtime_error("DSpark: dtype mismatch for " + name + ": expected " +
                                         safe_dtype_name(want) + ", got " +
                                         safe_dtype_name(info->dtype));
            }
            if (!want_shape.empty() && info->shape != want_shape) {
                std::string got;
                for (auto d : info->shape) got += std::to_string(d) + ",";
                std::string exp;
                for (auto d : want_shape) exp += std::to_string(d) + ",";
                throw std::runtime_error("DSpark: shape mismatch for " + name +
                                         ": expected [" + exp + "], got [" + got + "]");
            }
            return Loaded{reinterpret_cast<const uint8_t*>(sh.tensor_data(*info)), info};
        };

        auto upload_whole = [&](const std::string& name, SafeDType want,
                                std::vector<uint64_t> want_shape) -> void* {
            Loaded t = get(name, want, std::move(want_shape));
            return upload(t.data, t.info->nbytes, name.c_str());
        };

        const int dim = config.dim;
        const int hc = config.hc_mult;
        const uint64_t udim = static_cast<uint64_t>(dim);

        for (int stage = 0; stage < config.n_stages; ++stage) {
            const std::string p = "mtp." + std::to_string(stage) + ".";
            StageBlock& b = stages[stage];

            // hc_pre parameters. fn is [2*hc + hc*hc, hc*dim]: the pre gates,
            // the post gates, and the hc x hc combine matrix stacked in that
            // order (24 rows for hc=4), matching hc_pre_float_rows_kernel.
            const uint64_t hc_fn_rows = static_cast<uint64_t>(2 * hc + hc * hc);
            const uint64_t hc_fn_cols = static_cast<uint64_t>(hc) * udim;
            b.hc_attn_fn = static_cast<float*>(
                upload_whole(p + "hc_attn_fn", SafeDType::F32, {hc_fn_rows, hc_fn_cols}));
            b.hc_attn_scale = static_cast<float*>(
                upload_whole(p + "hc_attn_scale", SafeDType::F32, {3}));
            b.hc_attn_base = static_cast<float*>(
                upload_whole(p + "hc_attn_base", SafeDType::F32, {hc_fn_rows}));
            b.hc_ffn_fn = static_cast<float*>(
                upload_whole(p + "hc_ffn_fn", SafeDType::F32, {hc_fn_rows, hc_fn_cols}));
            b.hc_ffn_scale = static_cast<float*>(
                upload_whole(p + "hc_ffn_scale", SafeDType::F32, {3}));
            b.hc_ffn_base = static_cast<float*>(
                upload_whole(p + "hc_ffn_base", SafeDType::F32, {hc_fn_rows}));

            // Norms
            b.attn_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "attn_norm.weight", SafeDType::BF16, {udim}));
            b.ffn_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "ffn_norm.weight", SafeDType::BF16, {udim}));

            // --- Attention ---
            // wq_a / wkv are replicated across ranks (input-dim sharding is not
            // used here); wq_b / wo_a / wo_b / attn_sink are TP-sharded exactly
            // like the main model's layer weights.
            const uint64_t q_a = static_cast<uint64_t>(config.q_lora_rank);
            b.attn.wq_a_weight = static_cast<uint8_t*>(
                upload_whole(p + "attn.wq_a.weight", SafeDType::F8_E4M3, {q_a, udim}));
            b.attn.wq_a_scale = static_cast<uint8_t*>(
                upload_whole(p + "attn.wq_a.scale", SafeDType::F8_E8M0, {q_a / 128, udim / 128}));
            b.attn.q_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "attn.q_norm.weight", SafeDType::BF16, {q_a}));

            const uint64_t kv_dim_u = static_cast<uint64_t>(adims.kv_dim);
            b.attn.wkv_weight = static_cast<uint8_t*>(
                upload_whole(p + "attn.wkv.weight", SafeDType::F8_E4M3, {kv_dim_u, udim}));
            b.attn.wkv_scale = static_cast<uint8_t*>(
                upload_whole(p + "attn.wkv.scale", SafeDType::F8_E8M0,
                             {kv_dim_u / 128, udim / 128}));
            b.attn.kv_norm_weight = static_cast<uint16_t*>(
                upload_whole(p + "attn.kv_norm.weight", SafeDType::BF16, {kv_dim_u}));

            // wq_b: [n_heads*head_dim, q_a] -> local rows [q_dim, q_a]
            {
                const int row_start = tp_rank * adims.q_dim;
                Loaded w = get(p + "attn.wq_b.weight", SafeDType::F8_E4M3,
                               {static_cast<uint64_t>(config.n_heads) * config.head_dim, q_a});
                Loaded s = get(p + "attn.wq_b.scale", SafeDType::F8_E8M0,
                               {static_cast<uint64_t>(config.n_heads) * config.head_dim / 128,
                                q_a / 128});
                auto wl = slice_rows_u8(w.data, row_start, adims.q_dim, adims.q_a_dim);
                auto sl = slice_rows_u8(s.data, row_start / 128, adims.q_dim / 128,
                                        adims.q_a_dim / 128);
                b.attn.wq_b_weight = static_cast<uint8_t*>(upload(wl.data(), wl.size(), "wq_b"));
                b.attn.wq_b_scale = static_cast<uint8_t*>(upload(sl.data(), sl.size(), "wq_b scale"));
            }

            // wo_a: [o_groups*o_lora_rank, dim] -> local rows [attn_mid, dim]
            // wo_b: [dim, o_groups*o_lora_rank] -> local cols [dim, attn_mid]
            {
                const uint64_t o_full = static_cast<uint64_t>(config.o_groups) * config.o_lora_rank;
                const int row_start = tp_rank * adims.attn_mid;
                Loaded wa = get(p + "attn.wo_a.weight", SafeDType::F8_E4M3, {o_full, udim});
                Loaded sa = get(p + "attn.wo_a.scale", SafeDType::F8_E8M0,
                                {o_full / 128, udim / 128});
                auto wal = slice_rows_u8(wa.data, row_start, adims.attn_mid, dim);
                auto sal = slice_rows_u8(sa.data, row_start / 128, adims.attn_mid / 128, dim / 128);
                b.attn.wo_a_weight = static_cast<uint8_t*>(upload(wal.data(), wal.size(), "wo_a"));
                b.attn.wo_a_scale = static_cast<uint8_t*>(upload(sal.data(), sal.size(), "wo_a scale"));

                Loaded wb = get(p + "attn.wo_b.weight", SafeDType::F8_E4M3, {udim, o_full});
                Loaded sb = get(p + "attn.wo_b.scale", SafeDType::F8_E8M0,
                                {udim / 128, o_full / 128});
                auto wbl = slice_cols_u8(wb.data, dim, static_cast<int>(o_full), row_start,
                                         adims.attn_mid);
                auto sbl = slice_cols_u8(sb.data, dim / 128, static_cast<int>(o_full / 128),
                                         row_start / 128, adims.attn_mid / 128);
                b.attn.wo_b_weight = static_cast<uint8_t*>(upload(wbl.data(), wbl.size(), "wo_b"));
                b.attn.wo_b_scale = static_cast<uint8_t*>(upload(sbl.data(), sbl.size(), "wo_b scale"));
            }

            // attn_sink: [n_heads] -> local [heads]
            {
                Loaded t = get(p + "attn.attn_sink", SafeDType::F32,
                               {static_cast<uint64_t>(config.n_heads)});
                const float* src = reinterpret_cast<const float*>(t.data) + tp_rank * adims.heads;
                b.attn.attn_sink = static_cast<float*>(
                    upload(src, static_cast<size_t>(adims.heads) * sizeof(float), "attn_sink"));
            }

            // KV ring cache, zeroed: [window_size, head_dim]
            {
                const size_t bytes = static_cast<size_t>(adims.window_size) * adims.head_dim *
                                     sizeof(float);
                b.attn.kv_cache = static_cast<float*>(device_alloc(bytes, "dspark kv_cache"));
                check_device(device_memset(b.attn.kv_cache, 0, bytes), "zero dspark kv_cache");
            }

            // --- FFN ---
            const uint64_t n_exp = static_cast<uint64_t>(config.n_experts);
            const uint64_t inter = static_cast<uint64_t>(config.moe_inter);
            b.ffn.gate_weight = static_cast<uint16_t*>(
                upload_whole(p + "ffn.gate.weight", SafeDType::BF16, {n_exp, udim}));
            b.ffn.gate_bias = static_cast<float*>(
                upload_whole(p + "ffn.gate.bias", SafeDType::F32, {n_exp}));

            b.ffn.shared_w1_weight = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w1.weight", SafeDType::F8_E4M3, {inter, udim}));
            b.ffn.shared_w1_scale = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w1.scale", SafeDType::F8_E8M0,
                             {inter / 128, udim / 128}));
            b.ffn.shared_w3_weight = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w3.weight", SafeDType::F8_E4M3, {inter, udim}));
            b.ffn.shared_w3_scale = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w3.scale", SafeDType::F8_E8M0,
                             {inter / 128, udim / 128}));
            b.ffn.shared_w2_weight = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w2.weight", SafeDType::F8_E4M3, {udim, inter}));
            b.ffn.shared_w2_scale = static_cast<uint8_t*>(
                upload_whole(p + "ffn.shared_experts.w2.scale", SafeDType::F8_E8M0,
                             {udim / 128, inter / 128}));

            // Routed experts: this rank's slice, uploaded into one arena per
            // matrix. Experts are FP4 (int8-packed pairs) with e8m0 scales, the
            // same format the main model's grouped kernel consumes.
            {
                const std::string ep = p + "ffn.experts.";
                auto expert_name = [&](int e, const char* w, const char* suffix) {
                    return ep + std::to_string(e) + "." + w + "." + suffix;
                };

                // Size the arena from expert 0; every expert shares a shape, and
                // `get` below re-checks each one, so a ragged checkpoint fails
                // loudly rather than writing past a stride.
                Loaded w1q0 = get(expert_name(experts_start, "w1", "weight"), SafeDType::I8, {inter, udim / 2});
                Loaded w1s0 = get(expert_name(experts_start, "w1", "scale"), SafeDType::F8_E8M0, {inter, udim / 32});
                Loaded w2q0 = get(expert_name(experts_start, "w2", "weight"), SafeDType::I8, {udim, inter / 2});
                Loaded w2s0 = get(expert_name(experts_start, "w2", "scale"), SafeDType::F8_E8M0, {udim, inter / 32});
                Loaded w3q0 = get(expert_name(experts_start, "w3", "weight"), SafeDType::I8, {inter, udim / 2});
                Loaded w3s0 = get(expert_name(experts_start, "w3", "scale"), SafeDType::F8_E8M0, {inter, udim / 32});

                b.ffn.routed_w1q_stride = w1q0.info->nbytes;
                b.ffn.routed_w1s_stride = w1s0.info->nbytes;
                b.ffn.routed_w2q_stride = w2q0.info->nbytes;
                b.ffn.routed_w2s_stride = w2s0.info->nbytes;
                b.ffn.routed_w3q_stride = w3q0.info->nbytes;
                b.ffn.routed_w3s_stride = w3s0.info->nbytes;

                const size_t n = static_cast<size_t>(experts_per_rank);
                b.ffn.routed_w1q = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w1q_stride, "dspark routed w1q"));
                b.ffn.routed_w1s = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w1s_stride, "dspark routed w1s"));
                b.ffn.routed_w2q = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w2q_stride, "dspark routed w2q"));
                b.ffn.routed_w2s = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w2s_stride, "dspark routed w2s"));
                b.ffn.routed_w3q = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w3q_stride, "dspark routed w3q"));
                b.ffn.routed_w3s = static_cast<uint8_t*>(device_alloc(n * b.ffn.routed_w3s_stride, "dspark routed w3s"));

                for (int local = 0; local < experts_per_rank; ++local) {
                    const int e = experts_start + local;
                    struct Upload {
                        const char* w;
                        const char* suffix;
                        SafeDType dtype;
                        std::vector<uint64_t> shape;
                        uint8_t* dst;
                        size_t stride;
                    };
                    const Upload uploads[] = {
                        {"w1", "weight", SafeDType::I8,      {inter, udim / 2},  b.ffn.routed_w1q, b.ffn.routed_w1q_stride},
                        {"w1", "scale",  SafeDType::F8_E8M0, {inter, udim / 32}, b.ffn.routed_w1s, b.ffn.routed_w1s_stride},
                        {"w2", "weight", SafeDType::I8,      {udim, inter / 2},  b.ffn.routed_w2q, b.ffn.routed_w2q_stride},
                        {"w2", "scale",  SafeDType::F8_E8M0, {udim, inter / 32}, b.ffn.routed_w2s, b.ffn.routed_w2s_stride},
                        {"w3", "weight", SafeDType::I8,      {inter, udim / 2},  b.ffn.routed_w3q, b.ffn.routed_w3q_stride},
                        {"w3", "scale",  SafeDType::F8_E8M0, {inter, udim / 32}, b.ffn.routed_w3s, b.ffn.routed_w3s_stride},
                    };
                    for (const Upload& u : uploads) {
                        Loaded t = get(expert_name(e, u.w, u.suffix), u.dtype, u.shape);
                        if (t.info->nbytes != u.stride) {
                            throw std::runtime_error("DSpark: routed expert size mismatch for " +
                                                     expert_name(e, u.w, u.suffix));
                        }
                        check_device(memcpy_h2d(u.dst + static_cast<size_t>(local) * u.stride,
                                                t.data, u.stride),
                                     "upload dspark routed expert");
                    }
                }
            }

            // Stage 0 extras: main_proj + main_norm
            if (stage == 0) {
                const uint64_t n_target = static_cast<uint64_t>(config.target_layer_ids.size());
                stage0.main_proj_weight = static_cast<uint8_t*>(
                    upload_whole(p + "main_proj.weight", SafeDType::F8_E4M3,
                                 {udim, udim * n_target}));
                stage0.main_proj_scale = static_cast<uint8_t*>(
                    upload_whole(p + "main_proj.scale", SafeDType::F8_E8M0,
                                 {udim / 128, udim * n_target / 128}));
                stage0.main_norm_weight = static_cast<uint16_t*>(
                    upload_whole(p + "main_norm.weight", SafeDType::BF16, {udim}));
            }

            // Last stage extras: norm + hc_head + markov + confidence
            if (stage == config.n_stages - 1) {
                stage2_heads.norm_weight = static_cast<uint16_t*>(
                    upload_whole(p + "norm.weight", SafeDType::BF16, {udim}));
                stage2_heads.hc_head_fn = static_cast<float*>(
                    upload_whole(p + "hc_head_fn", SafeDType::F32,
                                 {static_cast<uint64_t>(hc), static_cast<uint64_t>(hc) * udim}));
                stage2_heads.hc_head_scale = static_cast<float*>(
                    upload_whole(p + "hc_head_scale", SafeDType::F32, {1}));
                stage2_heads.hc_head_base = static_cast<float*>(
                    upload_whole(p + "hc_head_base", SafeDType::F32,
                                 {static_cast<uint64_t>(hc)}));

                const uint64_t vocab = static_cast<uint64_t>(config.vocab_size);
                const uint64_t rank_u = static_cast<uint64_t>(config.markov_rank);
                // Both markov tensors are [vocab, rank]: w1 is the bigram
                // embedding table, w2 the output projection used transposed.
                // Vocab-major layout means TP would shard rows; keep them whole
                // for now and gather at the head, as the reference does.
                stage2_heads.markov_w1_weight = static_cast<uint16_t*>(
                    upload_whole(p + "markov_head.markov_w1.weight", SafeDType::BF16,
                                 {vocab, rank_u}));
                stage2_heads.markov_w2_weight = static_cast<uint16_t*>(
                    upload_whole(p + "markov_head.markov_w2.weight", SafeDType::BF16,
                                 {vocab, rank_u}));
                stage2_heads.confidence_proj_weight = static_cast<uint16_t*>(
                    upload_whole(p + "confidence_head.proj.weight", SafeDType::BF16,
                                 {1, udim + rank_u}));
            }
        }

        // Embedding is tied to the main model's table.
        embed_weight = static_cast<uint16_t*>(
            upload_whole("embed.weight", SafeDType::BF16,
                         {static_cast<uint64_t>(config.vocab_size), udim}));
        stage0.embed_weight = embed_weight;

        // Output head, also tied to the main model. The reference shards this
        // by vocab and all-gathers the logits; here every rank keeps the whole
        // table (1 GB bf16) and computes identical full-vocab logits. That
        // trades memory for having no collective in the draft's inner loop,
        // which matters because the loop runs block_size times per round --
        // and it keeps the draft's token ids identical across ranks by
        // construction rather than by agreement.
        head_weight = static_cast<uint16_t*>(
            upload_whole("head.weight", SafeDType::BF16,
                         {static_cast<uint64_t>(config.vocab_size), udim}));

        weights_loaded = true;
    }

    bool weights_loaded = false;

    // Stage 0: main_proj + main_norm + embed
    void forward_stage0(int input_token, const std::vector<float*>& main_hiddens,
                        const std::vector<int>& draft_input_ids) {
        const int dim = config.dim;
        const int bsz = config.block_size;

        if (static_cast<int>(draft_input_ids.size()) != bsz) {
            throw std::runtime_error("draft_input_ids size must match block_size");
        }

        // 1. Concat main_hidden_states from layers 40/41/42
        // main_hiddens[0/1/2] are [1, 1, dim] each
        // TODO: implement concat kernel or use a device copy
        for (int i = 0; i < 3; ++i) {
            memcpy_d2d(buffers.d_main_concat + i * dim, main_hiddens[i], dim * sizeof(float));
        }

        // 2. FP8 linear: [1, dim*3] @ [dim, dim*3]^T -> [1, dim]
        // Note: weight is stored as [out_dim=4096, in_dim=12288]
        // x: [1, 12288], weight: [4096, 12288], output: [1, 4096]
        if (stage0.main_proj_weight == nullptr || stage0.main_proj_scale == nullptr) {
            throw std::runtime_error("Stage 0 main_proj weights not loaded");
        }
        bool success = pocket::fp8_e4m3_e8m0_matmul_cuda(
            buffers.d_main_concat,        // input: [1, 12288]
            stage0.main_proj_weight,      // weight: [4096, 12288] F8_E4M3
            stage0.main_proj_scale,       // scale: [32, 96] F8_E8M0
            buffers.d_main_proj_out,      // output: [1, 4096]
            1,                            // batch
            4096,                         // rows (output dim)
            12288,                        // cols (input dim)
            nullptr                       // stream
        );
        if (!success) {
            throw std::runtime_error("FP8 matmul failed in main_proj");
        }

        // 3. RMSNorm: [1, dim] -> [1, dim]
        // Apply RMSNorm to main_proj output
        if (stage0.main_norm_weight == nullptr) {
            throw std::runtime_error("Stage 0 main_norm weights not loaded");
        }
        success = pocket::rmsnorm_bf16_gamma_cuda(
            buffers.d_main_proj_out,     // input: [1, 4096]
            stage0.main_norm_weight,     // gamma: [4096] BF16
            buffers.d_main_normed,       // output: [1, 4096]
            dim,                         // cols = 4096
            1e-6f,                       // eps
            nullptr                      // stream
        );
        if (!success) {
            throw std::runtime_error("RMSNorm failed in main_norm");
        }

        // 4. Embed draft_input_ids: [block_size] -> [block_size, dim]
        // Upload draft_input_ids to device
        memcpy_h2d(buffers.d_draft_input_ids, draft_input_ids.data(), bsz * sizeof(int));

        // Lookup embeddings using bf16_rows_to_float_cuda
        if (stage0.embed_weight == nullptr) {
            throw std::runtime_error("Stage 0 embed_weight not loaded");
        }
        success = pocket::bf16_rows_to_float_cuda(
            stage0.embed_weight,         // matrix: [vocab, dim] BF16
            buffers.d_draft_input_ids,   // row indices: [block_size]
            buffers.d_draft_input,       // output: [block_size, dim]
            bsz,                         // rows
            dim,                         // cols
            nullptr                      // stream
        );
        if (!success) {
            throw std::runtime_error("Embedding lookup failed");
        }

        // 5. hc_mult expansion: [block_size, dim] -> [block_size, hc_mult, dim]
        // Repeat each embedding hc_mult=4 times along dim=1
        const int hc = config.hc_mult;
        success = pocket::hc_repeat_rows_cuda(
            buffers.d_draft_input,       // input: [block_size, dim]
            buffers.d_draft_x,           // output: [block_size, hc_mult, dim]
            bsz,                         // rows
            dim,                         // dim
            nullptr                      // stream
        );
        if (!success) {
            throw std::runtime_error("hc_repeat failed");
        }
    }

    // Keys the draft attends to: the live ring window, then this block's own
    // draft keys. Every draft position sees the same set (the reference builds
    // one row and expands it), so causality across draft positions is
    // deliberately not enforced. `start_pos` is the committed token's position.
    // Returns the number of key slots per row.
    int build_topk_indices(int start_pos) {
        const int bsz = config.block_size;
        const int win = adims.window_size;
        // The ring holds positions 0..start_pos, capped at the window.
        const int committed = std::min(win, start_pos + 1);
        const int topk = committed + bsz;

        std::vector<int32_t> row(topk);
        for (int i = 0; i < committed; ++i) row[i] = i;
        for (int i = 0; i < bsz; ++i) row[committed + i] = win + i;

        // Same row for every draft position.
        std::vector<int32_t> all(static_cast<size_t>(bsz) * topk);
        for (int t = 0; t < bsz; ++t) {
            std::memcpy(all.data() + static_cast<size_t>(t) * topk, row.data(),
                        row.size() * sizeof(int32_t));
        }
        check_device(memcpy_h2d(buffers.d_topk_indices, all.data(), all.size() * sizeof(int32_t)),
                     "copy dspark topk indices");
        return topk;
    }

    // DSparkAttention: queries come from the draft tokens, keys/values from the
    // main model's hidden state. Mirrors DSparkAttention.forward in
    // src/models/deepseek_v4/dspark.py.
    //
    //   d_x       [block_size, dim]  attn_norm output for the draft tokens
    //   d_main_x  [dim]              main model's projected+normed hidden
    //   start_pos                    position of the committed token; the draft
    //                                tokens occupy start_pos+1 .. start_pos+bsz
    //   d_out     [block_size, dim]  attention output
    void forward_attention(StageBlock* block, const float* d_x, const float* d_main_x,
                           int start_pos, float* d_out) {
        using namespace pocket;
        const int bsz = config.block_size;
        const int dim = config.dim;
        const int win = adims.window_size;
        const int hd = adims.head_dim;
        const int rd = adims.rope_dim;
        const int draft_pos = start_pos + 1;  // position of draft token 0
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        if (start_pos <= 0) {
            // The reference asserts start_pos > 0 when building topk indices:
            // with an empty window there is nothing to draft from.
            throw std::runtime_error("DSpark attention requires start_pos > 0");
        }

        // --- Q from the draft tokens ---
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->attn.wq_a_weight, block->attn.wq_a_scale,
                                       buffers.d_q_a, bsz, adims.q_a_dim, dim))
            throw std::runtime_error("dspark wq_a failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_q_a, block->attn.q_norm_weight,
                                          buffers.d_q_normed, bsz, adims.q_a_dim,
                                          config.norm_eps))
            throw std::runtime_error("dspark q_norm failed");
        if (!fp8_e4m3_e8m0_matmul_cuda(buffers.d_q_normed, block->attn.wq_b_weight,
                                       block->attn.wq_b_scale, buffers.d_q, bsz,
                                       adims.q_dim, adims.q_a_dim))
            throw std::runtime_error("dspark wq_b failed");
        // Per-head RMSNorm (no gamma) then rope; draft token i sits at draft_pos+i.
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_q, bsz, adims.heads, hd, rd,
                                         draft_pos, config.rope_theta, false,
                                         config.norm_eps))
            throw std::runtime_error("dspark q rope failed");

        // --- KV for the committed position, from the main model's hidden ---
        if (!fp8_e4m3_e8m0_matmul_cuda(d_main_x, block->attn.wkv_weight, block->attn.wkv_scale,
                                       buffers.d_main_kv_a, 1, adims.kv_dim, dim))
            throw std::runtime_error("dspark main wkv failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_main_kv_a, block->attn.kv_norm_weight,
                                          buffers.d_main_kv, 1, adims.kv_dim, config.norm_eps))
            throw std::runtime_error("dspark main kv_norm failed");
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_main_kv, 1, 1, hd, rd,
                                         start_pos, config.rope_theta, false, 0.0f))
            throw std::runtime_error("dspark main kv rope failed");
        // act_quant covers only the non-rope prefix of each row.
        if (!fp8_act_quant_dequant_rows_strided_cuda(buffers.d_main_kv, 1, hd - rd, hd, 64))
            throw std::runtime_error("dspark main kv act_quant failed");
        // Ring write, before the concat below reads the window.
        if (!copy_rows_to_kv_cache_cuda(buffers.d_main_kv, block->attn.kv_cache, 1, hd,
                                        win, start_pos))
            throw std::runtime_error("dspark ring cache write failed");

        // --- KV from the draft tokens themselves (never cached) ---
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->attn.wkv_weight, block->attn.wkv_scale,
                                       buffers.d_kv_a, bsz, adims.kv_dim, dim))
            throw std::runtime_error("dspark draft wkv failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_kv_a, block->attn.kv_norm_weight,
                                          buffers.d_draft_kv, bsz, adims.kv_dim,
                                          config.norm_eps))
            throw std::runtime_error("dspark draft kv_norm failed");
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_draft_kv, bsz, 1, hd, rd,
                                         draft_pos, config.rope_theta, false, 0.0f))
            throw std::runtime_error("dspark draft kv rope failed");
        if (!fp8_act_quant_dequant_rows_strided_cuda(buffers.d_draft_kv, bsz, hd - rd, hd, 64))
            throw std::runtime_error("dspark draft kv act_quant failed");

        // --- Sparse attention over [ring window ++ draft keys] ---
        check_device(memcpy_d2d(buffers.d_kv_concat, block->attn.kv_cache,
                                static_cast<size_t>(win) * hd * sizeof(float)),
                     "dspark kv concat window");
        check_device(memcpy_d2d(buffers.d_kv_concat + static_cast<size_t>(win) * hd,
                                buffers.d_draft_kv, static_cast<size_t>(bsz) * hd * sizeof(float)),
                     "dspark kv concat draft");
        const int topk = build_topk_indices(start_pos);
        if (!prefill_sparse_attention_indexed_cuda(
                buffers.d_q, buffers.d_kv_concat, block->attn.attn_sink,
                buffers.d_topk_indices, buffers.d_attn_value, bsz, adims.heads,
                win + bsz, topk, hd, scale))
            throw std::runtime_error("dspark sparse attention failed");
        // Inverse rope on the attention output, at the draft positions.
        if (!head_rmsnorm_rope_rows_cuda(buffers.d_attn_value, bsz, adims.heads, hd, rd,
                                         draft_pos, config.rope_theta, true, 0.0f))
            throw std::runtime_error("dspark inverse rope failed");

        // --- Output projection: grouped wo_a, then wo_b ---
        for (int g = 0; g < adims.groups; ++g) {
            const float* group_x = buffers.d_attn_value + static_cast<size_t>(g) * adims.group_dim;
            const uint8_t* group_w = block->attn.wo_a_weight +
                static_cast<size_t>(g) * adims.group_rank * adims.group_dim;
            const uint8_t* group_s = block->attn.wo_a_scale +
                static_cast<size_t>(g) * (adims.group_rank / 128) * (adims.group_dim / 128);
            float* group_y = buffers.d_attn_mid + static_cast<size_t>(g) * adims.group_rank;
            // Each token's slice of d_attn_value / d_attn_mid is strided.
            if (!fp8_e4m3_e8m0_matmul_strided_cuda(group_x, group_w, group_s, group_y, bsz,
                                                   adims.group_rank, adims.group_dim,
                                                   adims.q_dim, adims.attn_mid))
                throw std::runtime_error("dspark wo_a failed");
        }
        if (!fp8_e4m3_e8m0_matmul_cuda(buffers.d_attn_mid, block->attn.wo_b_weight,
                                       block->attn.wo_b_scale, d_out, bsz, dim,
                                       adims.attn_mid))
            throw std::runtime_error("dspark wo_b failed");
        // Each rank ran its own heads, so d_out is a partial sum until this
        // reduce -- same place the main model reduces its attention output.
        tp_all_reduce(d_out, bsz * dim);
    }

    // Routed MoE + shared expert for `rows` tokens. d_x is the ffn_norm output
    // [rows, dim]; d_out receives shared + sum(route_weight * expert(x)).
    //
    // Mirrors the main model's prefill MoE: gate -> group routes by expert ->
    // one grouped FP4 kernel over all routes -> add the shared expert. The
    // routed weights are already resident (see StageBlock::FFN), so unlike the
    // main model there is no staging step on the critical path.
    void forward_moe(StageBlock* block, const float* d_x, float* d_out, int rows) {
        using namespace pocket;
        const int dim = config.dim;
        const int inter = config.moe_inter;
        const int topk = config.topk;

        // Gate: scores = sqrt_softplus(W x), select topk by score + bias, then
        // renormalize the *unbiased* scores over the selection.
        if (!gate_topk_bf16_rows_cuda(d_x, block->ffn.gate_weight, block->ffn.gate_bias,
                                      buffers.d_route_indices, buffers.d_route_weights, rows,
                                      config.n_experts, dim, topk, config.route_scale)) {
            throw std::runtime_error("dspark gate topk failed");
        }

        // Bucket the routes this rank owns by expert. Routes whose expert lives
        // on another rank are dropped here and contributed by that rank's
        // all-reduce (which TP>1 still needs wiring for -- see forward_attention).
        if (!moe_group_routes_cuda(buffers.d_route_indices, buffers.d_route_weights,
                                   buffers.d_group_route_tokens, buffers.d_group_route_weights,
                                   buffers.d_seg_starts, buffers.d_counts, buffers.d_offsets,
                                   buffers.d_total_routes, rows, topk, experts_start,
                                   experts_per_rank, buffers.d_token_slot_routes)) {
            throw std::runtime_error("dspark group routes failed");
        }

        int32_t total_routes = 0;
        check_device(memcpy_d2h(&total_routes, buffers.d_total_routes, sizeof(int32_t)), "copy dspark total routes");

        if (total_routes > 0) {
            std::vector<int32_t> h_counts(static_cast<size_t>(experts_per_rank));
            check_device(memcpy_d2h(h_counts.data(), buffers.d_counts,
                                    h_counts.size() * sizeof(int32_t)),
                         "copy dspark route counts");
            int max_count = 0;
            for (int32_t c : h_counts) max_count = std::max(max_count, static_cast<int>(c));

            // tile_count = 0 selects the padded path. With at most block_size
            // tokens the compact tiling has nothing to save, and the padded
            // path needs no host-built tile table.
            buffers.moe_ws.tile_count = 0;
            if (!moe_prefill_fp4_grouped_cuda_with_workspace(
                    d_x, buffers.d_group_route_tokens, buffers.d_group_route_weights,
                    buffers.d_seg_starts, block->ffn.routed_w1q, block->ffn.routed_w1s,
                    block->ffn.routed_w2q, block->ffn.routed_w2s, block->ffn.routed_w3q,
                    block->ffn.routed_w3s, d_out, rows, topk, total_routes,
                    experts_per_rank, max_count, dim, inter, config.swiglu_limit,
                    buffers.moe_ws, buffers.d_token_slot_routes)) {
                throw std::runtime_error("dspark grouped fp4 moe failed");
            }
        } else {
            check_device(device_memset(d_out, 0, static_cast<size_t>(rows) * dim * sizeof(float)),
                         "zero dspark moe out");
        }

        // Each rank only summed the routes landing on its own experts, so the
        // reduce goes here -- before the shared expert, which every rank
        // computes in full and would otherwise be counted tp_world_size times.
        tp_all_reduce(d_out, rows * dim);

        // Shared expert: SwiGLU FFN applied to every token, added on top.
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->ffn.shared_w1_weight,
                                       block->ffn.shared_w1_scale, buffers.d_shared_gate, rows,
                                       inter, dim))
            throw std::runtime_error("dspark shared w1 failed");
        if (!fp8_e4m3_e8m0_matmul_cuda(d_x, block->ffn.shared_w3_weight,
                                       block->ffn.shared_w3_scale, buffers.d_shared_up, rows,
                                       inter, dim))
            throw std::runtime_error("dspark shared w3 failed");
        if (!silu_mul_rows_cuda(buffers.d_shared_gate, buffers.d_shared_up,
                                buffers.d_shared_hidden, rows, inter))
            throw std::runtime_error("dspark shared silu failed");
        if (!fp8_e4m3_e8m0_matmul_cuda(buffers.d_shared_hidden, block->ffn.shared_w2_weight,
                                       block->ffn.shared_w2_scale, buffers.d_shared_out, rows,
                                       dim, inter))
            throw std::runtime_error("dspark shared w2 failed");
        if (!vector_accum_rows_cuda(buffers.d_shared_out, d_out, rows, dim, 1.0f))
            throw std::runtime_error("dspark shared accum failed");
    }

    // Stage 2 output heads: hc_head -> norm -> vocab logits, then one draft
    // token per position, each biased by the token before it.
    //
    //   d_x  [block_size, hc_mult, dim]  the last stage's block output
    //
    // Fills buffers.d_logits [block_size, vocab], buffers.d_out_token_ids
    // [block_size + 1] (input token, then the drafts), and
    // buffers.d_confidence [block_size].
    //
    // The loop is sequential by construction: position i's markov bias is a
    // lookup on the token argmaxed at position i-1, so the bias cannot be
    // precomputed. Everything else -- the hc_head, the norm, and the full
    // [block_size, vocab] logits -- is computed once for the whole block before
    // the loop starts, which is the only part that touches the 1 GB head.
    void forward_head(int input_token, const float* d_x) {
        using namespace pocket;
        const int bsz = config.block_size;
        const int dim = config.dim;
        const int vocab = config.vocab_size;
        const int mrank = config.markov_rank;
        const int conf_dim = dim + mrank;

        if (head_weight == nullptr || stage2_heads.norm_weight == nullptr) {
            throw std::runtime_error("dspark head weights not loaded");
        }

        if (!hc_head_float_rows_cuda(d_x, stage2_heads.hc_head_fn, stage2_heads.hc_head_scale,
                                     stage2_heads.hc_head_base, buffers.d_head_x, bsz, dim))
            throw std::runtime_error("dspark hc_head failed");
        if (!rmsnorm_bf16_gamma_rows_cuda(buffers.d_head_x, stage2_heads.norm_weight,
                                          buffers.d_head_normed, bsz, dim, config.norm_eps))
            throw std::runtime_error("dspark head norm failed");
        if (!bf16_matvec_rows_cuda(buffers.d_head_normed, head_weight, buffers.d_logits,
                                   bsz, vocab, dim))
            throw std::runtime_error("dspark head logits failed");

        // The confidence head reads concat(hc_head output, markov embedding).
        // Copy the first half in now; the loop fills each row's second half as
        // that position's markov embedding is looked up.
        check_device(memcpy_2d_d2d(buffers.d_conf_in, conf_dim * sizeof(float),
                                   buffers.d_head_x, dim * sizeof(float),
                                   dim * sizeof(float), bsz),
                     "dspark confidence hidden copy");

        check_device(memcpy_h2d(buffers.d_out_token_ids, &input_token, sizeof(int)),
                     "dspark seed output token");

        for (int i = 0; i < bsz; ++i) {
            // Bigram embedding of the token at position i, written straight
            // into this row's slot in the confidence input.
            float* d_embed = buffers.d_conf_in + static_cast<size_t>(i) * conf_dim + dim;
            if (!bf16_rows_to_float_cuda(stage2_heads.markov_w1_weight,
                                         buffers.d_out_token_ids + i, d_embed, 1, mrank))
                throw std::runtime_error("dspark markov embed failed");
            // markov_w2 is stored [vocab, rank], so this is the projection to
            // vocab without a transpose.
            if (!bf16_matvec_cuda(d_embed, stage2_heads.markov_w2_weight,
                                  buffers.d_markov_bias, vocab, mrank))
                throw std::runtime_error("dspark markov bias failed");

            float* d_row = buffers.d_logits + static_cast<size_t>(i) * vocab;
            if (!vector_accum_cuda(buffers.d_markov_bias, d_row, vocab, 1.0f))
                throw std::runtime_error("dspark markov bias add failed");
            // Greedy only. The reference also has a temperature path, but the
            // verify loop compares against the target's greedy choice, so
            // sampling here would only lower the accept rate.
            if (!argmax_fp32_cuda(d_row, buffers.d_out_token_ids + i + 1,
                                  buffers.d_argmax_logit, vocab, 0))
                throw std::runtime_error("dspark draft argmax failed");
        }

        if (!bf16_matvec_rows_cuda(buffers.d_conf_in, stage2_heads.confidence_proj_weight,
                                   buffers.d_confidence, bsz, 1, conf_dim))
            throw std::runtime_error("dspark confidence head failed");
    }

    void forward_block(float* x, const float* d_main_x, int start_pos,
                       const std::vector<int>& draft_input_ids, int block_id) {
        const int bsz = config.block_size;
        const int hc = config.hc_mult;
        const int dim = config.dim;
        const int rows = bsz;  // Process all block_size tokens together

        // Get block-specific weights (one StageBlock per mtp.N stage)
        if (block_id < 0 || block_id >= config.n_stages || block_id >= kNumStages) {
            throw std::runtime_error("DSpark: block_id out of range");
        }
        StageBlock* block = &stages[block_id];

        // 1. Attention path: hc_pre + attn_norm + attention + hc_post

        // 1.1 hc_pre for attention
        // Input: x [rows, hc, dim]
        // Output: x_attn [rows, dim], post_attn [rows, hc], comb_attn [rows, hc, hc]
        bool success = pocket::hc_pre_float_rows_cuda(
            x,                          // d_h4_rows: [rows, hc, dim]
            block->hc_attn_fn,          // d_fn: [hc, hc*dim]
            block->hc_attn_scale,       // d_scale: [hc]
            block->hc_attn_base,        // d_base: [hc]
            buffers.d_attn_x,           // d_x_rows: [rows, dim]
            buffers.d_attn_post,        // d_post_rows: [rows, hc]
            buffers.d_attn_comb,        // d_comb_rows: [rows, hc, hc]
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_pre attention failed");
        }

        // 1.2 attn_norm: RMSNorm on [rows, dim]
        success = pocket::rmsnorm_bf16_gamma_rows_cuda(
            buffers.d_attn_x,           // input: [rows, dim]
            block->attn_norm_weight,    // gamma: [dim] BF16
            buffers.d_attn_normed,      // output: [rows, dim]
            rows,
            dim,
            config.norm_eps,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("attn_norm failed");
        }

        // 1.3 DSparkAttention forward. main_x carries the main model's hidden;
        // start_pos is the committed token's position.
        forward_attention(block, buffers.d_attn_normed, d_main_x, start_pos,
                          buffers.d_attn_out);

        // 1.4 hc_post: merge attention output back
        // Output: x [rows, hc, dim]
        success = pocket::hc_post_float_rows_cuda(
            buffers.d_attn_out,         // d_x_rows: [rows, dim]
            x,                          // d_residual_h4_rows: [rows, hc, dim]
            buffers.d_attn_post,        // d_post_rows: [rows, hc]
            buffers.d_attn_comb,        // d_comb_rows: [rows, hc, hc]
            x,                          // d_y_h4_rows: [rows, hc, dim] (in-place)
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_post attention failed");
        }

        // 2. FFN path: hc_pre + ffn_norm + ffn + hc_post

        // 2.1 hc_pre for FFN
        success = pocket::hc_pre_float_rows_cuda(
            x,                          // d_h4_rows: [rows, hc, dim]
            block->hc_ffn_fn,           // d_fn: [hc, hc*dim]
            block->hc_ffn_scale,        // d_scale: [hc]
            block->hc_ffn_base,         // d_base: [hc]
            buffers.d_ffn_x,            // d_x_rows: [rows, dim]
            buffers.d_ffn_post,         // d_post_rows: [rows, hc]
            buffers.d_ffn_comb,         // d_comb_rows: [rows, hc, hc]
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_pre FFN failed");
        }

        // 2.2 ffn_norm: RMSNorm on [rows, dim]
        success = pocket::rmsnorm_bf16_gamma_rows_cuda(
            buffers.d_ffn_x,            // input: [rows, dim]
            block->ffn_norm_weight,     // gamma: [dim] BF16
            buffers.d_ffn_normed,       // output: [rows, dim]
            rows,
            dim,
            config.norm_eps,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("ffn_norm failed");
        }

        // 2.3 FFN forward: shared expert + routed MoE, same structure as a
        // main-model layer. The reference inherits Block's MoE unchanged, so
        // this must match it: sqrt-softplus gate scores, topk by score+bias,
        // weights renormalized from the unbiased scores, times route_scale.
        forward_moe(block, buffers.d_ffn_normed, buffers.d_ffn_out, rows);

        // 2.4 hc_post: merge FFN output back
        success = pocket::hc_post_float_rows_cuda(
            buffers.d_ffn_out,          // d_x_rows: [rows, dim]
            x,                          // d_residual_h4_rows: [rows, hc, dim]
            buffers.d_ffn_post,         // d_post_rows: [rows, hc]
            buffers.d_ffn_comb,         // d_comb_rows: [rows, hc, hc]
            x,                          // d_y_h4_rows: [rows, hc, dim] (in-place)
            rows,
            dim,
            nullptr                     // stream
        );
        if (!success) {
            throw std::runtime_error("hc_post FFN failed");
        }
    }

    // Prime every stage's KV ring from the main model's hiddens for a run of
    // committed positions. See the header for why this is separate from
    // draft(): the ring has to hold every committed position, not just the one
    // being drafted from, and a missing one shows up only as a draft that
    // matches nothing rather than as an error.
    void write_main_kv(const float* h_main_hidden, int rows, int start_pos) {
        using namespace pocket;
        if (h_main_hidden == nullptr || rows <= 0) return;
        if (start_pos < 0) throw std::runtime_error("write_main_kv: negative start_pos");

        const int dim = config.dim;
        const int win = adims.window_size;
        const int hd = adims.head_dim;
        const int rd = adims.rope_dim;
        const int n_target = static_cast<int>(config.target_layer_ids.size());
        const size_t stride = static_cast<size_t>(n_target) * dim;

        // Only the last `win` positions can survive the ring, so drop the rest
        // up front rather than writing slots that will be overwritten.
        if (rows > win) {
            h_main_hidden += static_cast<size_t>(rows - win) * stride;
            start_pos += rows - win;
            rows = win;
        }

        // Scratch sized to this call. The per-round buffers are single-row, and
        // priming happens once per committed block rather than per draft
        // position, so widening them permanently would cost more than it saves.
        float* d_concat = nullptr;
        float* d_proj = nullptr;
        float* d_normed = nullptr;
        float* d_kv_a = nullptr;
        float* d_kv = nullptr;
        check_device(device_malloc_into(d_concat, static_cast<size_t>(rows) * stride * sizeof(float)),
                     "device_malloc write_main_kv concat");
        auto cleanup = [&] {
            if (d_concat) device_free(d_concat);
            if (d_proj) device_free(d_proj);
            if (d_normed) device_free(d_normed);
            if (d_kv_a) device_free(d_kv_a);
            if (d_kv) device_free(d_kv);
        };
        try {
            check_device(device_malloc_into(d_proj, static_cast<size_t>(rows) * dim * sizeof(float)),
                         "device_malloc write_main_kv proj");
            check_device(device_malloc_into(d_normed, static_cast<size_t>(rows) * dim * sizeof(float)),
                         "device_malloc write_main_kv normed");
            check_device(device_malloc_into(d_kv_a, static_cast<size_t>(rows) * adims.kv_dim * sizeof(float)),
                         "device_malloc write_main_kv kv_a");
            check_device(device_malloc_into(d_kv, static_cast<size_t>(rows) * hd * sizeof(float)),
                         "device_malloc write_main_kv kv");
            check_device(memcpy_h2d(d_concat, h_main_hidden,
                                    static_cast<size_t>(rows) * stride * sizeof(float)),
                         "copy write_main_kv hidden");

            // main_proj + main_norm, shared by every stage exactly as in draft().
            if (!fp8_e4m3_e8m0_matmul_cuda(d_concat, stage0.main_proj_weight,
                                           stage0.main_proj_scale, d_proj, rows, dim,
                                           static_cast<int>(stride)))
                throw std::runtime_error("write_main_kv main_proj failed");
            if (!rmsnorm_bf16_gamma_rows_cuda(d_proj, stage0.main_norm_weight, d_normed, rows,
                                              dim, config.norm_eps))
                throw std::runtime_error("write_main_kv main_norm failed");

            for (int s = 0; s < config.n_stages; ++s) {
                StageBlock* block = &stages[s];
                if (!fp8_e4m3_e8m0_matmul_cuda(d_normed, block->attn.wkv_weight,
                                               block->attn.wkv_scale, d_kv_a, rows,
                                               adims.kv_dim, dim))
                    throw std::runtime_error("write_main_kv wkv failed");
                if (!rmsnorm_bf16_gamma_rows_cuda(d_kv_a, block->attn.kv_norm_weight, d_kv, rows,
                                                  adims.kv_dim, config.norm_eps))
                    throw std::runtime_error("write_main_kv kv_norm failed");
                // rope with no norm, then act_quant over the non-rope prefix --
                // the same treatment draft() gives the single committed position.
                if (!head_rmsnorm_rope_rows_cuda(d_kv, rows, 1, hd, rd, start_pos,
                                                 config.rope_theta, false, 0.0f))
                    throw std::runtime_error("write_main_kv rope failed");
                if (!fp8_act_quant_dequant_rows_strided_cuda(d_kv, rows, hd - rd, hd, 64))
                    throw std::runtime_error("write_main_kv act_quant failed");
                if (!copy_rows_to_kv_cache_cuda(d_kv, block->attn.kv_cache, rows, hd, win,
                                                start_pos))
                    throw std::runtime_error("write_main_kv ring write failed");
            }
        } catch (...) {
            cleanup();
            throw;
        }
        cleanup();
    }

    DraftOutput forward(int input_token, int start_pos,
                       const std::vector<float*>& main_hidden_states) {
        const int bsz = config.block_size;

        // Generate draft_input_ids: [input_token, noise, noise, ...]
        // For now use noise_token_id from config (typically 128000)
        std::vector<int> draft_input_ids(bsz);
        draft_input_ids[0] = input_token;
        for (int i = 1; i < bsz; ++i) {
            draft_input_ids[i] = config.noise_token_id;
        }

        // Stage 0: main_proj + main_norm + embed
        forward_stage0(input_token, main_hidden_states, draft_input_ids);

        // Stage 1-2: DSparkBlock forward. The checkpoint carries one block per
        // `mtp.N.` prefix, so every stage runs — not just the first two.
        // main_x is computed once in stage 0 and shared by every stage, matching
        // DSpark.forward in the reference.
        float* x = buffers.d_draft_x;  // [block_size, hc_mult, dim]
        for (int block_id = 0; block_id < config.n_stages; ++block_id) {
            // forward_block modifies x in-place
            forward_block(x, buffers.d_main_normed, start_pos, draft_input_ids, block_id);
        }

        // Stage 2 heads: hc_head + norm + vocab logits, then the markov-biased
        // greedy loop that actually emits the draft.
        forward_head(input_token, x);

        DraftOutput output(config.block_size);
        check_device(memcpy_d2h(output.tokens.data(), buffers.d_out_token_ids,
                                output.tokens.size() * sizeof(int)),
                     "copy dspark draft tokens");
        check_device(memcpy_d2h(output.confidence.data(), buffers.d_confidence,
                                output.confidence.size() * sizeof(float)),
                     "copy dspark draft confidence");
        return output;
    }
};

// ============================================================================
// DSparkEngine public interface
// ============================================================================

DSparkEngine::DSparkEngine(const char* checkpoint_dir, int tp_rank, int tp_world_size) {
    impl_ = new Impl(checkpoint_dir, tp_rank, tp_world_size);
    try {
        impl_->load_weights();
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

DSparkEngine::DSparkEngine(const char* checkpoint_dir, int tp_rank, int tp_world_size,
                           int device, const char* nccl_id_path) {
    impl_ = new Impl(checkpoint_dir, tp_rank, tp_world_size, device, nccl_id_path);
    try {
        impl_->load_weights();
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

DSparkEngine::~DSparkEngine() {
    delete impl_;
}

DraftOutput DSparkEngine::draft(int input_token, int start_pos,
                                const std::vector<float*>& main_hidden_states) {
    if (main_hidden_states.size() != impl_->config.target_layer_ids.size()) {
        throw std::runtime_error(
            "main_hidden_states size mismatch: expected " +
            std::to_string(impl_->config.target_layer_ids.size()) +
            ", got " + std::to_string(main_hidden_states.size()));
    }

    return impl_->forward(input_token, start_pos, main_hidden_states);
}

const Config& DSparkEngine::config() const {
    return impl_->config;
}

void DSparkEngine::write_main_kv(const float* h_main_hidden, int rows, int start_pos) {
    impl_->write_main_kv(h_main_hidden, rows, start_pos);
}

void DSparkEngine::debug_set_kv_cache(int stage_id, const float* h_cache) {
    if (stage_id < 0 || stage_id >= impl_->config.n_stages) {
        throw std::runtime_error("debug_set_kv_cache: stage_id out of range");
    }
    const size_t bytes = static_cast<size_t>(impl_->adims.window_size) *
                         impl_->adims.head_dim * sizeof(float);
    check_device(memcpy_h2d(impl_->stages[stage_id].attn.kv_cache, h_cache, bytes),
                 "debug_set_kv_cache");
}

void DSparkEngine::debug_attention(int stage_id, const float* h_x, const float* h_main_x,
                                   int start_pos, float* h_out) {
    if (stage_id < 0 || stage_id >= impl_->config.n_stages) {
        throw std::runtime_error("debug_attention: stage_id out of range");
    }
    Impl& impl = *impl_;
    const int bsz = impl.config.block_size;
    const int dim = impl.config.dim;

    // d_attn_normed / d_attn_out are the same scratch the real path uses.
    check_device(memcpy_h2d(impl.buffers.d_attn_normed, h_x,
                            static_cast<size_t>(bsz) * dim * sizeof(float)),
                 "debug_attention x");
    check_device(memcpy_h2d(impl.buffers.d_main_normed, h_main_x,
                            static_cast<size_t>(dim) * sizeof(float)),
                 "debug_attention main_x");

    impl.forward_attention(&impl.stages[stage_id], impl.buffers.d_attn_normed,
                           impl.buffers.d_main_normed, start_pos,
                           impl.buffers.d_attn_out);
    check_device(device_synchronize(), "debug_attention sync");

    check_device(memcpy_d2h(h_out, impl.buffers.d_attn_out,
                            static_cast<size_t>(bsz) * dim * sizeof(float)),
                 "debug_attention out");
}

void DSparkEngine::debug_moe(int stage_id, const float* h_x, int rows, float* h_out) {
    if (stage_id < 0 || stage_id >= impl_->config.n_stages) {
        throw std::runtime_error("debug_moe: stage_id out of range");
    }
    Impl& impl = *impl_;
    const int dim = impl.config.dim;
    if (rows <= 0 || rows > impl.config.block_size) {
        throw std::runtime_error("debug_moe: rows must be in [1, block_size]");
    }
    const size_t n = static_cast<size_t>(rows) * dim;

    check_device(memcpy_h2d(impl.buffers.d_ffn_normed, h_x, n * sizeof(float)),
                 "debug_moe x");
    impl.forward_moe(&impl.stages[stage_id], impl.buffers.d_ffn_normed,
                     impl.buffers.d_ffn_out, rows);
    check_device(device_synchronize(), "debug_moe sync");
    check_device(memcpy_d2h(h_out, impl.buffers.d_ffn_out, n * sizeof(float)),
                 "debug_moe out");
}

void DSparkEngine::debug_head(const float* h_x, int input_token, int* h_tokens,
                              float* h_confidence, float* h_logits) {
    Impl& impl = *impl_;
    const int bsz = impl.config.block_size;
    const int dim = impl.config.dim;
    const int hc = impl.config.hc_mult;

    // d_draft_x is the same [block_size, hc_mult, dim] scratch the real path
    // hands to the last stage.
    check_device(memcpy_h2d(impl.buffers.d_draft_x, h_x,
                            static_cast<size_t>(bsz) * hc * dim * sizeof(float)),
                 "debug_head x");
    impl.forward_head(input_token, impl.buffers.d_draft_x);
    check_device(device_synchronize(), "debug_head sync");

    check_device(memcpy_d2h(h_tokens, impl.buffers.d_out_token_ids,
                            (static_cast<size_t>(bsz) + 1) * sizeof(int)),
                 "debug_head tokens");
    check_device(memcpy_d2h(h_confidence, impl.buffers.d_confidence,
                            static_cast<size_t>(bsz) * sizeof(float)),
                 "debug_head confidence");
    if (h_logits != nullptr) {
        check_device(memcpy_d2h(h_logits, impl.buffers.d_logits,
                                static_cast<size_t>(bsz) * impl.config.vocab_size * sizeof(float)),
                     "debug_head logits");
    }
}

} // namespace dspark
