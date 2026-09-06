#include "qwen_engine.hpp"

#include <chrono>
#include <unordered_map>

#include "cmd_channel.hpp"
#include "cuda_ops.hpp"
#include "device_runtime.hpp"
#include "qwen_block_pool.hpp"
#include "qwen_block_table.hpp"
#include "qwen_ops.hpp"
#include "qwen_dspark.hpp"
#include "qwen_dflash2.hpp"
#include "qwen_sampler.hpp"
#include "qwen_target_head.hpp"
#include "tp_comm.hpp"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace dsv4 {
namespace {

constexpr int kKvScaleBlock = 64;

void check_device(bool ok, const char* what) {
    if (!ok) {
        const std::string detail = device_last_error();
        throw std::runtime_error(detail.empty() ? std::string(what)
                                                : std::string(what) + ": " + detail);
    }
}

void require_launch(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(
            std::string("Qwen ") + device_backend_name() + " launch failed: " + what);
    }
}

bool qwen_env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return false;
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 &&
           std::strcmp(value, "OFF") != 0;
}

// Opt-out switch for defaults that are on unless explicitly disabled.
bool qwen_env_enabled_default(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return true;
    return qwen_env_enabled(name);
}

int qwen_env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 1'000'000) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

// The wide 128x64 tile only pays off once the batch fills its row span; below
// that the WMMA path stays cheaper. Enabled by default above the row threshold.
bool qwen_nvfp4_wide_n64_enabled(int rows) {
    return rows >= qwen_env_int("DSV4_QWEN_NVFP4_WIDE_N64_MIN_ROWS", 128) &&
           qwen_env_enabled_default("DSV4_QWEN_NVFP4_WIDE_N64");
}

struct DeviceLinear {
    QwenLinearKind kind = QwenLinearKind::DenseF16;
    std::vector<uint64_t> logical_shape;
    QwenDeviceTensor weight;
    QwenDeviceTensor scale;
    // Optional dense FP16 expansion used only by batched target verification.
    // The raw FP8 matrix remains canonical and serves decode/prefill.
    QwenDeviceTensor verify_weight;
    // NVFP4 tensor-level scaling. The factor is reciprocal(weight_global_scale);
    // input_global_scale is calibration metadata the SM75 path does not apply.
    float weight_global_factor = 1.0f;
    float input_global_scale = 1.0f;
};

struct DeviceLinearAttention {
    DeviceLinear qkv;
    DeviceLinear z;
    DeviceLinear out;
    DeviceLinear a;
    DeviceLinear b;
    // Row-concat of a and b; empty when the two cannot be fused.
    DeviceLinear ab;
    QwenDeviceTensor conv;
    QwenDeviceTensor a_log;
    QwenDeviceTensor dt_bias;
    QwenDeviceTensor norm;
    QwenDeviceTensor state;
    QwenDeviceTensor conv_tail;
};

struct DeviceFullAttention {
    DeviceLinear q;
    DeviceLinear k;
    DeviceLinear v;
    // Row-concat of K and V for batched verify/prefill. Decode keeps the
    // individual projections so its established one-row path is unchanged.
    DeviceLinear kv;
    DeviceLinear out;
    QwenDeviceTensor q_norm;
    QwenDeviceTensor k_norm;
    QwenDeviceTensor k_cache;
    QwenDeviceTensor v_cache;
    QwenDeviceTensor k_scale;
    QwenDeviceTensor v_scale;
    // TurboQuant K8V4 combined cache: 196-byte slots with FP8 E5M2 key + 4-bit
    // value + FP16 metadata. Only allocated when kv_cache_dtype is TurboQuantK8V4.
    QwenDeviceTensor turboquant_cache;
};

struct DeviceLayer {
    QwenDeviceTensor input_norm;
    QwenDeviceTensor post_norm;
    DeviceLinearAttention linear;
    DeviceFullAttention full;
    DeviceLinear gate;
    DeviceLinear up;
    DeviceLinear down;
};

QwenDeviceTensor upload(const SafeTensorsIndex& index, const QwenTensorRef& ref) {
    return qwen_upload_tensor(index, ref);
}

DeviceLinear upload_linear(const SafeTensorsIndex& index, const QwenLinearRef& ref) {
    DeviceLinear output;
    output.kind = ref.kind;
    output.logical_shape = ref.logical_local_shape;
#ifdef POCKET_BACKEND_ASCEND
    // NVFP4 repacks into the SM75 WMMA block layout, and only the CUDA kernels
    // read that layout. Reject the kind at load time so the Ascend build carries
    // no reference to the repacking uploader.
    if (ref.kind == QwenLinearKind::NvFp4Group16) {
        throw std::runtime_error(
            "Qwen NVFP4 weights are not implemented on the Ascend backend");
    }
    {
#else
    if (ref.kind == QwenLinearKind::NvFp4Group16) {
        output.weight = qwen_upload_nvfp4_linear_cuda(
            index, ref, &output.weight_global_factor,
            &output.input_global_scale);
    } else {
#endif
        output.weight = upload(index, ref.weight);
        if (ref.has_scale) output.scale = upload(index, ref.scale);
    }
    return output;
}

void allocate(QwenDeviceTensor& tensor, size_t bytes,
              const std::vector<uint64_t>& shape, SafeDType dtype);

// Concatenates two linear weights along output rows so one GEMM can produce both
// results. FP8 scales are row-concatenated in lockstep. Only the input width and
// dtypes must match; output row counts may differ (Q/K/V fusion relies on this).
DeviceLinear fuse_linear_rows(const DeviceLinear& first,
                              const DeviceLinear& second) {
    DeviceLinear fused;
    if (first.weight.data == nullptr || second.weight.data == nullptr) return fused;
    if (first.weight.shape.size() != 2 || second.weight.shape.size() != 2) return fused;
    if (first.weight.shape[1] != second.weight.shape[1] ||
        first.weight.device_dtype != second.weight.device_dtype) {
        return fused;
    }
    // The two operands must dispatch to the same kernel family, and both must
    // carry a logical shape: `projection` reads kind and logical_shape, not the
    // storage shape, so a fused linear that leaves them defaulted would either
    // pick the wrong kernel or throw. NVFP4 is never fused because its packed
    // rows and per-tensor factors do not concatenate.
    if (first.kind != second.kind ||
        (first.kind != QwenLinearKind::DenseF16 &&
         first.kind != QwenLinearKind::Fp8Block128) ||
        first.logical_shape.size() != 2 || second.logical_shape.size() != 2 ||
        first.logical_shape[1] != second.logical_shape[1]) {
        return fused;
    }
    // Block-128 FP8 scale rows track output-row blocks and therefore concatenate
    // with the weight rows. Other compressed formats have different layouts and
    // are rejected above until a fused consumer for them is validated.
    const bool has_fp8_scale = first.kind == QwenLinearKind::Fp8Block128;
    if (has_fp8_scale &&
        (first.scale.data == nullptr || second.scale.data == nullptr ||
         first.scale.shape.size() != 2 || second.scale.shape.size() != 2 ||
         first.scale.shape[1] != second.scale.shape[1] ||
         first.scale.device_dtype != second.scale.device_dtype)) {
        return fused;
    }

    fused.kind = first.kind;
    fused.logical_shape = {first.logical_shape[0] + second.logical_shape[0],
                           first.logical_shape[1]};
    const uint64_t first_rows = first.weight.shape[0];
    const uint64_t second_rows = second.weight.shape[0];
    const uint64_t columns = first.weight.shape[1];
    allocate(fused.weight, first.weight.nbytes + second.weight.nbytes,
             {first_rows + second_rows, columns}, first.weight.device_dtype);
    check_device(memcpy_d2d(fused.weight.data, first.weight.data,
                            first.weight.nbytes),
                 "Qwen fused projection first weight copy");
    check_device(memcpy_d2d(static_cast<uint8_t*>(fused.weight.data) + first.weight.nbytes,
                            second.weight.data, second.weight.nbytes),
                 "Qwen fused projection second weight copy");
    if (has_fp8_scale) {
        const uint64_t scale_columns = first.scale.shape[1];
        allocate(fused.scale, first.scale.nbytes + second.scale.nbytes,
                 {first.scale.shape[0] + second.scale.shape[0], scale_columns},
                 first.scale.device_dtype);
        check_device(memcpy_d2d(fused.scale.data, first.scale.data,
                                first.scale.nbytes),
                     "Qwen fused projection first scale copy");
        check_device(memcpy_d2d(static_cast<uint8_t*>(fused.scale.data) + first.scale.nbytes,
                                second.scale.data, second.scale.nbytes),
                     "Qwen fused projection second scale copy");
    }
    return fused;
}

void allocate(QwenDeviceTensor& tensor, size_t bytes,
              const std::vector<uint64_t>& shape, SafeDType dtype) {
    if (bytes == 0 || safe_dtype_size(dtype) == 0) {
        throw std::runtime_error("Qwen attempted to allocate an invalid tensor");
    }
    if (tensor.data != nullptr && tensor.capacity >= bytes) {
        tensor.device_dtype = dtype;
        tensor.shape = shape;
        tensor.nbytes = bytes;
        return;
    }
    if (tensor.data != nullptr) {
        device_free(tensor.data);
        tensor.data = nullptr;
        tensor.data = nullptr;
    }
    tensor.data = device_malloc(bytes);
    check_device(tensor.data != nullptr, "device_malloc Qwen runtime tensor");
    tensor.device_dtype = dtype;
    tensor.shape = shape;
    tensor.nbytes = bytes;
    tensor.capacity = bytes;
}

void allocate_elements(QwenDeviceTensor& tensor, size_t elements,
                       const std::vector<uint64_t>& shape, SafeDType dtype) {
    const uint64_t item_size = safe_dtype_size(dtype);
    if (elements > static_cast<size_t>(UINT64_MAX / item_size)) {
        throw std::runtime_error("Qwen tensor byte extent overflow");
    }
    allocate(tensor, elements * item_size, shape, dtype);
}

void allocate_float(QwenDeviceTensor& tensor, size_t elements,
                    const std::vector<uint64_t>& shape) {
    allocate_elements(tensor, elements, shape, SafeDType::F32);
}

void allocate_half(QwenDeviceTensor& tensor, size_t elements,
                   const std::vector<uint64_t>& shape) {
    allocate_elements(tensor, elements, shape, SafeDType::F16);
}

void zero_tensor(QwenDeviceTensor& tensor) {
    if (tensor.data != nullptr && tensor.nbytes != 0) {
        check_device(device_memset(tensor.data, 0, tensor.nbytes),
                       "device_memset Qwen runtime tensor");
    }
}

// Phase 3.4: Zeroes one leading-dimension row of a per-slot arena. Used for the
// recurrent state, where resetting one request must not disturb the others.
void zero_slot_region(QwenDeviceTensor& tensor, int slot_id) {
    if (tensor.data == nullptr || tensor.nbytes == 0) return;
    const uint64_t slots = tensor.shape.empty() ? 1 : tensor.shape[0];
    if (slots <= 1) {
        zero_tensor(tensor);
        return;
    }
    const size_t slot_bytes = tensor.nbytes / static_cast<size_t>(slots);
    if (slot_id < 0 || static_cast<uint64_t>(slot_id) >= slots) {
        throw std::runtime_error("Qwen recurrent slot index out of range");
    }
    check_device(device_memset(static_cast<uint8_t*>(tensor.data) +
                                   static_cast<size_t>(slot_id) * slot_bytes,
                               0, slot_bytes),
                 "device_memset Qwen recurrent slot");
}

// One rollback point for the recurrent half of the network. The 16 full
// attention layers need nothing here because their KV cache is indexed by
// absolute position and is never invalidated by a longer prompt.
struct QwenRecurrentSnapshot {
    int position = 0;
    // Device-resident contiguous copies avoid a PCIe round trip and thousands
    // of small allocations. The tensors are local to one TP rank.
    QwenDeviceTensor state;
    QwenDeviceTensor conv_tail;
    QwenDeviceTensor target_hidden;
    bool periodic = false;
    // A request-boundary snapshot can answer an exact shorter-prefix prefill
    // without re-running its final token.
    QwenForwardResult result;
    bool has_result = false;

    uint64_t bytes() const {
        return state.capacity + conv_tail.capacity + target_hidden.capacity;
    }
};

struct QwenVerifyBatch {
    std::vector<int> top_tokens;
    std::vector<float> top_logits;
    std::vector<float> local_logits;
    int position_after = 0;
};

struct QwenWorkspace {
    std::vector<QwenDeviceTensor> slots;
    size_t cursor = 0;

    QwenWorkspace() { slots.reserve(32); }

    void begin() { cursor = 0; }

    QwenDeviceTensor& tensor(size_t elements, const std::vector<uint64_t>& shape,
                             SafeDType dtype) {
        if (cursor == slots.size()) slots.emplace_back();
        QwenDeviceTensor& output = slots[cursor++];
        allocate_elements(output, elements, shape, dtype);
        return output;
    }

    QwenDeviceTensor& half_tensor(size_t elements,
                                  const std::vector<uint64_t>& shape) {
        return tensor(elements, shape, SafeDType::F16);
    }

    QwenDeviceTensor& float_tensor(size_t elements,
                                   const std::vector<uint64_t>& shape) {
        return tensor(elements, shape, SafeDType::F32);
    }

    uint64_t capacity_bytes() const {
        uint64_t total = 0;
        for (const QwenDeviceTensor& slot : slots) total += slot.capacity;
        return total;
    }
};

}  // namespace

const char* qwen_kv_cache_dtype_name(QwenKvCacheDType dtype) {
    switch (dtype) {
        case QwenKvCacheDType::Fp16: return "fp16";
        case QwenKvCacheDType::Fp8: return "fp8";
        case QwenKvCacheDType::TurboQuantK8V4: return "turboquant_k8v4";
        case QwenKvCacheDType::Int8PerTokenHead: return "int8_per_token_head";
    }
    return "unknown";
}

QwenKvCacheDType parse_qwen_kv_cache_dtype(const std::string& value) {
    if (value == "fp16") return QwenKvCacheDType::Fp16;
    if (value == "fp8") return QwenKvCacheDType::Fp8;
    if (value == "turboquant_k8v4") return QwenKvCacheDType::TurboQuantK8V4;
    if (value == "int8_per_token_head") return QwenKvCacheDType::Int8PerTokenHead;
    throw std::runtime_error("Qwen KV cache dtype must be fp16, fp8, turboquant_k8v4, or int8_per_token_head");
}

struct QwenEngine::Impl {
    SafeTensorsIndex& index;
    QwenConfig& config;
    QwenEngineOptions options;
    int max_context = 0;
    std::vector<DeviceLayer> layers;
    QwenDeviceTensor transaction_packed;
    QwenDeviceTensor transaction_regions;
    std::vector<QwenCopyRegion> host_transaction_regions;
    uint64_t transaction_total_blocks = 0;
    QwenDeviceTensor embed;
    QwenDeviceTensor final_norm;
    DeviceLinear lm_head;
    bool mtp_enabled = false;
    bool dspark_enabled = false;
    bool dflash2_enabled = false;
    // The external drafters stay declared on every backend. Their bookkeeping is
    // woven through run_chunk and the speculative steps, so compiling the members
    // out fragments code that is otherwise backend-neutral; the Ascend build
    // instead links throwing stubs for the two runtimes
    // (engine/backend_unimplemented_ascend.cpp)
    // and rejects a drafter checkpoint at construction.
    // Caps the verified draft block width. 0 verifies the full seven-draft
    // proposal, which is the upstream behaviour and stays the default.
    int dflash2_draft_width = qwen_env_int("DSV4_DFLASH2_DRAFT_WIDTH", 0);
    // A fixed cap cannot serve both high- and low-acceptance workloads, so the
    // width is also allowed to track measured acceptance. The controller keeps an
    // exponentially weighted accepted-draft count and verifies one row past it,
    // which is the cheapest width that can still grow when acceptance recovers.
    bool dflash2_adaptive_width = qwen_env_enabled("DSV4_DFLASH2_ADAPTIVE_WIDTH");
    double dflash2_accept_ewma = 0.0;
    uint64_t dflash2_width_samples = 0;
    std::unique_ptr<QwenDSparkConfig> dspark_config;
    std::unique_ptr<QwenDFlash2Config> dflash2_config;
    std::unique_ptr<SafeTensorsIndex> dflash2_index;
    std::unique_ptr<QwenDFlash2WeightMap> dflash2_weights;
    std::unique_ptr<QwenDFlash2Runtime> dflash2;
    QwenDeviceTensor dflash2_target_taps;
    std::vector<int> dflash2_debug_target_layer_ids;
    QwenDFlash2DebugCallback dflash2_target_debug_callback;
    std::unique_ptr<SafeTensorsIndex> dspark_index;
    std::unique_ptr<QwenDSparkWeightMap> dspark_weights;
    std::unique_ptr<QwenDSparkRuntime> dspark;
    QwenDeviceTensor dspark_target_taps;
    DeviceLinear mtp_fc;
    QwenDeviceTensor mtp_pre_fc_norm_embedding;
    QwenDeviceTensor mtp_pre_fc_norm_hidden;
    QwenDeviceTensor mtp_norm;
    DeviceLayer mtp_layer;
    QwenDeviceTensor target_hidden_rows;
    QwenDeviceTensor target_last_hidden;
    QwenDeviceTensor mtp_seed_hidden;
    QwenDeviceTensor mtp_embedding;
    QwenDeviceTensor mtp_normalized_embedding;
    QwenDeviceTensor mtp_normalized_hidden;
    QwenDeviceTensor mtp_concat;
    QwenDeviceTensor mtp_fused;
    QwenDeviceTensor mtp_next_hidden;
    QwenDeviceTensor mtp_normalized_output;
    int mtp_position = 0;
    int mtp_seed_input_token = 0;
    int mtp_next_token = 0;
    float mtp_next_logit = 0.0f;
    float mtp_next_checksum = 0.0f;
    bool has_target_last_hidden = false;
    bool mtp_seed_ready = false;
    std::vector<int> local_tokens;
    QwenDeviceTensor d_tokens;
    QwenDeviceTensor hidden_a;
    QwenDeviceTensor hidden_b;
    QwenDeviceTensor argmax_token;
    QwenDeviceTensor argmax_logit;
    // Batched decode row metadata. Each row of a batched decode step sits at a
    // different position in a different slot, so the kernels read these arrays
    // instead of the scalar position/slot the single-sequence path passes down.
    // Null in `batch_rows` is what tells every layer it is on the single-row
    // path, which keeps the existing launches byte-for-byte unchanged.
    struct BatchRows {
        // Device arrays, all `rows` long.
        const int* slot_ids = nullptr;
        // Position of each row's new token, i.e. its context length minus one.
        const int* positions = nullptr;
        // Context length after the new token is appended: positions[i] + 1.
        const int* context_lens = nullptr;
        // Host mirrors. The launchers need the widest context to size the split
        // geometry and the scratch, and that is a host-side decision.
        int rows = 0;
        int max_context_len = 0;
    };
    const BatchRows* batch_rows = nullptr;
    QwenDeviceTensor batch_slot_ids;
    QwenDeviceTensor batch_positions;
    QwenDeviceTensor batch_context_lens;
    // Sampling scratch. Allocated lazily on the first sampled step so greedy
    // runs pay nothing.
    QwenDeviceTensor sample_cand_token;
    QwenDeviceTensor sample_cand_logit;
    QwenDeviceTensor sample_merged_token;
    QwenDeviceTensor sample_merged_logit;
    QwenDeviceTensor sample_uniforms;
    QwenDeviceTensor sample_rng_states;
    // Host RNG. Seeding this identically on every rank and broadcasting the
    // drawn uniforms is what makes a TP group agree on the sampled token.
    std::mt19937_64 sampling_rng;
    bool sampling_rng_ready = false;
    std::vector<float> host_uniform_scratch;
    QwenWorkspace workspace;
    // Separate NCCL stream. Every collective is bracketed by events on the
    // default compute stream, so enabling this cannot expose a partially reduced
    // tensor to the next layer. It is on by default because the sliced
    // projection+all-reduce pipeline below needs it to hide any of the
    // collective; `=0` restores the single-stream behaviour.
    const bool use_nccl_comm_stream =
        qwen_env_enabled_default("QWEN_NCCL_COMM_STREAM");
    void* nccl_comm_stream = nullptr;
    void* nccl_comm_ready = nullptr;
    void* nccl_comm_done = nullptr;
    bool nccl_comm_stream_initialized = false;
    // Command channel for TP worker loop
    std::unique_ptr<CmdChannel> cmd;
    // The overlapped path has two collectives in flight across a slice boundary,
    // so it cannot share the single ready/done pair above: recording ready for
    // slice i+1 could overwrite it before slice i's waiter has resolved. These are
    // allocated per slice index and reused across layers.
    std::vector<void*> comm_slice_ready;
    std::vector<void*> comm_slice_done;
    // Reusable per-token INT8 activation buffers for the NVFP4 integer kernels.
    QwenDeviceTensor nvfp4_q8;
    QwenDeviceTensor nvfp4_q8_scale;
    QwenLinearKindCounts active_linear_kinds;
    uint64_t uploaded_weight_bytes = 0;
    uint64_t uploaded_scale_bytes = 0;
    uint64_t verify_weight_bytes = 0;
    uint64_t cache_data_bytes = 0;
    uint64_t cache_scale_bytes = 0;
    QwenRuntimeTelemetry telemetry;

    // Prefix-reuse bookkeeping for one KV slot.  Every field describes the state
    // materialized in that slot's rows, so it has to be per-slot: a single shared
    // copy lets one sequence's tokens be matched as another's prefix, and the
    // reuse then resumes from recurrent state that belongs to a different
    // sequence.  Snapshots are per-slot for the same reason, and because
    // capture/restore address one slot's rows.
    struct SlotPrefixState {
        // Prompt whose KV cache and recurrent state are currently materialized.
        std::vector<int> cached_prompt;
        // Ordered by ascending position. Index 0 is always the implicit empty
        // state, which is not stored.
        std::vector<QwenRecurrentSnapshot> snapshots;
        QwenForwardResult cached_result;
        bool has_cached_result = false;
    };
    std::vector<SlotPrefixState> slot_prefix;
    // Hit/miss counters stay engine-wide: they are reported as one aggregate
    // statistic and carry no per-sequence correctness meaning.
    int prefix_hits = 0;
    int prefix_misses = 0;

    SlotPrefixState& prefix_for(int slot_id) {
        if (slot_id < 0 || slot_id >= static_cast<int>(slot_prefix.size())) {
            throw std::runtime_error("Qwen prefix state slot " +
                                     std::to_string(slot_id) + " is out of range");
        }
        return slot_prefix[static_cast<size_t>(slot_id)];
    }

    // Position of one slot's sequence.  Outside batch mode there is only the
    // engine-wide position_, which the caller passes in as the fallback so the
    // single-session path keeps behaving exactly as before.
    int slot_position(int slot_id, int engine_position) const {
        if (!batch_mode_enabled || slot_id < 0 ||
            slot_id >= static_cast<int>(slot_positions.size())) {
            return engine_position;
        }
        return slot_positions[static_cast<size_t>(slot_id)];
    }

    // Records a slot's new position.  engine_position is updated too because the
    // non-batch code paths and telemetry still read position_.
    void set_slot_position(int slot_id, int position, int& engine_position) {
        engine_position = position;
        if (batch_mode_enabled && slot_id >= 0 &&
            slot_id < static_cast<int>(slot_positions.size())) {
            slot_positions[static_cast<size_t>(slot_id)] = position;
        }
    }

    // ========== Batch slot management (Phase 3.1) ==========
    int max_batch_size = 1;  // Default single session
    bool batch_mode_enabled = false;
    std::unordered_map<uint64_t, int> request_to_slot;
    std::vector<int> free_slots;
    std::vector<int> slot_positions;  // Phase 3.4: per-slot position tracker

    // Phase 3.3: Removed current_slot_id (now passed explicitly as parameter)

    // Calculate element offset for a given slot_id in KV cache (Phase 3.2)
    size_t kv_slot_offset_elements(int slot_id, int local_kv_heads, int head_dim) const {
        if (max_batch_size == 1) return 0;  // Fast path for single session
        return static_cast<size_t>(slot_id) * max_context * local_kv_heads * head_dim;
    }

    // ========== Paged KV cache ==========
    // Present only when options.kv_paged is set. The pool owns the block
    // bookkeeping and the table owns the per-slot logical-to-physical map; the
    // device arenas stay in the per-layer k_cache/v_cache tensors, sized to the
    // pool rather than to max_batch_size * max_context.
    std::unique_ptr<QwenBlockPool> block_pool;
    std::unique_ptr<QwenBlockTable> block_table;
    // Device mirror of the table, `[max_slots, max_blocks_per_seq]` int32.
    QwenDeviceTensor block_table_device;

    bool kv_paged() const { return block_table != nullptr; }

    // Grows `slot` to cover `tokens` logical positions, throwing when the pool
    // is exhausted. The scheduler-facing entry points convert that into
    // backpressure; inside a forward there is nothing to fall back to, since the
    // K/V for those positions has to land somewhere.
    void reserve_paged_slot(int slot_id, int tokens) {
        if (!kv_paged()) return;
        if (!block_table->ensure_capacity(slot_id, tokens)) {
            throw std::runtime_error(
                "Qwen paged KV cache exhausted: slot " +
                std::to_string(slot_id) + " needs " + std::to_string(tokens) +
                " tokens but only " +
                std::to_string(block_pool->free_blocks()) +
                " of " + std::to_string(block_pool->total_blocks()) +
                " blocks are free");
        }
    }

    // Uploads the table when a row changed. A decode step that crosses no block
    // boundary leaves it clean and pays nothing.
    void sync_block_table() {
        if (!kv_paged()) return;
        if (!block_table->dirty()) return;
        const std::vector<int32_t>& image = block_table->device_image();
        check_device(memcpy_h2d(block_table_device.data, image.data(),
                                image.size() * sizeof(int32_t)),
                     "Qwen block table upload");
    }

    const int* block_table_data() const {
        return kv_paged()
            ? static_cast<const int*>(block_table_device.data) : nullptr;
    }

    int paged_block_size() const {
        return kv_paged() ? block_table->block_size() : 0;
    }

    int paged_blocks_per_seq() const {
        return kv_paged() ? block_table->max_blocks_per_seq() : 0;
    }

    // One slot's row of the device image, which is what the single-sequence
    // paged kernels take: they address one sequence and need no slot
    // indirection.
    const int* block_table_row(int slot_id) const {
        if (!kv_paged()) return nullptr;
        return static_cast<const int*>(block_table_device.data) +
               static_cast<size_t>(slot_id) * block_table->max_blocks_per_seq();
    }

    void release_paged_slot(int slot_id) {
        if (!kv_paged()) return;
        block_table->release(slot_id);
    }

    // Tears down one slot's paged state as a unit: returns its blocks to the
    // pool, uploads the table, and clears the reuse state that described KV no
    // longer backed by those blocks. Rank 0's free_slot and the worker loop's
    // FreeSlot command both call this, so a slot ends up in exactly the same
    // state on every rank. A stale cached_prompt on a worker would otherwise let
    // an exact-repeat prompt hit the previous request's logits while rank 0
    // recomputes, silently desynchronizing the two.
    void release_slot_paged_state(int slot_id, int& engine_position) {
        if (!kv_paged()) return;
        release_paged_slot(slot_id);
        sync_block_table();
        SlotPrefixState& prefix = prefix_for(slot_id);
        prefix.cached_prompt.clear();
        prefix.snapshots.clear();
        prefix.cached_result = QwenForwardResult{};
        prefix.has_cached_result = false;
        set_slot_position(slot_id, 0, engine_position);
    }

    // Phase 3.4: Element offsets into the per-slot recurrent state arenas. Both
    // tensors carry the slot count as their leading dimension, so the stride is
    // whatever remains. Deriving it from the shape keeps these correct without
    // restating the head geometry, which differs per rank under TP.
    static size_t slot_stride_elements(const QwenDeviceTensor& tensor) {
        if (tensor.data == nullptr || tensor.shape.empty()) return 0;
        size_t stride = 1;
        for (size_t dim = 1; dim < tensor.shape.size(); ++dim) {
            stride *= static_cast<size_t>(tensor.shape[dim]);
        }
        return stride;
    }

    size_t recurrent_slot_offset(int slot_id,
                                 const QwenDeviceTensor& tensor) const {
        if (max_batch_size == 1 || slot_id == 0) return 0;  // Fast path
        return static_cast<size_t>(slot_id) * slot_stride_elements(tensor);
    }

    // Bytes belonging to a single slot. The prefix-cache snapshot and the
    // packed transaction both operate on one session's state, so they must size
    // themselves per slot rather than over the whole arena.
    static size_t recurrent_slot_bytes(const QwenDeviceTensor& tensor) {
        if (tensor.data == nullptr || tensor.nbytes == 0) return 0;
        const uint64_t slots = tensor.shape.empty() ? 1 : tensor.shape[0];
        if (slots <= 1) return tensor.nbytes;
        return tensor.nbytes / static_cast<size_t>(slots);
    }

    // Byte offset of one slot's rows within a recurrent tensor.  Snapshot
    // capture and restore both need it so they touch only the slot they were
    // asked about; addressing slot 0 unconditionally would copy another
    // sequence's state.
    static size_t recurrent_slot_offset(const QwenDeviceTensor& tensor, int slot_id) {
        if (tensor.data == nullptr || tensor.nbytes == 0) return 0;
        const uint64_t slots = tensor.shape.empty() ? 1 : tensor.shape[0];
        if (slots <= 1) return 0;
        if (slot_id < 0 || static_cast<uint64_t>(slot_id) >= slots) {
            throw std::runtime_error("Qwen recurrent slot index out of range");
        }
        return static_cast<size_t>(slot_id) * (tensor.nbytes / static_cast<size_t>(slots));
    }

    void ensure_nccl_comm_stream() {
        if (!use_nccl_comm_stream || nccl_comm_stream_initialized) return;
        nccl_comm_stream = stream_create();
        check_device(nccl_comm_stream != nullptr,
                     "Qwen NCCL communication stream");
        nccl_comm_ready = event_create();
        check_device(nccl_comm_ready != nullptr, "Qwen NCCL ready event");
        nccl_comm_done = event_create();
        check_device(nccl_comm_done != nullptr, "Qwen NCCL done event");
        nccl_comm_stream_initialized = true;
    }

    // Grow the per-slice event pool to hold at least `slices` entries.
    void ensure_comm_slice_events(int slices) {
        while (static_cast<int>(comm_slice_ready.size()) <
               static_cast<size_t>(slices)) {
            void* ready = event_create();
            check_device(ready != nullptr, "Qwen comm slice ready event");
            void* done = event_create();
            check_device(done != nullptr, "Qwen comm slice done event");
            comm_slice_ready.push_back(ready);
            comm_slice_done.push_back(done);
        }
    }

    // The compute kernels all use the legacy default stream. Bracket a
    // communication-stream collective with events so the collective sees the
    // complete producer work and the next compute kernel sees its result.
    void begin_nccl_collective() {
        ensure_nccl_comm_stream();
        check_device(event_record(nccl_comm_ready, nullptr),
                     "Qwen record NCCL ready event");
        check_device(stream_wait_event(nccl_comm_stream, nccl_comm_ready),
                     "Qwen NCCL stream wait");
    }

    void end_nccl_collective() {
        check_device(event_record(nccl_comm_done, nccl_comm_stream),
                     "Qwen record NCCL done event");
        check_device(stream_wait_event(nullptr, nccl_comm_done),
                     "Qwen compute stream wait");
    }

    ~Impl() {
        if (nccl_comm_stream != nullptr) {
            // The default stream wait above normally makes this unnecessary;
            // keep destruction safe if construction or a caller failed midway.
            stream_synchronize(nccl_comm_stream);
        }
        for (void* event : comm_slice_ready) {
            if (event != nullptr) event_destroy(event);
        }
        for (void* event : comm_slice_done) {
            if (event != nullptr) event_destroy(event);
        }
        if (nccl_comm_done != nullptr) event_destroy(nccl_comm_done);
        if (nccl_comm_ready != nullptr) event_destroy(nccl_comm_ready);
        if (nccl_comm_stream != nullptr) stream_destroy(nccl_comm_stream);
    }

    Impl(SafeTensorsIndex& index_, QwenConfig& config_,
         const QwenWeightMap& map, const QwenEngineOptions& options_,
         int max_context_, int active_layers)
        : index(index_), config(config_), options(options_),
          max_context(max_context_) {
        // Phase 3.2: Initialize max_batch_size from options
        max_batch_size = options_.max_batch_size;
        batch_mode_enabled = (max_batch_size > 1);
        slot_positions.resize(max_batch_size, 0);  // Phase 3.4: per-slot positions
        // One prefix-reuse record per slot, sized with the KV arena.
        slot_prefix.resize(static_cast<size_t>(max_batch_size));
        // Phase 3.4: The KV arena is sized here, so the free list must be
        // seeded here as well. Seeding it only from allocate_batch_slots()
        // left the batch API unreachable: that function throws once
        // batch_mode_enabled is set, so allocate_slot() only ever saw an
        // empty list and returned -1. Descending order makes back() hand out
        // slot 0 first, matching single-session addressing.
        for (int slot = max_batch_size - 1; batch_mode_enabled && slot >= 0;
             --slot) {
            free_slots.push_back(slot);
        }

        telemetry.checkpoint_linear_kinds =
            map.checkpoint_linear_kind_counts();
        telemetry.host_global_metadata_bytes = map.host_global_metadata_bytes();
        if (telemetry.checkpoint_linear_kinds.nvfp4_group16 != 0) {
            const char* requested = std::getenv("DSV4_QWEN_NVFP4");
            const bool reference = requested != nullptr &&
                std::strcmp(requested, "reference") == 0;
            const bool dp4a = requested != nullptr &&
                std::strcmp(requested, "dp4a") == 0;
            const bool wmma = requested != nullptr &&
                std::strcmp(requested, "wmma") == 0;
            const bool automatic = requested == nullptr || *requested == '\0' ||
                std::strcmp(requested, "auto") == 0;
            if (!reference && !dp4a && !wmma && !automatic) {
                throw std::runtime_error(
                    "DSV4_QWEN_NVFP4 must be auto, dp4a, wmma, or reference");
            }
            telemetry.nvfp4_decode_path = reference
                ? "packed_reference" : wmma ? "q8_group32_wmma" :
                    "q8_group32_dp4a";
            if (reference) {
                telemetry.nvfp4_prefill_path = "packed_reference";
            } else if (dp4a) {
                telemetry.nvfp4_prefill_path = "q8_group32_dp4a";
            } else {
                const char* fused =
                    std::getenv("DSV4_QWEN_NVFP4_FUSED_SWIGLU");
                const char* shared =
                    std::getenv("DSV4_QWEN_NVFP4_SHARED_Q8_SWIGLU");
                // Report the path the configured chunk size will actually
                // take, since the wide tile only engages above a row count.
                const bool wide = qwen_nvfp4_wide_n64_enabled(
                    std::max(1, options_.prefill_chunk_tokens));
                telemetry.nvfp4_prefill_path =
                    fused != nullptr && std::strcmp(fused, "1") == 0
                        ? "q8_group32_wmma_fused_gate_up_swiglu_experimental"
                        : wide
                              ? "q8_group32_wide_n64_shared_gate_up_q8"
                        : shared == nullptr || std::strcmp(shared, "0") != 0
                              ? "q8_group32_wmma_shared_gate_up_q8"
                              : "q8_group32_wmma";
            }
        }
        if (telemetry.checkpoint_linear_kinds.fp8_channel != 0) {
            telemetry.fp8_channel_decode_path = "online_matvec";
            const char* wide = std::getenv("QWEN_FP8_CHANNEL_PREFILL_WIDE_N64");
            telemetry.fp8_channel_prefill_path =
                wide == nullptr || std::strcmp(wide, "0") != 0
                    ? "wide_n64_online" : "tiled_online";
        }
        telemetry.target_head_path =
            map.lm_head().kind == QwenLinearKind::DenseF16
                ? "dense_f16"
                : map.lm_head().kind == QwenLinearKind::Fp8Channel
                      ? "fp8_channel_online"
                      : qwen_linear_kind_name(map.lm_head().kind);
        embed = upload(index, map.embed_tokens());
        final_norm = upload(index, map.final_norm());
        lm_head = upload_linear(index, map.lm_head());
        count_active_linear(map.lm_head().kind);
        uploaded_weight_bytes += map.embed_tokens().device_nbytes +
                                 map.final_norm().device_nbytes;
        uploaded_weight_bytes += map.lm_head().weight.device_nbytes;
        if (map.lm_head().has_scale) {
            uploaded_scale_bytes += map.lm_head().scale.device_nbytes;
        }
        const size_t layer_limit = active_layers > 0
            ? std::min(static_cast<size_t>(active_layers), map.layers().size())
            : map.layers().size();
        layers.reserve(layer_limit);
        const int world = options.tp_world;
        const int local_key_heads =
            static_cast<int>(config.linear_attention.key_heads / world);
        const int local_value_heads =
            static_cast<int>(config.linear_attention.value_heads / world);
        const int local_key_dim = local_key_heads *
            static_cast<int>(config.linear_attention.key_head_dim);
        const int local_value_dim = local_value_heads *
            static_cast<int>(config.linear_attention.value_head_dim);
        const int local_qkv_dim = 2 * local_key_dim + local_value_dim;
        const int local_kv_heads =
            static_cast<int>(config.full_attention.num_key_value_heads / world);
        const int head_dim = static_cast<int>(config.full_attention.head_dim);
        const bool speculative_target = options.mtp ||
            !options.dspark_checkpoint.empty() ||
            !options.dflash2_checkpoint.empty();
        const bool fuse_kv_projection = speculative_target &&
            qwen_env_enabled("QWEN_FUSE_KV_PROJECTION");

        // The paged pool is sized before any layer is uploaded, because every
        // full-attention layer then allocates its arena from the same block
        // count. Counting the layers up front rather than growing per layer is
        // what makes the budget a single decision.
        if (options.kv_paged) {
            size_t full_attention_layers = 0;
            for (size_t index = 0; index < layer_limit; ++index) {
                if (!map.layers()[index].linear_attention.in_proj_qkv.weight.found) {
                    ++full_attention_layers;
                }
            }
            const size_t elements_per_token =
                static_cast<size_t>(local_kv_heads) * head_dim;
            // Bytes one block costs across every full-attention layer, K and V
            // together. That is the unit the budget divides.
            const size_t bytes_per_block = static_cast<size_t>(
                options.kv_block_size) * elements_per_token * sizeof(uint16_t) *
                2 * full_attention_layers;
            if (full_attention_layers == 0 || bytes_per_block == 0) {
                throw std::runtime_error(
                    "Qwen paged KV cache requires at least one full-attention "
                    "layer");
            }
            // A zero budget reproduces what the contiguous arena would have
            // reserved, so turning paging on is memory-neutral and only changes
            // how the same bytes are handed out.
            const uint64_t budget = options.kv_cache_bytes != 0
                ? options.kv_cache_bytes
                : static_cast<uint64_t>(options.max_batch_size) * max_context *
                      elements_per_token * sizeof(uint16_t) * 2 *
                      full_attention_layers;
            const int num_blocks = static_cast<int>(budget / bytes_per_block);
            // A slot must be able to reach max_context, or a long request would
            // fail at a boundary no scheduler could anticipate.
            const int blocks_per_seq =
                (max_context + options.kv_block_size - 1) /
                options.kv_block_size;
            if (num_blocks < blocks_per_seq) {
                throw std::runtime_error(
                    "Qwen paged KV cache budget covers " +
                    std::to_string(num_blocks) + " blocks but a single " +
                    std::to_string(max_context) +
                    "-token sequence needs " + std::to_string(blocks_per_seq) +
                    "; raise QwenEngineOptions::kv_cache_bytes");
            }
            block_pool = std::make_unique<QwenBlockPool>(
                num_blocks, options.kv_block_size);
            block_table = std::make_unique<QwenBlockTable>(
                block_pool.get(), options.max_batch_size, max_context);
            const size_t image_elements =
                static_cast<size_t>(block_table->max_slots()) *
                block_table->max_blocks_per_seq();
            allocate(block_table_device, image_elements * sizeof(int32_t),
                     {static_cast<uint64_t>(block_table->max_slots()),
                      static_cast<uint64_t>(block_table->max_blocks_per_seq())},
                     // Tagged like the other int row buffers in this engine
                     // (batch_positions, batch_slot_ids): the allocation is
                     // sized in bytes and nothing reads the dtype back.
                     SafeDType::I64);
            telemetry.kv_paged_blocks = num_blocks;
            telemetry.kv_paged_block_size = options.kv_block_size;
        }

        for (size_t layer_index = 0; layer_index < layer_limit; ++layer_index) {
            const QwenLayerWeights& source = map.layers()[layer_index];
            for (const QwenTensorRef* ref : {
                     &source.input_layernorm, &source.post_attention_layernorm,
                     &source.mlp.gate_proj.weight, &source.mlp.up_proj.weight,
                     &source.mlp.down_proj.weight,
                     &source.linear_attention.in_proj_qkv.weight,
                     &source.linear_attention.in_proj_z.weight,
                     &source.linear_attention.out_proj.weight,
                     &source.linear_attention.in_proj_a.weight,
                     &source.linear_attention.in_proj_b.weight,
                     &source.linear_attention.conv1d,
                     &source.linear_attention.a_log,
                     &source.linear_attention.dt_bias,
                     &source.linear_attention.norm,
                     &source.full_attention.q_proj.weight,
                     &source.full_attention.k_proj.weight,
                     &source.full_attention.v_proj.weight,
                     &source.full_attention.o_proj.weight,
                     &source.full_attention.q_norm,
                     &source.full_attention.k_norm}) {
                if (ref->found) uploaded_weight_bytes += ref->device_nbytes;
            }
            for (const QwenLinearRef* ref : {
                     &source.mlp.gate_proj, &source.mlp.up_proj,
                     &source.mlp.down_proj,
                     &source.linear_attention.in_proj_qkv,
                     &source.linear_attention.in_proj_z,
                     &source.linear_attention.out_proj,
                     &source.full_attention.q_proj,
                     &source.full_attention.k_proj,
                     &source.full_attention.v_proj,
                     &source.full_attention.o_proj}) {
                if (ref->has_scale) {
                    uploaded_scale_bytes += ref->scale.device_nbytes;
                }
            }

            for (const QwenLinearRef* ref : {
                     &source.mlp.gate_proj, &source.mlp.up_proj,
                     &source.mlp.down_proj}) {
                count_active_linear(ref->kind);
            }
            if (source.linear_attention.in_proj_qkv.weight.found) {
                for (const QwenLinearRef* ref : {
                         &source.linear_attention.in_proj_qkv,
                         &source.linear_attention.in_proj_z,
                         &source.linear_attention.out_proj,
                         &source.linear_attention.in_proj_a,
                         &source.linear_attention.in_proj_b}) {
                    count_active_linear(ref->kind);
                }
            } else {
                for (const QwenLinearRef* ref : {
                         &source.full_attention.q_proj,
                         &source.full_attention.k_proj,
                         &source.full_attention.v_proj,
                         &source.full_attention.o_proj}) {
                    count_active_linear(ref->kind);
                }
            }

            DeviceLayer destination;
            destination.input_norm = upload(index, source.input_layernorm);
            destination.post_norm = upload(index, source.post_attention_layernorm);
            destination.gate = upload_linear(index, source.mlp.gate_proj);
            destination.up = upload_linear(index, source.mlp.up_proj);
            destination.down = upload_linear(index, source.mlp.down_proj);
            if (source.linear_attention.in_proj_qkv.weight.found) {
                destination.linear.qkv =
                    upload_linear(index, source.linear_attention.in_proj_qkv);
                destination.linear.z =
                    upload_linear(index, source.linear_attention.in_proj_z);
                destination.linear.out =
                    upload_linear(index, source.linear_attention.out_proj);
                destination.linear.a =
                    upload_linear(index, source.linear_attention.in_proj_a);
                destination.linear.b =
                    upload_linear(index, source.linear_attention.in_proj_b);
                // in_proj_a and in_proj_b are both BF16 [num_v_heads, hidden]
                // and read the same activation, so a row-concat lets one GEMM
                // replace two. Each alone is far too small to fill the device
                // (12 rows per rank), and together they cost ~6.9 ms/step of
                // verify. Rows [0, n) are a; rows [n, 2n) are b.
                destination.linear.ab = fuse_linear_rows(
                    destination.linear.a, destination.linear.b);
                destination.linear.conv =
                    upload(index, source.linear_attention.conv1d);
                destination.linear.a_log =
                    upload(index, source.linear_attention.a_log);
                destination.linear.dt_bias =
                    upload(index, source.linear_attention.dt_bias);
                destination.linear.norm =
                    upload(index, source.linear_attention.norm);
                // Phase 3.4: The recurrent state carries the whole history of a
                // linear-attention layer, so batched requests need one copy
                // each. It is allocated as [slot, value_heads, k_dim, v_dim]
                // and indexed by offset, mirroring the KV cache arena. At TP4
                // this is 36.7 MiB per slot per rank across all 48 linear
                // layers, cheap enough to hold resident; copying it in and out
                // on every slot switch would cost that much D2D traffic per
                // token instead.
                const int state_slots = options.max_batch_size;
                const size_t state_elements_per_slot =
                    static_cast<size_t>(local_value_heads) *
                        config.linear_attention.key_head_dim *
                        config.linear_attention.value_head_dim;
                allocate_float(destination.linear.state,
                    static_cast<size_t>(state_slots) * state_elements_per_slot,
                    {static_cast<uint64_t>(state_slots),
                     static_cast<uint64_t>(local_value_heads),
                     config.linear_attention.key_head_dim,
                     config.linear_attention.value_head_dim});
                const uint64_t tail_rows = static_cast<uint64_t>(std::max(
                    0, static_cast<int>(config.linear_attention.conv_kernel_dim) - 1));
                allocate_half(destination.linear.conv_tail,
                    static_cast<size_t>(state_slots) * tail_rows * local_qkv_dim,
                    {static_cast<uint64_t>(state_slots), tail_rows,
                     static_cast<uint64_t>(local_qkv_dim)});
                zero_tensor(destination.linear.state);
                zero_tensor(destination.linear.conv_tail);
            } else {
                destination.full.q = upload_linear(index, source.full_attention.q_proj);
                destination.full.k = upload_linear(index, source.full_attention.k_proj);
                destination.full.v = upload_linear(index, source.full_attention.v_proj);
                if (fuse_kv_projection) {
                    destination.full.kv = fuse_linear_rows(
                        destination.full.k, destination.full.v);
                }
                destination.full.out = upload_linear(index, source.full_attention.o_proj);
                destination.full.q_norm = upload(index, source.full_attention.q_norm);
                destination.full.k_norm = upload(index, source.full_attention.k_norm);

                // Phase 3.2: Multi-slot KV cache allocation
                const int slot_count = options.max_batch_size;
                const size_t cache_elements = static_cast<size_t>(slot_count) *
                    static_cast<size_t>(max_context) * local_kv_heads * head_dim;
                const std::vector<uint64_t> cache_shape = {
                    static_cast<uint64_t>(slot_count),
                    static_cast<uint64_t>(max_context),
                    static_cast<uint64_t>(local_kv_heads),
                    static_cast<uint64_t>(head_dim)};

                if (block_pool != nullptr) {
                    // Paged: one arena of blocks shared by every slot. The
                    // leading dimension is the block count, not the slot count,
                    // which is the whole difference: a slot's extent is now what
                    // it has been handed rather than what it reserved.
                    // Construction rejects the quantized dtypes, so this branch
                    // is FP16 by definition.
                    const size_t paged_elements =
                        static_cast<size_t>(block_pool->total_blocks()) *
                        block_pool->block_size() * local_kv_heads * head_dim;
                    const std::vector<uint64_t> paged_shape = {
                        static_cast<uint64_t>(block_pool->total_blocks()),
                        static_cast<uint64_t>(block_pool->block_size()),
                        static_cast<uint64_t>(local_kv_heads),
                        static_cast<uint64_t>(head_dim)};
                    allocate_half(destination.full.k_cache, paged_elements,
                                  paged_shape);
                    allocate_half(destination.full.v_cache, paged_elements,
                                  paged_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                } else if (options.kv_cache_dtype == QwenKvCacheDType::Fp16) {
                    allocate_half(destination.full.k_cache, cache_elements, cache_shape);
                    allocate_half(destination.full.v_cache, cache_elements, cache_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                } else if (options.kv_cache_dtype == QwenKvCacheDType::Fp8) {
                    allocate_elements(destination.full.k_cache, cache_elements,
                                      cache_shape, SafeDType::F8_E4M3);
                    allocate_elements(destination.full.v_cache, cache_elements,
                                      cache_shape, SafeDType::F8_E4M3);
                    const size_t scale_elements = static_cast<size_t>(slot_count) *
                        static_cast<size_t>(max_context) * local_kv_heads * (head_dim / kKvScaleBlock);
                    const std::vector<uint64_t> scale_shape = {
                        static_cast<uint64_t>(slot_count),
                        static_cast<uint64_t>(max_context),
                        static_cast<uint64_t>(local_kv_heads),
                        static_cast<uint64_t>(head_dim / kKvScaleBlock)};
                    allocate_half(destination.full.k_scale, scale_elements, scale_shape);
                    allocate_half(destination.full.v_scale, scale_elements, scale_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                    cache_scale_bytes += destination.full.k_scale.nbytes +
                                         destination.full.v_scale.nbytes;
                } else if (options.kv_cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
                    // TurboQuantK8V4: one combined slot per token/head, sized
                    // from head_dim (196 bytes at 128, 388 at 256).
                    const int slot_bytes =
                        qwen_turboquant_k8v4_slot_bytes(head_dim);
                    const size_t slot_elements = static_cast<size_t>(slot_count) *
                        static_cast<size_t>(max_context) * local_kv_heads * slot_bytes;
                    const std::vector<uint64_t> slot_shape = {
                        static_cast<uint64_t>(slot_count),
                        static_cast<uint64_t>(max_context),
                        static_cast<uint64_t>(local_kv_heads),
                        static_cast<uint64_t>(slot_bytes)};
                    allocate_elements(destination.full.turboquant_cache, slot_elements,
                                      slot_shape, SafeDType::I8);
                    cache_data_bytes += destination.full.turboquant_cache.nbytes;
                } else if (options.kv_cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
                    // INT8 per-token-head: separate INT8 K/V arrays + per-token per-head scales
                    allocate_elements(destination.full.k_cache, cache_elements,
                                      cache_shape, SafeDType::I8);
                    allocate_elements(destination.full.v_cache, cache_elements,
                                      cache_shape, SafeDType::I8);
                    const size_t scale_elements = static_cast<size_t>(slot_count) *
                        static_cast<size_t>(max_context) * local_kv_heads;
                    const std::vector<uint64_t> scale_shape = {
                        static_cast<uint64_t>(slot_count),
                        static_cast<uint64_t>(max_context),
                        static_cast<uint64_t>(local_kv_heads)};
                    allocate_half(destination.full.k_scale, scale_elements, scale_shape);
                    allocate_half(destination.full.v_scale, scale_elements, scale_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                    cache_scale_bytes += destination.full.k_scale.nbytes +
                                         destination.full.v_scale.nbytes;
                } else {
                    // FP16: default exact path
                    allocate_half(destination.full.k_cache, cache_elements, cache_shape);
                    allocate_half(destination.full.v_cache, cache_elements, cache_shape);
                    cache_data_bytes += destination.full.k_cache.nbytes +
                                        destination.full.v_cache.nbytes;
                }
            }
            layers.push_back(std::move(destination));
        }

        // On SM75 the verify batch is too narrow to amortize online FP8 decode.
        // A dense FP16 cache lets cuBLAS reuse tensor-core tiles across all rows,
        // but expanding the whole model would consume ~17 GiB/rank. Keep only the
        // most frequently executed target projections: qkv/z/out/down on the 48
        // linear-attention layers and q/k/v/down on the 16 full-attention layers.
        // Gate/up stay raw FP8 because their fused kernel is close to the two-GEMM
        // resident alternative while costing another ~5.3 GiB; full-attention out
        // saves only ~0.15 ms/verify for another 240 MiB and stays raw FP8. Set
        // QWEN_VERIFY_RESIDENT_FP16=0 to keep the former all-FP8 verify path.
        // Plain decoding never consumes these matrices, so do not spend the fixed
        // cache unless this engine actually owns a speculative target.
        const bool verify_resident_weights = speculative_target &&
            qwen_env_enabled_default("QWEN_VERIFY_RESIDENT_FP16");
#ifdef POCKET_BACKEND_ASCEND
        // The resident cache only ever expands FP8 weights, and no FP8 kind is
        // loadable on this backend, so there is nothing for it to materialize.
        (void)verify_resident_weights;
#else
        if (verify_resident_weights) {
            auto materialize_verify_weight = [this](DeviceLinear& linear) {
                if (linear.kind != QwenLinearKind::Fp8Block128 ||
                    linear.weight.data == nullptr || linear.weight.shape.size() != 2 ||
                    linear.scale.data == nullptr || linear.scale.shape.size() != 2) {
                    return;
                }
                const int rows = static_cast<int>(linear.weight.shape[0]);
                const int cols = static_cast<int>(linear.weight.shape[1]);
                const size_t elements = static_cast<size_t>(rows) * cols;
                allocate_half(linear.verify_weight, elements,
                              {linear.weight.shape[0], linear.weight.shape[1]});
                require_launch(qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
                    linear.weight.fp8_data(), linear.scale.f16_data(),
                    linear.verify_weight.f16_data(), rows, cols, cols,
                    static_cast<int>(linear.scale.shape[1])),
                    "Qwen resident verify weight expansion");
                verify_weight_bytes += linear.verify_weight.nbytes;
            };
            for (DeviceLayer& layer : layers) {
                materialize_verify_weight(layer.down);
                if (layer.linear.qkv.weight.data != nullptr) {
                    materialize_verify_weight(layer.linear.qkv);
                    materialize_verify_weight(layer.linear.z);
                    materialize_verify_weight(layer.linear.out);
                } else {
                    materialize_verify_weight(layer.full.q);
                    if (layer.full.kv.weight.data != nullptr) {
                        materialize_verify_weight(layer.full.kv);
                    } else {
                        materialize_verify_weight(layer.full.k);
                        materialize_verify_weight(layer.full.v);
                    }
                }
            }
        }
#endif

#ifdef POCKET_BACKEND_ASCEND
        if (!options.dspark_checkpoint.empty() ||
            !options.dflash2_checkpoint.empty()) {
            throw std::runtime_error(
                "Qwen DSpark and DFlash2 drafters are not implemented on the "
                "Ascend backend");
        }
#else
        if (!options.dspark_checkpoint.empty()) {
            if (active_layers > 0 &&
                active_layers != static_cast<int>(config.num_hidden_layers)) {
                throw std::runtime_error(
                    "Qwen DSpark requires the complete target layer stack");
            }
            dspark_config = std::make_unique<QwenDSparkConfig>(
                QwenDSparkConfig::from_directory(options.dspark_checkpoint));
            dspark_config->validate_for_target(
                config.hidden_size, config.vocab_size, config.num_hidden_layers);
            dspark_index = std::make_unique<SafeTensorsIndex>(
                SafeTensorsIndex::from_single_file(options.dspark_checkpoint));
            dspark_weights = std::make_unique<QwenDSparkWeightMap>(
                *dspark_index, *dspark_config, options.tp_world, options.tp_rank);
            // The target embedding is vocab-sharded. QwenDSparkRuntime gathers
            // local rows then performs the same TP all-reduce as target prefill.
            const QwenTargetHeadAdapter target_head{
                lm_head.kind, &lm_head.weight,
                lm_head.scale.data != nullptr ? &lm_head.scale : nullptr,
                static_cast<int>(lm_head.logical_shape.at(0)),
                static_cast<int>(lm_head.logical_shape.at(1)),
                static_cast<uint64_t>(options.tp_rank) * config.vocab_size /
                    options.tp_world};
            dspark = std::make_unique<QwenDSparkRuntime>(
                options.dspark_checkpoint, *dspark_config, *dspark_weights,
                embed, target_head, options.tp_world, options.tp_rank,
                options.device, options.nccl_id_path, max_context);
            dspark_enabled = true;
            uploaded_weight_bytes += dspark->resident_weight_bytes();
            cache_data_bytes += dspark->context_cache_bytes();
        }
        if (!options.dflash2_checkpoint.empty()) {
            if (active_layers > 0 &&
                active_layers != static_cast<int>(config.num_hidden_layers)) {
                throw std::runtime_error(
                    "Qwen DFlash2 requires the complete target layer stack");
            }
            dflash2_config = std::make_unique<QwenDFlash2Config>(
                QwenDFlash2Config::from_directory(options.dflash2_checkpoint));
            dflash2_config->validate_for_target(
                config.hidden_size, config.vocab_size, config.num_hidden_layers);
            dflash2_index = std::make_unique<SafeTensorsIndex>(
                SafeTensorsIndex::from_single_file(options.dflash2_checkpoint));
            dflash2_weights = std::make_unique<QwenDFlash2WeightMap>(
                *dflash2_index, *dflash2_config, options.tp_world, options.tp_rank);
            const QwenTargetHeadAdapter target_head{
                lm_head.kind, &lm_head.weight,
                lm_head.scale.data != nullptr ? &lm_head.scale : nullptr,
                static_cast<int>(lm_head.logical_shape.at(0)),
                static_cast<int>(lm_head.logical_shape.at(1)),
                static_cast<uint64_t>(options.tp_rank) * config.vocab_size /
                    options.tp_world};
            dflash2 = std::make_unique<QwenDFlash2Runtime>(
                options.dflash2_checkpoint, *dflash2_config, *dflash2_weights,
                embed, target_head, options.tp_world, options.tp_rank,
                options.device, options.nccl_id_path, max_context);
            dflash2_enabled = true;
            uploaded_weight_bytes += dflash2->resident_weight_bytes();
            cache_data_bytes += dflash2->context_cache_bytes();
        }
#endif

        if (options.mtp) {
            if (!map.mtp().found) {
                throw std::runtime_error(
                    "Qwen MTP requested but checkpoint has no native MTP weights");
            }
            if (config.mtp_num_hidden_layers != 1) {
                throw std::runtime_error(
                    "Qwen runtime currently supports exactly one native MTP layer");
            }
            mtp_enabled = true;
            const QwenMtpWeights& source = map.mtp();
            mtp_pre_fc_norm_embedding = upload(index, source.pre_fc_norm_embedding);
            mtp_pre_fc_norm_hidden = upload(index, source.pre_fc_norm_hidden);
            mtp_norm = upload(index, source.norm);
            mtp_fc = upload_linear(index, source.fc);
            const QwenLayerWeights& mtp_source = source.layer;
            mtp_layer.input_norm = upload(index, mtp_source.input_layernorm);
            mtp_layer.post_norm = upload(index, mtp_source.post_attention_layernorm);
            mtp_layer.gate = upload_linear(index, mtp_source.mlp.gate_proj);
            mtp_layer.up = upload_linear(index, mtp_source.mlp.up_proj);
            mtp_layer.down = upload_linear(index, mtp_source.mlp.down_proj);
            mtp_layer.full.q = upload_linear(index, mtp_source.full_attention.q_proj);
            mtp_layer.full.k = upload_linear(index, mtp_source.full_attention.k_proj);
            mtp_layer.full.v = upload_linear(index, mtp_source.full_attention.v_proj);
            mtp_layer.full.out = upload_linear(index, mtp_source.full_attention.o_proj);
            mtp_layer.full.q_norm = upload(index, mtp_source.full_attention.q_norm);
            mtp_layer.full.k_norm = upload(index, mtp_source.full_attention.k_norm);
            uploaded_weight_bytes += source.pre_fc_norm_embedding.device_nbytes +
                source.pre_fc_norm_hidden.device_nbytes + source.norm.device_nbytes +
                mtp_source.input_layernorm.device_nbytes +
                mtp_source.post_attention_layernorm.device_nbytes +
                mtp_source.full_attention.q_norm.device_nbytes +
                mtp_source.full_attention.k_norm.device_nbytes;
            const auto count_mtp_linear = [this](const QwenLinearRef& linear) {
                uploaded_weight_bytes += linear.weight.device_nbytes;
                if (linear.has_scale) {
                    uploaded_scale_bytes += linear.scale.device_nbytes;
                }
            };
            count_mtp_linear(source.fc);
            count_active_linear(source.fc.kind);
            for (const QwenLinearRef* linear : {
                     &mtp_source.full_attention.q_proj,
                     &mtp_source.full_attention.k_proj,
                     &mtp_source.full_attention.v_proj,
                     &mtp_source.full_attention.o_proj,
                     &mtp_source.mlp.gate_proj,
                     &mtp_source.mlp.up_proj,
                     &mtp_source.mlp.down_proj}) {
                count_mtp_linear(*linear);
                count_active_linear(linear->kind);
            }
            const size_t mtp_cache_elements = static_cast<size_t>(max_context) *
                local_kv_heads * head_dim;
            const std::vector<uint64_t> mtp_cache_shape = {
                static_cast<uint64_t>(max_context),
                static_cast<uint64_t>(local_kv_heads),
                static_cast<uint64_t>(head_dim)};
            if (options.kv_cache_dtype == QwenKvCacheDType::Fp16) {
                allocate_half(mtp_layer.full.k_cache, mtp_cache_elements,
                              mtp_cache_shape);
                allocate_half(mtp_layer.full.v_cache, mtp_cache_elements,
                              mtp_cache_shape);
                cache_data_bytes += mtp_layer.full.k_cache.nbytes +
                                    mtp_layer.full.v_cache.nbytes;
            } else if (options.kv_cache_dtype == QwenKvCacheDType::Fp8) {
                allocate_elements(mtp_layer.full.k_cache, mtp_cache_elements,
                                  mtp_cache_shape, SafeDType::F8_E4M3);
                allocate_elements(mtp_layer.full.v_cache, mtp_cache_elements,
                                  mtp_cache_shape, SafeDType::F8_E4M3);
                const size_t scale_elements = static_cast<size_t>(max_context) *
                    local_kv_heads * (head_dim / kKvScaleBlock);
                const std::vector<uint64_t> scale_shape = {
                    static_cast<uint64_t>(max_context),
                    static_cast<uint64_t>(local_kv_heads),
                    static_cast<uint64_t>(head_dim / kKvScaleBlock)};
                allocate_half(mtp_layer.full.k_scale, scale_elements, scale_shape);
                allocate_half(mtp_layer.full.v_scale, scale_elements, scale_shape);
                cache_data_bytes += mtp_layer.full.k_cache.nbytes +
                                    mtp_layer.full.v_cache.nbytes;
                cache_scale_bytes += mtp_layer.full.k_scale.nbytes +
                                     mtp_layer.full.v_scale.nbytes;
            } else {
                // TurboQuantK8V4: one combined slot per token/head, sized
                // from head_dim (196 bytes at 128, 388 at 256).
                const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
                const size_t slot_elements = static_cast<size_t>(max_context) *
                    local_kv_heads * slot_bytes;
                const std::vector<uint64_t> slot_shape = {
                    static_cast<uint64_t>(max_context),
                    static_cast<uint64_t>(local_kv_heads),
                    static_cast<uint64_t>(slot_bytes)};
                allocate_elements(mtp_layer.full.turboquant_cache, slot_elements,
                                  slot_shape, SafeDType::I8);
                cache_data_bytes += mtp_layer.full.turboquant_cache.nbytes;
            }
        }
    }

    bool has_recurrent_state() const {
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data != nullptr) return true;
        }
        return false;
    }

    // Phase 3.4: Clears one slot's recurrent state. The MTP and drafter fields
    // below are still single-session, so slot != 0 only resets the arena rows
    // it owns; see the shared-state limits in the Phase 3.4 notes.
    void zero_recurrent_state(int slot_id = 0) {
        for (DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            zero_slot_region(layer.linear.state, slot_id);
            zero_slot_region(layer.linear.conv_tail, slot_id);
        }
        if (dspark_enabled) dspark->reset();
        if (dflash2_enabled) dflash2->reset();
        has_target_last_hidden = false;
        mtp_seed_ready = false;
        mtp_position = 0;
        mtp_seed_input_token = 0;
        mtp_next_token = 0;
        mtp_next_logit = 0.0f;
        mtp_next_checksum = 0.0f;
    }

    void append_transaction_region(void* data, uint64_t bytes,
                                   uint64_t& packed_offset) {
        if (data == nullptr || bytes == 0) return;
        if (packed_offset > UINT64_MAX - bytes) {
            throw std::runtime_error("Qwen transaction byte extent overflow");
        }
        QwenCopyRegion region;
        region.device_address = reinterpret_cast<uint64_t>(data);
        region.packed_offset = packed_offset;
        region.bytes = bytes;
        region.first_block = transaction_total_blocks;
        host_transaction_regions.push_back(region);
        const uint64_t blocks = qwen_copy_region_blocks(bytes);
        if (transaction_total_blocks > UINT64_MAX - blocks) {
            throw std::runtime_error("Qwen transaction block extent overflow");
        }
        transaction_total_blocks += blocks;
        packed_offset += bytes;
    }

    void initialize_packed_transaction_state() {
        if (!host_transaction_regions.empty()) return;
        uint64_t packed_bytes = 0;
        transaction_total_blocks = 0;
        host_transaction_regions.reserve(layers.size() * 2 +
                                         (mtp_enabled ? 1 : 0));
        for (DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            // Phase 3.4: MTP rollback is single-session, so the transaction
            // covers slot 0's rows only rather than the whole arena.
            append_transaction_region(layer.linear.state.data,
                                      recurrent_slot_bytes(layer.linear.state),
                                      packed_bytes);
            append_transaction_region(layer.linear.conv_tail.data,
                                      recurrent_slot_bytes(layer.linear.conv_tail),
                                      packed_bytes);
        }
        if (mtp_enabled) {
            if (!has_target_last_hidden || target_last_hidden.data == nullptr) {
                throw std::runtime_error(
                    "Qwen MTP transaction requires the committed target hidden");
            }
            append_transaction_region(
                target_last_hidden.data,
                static_cast<uint64_t>(config.hidden_size) * sizeof(uint16_t),
                packed_bytes);
        }
        if (host_transaction_regions.empty() || packed_bytes == 0 ||
            transaction_total_blocks == 0) {
            throw std::runtime_error("Qwen transaction has no recurrent state");
        }
        allocate(transaction_packed, packed_bytes, {packed_bytes}, SafeDType::I8);
        const uint64_t descriptor_bytes =
            host_transaction_regions.size() * sizeof(QwenCopyRegion);
        allocate(transaction_regions, descriptor_bytes,
                 {static_cast<uint64_t>(host_transaction_regions.size())},
                 SafeDType::I8);
        check_device(memcpy_h2d(transaction_regions.data, host_transaction_regions.data(),
                                descriptor_bytes),
                     "Qwen transaction descriptor upload");
    }

    // Copies the recurrent half of the network to device-resident snapshots.
    // Only the 48 DeltaNet layers carry order-dependent state; full attention
    // is skipped because its KV cache is addressed by absolute position.
    void prepare_transaction_state() {
        PhaseScope scope(this, "transaction_snapshot");
        initialize_packed_transaction_state();
        require_launch(qwen_gather_copy_regions(
            static_cast<const QwenCopyRegion*>(transaction_regions.data),
            static_cast<int>(host_transaction_regions.size()),
            static_cast<uint8_t*>(transaction_packed.data),
            transaction_total_blocks),
            "Qwen packed transaction snapshot");
    }

    void restore_transaction_state(int position) {
        PhaseScope scope(this, "transaction_restore");
        if (host_transaction_regions.empty() ||
            transaction_regions.data == nullptr ||
            transaction_packed.data == nullptr) {
            throw std::runtime_error("Qwen transaction state is unavailable");
        }
        require_launch(qwen_scatter_copy_regions(
            static_cast<const QwenCopyRegion*>(transaction_regions.data),
            static_cast<int>(host_transaction_regions.size()),
            static_cast<const uint8_t*>(transaction_packed.data),
            transaction_total_blocks),
            "Qwen packed transaction restore");
        if (mtp_enabled) {
            has_target_last_hidden = true;
            mtp_seed_ready = false;
            mtp_position = position;
        }
        if (dspark_enabled) dspark->crop_context(position);
        if (dflash2_enabled) dflash2->crop_context(position);
    }

    QwenRecurrentSnapshot capture_recurrent_state(
        int position, const QwenForwardResult* result = nullptr,
        bool periodic = false, int slot_id = 0) {
        QwenRecurrentSnapshot snapshot;
        snapshot.position = position;
        snapshot.periodic = periodic;
        if (result != nullptr) {
            snapshot.result = *result;
            snapshot.has_result = true;
        }
        // One slot's worth per layer: a snapshot must not size itself over the
        // whole multi-slot arena.
        size_t state_bytes = 0;
        size_t tail_bytes = 0;
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            state_bytes += recurrent_slot_bytes(layer.linear.state);
            tail_bytes += recurrent_slot_bytes(layer.linear.conv_tail);
        }
        if (state_bytes != 0) {
            allocate(snapshot.state, state_bytes, {state_bytes / sizeof(float)},
                     SafeDType::F32);
        }
        if (tail_bytes != 0) {
            allocate(snapshot.conv_tail, tail_bytes,
                     {tail_bytes / sizeof(uint16_t)}, SafeDType::F16);
        }
        if (mtp_enabled && position > 0) {
            if (!has_target_last_hidden || target_last_hidden.data == nullptr) {
                throw std::runtime_error(
                    "Qwen MTP snapshot requires the committed target hidden");
            }
            const size_t hidden_elements = static_cast<size_t>(config.hidden_size);
            allocate_half(snapshot.target_hidden, hidden_elements,
                          {config.hidden_size});
            check_device(memcpy_d2d(snapshot.target_hidden.data, target_last_hidden.data,
                                    hidden_elements * sizeof(uint16_t)),
                         "Qwen target hidden snapshot copy");
        }
        size_t state_offset = 0;
        size_t tail_offset = 0;
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            const size_t layer_state_bytes = recurrent_slot_bytes(layer.linear.state);
            const size_t layer_tail_bytes = recurrent_slot_bytes(layer.linear.conv_tail);
            check_device(memcpy_d2d(static_cast<uint8_t*>(snapshot.state.data) + state_offset,
                                    static_cast<const uint8_t*>(layer.linear.state.data) +
                                        recurrent_slot_offset(layer.linear.state, slot_id),
                                    layer_state_bytes),
                         "Qwen recurrent state snapshot copy");
            state_offset += layer_state_bytes;
            if (layer_tail_bytes != 0) {
                check_device(memcpy_d2d(static_cast<uint8_t*>(snapshot.conv_tail.data) + tail_offset,
                                        static_cast<const uint8_t*>(layer.linear.conv_tail.data) +
                                            recurrent_slot_offset(layer.linear.conv_tail, slot_id),
                                        layer_tail_bytes),
                             "Qwen convolution tail snapshot copy");
                tail_offset += layer_tail_bytes;
            }
        }
        return snapshot;
    }

    void restore_recurrent_state(const QwenRecurrentSnapshot& snapshot,
                                 bool restore_target_hidden = true,
                                 int slot_id = 0) {
        size_t expected_state_bytes = 0;
        size_t expected_tail_bytes = 0;
        for (const DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            expected_state_bytes += recurrent_slot_bytes(layer.linear.state);
            expected_tail_bytes += recurrent_slot_bytes(layer.linear.conv_tail);
        }
        if (snapshot.state.nbytes != expected_state_bytes ||
            snapshot.conv_tail.nbytes != expected_tail_bytes) {
            throw std::runtime_error("Qwen recurrent snapshot extent mismatch");
        }
        if (restore_target_hidden && mtp_enabled && snapshot.position > 0) {
            const size_t hidden_bytes = static_cast<size_t>(config.hidden_size) *
                sizeof(uint16_t);
            if (snapshot.target_hidden.nbytes != hidden_bytes) {
                throw std::runtime_error(
                    "Qwen MTP snapshot target hidden extent mismatch");
            }
            allocate_half(target_last_hidden, config.hidden_size,
                          {config.hidden_size});
            check_device(memcpy_d2d(target_last_hidden.data, snapshot.target_hidden.data,
                                    hidden_bytes),
                         "Qwen target hidden snapshot restore");
            has_target_last_hidden = true;
            mtp_seed_ready = false;
            mtp_position = snapshot.position;
        } else if (snapshot.position == 0) {
            has_target_last_hidden = false;
            mtp_seed_ready = false;
            mtp_position = 0;
        }
        if (dspark_enabled) {
            // DSpark K/V is position-indexed like target GQA. Cropping only moves
            // the logical committed boundary; replay overwrites the suffix.
            if (snapshot.position <= dspark->committed_position()) {
                dspark->crop_context(snapshot.position);
            } else {
                throw std::runtime_error(
                    "Qwen DSpark snapshot exceeds committed context");
            }
        }
        if (dflash2_enabled) {
            if (snapshot.position <= dflash2->committed_position()) {
                dflash2->crop_context(snapshot.position);
            } else {
                throw std::runtime_error(
                    "Qwen DFlash2 snapshot exceeds committed context");
            }
        }
        size_t state_offset = 0;
        size_t tail_offset = 0;
        for (DeviceLayer& layer : layers) {
            if (layer.linear.state.data == nullptr) continue;
            // Restores into the requested slot, matching capture_recurrent_state.
            const size_t layer_state_bytes = recurrent_slot_bytes(layer.linear.state);
            const size_t layer_tail_bytes = recurrent_slot_bytes(layer.linear.conv_tail);
            check_device(memcpy_d2d(static_cast<uint8_t*>(layer.linear.state.data) +
                                        recurrent_slot_offset(layer.linear.state, slot_id),
                                    static_cast<const uint8_t*>(snapshot.state.data) + state_offset,
                                    layer_state_bytes),
                         "Qwen recurrent state snapshot restore");
            state_offset += layer_state_bytes;
            if (layer_tail_bytes != 0) {
                check_device(memcpy_d2d(static_cast<uint8_t*>(layer.linear.conv_tail.data) +
                                            recurrent_slot_offset(layer.linear.conv_tail, slot_id),
                                        static_cast<const uint8_t*>(snapshot.conv_tail.data) + tail_offset,
                                        layer_tail_bytes),
                             "Qwen convolution tail snapshot restore");
                tail_offset += layer_tail_bytes;
            }
        }
    }

    // Keeps snapshots ordered and bounded. The newest position wins because a
    // monotonically growing prompt resumes from the deepest available point.
    void record_snapshot(int position, const QwenForwardResult* result = nullptr,
                         bool periodic = false, int slot_id = 0) {
        if (!options.prefix_cache ||
            options.state_snapshot_interval_tokens <= 0 ||
            options.max_state_snapshots <= 0 || position <= 0) {
            return;
        }
        if (!has_recurrent_state() && !mtp_enabled && !dspark_enabled) return;
        std::vector<QwenRecurrentSnapshot>& ring = prefix_for(slot_id).snapshots;
        for (QwenRecurrentSnapshot& entry : ring) {
            if (entry.position != position) continue;
            entry.periodic = entry.periodic || periodic;
            if (result != nullptr && !entry.has_result) {
                entry.result = *result;
                entry.has_result = true;
            }
            return;
        }
        ring.push_back(capture_recurrent_state(position, result, periodic, slot_id));
        std::sort(ring.begin(), ring.end(),
                  [](const QwenRecurrentSnapshot& a,
                     const QwenRecurrentSnapshot& b) {
                      return a.position < b.position;
                  });
        // Preserve periodic coverage. Request-boundary snapshots improve exact
        // repeats but are expendable; evict their oldest entries first.
        // The budget is per slot, so a batch cannot let one sequence starve
        // another's rollback points.
        while (static_cast<int>(ring.size()) > options.max_state_snapshots) {
            auto evict = std::find_if(
                ring.begin(), ring.end(),
                [](const QwenRecurrentSnapshot& entry) {
                    return !entry.periodic;
                });
            if (evict == ring.end()) evict = ring.begin();
            ring.erase(evict);
        }
    }

    // Deepest snapshot at or before limit, or nullptr for the empty state.
    const QwenRecurrentSnapshot* snapshot_at_or_before(int limit, int slot_id = 0) {
        const QwenRecurrentSnapshot* best = nullptr;
        for (const QwenRecurrentSnapshot& entry : prefix_for(slot_id).snapshots) {
            if (entry.position <= limit &&
                (best == nullptr || entry.position > best->position)) {
                best = &entry;
            }
        }
        return best;
    }

    // A prompt result needs the hidden/logits for its final token. If the
    // chosen state is already after that token, use an earlier snapshot so the
    // final token is executed again rather than returning stale logits.
    const QwenRecurrentSnapshot* snapshot_strictly_before(int limit, int slot_id = 0) {
        const QwenRecurrentSnapshot* best = nullptr;
        for (const QwenRecurrentSnapshot& entry : prefix_for(slot_id).snapshots) {
            if (entry.position < limit &&
                (best == nullptr || entry.position > best->position)) {
                best = &entry;
            }
        }
        return best;
    }

    // Totalled across slots: this is reported as engine-wide memory use.
    uint64_t snapshot_bytes() const {
        uint64_t total = 0;
        for (const SlotPrefixState& slot : slot_prefix) {
            for (const QwenRecurrentSnapshot& entry : slot.snapshots) {
                total += entry.bytes();
            }
        }
        return total;
    }

    int snapshot_count(int slot_id) {
        return static_cast<int>(prefix_for(slot_id).snapshots.size());
    }

    // Snapshots let a later shorter-prefix or branched request resume mid-prompt,
    // but a snapshot position splits the prefill chunk, and a short chunk is far
    // less efficient per token. So a snapshot grid finer than the chunk is not
    // free: it silently caps the chunk size the prompt actually runs at.
    //
    // Measured on the real 64-layer TP4 model, chunk 4096, FP16 cache, identical
    // generated tokens throughout:
    //
    //   context   dense grid split   grid at chunk   prefill
    //     8192      1395 tok/s         1799 tok/s     1.29x
    //    65536      1415 tok/s         1457 tok/s     1.03x
    //
    // The 8K case is the extreme one: the 256-token early grid cut the first 4096
    // tokens into 16 chunks costing 3.59 s where a single 4096-row chunk does the
    // same work in 1.40 s. Both grids are therefore rounded up to the chunk, and
    // resume granularity becomes exactly the chunk boundary.
    int snapshot_interval_tokens() const {
        const int interval = options.state_snapshot_interval_tokens;
        if (interval <= 0) return 0;
        const int chunk = std::max(1, options.prefill_chunk_tokens);
        if (interval >= chunk) return interval;
        // Round up to a whole number of chunks so every snapshot position is also
        // a chunk boundary and no chunk is ever split.
        return ((chunk + interval - 1) / interval) * interval;
    }

    int early_snapshot_interval() const {
        const int interval = snapshot_interval_tokens();
        if (interval <= 0) return 0;
        const int early = std::min(256, interval);
        if (options.prefill_chunk_tokens > early) return 0;
        return early;
    }

    bool is_periodic_snapshot_position(int position) const {
        const int interval = snapshot_interval_tokens();
        if (interval <= 0 || position <= 0) return false;
        if (position % interval == 0) return true;
        const int early_interval = early_snapshot_interval();
        if (early_interval <= 0) return false;
        return position <= 4096 && position % early_interval == 0;
    }

    int next_periodic_snapshot_after(int position) const {
        const int interval = snapshot_interval_tokens();
        if (interval <= 0) return 0;
        const int next_regular = ((position / interval) + 1) * interval;
        if (position >= 4096) return next_regular;
        const int early_interval = early_snapshot_interval();
        if (early_interval <= 0) return next_regular;
        const int next_early = ((position / early_interval) + 1) * early_interval;
        return std::min(next_regular, next_early);
    }

    void drop_snapshots_after(int position, int slot_id = 0) {
        std::vector<QwenRecurrentSnapshot>& ring = prefix_for(slot_id).snapshots;
        ring.erase(
            std::remove_if(ring.begin(), ring.end(),
                           [position](const QwenRecurrentSnapshot& entry) {
                               return entry.position > position;
                           }),
            ring.end());
    }

    // Opt-in prefill phase attribution. Each scope synchronises the device, so it
    // is only ever enabled for profiling runs; the default path adds no sync.
    bool phase_profile = qwen_env_enabled("QWEN_PHASE_PROFILE");
    std::map<std::string, double> phase_seconds;
    std::map<std::string, uint64_t> phase_calls;

    class PhaseScope {
    public:
        PhaseScope(Impl* owner, std::string name) : owner_(owner), name_(std::move(name)) {
            // The profiler range is pushed without any synchronization, so a
            // capture can attribute kernels to a site by launch correlation while
            // the stream pipeline stays exactly as production runs it. Only the
            // phase_profile timer needs the device drained.
            if (owner_->range_profile) {
                device_range_push(name_.c_str());
                range_pushed_ = true;
            }
            if (owner_->phase_profile) {
                device_synchronize();
                started_ = std::chrono::steady_clock::now();
            }
        }
        ~PhaseScope() {
            if (owner_->phase_profile) {
                device_synchronize();
                owner_->phase_seconds[name_] += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started_).count();
                ++owner_->phase_calls[name_];
            }
            if (range_pushed_) device_range_pop();
        }
        PhaseScope(const PhaseScope&) = delete;
        PhaseScope& operator=(const PhaseScope&) = delete;

    private:
        Impl* owner_;
        std::string name_;
        std::chrono::steady_clock::time_point started_;
        bool range_pushed_ = false;
    };

    class RangeScope {
    public:
        explicit RangeScope(const char* name) { device_range_push(name); }
        ~RangeScope() { device_range_pop(); }
        RangeScope(const RangeScope&) = delete;
        RangeScope& operator=(const RangeScope&) = delete;
    };

    // Application-level profiler ranges are opt-in: they only push/pop host-side
    // markers so a Nsight capture can separate warmup, prefill and decode
    // without the CUDA synchronization that `QWEN_PHASE_PROFILE` forces.
    bool range_profile = qwen_env_enabled("QWEN_RANGE_PROFILE") ||
                         qwen_env_enabled("QWEN_NVTX_PROFILE");

    void report_phase_profile(const char* tag) const {
        if (!phase_profile) return;
        double total = 0.0;
        for (const auto& entry : phase_seconds) total += entry.second;
        for (const auto& entry : phase_seconds) {
            std::cout << "qwen_phase tag=" << tag << " rank=" << options.tp_rank
                      << " phase=" << entry.first
                      << " seconds=" << entry.second
                      << " calls=" << phase_calls.at(entry.first)
                      << " share=" << (total > 0.0 ? entry.second / total : 0.0)
                      << "\n";
        }
        std::cout << "qwen_phase tag=" << tag << " rank=" << options.tp_rank
                  << " phase=TOTAL seconds=" << total << "\n";
        std::cout.flush();
    }

    // Number of row slices used by the overlapped projection+all-reduce path.
    // 0 or 1 disables slicing entirely and keeps the serial behaviour.
    //
    // Four slices is the default. Measured serially on the real 64-layer TP4
    // checkpoint with the wider q64 attention tile in place, comm stream on, and
    // both the mlp.down and lin.out sites overlapped:
    //
    //   context   serial    slices=4   slices=8   prefill
    //    65536    1275.7     1325.8     1304.1     1.04x
    //
    // Eight slices is worse than four: the extra launches cost more than the
    // finer-grained overlap recovers. Phase profiling attributes the win to
    // tp_all_reduce dropping 14.64 -> 2.88 s across the whole 65K prefill.
    // The collective is bytes-bound at ~8.8 GB/s of ring bandwidth per rank, so
    // it cannot be made faster, only hidden. Set 0 or 1 to disable.
    int comm_overlap_slices() const {
        static const int slices =
            qwen_env_int("QWEN_COMM_OVERLAP_SLICES", 4);
        return slices;
    }

    // Pipelined projection and all-reduce.
    //
    // The collective is bytes-bound, not launch-bound: raising the prefill chunk
    // from 512 to 4096 cut the call count 3.1x (9288 -> 2967) but the time only
    // 6% (7.72 -> 7.24 s), and ar.mlp moves 41.9 MB in 2.43 ms, about 26 GB/s of
    // ring bandwidth. So it cannot be made faster, only hidden.
    //
    // This splits the output rows into slices. After slice i's GEMM is issued,
    // its all-reduce goes on the communication stream while slice i+1's GEMM runs
    // on the compute stream. Only the final slice's collective is exposed. Each
    // slice is an independent contiguous row range, and NCCL reduces each element
    // over the same four ranks in the same order regardless of slicing, so the
    // reduction itself is unchanged.
    //
    // The projection is a different matter: slicing changes the GEMM's batch
    // dimension, and cuBLAS may pick a different algorithm for a different batch,
    // so the projection output is not guaranteed bit-identical. Slices are
    // therefore kept at or above the batch>=96 cuBLAS dispatch gate, and the real
    // token-parity gate is what decides whether this ships.
    //
    // Requires the row-major [rows, hidden] layout: a row slice must be a
    // contiguous byte range in both the projection output and the reduced buffer.
    bool projection_all_reduce_overlapped(
            const DeviceLinear& linear, const uint16_t* input, uint16_t* output,
            int rows, int hidden, const char* proj_site, const char* ar_site) {
        const int slices = comm_overlap_slices();
        if (options.tp_world == 1 || slices <= 1 || rows < slices * 2) {
            return false;
        }
#ifndef DSV4_HAVE_TP_COMM
        return false;
#else
        if (!use_nccl_comm_stream || options.nccl_id_path.empty()) return false;
        // Rows per slice, rounded up so the tail slice is the short one. Below the
        // cuBLAS batch>=96 gate the projection would fall back to the handwritten
        // tile kernel, which is 3-5x slower and would cost far more than the
        // collective being hidden, so refuse to slice that finely.
        const int per = (rows + slices - 1) / slices;
        if (per < 96) return false;
        ensure_nccl_comm_stream();
        const int slice_count = (rows + per - 1) / per;
        ensure_comm_slice_events(slice_count);
        int index = 0;
        bool pending = false;
        int pending_index = 0;
        for (int start = 0; start < rows; start += per, ++index) {
            const int count = std::min(per, rows - start);
            const size_t offset = static_cast<size_t>(start) * hidden;
            projection(linear, input + static_cast<size_t>(start) * linear_columns(linear),
                       output + offset, count, proj_site);
            // Wait for the previous slice's collective only now, so it had the
            // whole of this slice's GEMM to run underneath.
            if (pending) {
                check_device(stream_wait_event(
                    nullptr, comm_slice_done[pending_index]),
                    "Qwen overlapped all-reduce wait");
                pending = false;
            }
            // Deliberately no PhaseScope around the collective. PhaseScope calls
            // device_synchronize() on entry and exit, which would drain the
            // comm stream and serialize exactly the pipeline being built, so a
            // profiled run would report a slowdown caused by the profiler. The
            // enclosing per-layer scopes still attribute the exposed cost.
            check_device(event_record(comm_slice_ready[index], nullptr),
                         "Qwen overlapped ready event");
            check_device(stream_wait_event(
                nccl_comm_stream, comm_slice_ready[index]),
                "Qwen overlapped comm wait");
            tp_all_reduce_sum_f16_inplace(
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path.c_str(), output + offset,
                count * hidden, nccl_comm_stream);
            check_device(event_record(comm_slice_done[index], nccl_comm_stream),
                         "Qwen overlapped done event");
            pending = true;
            pending_index = index;
        }
        if (pending) {
            check_device(stream_wait_event(
                nullptr, comm_slice_done[pending_index]),
                "Qwen overlapped final all-reduce wait");
        }
        return true;
#endif
    }

    static int linear_columns(const DeviceLinear& linear) {
        return static_cast<int>(linear.weight.shape[1]);
    }

    void all_reduce_half(uint16_t* values, int count, const char* site = "other") {
        PhaseScope scope(this, std::string("ar.") + site);
        if (options.tp_world == 1) return;
#ifdef DSV4_HAVE_TP_COMM
        if (options.nccl_id_path.empty()) {
            throw std::runtime_error("Qwen TP requires --nccl-id-path");
        }
        {
            PhaseScope scope(this, "tp_all_reduce");
            if (use_nccl_comm_stream) {
                begin_nccl_collective();
                tp_all_reduce_sum_f16_inplace(
                    options.tp_world, options.tp_rank, options.device,
                    options.nccl_id_path.c_str(), values, count,
                    nccl_comm_stream);
                end_nccl_collective();
            } else {
                tp_all_reduce_sum_f16_inplace(
                    options.tp_world, options.tp_rank, options.device,
                    options.nccl_id_path.c_str(), values, count);
            }
        }
#else
        (void)values;
        (void)count;
        throw std::runtime_error("Qwen TP requires an NCCL-enabled build");
#endif
    }

    void count_active_linear(QwenLinearKind kind) {
        switch (kind) {
            case QwenLinearKind::DenseF16:
                ++active_linear_kinds.dense_f16;
                break;
            case QwenLinearKind::Fp8Block128:
                ++active_linear_kinds.fp8_block128;
                break;
            case QwenLinearKind::Fp8Channel:
                ++active_linear_kinds.fp8_channel;
                break;
            case QwenLinearKind::NvFp4Group16:
                ++active_linear_kinds.nvfp4_group16;
                break;
        }
        telemetry.active_linear_kinds = active_linear_kinds;
    }

#ifndef POCKET_BACKEND_ASCEND
    void allocate_nvfp4_q8(int rows, int columns) {
        allocate(nvfp4_q8, static_cast<size_t>(rows) * columns,
                 {static_cast<uint64_t>(rows),
                  static_cast<uint64_t>(columns)}, SafeDType::I8);
        allocate_float(nvfp4_q8_scale,
                       static_cast<size_t>(rows) * (columns / 32),
                       {static_cast<uint64_t>(rows),
                        static_cast<uint64_t>(columns / 32)});
        telemetry.nvfp4_q8_workspace_peak_bytes = std::max(
            telemetry.nvfp4_q8_workspace_peak_bytes,
            nvfp4_q8.capacity + nvfp4_q8_scale.capacity);
    }

    bool nvfp4_integer_projection(const DeviceLinear& linear,
                                  const uint16_t* input, uint16_t* output,
                                  int rows, bool wmma) {
        const int output_rows = static_cast<int>(linear.logical_shape[0]);
        const int columns = static_cast<int>(linear.logical_shape[1]);
        allocate_nvfp4_q8(rows, columns);
        if (!qwen_nvfp4_quantize_q8_group32_f16_cuda(
                input, static_cast<int8_t*>(nvfp4_q8.data),
                nvfp4_q8_scale.f32_data(), rows, columns, columns)) {
            return false;
        }
        if (wmma && qwen_nvfp4_wide_n64_enabled(rows)) {
            return qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
                static_cast<const int8_t*>(nvfp4_q8.data),
                nvfp4_q8_scale.f32_data(), linear.weight.u8_data(), output,
                rows, output_rows, columns, columns, output_rows,
                columns / 64, linear.weight_global_factor);
        }
        return wmma
            ? qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
                  static_cast<const int8_t*>(nvfp4_q8.data),
                  nvfp4_q8_scale.f32_data(), linear.weight.u8_data(), output,
                  rows, output_rows, columns, columns, output_rows,
                  columns / 64, linear.weight_global_factor)
            : qwen_nvfp4_group16_matmul_q8_f16_cuda(
                  static_cast<const int8_t*>(nvfp4_q8.data),
                  nvfp4_q8_scale.f32_data(), linear.weight.u8_data(), output,
                  rows, output_rows, columns, columns, output_rows,
                  columns / 64, linear.weight_global_factor);
    }

    bool nvfp4_shared_q8_swiglu_projection(
        const DeviceLinear& gate, const DeviceLinear& up,
        const uint16_t* input, uint16_t* gate_output,
        uint16_t* up_output, int rows) {
        const int output_rows = static_cast<int>(gate.logical_shape[0]);
        const int columns = static_cast<int>(gate.logical_shape[1]);
        allocate_nvfp4_q8(rows, columns);
        if (!qwen_nvfp4_quantize_q8_group32_f16_cuda(
                input, static_cast<int8_t*>(nvfp4_q8.data),
                nvfp4_q8_scale.f32_data(), rows, columns, columns)) {
            return false;
        }
        const int8_t* q8 = static_cast<const int8_t*>(nvfp4_q8.data);
        const float* q8_scale = nvfp4_q8_scale.f32_data();
        if (qwen_nvfp4_wide_n64_enabled(rows)) {
            return qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
                       q8, q8_scale, gate.weight.u8_data(), gate_output,
                       rows, output_rows, columns, columns, output_rows,
                       columns / 64, gate.weight_global_factor) &&
                   qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
                       q8, q8_scale, up.weight.u8_data(), up_output,
                       rows, output_rows, columns, columns, output_rows,
                       columns / 64, up.weight_global_factor);
        }
        return qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
                   q8, q8_scale, gate.weight.u8_data(), gate_output,
                   rows, output_rows, columns, columns, output_rows,
                   columns / 64, gate.weight_global_factor) &&
               qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
                   q8, q8_scale, up.weight.u8_data(), up_output,
                   rows, output_rows, columns, columns, output_rows,
                   columns / 64, up.weight_global_factor);
    }

    bool nvfp4_fused_swiglu_projection(
        const DeviceLinear& gate, const DeviceLinear& up,
        const uint16_t* input, uint16_t* output, int rows) {
        const int output_rows = static_cast<int>(gate.logical_shape[0]);
        const int columns = static_cast<int>(gate.logical_shape[1]);
        allocate_nvfp4_q8(rows, columns);
        if (!qwen_nvfp4_quantize_q8_group32_f16_cuda(
                input, static_cast<int8_t*>(nvfp4_q8.data),
                nvfp4_q8_scale.f32_data(), rows, columns, columns)) {
            return false;
        }
        return qwen_nvfp4_group16_swiglu_q8_wmma_f16_cuda(
            static_cast<const int8_t*>(nvfp4_q8.data),
            nvfp4_q8_scale.f32_data(), gate.weight.u8_data(),
            gate.weight_global_factor, up.weight.u8_data(),
            up.weight_global_factor, output, rows, output_rows, columns,
            columns, output_rows, columns / 64);
    }
#endif

    void projection(const DeviceLinear& linear, const uint16_t* input,
                    uint16_t* output, int rows, const char* site = "other") {
        PhaseScope scope(this, std::string(rows == 1 ? "pd." : "pr.") + site);
        if (linear.logical_shape.size() != 2) {
            throw std::runtime_error("Qwen linear has invalid logical shape");
        }
        const int output_rows = static_cast<int>(linear.logical_shape[0]);
        const int columns = static_cast<int>(linear.logical_shape[1]);
#ifdef POCKET_BACKEND_ASCEND
        // Only the dense FP16 kind has an Ascend implementation. The quantized
        // kinds are rejected here rather than left to a runtime branch: a
        // reference to their CUDA kernels in this object would make the whole
        // executable require the CUDA toolchain to link.
        if (linear.kind != QwenLinearKind::DenseF16) {
            throw std::runtime_error(
                std::string("Qwen Ascend path is not implemented for ") +
                qwen_linear_kind_name(linear.kind));
        }
        require_launch(qwen_fp16_matmul_rows_f16(
            input, linear.weight.f16_data(), output, rows, output_rows,
            columns, columns, output_rows, columns),
            "FP16 activation/weight projection");
        return;
#else
        if (linear.kind == QwenLinearKind::Fp8Block128) {
            if (rows == 1) {
                require_launch(qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    input, linear.weight.fp8_data(), linear.scale.f16_data(),
                    output, output_rows, columns, columns,
                    static_cast<int>(linear.scale.shape[1])),
                    "FP8 FP16-activation decode projection");
            } else if (rows <= 8 && linear.verify_weight.data != nullptr) {
                require_launch(qwen_fp16_matmul_rows_f16_cublas_cuda(
                    input, linear.verify_weight.f16_data(), output, rows,
                    output_rows, columns, columns, output_rows, columns),
                    "resident FP16 verify projection");
            } else {
                require_launch(qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                    input, linear.weight.fp8_data(), linear.scale.f16_data(),
                    output, rows, output_rows, columns, columns, output_rows,
                    columns, static_cast<int>(linear.scale.shape[1])),
                    "FP8 FP16-activation projection");
            }
        } else if (linear.kind == QwenLinearKind::Fp8Channel) {
            require_launch(rows == 1
                ? qwen_fp8_e4m3_channel_matvec_f16_cuda(
                      input, linear.weight.fp8_data(), linear.scale.f16_data(),
                      output, output_rows, columns, columns)
                : qwen_fp8_e4m3_channel_matmul_rows_f16_cuda(
                      input, linear.weight.fp8_data(), linear.scale.f16_data(),
                      output, rows, output_rows, columns, columns, output_rows,
                      columns),
                "channel FP8 FP16-activation projection");
        } else if (linear.kind == QwenLinearKind::NvFp4Group16) {
            const char* requested = std::getenv("DSV4_QWEN_NVFP4");
            const bool reference = requested != nullptr &&
                std::strcmp(requested, "reference") == 0;
            const bool explicit_dp4a = requested != nullptr &&
                std::strcmp(requested, "dp4a") == 0;
            const bool explicit_wmma = requested != nullptr &&
                std::strcmp(requested, "wmma") == 0;
            const bool automatic = requested == nullptr || *requested == '\0' ||
                std::strcmp(requested, "auto") == 0;
            if (!reference && !explicit_dp4a && !explicit_wmma && !automatic) {
                throw std::runtime_error(
                    "DSV4_QWEN_NVFP4 must be auto, dp4a, wmma, or reference");
            }
            const int blocks_per_row = columns / 64;
            if (!reference) {
                const bool use_wmma = explicit_wmma || (automatic && rows >= 8);
                const bool use_wide =
                    use_wmma && qwen_nvfp4_wide_n64_enabled(rows);
                require_launch(nvfp4_integer_projection(
                    linear, input, output, rows, use_wmma),
                    use_wide ? "NVFP4 group-16 Q8 wide-N64 projection" :
                    use_wmma ? "NVFP4 group-16 Q8 WMMA projection" :
                               "NVFP4 group-16 Q8 DP4A projection");
            } else {
                require_launch(rows == 1
                    ? qwen_nvfp4_group16_matvec_f16_cuda(
                          input, linear.weight.u8_data(), output, output_rows,
                          columns, blocks_per_row, linear.weight_global_factor)
                    : qwen_nvfp4_group16_matmul_rows_f16_cuda(
                          input, linear.weight.u8_data(), output, rows,
                          output_rows, columns, columns, output_rows,
                          blocks_per_row, linear.weight_global_factor),
                    "NVFP4 group-16 reference projection");
            }
        } else if (linear.kind == QwenLinearKind::DenseF16) {
            // The fused DeltaNet a/b matrix is only 24 rows per TP4 rank. At
            // verify width 8, tensor cores are ~5x faster than the generic
            // warp-per-output-row reduction. Decode keeps its established
            // reduction order, and wider prefill remains on the existing path.
            const bool verify_cublas = rows >= 4 && rows <= 8 &&
                output_rows <= 32 && columns % 8 == 0 &&
                qwen_env_enabled_default("QWEN_VERIFY_SMALL_FP16_CUBLAS");
            if (verify_cublas) {
                require_launch(qwen_fp16_matmul_rows_f16_cublas_cuda(
                    input, linear.weight.f16_data(), output, rows, output_rows,
                    columns, columns, output_rows, columns),
                    "small FP16 verify cuBLAS projection");
            } else {
                require_launch(qwen_fp16_matmul_rows_f16(
                    input, linear.weight.f16_data(), output, rows, output_rows,
                    columns, columns, output_rows, columns),
                    "FP16 activation/weight projection");
            }
        } else {
            throw std::runtime_error(
                std::string("Qwen linear CUDA path is not implemented for ") +
                qwen_linear_kind_name(linear.kind));
        }
#endif
    }

    void norm(const QwenDeviceTensor& gamma, const uint16_t* input,
              uint16_t* output, int rows, int columns) {
        PhaseScope scope(this, "norm");
        require_launch(qwen_rmsnorm_fp16_gamma_rows_f16(
            input, gamma.f16_data(), output, rows, columns,
            static_cast<float>(config.rms_norm_eps)), "Qwen FP16 RMSNorm");
    }

    void add(uint16_t* output, const uint16_t* input, int count) {
        PhaseScope scope(this, "add");
        require_launch(qwen_add_inplace_f16(output, input, count),
                       "Qwen FP16 residual add");
    }

    void begin_workspace() { workspace.begin(); }

    QwenDeviceTensor& workspace_half(size_t elements,
                                     const std::vector<uint64_t>& shape) {
        return workspace.half_tensor(elements, shape);
    }

    QwenDeviceTensor& workspace_float(size_t elements,
                                      const std::vector<uint64_t>& shape) {
        return workspace.float_tensor(elements, shape);
    }

    void linear_attention(DeviceLayer& layer, const uint16_t* hidden,
                          uint16_t* output, int rows, int position_offset,
                          int slot_id) {
        // Phase 3.4: Every kernel below advances this slot's own recurrent
        // state. Resolved once here so the many call sites cannot disagree.
        const size_t state_offset = recurrent_slot_offset(slot_id, layer.linear.state);
        const size_t tail_offset = recurrent_slot_offset(slot_id, layer.linear.conv_tail);
        float* const state = layer.linear.state.f32_data() + state_offset;
        uint16_t* const conv_tail = layer.linear.conv_tail.f16_data() + tail_offset;
        const int key_heads = static_cast<int>(
            config.linear_attention.key_heads / options.tp_world);
        const int value_heads = static_cast<int>(
            config.linear_attention.value_heads / options.tp_world);
        const int key_dim = key_heads *
            static_cast<int>(config.linear_attention.key_head_dim);
        const int value_dim = value_heads *
            static_cast<int>(config.linear_attention.value_head_dim);
        const int packed_dim = 2 * key_dim + value_dim;
        const int kernel = static_cast<int>(config.linear_attention.conv_kernel_dim);
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t packed_elements = static_cast<size_t>(rows) * packed_dim;
        const size_t key_elements = static_cast<size_t>(rows) * key_dim;
        const size_t value_elements = static_cast<size_t>(rows) * value_dim;
        const size_t gate_elements = static_cast<size_t>(rows) * value_heads;
        QwenDeviceTensor& packed = workspace_half(
            packed_elements, {static_cast<uint64_t>(rows),
                              static_cast<uint64_t>(packed_dim)});
        QwenDeviceTensor& convolved = workspace_half(packed_elements, packed.shape);
        QwenDeviceTensor& q = workspace_half(
            key_elements, {static_cast<uint64_t>(rows),
                           static_cast<uint64_t>(key_dim)});
        QwenDeviceTensor& k = workspace_half(key_elements, q.shape);
        QwenDeviceTensor& v = workspace_half(
            value_elements, {static_cast<uint64_t>(rows),
                             static_cast<uint64_t>(value_dim)});
        QwenDeviceTensor& a = workspace_half(
            gate_elements, {static_cast<uint64_t>(rows),
                            static_cast<uint64_t>(value_heads)});
        QwenDeviceTensor& b = workspace_half(gate_elements, a.shape);
        QwenDeviceTensor& gates = workspace_half(gate_elements, a.shape);
        QwenDeviceTensor& beta = workspace_half(gate_elements, a.shape);
        QwenDeviceTensor& core = workspace_half(value_elements, v.shape);
        QwenDeviceTensor& z = workspace_half(value_elements, v.shape);
        QwenDeviceTensor& normalized = workspace_half(value_elements, v.shape);
        // FlashQLA is an SM75 register/subgroup layout of the same recurrence,
        // not a distinct operation, so the Ascend build simply does not have the
        // selector: the normalized sequence kernel below produces the same
        // result from the same FP32 normalized Q/K.
#ifndef POCKET_BACKEND_ASCEND
        QwenDeviceTensor* q_normalized = nullptr;
        QwenDeviceTensor* k_normalized = nullptr;
        // Every sequence-recurrence kernel below reads `rows` as consecutive
        // tokens of one stream. A batched decode step is the opposite: `rows`
        // independent sequences each advancing one token, so none of them apply
        // and the batched step kernel handles it instead.
        const bool flashqla = batch_rows == nullptr && rows >= 8 &&
            qwen_env_enabled_default("QWEN_GATED_DELTA_FLASHQLA_SM75");
        if (flashqla) {
            q_normalized = &workspace_float(
                key_elements,
                {static_cast<uint64_t>(rows), static_cast<uint64_t>(key_dim)});
            k_normalized = &workspace_float(key_elements, q_normalized->shape);
        }
#endif

#ifndef POCKET_BACKEND_ASCEND
        bool dual_qkvz = rows == 1 &&
            layer.linear.qkv.kind == QwenLinearKind::Fp8Block128 &&
            layer.linear.z.kind == QwenLinearKind::Fp8Block128 &&
            qwen_env_enabled("QWEN_FUSE_QKVZ_DECODE");
#else
        const bool dual_qkvz = false;
#endif
#ifndef POCKET_BACKEND_ASCEND
        if (dual_qkvz) {
            PhaseScope dual_scope(this, "pd.lin.qkvz");
            require_launch(qwen_fp8_e4m3_fp16scale_matvec_dual_f16_cuda(
                hidden,
                layer.linear.qkv.weight.fp8_data(),
                layer.linear.qkv.scale.f16_data(), packed.f16_data(), packed_dim,
                hidden_size, static_cast<int>(layer.linear.qkv.scale.shape[1]),
                layer.linear.z.weight.fp8_data(),
                layer.linear.z.scale.f16_data(), z.f16_data(), value_dim,
                hidden_size, static_cast<int>(layer.linear.z.scale.shape[1]),
                hidden_size), "dual FP8 QKV/Z decode projection");
        } else
#endif
        {
            projection(layer.linear.qkv, hidden, packed.f16_data(), rows,
                       "lin.qkv");
        }
#ifndef POCKET_BACKEND_ASCEND
        if (batch_rows != nullptr) {
            // One token per sequence against that sequence's own tail. The
            // pointer is the arena base here, not this slot's rows, because the
            // kernel resolves the slot per row.
            require_launch(qwen_causal_depthwise_conv_silu_f16_batched_cuda(
                packed.f16_data(), layer.linear.conv.f16_data(),
                layer.linear.conv_tail.f16_data(), convolved.f16_data(),
                batch_rows->slot_ids, rows, packed_dim, kernel,
                slot_stride_elements(layer.linear.conv_tail)),
                "FP16 batched linear causal convolution");
        } else
#endif
        require_launch(qwen_causal_depthwise_conv_silu_f16(
            packed.f16_data(), layer.linear.conv.f16_data(),
            conv_tail, convolved.f16_data(), rows,
            packed_dim, kernel, true), "FP16 linear causal convolution");
        require_launch(qwen_split_packed_qkv_f16(
            convolved.f16_data(), q.f16_data(), k.f16_data(), v.f16_data(),
            rows, key_dim, value_dim), "FP16 linear QKV split");
        if (layer.linear.ab.weight.data != nullptr &&
            qwen_env_enabled_default("QWEN_FUSE_AB_PROJECTION")) {
            // One GEMM over the row-concatenated [2*gate, hidden] weight. The
            // output is [rows, 2*gate] interleaved per row, so a and b are
            // strided slices of it rather than contiguous halves.
            QwenDeviceTensor& ab = workspace_half(
                gate_elements * 2,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(value_heads * 2)});
            projection(layer.linear.ab, hidden, ab.f16_data(), rows, "lin.ab");
            require_launch(qwen_split_rows_pair_f16(
                ab.f16_data(), a.f16_data(), b.f16_data(), rows, value_heads),
                "FP16 fused a/b split");
        } else {
            projection(layer.linear.a, hidden, a.f16_data(), rows, "lin.a");
            projection(layer.linear.b, hidden, b.f16_data(), rows, "lin.b");
        }
        require_launch(qwen_linear_attn_gates_f16(
            a.f16_data(), b.f16_data(), layer.linear.a_log.f16_data(),
            layer.linear.dt_bias.f16_data(), gates.f16_data(), beta.f16_data(),
            rows, value_heads), "FP16 linear attention gates");
        const float q_scale = 1.0f / std::sqrt(
            static_cast<float>(config.linear_attention.key_head_dim));
        std::optional<PhaseScope> delta_scope;
        delta_scope.emplace(this, "gated_delta");
        bool sequenced = false;
#ifndef POCKET_BACKEND_ASCEND
        if (batch_rows != nullptr) {
            sequenced = qwen_gated_delta_step_batched_f16_cuda(
                layer.linear.state.f32_data(), q.f16_data(), k.f16_data(),
                v.f16_data(), gates.f16_data(), beta.f16_data(),
                core.f16_data(), batch_rows->slot_ids, rows, value_heads,
                key_heads,
                static_cast<int>(config.linear_attention.key_head_dim),
                static_cast<int>(config.linear_attention.value_head_dim),
                q_scale, slot_stride_elements(layer.linear.state));
            require_launch(sequenced, "FP16 batched linear recurrent state");
        } else if (flashqla) {
            // FlashQLA SM75 subgroup-sharded kernel. Still a serial recurrence,
            // but sharding the [128, 128] state over 16-lane subgroups replaces
            // the baseline's 128-element per-thread register vector: 1.85x on the
            // primitive and +5-7% real TP4 prefill at token-exact parity.
            // Consume the established FP32 normalized Q/K tensors so the norm
            // reduction order remains identical to the reference path.
            sequenced = qwen_normalize_gated_delta_qk_f16(
                q.f16_data(), k.f16_data(), q_normalized->f32_data(),
                k_normalized->f32_data(), rows, key_heads,
                static_cast<int>(config.linear_attention.key_head_dim)) &&
                qwen_gated_delta_flashqla_sm75_f16_cuda(
                    state, q_normalized->f32_data(),
                    k_normalized->f32_data(), v.f16_data(), gates.f16_data(),
                    beta.f16_data(), core.f16_data(), rows, value_heads,
                    key_heads,
                    static_cast<int>(config.linear_attention.key_head_dim),
                    static_cast<int>(config.linear_attention.value_head_dim),
                    q_scale);
        } else
#endif
        if (rows >= 5 && key_heads < value_heads &&
            qwen_env_enabled_default("QWEN_GATED_DELTA_PRENORMALIZE")) {
            QwenDeviceTensor& q_normalized = workspace_float(
                key_elements, {static_cast<uint64_t>(rows),
                               static_cast<uint64_t>(key_dim)});
            QwenDeviceTensor& k_normalized = workspace_float(
                key_elements, q_normalized.shape);
            sequenced = qwen_normalize_gated_delta_qk_f16(
                q.f16_data(), k.f16_data(), q_normalized.f32_data(),
                k_normalized.f32_data(), rows, key_heads,
                static_cast<int>(config.linear_attention.key_head_dim)) &&
                (qwen_env_enabled("QWEN_GATED_DELTA_SHARED_STATE")
                 ? qwen_gated_delta_sequence_normalized_shared_f16(
                        state, q_normalized.f32_data(),
                        k_normalized.f32_data(), v.f16_data(), gates.f16_data(),
                        beta.f16_data(), core.f16_data(), rows, value_heads,
                        key_heads,
                        static_cast<int>(config.linear_attention.key_head_dim),
                        static_cast<int>(config.linear_attention.value_head_dim),
                        q_scale)
                    : qwen_gated_delta_sequence_normalized_f16(
                        state, q_normalized.f32_data(),
                        k_normalized.f32_data(), v.f16_data(), gates.f16_data(),
                        beta.f16_data(), core.f16_data(), rows, value_heads,
                        key_heads,
                        static_cast<int>(config.linear_attention.key_head_dim),
                        static_cast<int>(config.linear_attention.value_head_dim),
                        q_scale));
        } else {
            // Also covers rows == 1. Decode used to fall through to the step
            // loop below, which round-trips the [128, 128] state through global
            // memory twice per token; the sequence kernel holds it in registers
            // and is bit-exact against the step kernel, so there is no reason to
            // reserve it for multi-token chunks. See the step-vs-sequence parity
            // check in test_qwen_half_ops.
            sequenced = qwen_gated_delta_sequence_f16(
                state, q.f16_data(), k.f16_data(),
                v.f16_data(), gates.f16_data(), beta.f16_data(),
                core.f16_data(), rows, value_heads, key_heads,
                static_cast<int>(config.linear_attention.key_head_dim),
                static_cast<int>(config.linear_attention.value_head_dim), q_scale);
        }
        for (int token = 0; !sequenced && token < rows; ++token) {
            require_launch(qwen_gated_delta_step_f16(
                state,
                q.f16_data() + static_cast<size_t>(token) * key_dim,
                k.f16_data() + static_cast<size_t>(token) * key_dim,
                v.f16_data() + static_cast<size_t>(token) * value_dim,
                gates.f16_data() + static_cast<size_t>(token) * value_heads,
                beta.f16_data() + static_cast<size_t>(token) * value_heads,
                core.f16_data() + static_cast<size_t>(token) * value_dim,
                value_heads, key_heads,
                static_cast<int>(config.linear_attention.key_head_dim),
                static_cast<int>(config.linear_attention.value_head_dim), q_scale),
                "FP16 linear recurrent state");
        }
        delta_scope.reset();
        if (!dual_qkvz) {
            projection(layer.linear.z, hidden, z.f16_data(), rows, "lin.z");
        }
        require_launch(qwen_gated_rmsnorm_fp16_gamma_rows_f16(
            core.f16_data(), layer.linear.norm.f16_data(), z.f16_data(),
            normalized.f16_data(), rows * value_heads,
            static_cast<int>(config.linear_attention.value_head_dim),
            static_cast<float>(config.rms_norm_eps)),
            "FP16 linear gated RMSNorm");
        // ar.lin.out is the second-largest collective (2.69 s at 32K) and runs on
        // the 48 DeltaNet layers, so it is the other worthwhile overlap site.
        if (!projection_all_reduce_overlapped(
                layer.linear.out, normalized.f16_data(), output, rows,
                hidden_size, "lin.out", "lin.out")) {
            projection(layer.linear.out, normalized.f16_data(), output, rows,
                       "lin.out");
            all_reduce_half(output, rows * hidden_size, "lin.out");
        }
        (void)position_offset;
    }

    void full_attention(DeviceLayer& layer, const uint16_t* hidden,
                        uint16_t* output, int rows, int position_offset,
                        int slot_id) {
        PhaseScope scope(this, "full_attention");
        const int q_heads = static_cast<int>(
            config.full_attention.num_heads / options.tp_world);
        const int kv_heads = static_cast<int>(
            config.full_attention.num_key_value_heads / options.tp_world);
        const int head_dim = static_cast<int>(config.full_attention.head_dim);
        const int attention_dim = q_heads * head_dim;
        const int q_projection_dim = attention_dim * 2;
        const size_t q_projection_elements =
            static_cast<size_t>(rows) * q_projection_dim;
        const size_t attention_elements =
            static_cast<size_t>(rows) * attention_dim;
        const size_t kv_elements =
            static_cast<size_t>(rows) * kv_heads * head_dim;
        QwenDeviceTensor& q_projection = workspace_half(
            q_projection_elements, {static_cast<uint64_t>(rows),
                                    static_cast<uint64_t>(q_projection_dim)});
        QwenDeviceTensor& q = workspace_half(
            attention_elements, {static_cast<uint64_t>(rows),
                                 static_cast<uint64_t>(attention_dim)});
        QwenDeviceTensor& gate = workspace_half(attention_elements, q.shape);
        QwenDeviceTensor& k = workspace_half(
            kv_elements, {static_cast<uint64_t>(rows),
                          static_cast<uint64_t>(kv_heads),
                          static_cast<uint64_t>(head_dim)});
        QwenDeviceTensor& v = workspace_half(kv_elements, k.shape);
        QwenDeviceTensor* kv_projection = nullptr;
        // Keep the candidate isolated to the fixed-width target verify batch;
        // prefill and one-row decode retain their established K/V projections.
        const bool fused_kv = rows > 1 && rows <= 8 &&
            layer.full.kv.weight.data != nullptr;
        if (fused_kv) {
            kv_projection = &workspace_half(
                static_cast<size_t>(rows) * 2 * kv_heads * head_dim,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(2 * kv_heads * head_dim)});
        }
        QwenDeviceTensor& q_norm = workspace_half(attention_elements, q.shape);
        QwenDeviceTensor& k_norm = workspace_half(kv_elements, k.shape);
        QwenDeviceTensor& attention = workspace_half(attention_elements, q.shape);
        QwenDeviceTensor& merged = workspace_half(attention_elements, q.shape);

        const int hidden_size = static_cast<int>(config.hidden_size);
#ifndef POCKET_BACKEND_ASCEND
        const bool grouped_qkv = rows == 1 && !fused_kv &&
            layer.full.q.kind == QwenLinearKind::Fp8Block128 &&
            layer.full.k.kind == QwenLinearKind::Fp8Block128 &&
            layer.full.v.kind == QwenLinearKind::Fp8Block128 &&
            qwen_env_enabled("QWEN_FUSE_FULL_QKV_DECODE");
        if (grouped_qkv) {
            PhaseScope sub(this, "pd.full.qkv");
            require_launch(qwen_fp8_e4m3_fp16scale_matvec_triple_f16_cuda(
                hidden, layer.full.q.weight.fp8_data(),
                layer.full.q.scale.f16_data(), q_projection.f16_data(),
                q_projection_dim, hidden_size,
                static_cast<int>(layer.full.q.scale.shape[1]),
                layer.full.k.weight.fp8_data(), layer.full.k.scale.f16_data(),
                k.f16_data(), kv_heads * head_dim, hidden_size,
                static_cast<int>(layer.full.k.scale.shape[1]),
                layer.full.v.weight.fp8_data(), layer.full.v.scale.f16_data(),
                v.f16_data(), kv_heads * head_dim, hidden_size,
                static_cast<int>(layer.full.v.scale.shape[1]), hidden_size),
                "triple FP8 full Q/K/V decode projection");
        } else
#endif
        {
            { PhaseScope sub(this, "full.q_proj"); projection(layer.full.q, hidden, q_projection.f16_data(), rows, "full.q"); }
            if (fused_kv) {
                { PhaseScope sub(this, "full.kv_proj"); projection(
                    layer.full.kv, hidden, kv_projection->f16_data(), rows,
                    "full.kv"); }
                require_launch(qwen_split_rows_pair_f16(
                    kv_projection->f16_data(), k.f16_data(), v.f16_data(), rows,
                    kv_heads * head_dim), "FP16 fused full K/V split");
            } else {
                { PhaseScope sub(this, "full.k_proj"); projection(layer.full.k, hidden, k.f16_data(), rows, "full.k"); }
                { PhaseScope sub(this, "full.v_proj"); projection(layer.full.v, hidden, v.f16_data(), rows, "full.v"); }
            }
        }
        require_launch(qwen_split_q_gate_f16(
            q_projection.f16_data(), q.f16_data(), gate.f16_data(), rows,
            q_heads, head_dim), "FP16 full Q/gate split");
        norm(layer.full.q_norm, q.f16_data(), q_norm.f16_data(),
             rows * q_heads, head_dim);
        norm(layer.full.k_norm, k.f16_data(), k_norm.f16_data(),
             rows * kv_heads, head_dim);
#ifndef POCKET_BACKEND_ASCEND
        if (batch_rows != nullptr) {
            // Each row is a different sequence at its own position, so the
            // rotation angle is per row rather than position_offset + row.
            require_launch(qwen_partial_rope_rows_f16_batched_cuda(
                q_norm.f16_data(), k_norm.f16_data(), batch_rows->positions,
                rows, static_cast<int>(config.partial_rotary_dim()),
                static_cast<float>(config.rope_theta), q_heads, kv_heads,
                head_dim), "FP16 batched partial RoPE");
        } else
#endif
        require_launch(qwen_partial_rope_rows_f16(
            q_norm.f16_data(), k_norm.f16_data(), position_offset, rows,
            static_cast<int>(config.partial_rotary_dim()),
            static_cast<float>(config.rope_theta), q_heads, kv_heads, head_dim),
            "FP16 partial RoPE");

        const QwenKvCacheDType cache_dtype = options.kv_cache_dtype;
        const int attention_window = options.attention_window;
        const int sink_tokens = options.attention_sink_tokens;
#ifdef POCKET_BACKEND_ASCEND
        // The engine constructor rejects every non-FP16 cache dtype on this
        // backend, so only the FP16 append and the three neutral attention
        // launchers are compiled. The quantized caches are not merely disabled:
        // naming their kernels here would put a CUDA dependency in this object.

        // Phase 3.3: Use slot_id parameter instead of global current_slot_id
        const size_t slot_offset = kv_slot_offset_elements(slot_id, kv_heads, head_dim);

        require_launch(qwen_append_kv_cache_f16(
            k_norm.f16_data(), v.f16_data(),
            layer.full.k_cache.f16_data() + slot_offset,
            layer.full.v_cache.f16_data() + slot_offset,
            rows, kv_heads, head_dim,
            position_offset, max_context), "append FP16 full KV cache");

        if (rows == 1) {
            const int context_length = position_offset + 1;
            QwenDeviceTensor& scores = workspace_float(
                static_cast<size_t>(q_heads) * context_length,
                {static_cast<uint64_t>(q_heads),
                 static_cast<uint64_t>(context_length)});
            require_launch(qwen_gqa_decode_attention_f16(
                q_norm.f16_data(),
                layer.full.k_cache.f16_data() + slot_offset,
                layer.full.v_cache.f16_data() + slot_offset,
                attention.f16_data(),
                scores.f32_data(), q_heads, kv_heads, head_dim,
                context_length, max_context), "decode FP16-cache GQA");
        } else if (rows <= 8) {
            const int context_length = position_offset + rows;
            // Same split geometry the CUDA verify path uses, so a verify block
            // reduces its partials in the same order on both backends.
            const int default_splits = context_length >= 1024 ? 64 : 32;
            int target_splits = qwen_env_int(
                "QWEN_GQA_VERIFY_SPLITS", default_splits);
            if (target_splits <= 0) target_splits = default_splits;
            const int verify_splits = std::max(
                1, std::min(target_splits, context_length));
            QwenDeviceTensor& partials = workspace_float(
                static_cast<size_t>(rows) * q_heads * verify_splits *
                    static_cast<size_t>(head_dim + 2),
                {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_heads),
                 static_cast<uint64_t>(verify_splits),
                 static_cast<uint64_t>(head_dim + 2)});
            PhaseScope sub(this, "full.attn_kernel");
            require_launch(qwen_gqa_verify_attention_f16(
                q_norm.f16_data(),
                layer.full.k_cache.f16_data() + slot_offset,
                layer.full.v_cache.f16_data() + slot_offset,
                attention.f16_data(),
                partials.f32_data(), rows, q_heads, kv_heads, head_dim,
                position_offset, max_context, verify_splits),
                "verify split FP16-cache GQA");
        } else {
            require_launch(qwen_gqa_prefill_attention_f16(
                q_norm.f16_data(),
                layer.full.k_cache.f16_data() + slot_offset,
                layer.full.v_cache.f16_data() + slot_offset,
                attention.f16_data(), rows,
                q_heads, kv_heads, head_dim, position_offset, max_context),
                "prefill FP16-cache GQA");
        }
        (void)sink_tokens;
#else
        // Phase 3.3: Use slot_id parameter instead of global current_slot_id
        const size_t slot_offset = kv_slot_offset_elements(slot_id, kv_heads, head_dim);
        // Distance between slots in the KV arena. kv_slot_offset_elements folds
        // this to zero for a single session, so the stride is recomputed rather
        // than divided out of it.
        const size_t kv_slot_stride = static_cast<size_t>(max_context) *
            kv_heads * head_dim;

        if (batch_rows != nullptr) {
            // Each row appends its new K/V at its own position in its own slot.
            // Under paging the slot stride is replaced by the block table, which
            // the caller has already grown and uploaded for these positions.
            require_launch(qwen_append_kv_cache_f16_batched_cuda(
                k_norm.f16_data(), v.f16_data(),
                layer.full.k_cache.f16_data(), layer.full.v_cache.f16_data(),
                rows, kv_heads, head_dim, batch_rows->positions,
                batch_rows->slot_ids, max_context, kv_slot_stride,
                block_table_data(), paged_block_size(),
                paged_blocks_per_seq()),
                "append batched FP16 full KV cache");
        } else if (kv_paged()) {
            // Single-sequence paged append: consecutive positions of one
            // sequence, scattered across the blocks its row names.
            require_launch(qwen_append_kv_cache_f16_paged_cuda(
                k_norm.f16_data(), v.f16_data(),
                layer.full.k_cache.f16_data(), layer.full.v_cache.f16_data(),
                rows, kv_heads, head_dim, position_offset,
                block_table_row(slot_id), paged_block_size()),
                "append paged FP16 full KV cache");
        } else if (cache_dtype == QwenKvCacheDType::Fp8) {
            require_launch(qwen_append_kv_cache_fp8_cuda(
                k_norm.f16_data(), v.f16_data(),
                layer.full.k_cache.fp8_data() + slot_offset,
                layer.full.v_cache.fp8_data() + slot_offset,
                layer.full.k_scale.f16_data() + slot_offset / kKvScaleBlock,
                layer.full.v_scale.f16_data() + slot_offset / kKvScaleBlock,
                rows, kv_heads, head_dim,
                kKvScaleBlock, position_offset, max_context),
                "append FP8 full KV cache");
        } else if (cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
            const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
            const size_t turboquant_slot_offset = static_cast<size_t>(slot_id) *
                max_context * kv_heads * slot_bytes;
            require_launch(qwen_append_kv_cache_turboquant_k8v4_cuda(
                k_norm.f16_data(), v.f16_data(),
                layer.full.turboquant_cache.byte_data() + turboquant_slot_offset,
                rows, kv_heads, head_dim, position_offset, max_context),
                "append TurboQuant K8V4 full KV cache");
        } else if (cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
            const size_t scale_offset = static_cast<size_t>(slot_id) * max_context * kv_heads;
            require_launch(qwen_append_kv_cache_int8_per_token_head_cuda(
                k_norm.f16_data(), v.f16_data(),
                layer.full.k_cache.int8_data() + slot_offset,
                layer.full.v_cache.int8_data() + slot_offset,
                layer.full.k_scale.f16_data() + scale_offset,
                layer.full.v_scale.f16_data() + scale_offset,
                rows, kv_heads, head_dim,
                position_offset, max_context),
                "append INT8 per-token-head full KV cache");
        } else {
            require_launch(qwen_append_kv_cache_f16(
                k_norm.f16_data(), v.f16_data(),
                layer.full.k_cache.f16_data() + slot_offset,
                layer.full.v_cache.f16_data() + slot_offset,
                rows, kv_heads, head_dim,
                position_offset, max_context), "append FP16 full KV cache");
        }

        // Where the read kernels find this sequence's history. Contiguous, that
        // is the slot's own span of the arena. Paged and single-sequence, the
        // blocks are gathered once per layer into the dense
        // `[context, kv_heads, head_dim]` layout every read kernel already
        // indexes, so the five tuned prefill and decode variants stay untouched.
        // This is the same dequant-once trade the FP8 and TurboQuant paths make:
        // one O(context) pass instead of translating inside O(rows * context)
        // inner loops. Batched decode does not come through here; it reads the
        // blocks natively, which is the path that has to scale with batch size.
        // Only the FP16 cache stores halves here; the quantized caches keep
        // their own packed buffers and reach for them inside their own branches
        // below, so binding these eagerly would trip f16_data()'s dtype check.
        const bool fp16_cache = cache_dtype == QwenKvCacheDType::Fp16;
        const uint16_t* k_read = fp16_cache
            ? layer.full.k_cache.f16_data() + slot_offset : nullptr;
        const uint16_t* v_read = fp16_cache
            ? layer.full.v_cache.f16_data() + slot_offset : nullptr;
        if (kv_paged() && batch_rows == nullptr) {
            const int context_length = position_offset + rows;
            const std::vector<uint64_t> dense_shape = {
                static_cast<uint64_t>(context_length),
                static_cast<uint64_t>(kv_heads),
                static_cast<uint64_t>(head_dim)};
            const size_t dense_elements =
                static_cast<size_t>(context_length) * kv_heads * head_dim;
            QwenDeviceTensor& k_dense =
                workspace_half(dense_elements, dense_shape);
            QwenDeviceTensor& v_dense =
                workspace_half(dense_elements, dense_shape);
            PhaseScope sub(this, "full.kv_gather");
            require_launch(qwen_gather_kv_cache_f16_paged_cuda(
                layer.full.k_cache.f16_data(), layer.full.v_cache.f16_data(),
                k_dense.f16_data(), v_dense.f16_data(), context_length,
                kv_heads, head_dim, block_table_row(slot_id),
                paged_block_size()), "gather paged FP16 KV cache");
            k_read = k_dense.f16_data();
            v_read = v_dense.f16_data();
        }

        if (batch_rows != nullptr) {
            // One launch for the whole batch. The split geometry comes from the
            // longest row, so the scratch is sized from that same query the
            // launcher uses; shorter rows leave their trailing splits empty.
            const int splits = qwen_gqa_decode_batched_split_count(
                batch_rows->max_context_len, kv_heads, attention_window,
                sink_tokens);
            require_launch(splits > 0, "batched decode split geometry");
            QwenDeviceTensor& partials = workspace_float(
                static_cast<size_t>(rows) * q_heads * splits *
                    static_cast<size_t>(head_dim + 2),
                {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_heads),
                 static_cast<uint64_t>(splits),
                 static_cast<uint64_t>(head_dim + 2)});
            PhaseScope sub(this, "full.attn_kernel");
            require_launch(qwen_gqa_decode_attention_f16_batched_cuda(
                q_norm.f16_data(), layer.full.k_cache.f16_data(),
                layer.full.v_cache.f16_data(), attention.f16_data(),
                partials.f32_data(), batch_rows->context_lens,
                batch_rows->slot_ids, rows, batch_rows->max_context_len,
                kv_slot_stride, q_heads, kv_heads, head_dim, max_context,
                attention_window, sink_tokens, block_table_data(),
                paged_block_size(), paged_blocks_per_seq()),
                "batched decode FP16-cache GQA");
        } else if (rows == 1) {
            const int context_length = position_offset + 1;
            const bool optimized_attention =
                qwen_env_enabled_default("DSV4_QWEN_GQA_OPTIMIZED") ||
                attention_window > 0;
            // The split/merge decode path used to lose below 16384 context
            // because it walked 2048 positions per split serially, leaving the
            // device idle. At 128 positions per split it wins everywhere it is
            // allowed to run: 3.6x the reference kernels at 4096 context and
            // 16.1x at 65536. Its own guard still declines contexts under 4096.
            const bool optimized_decode = cache_dtype == QwenKvCacheDType::Fp16 &&
                optimized_attention &&
                (context_length >= 4096 || attention_window > 0);
            int attended_positions = context_length;
            if (attention_window > 0) {
                const int sink_count = std::min(sink_tokens, context_length);
                const int window_start = std::max(
                    context_length - attention_window, sink_count);
                attended_positions = sink_count + (context_length - window_start);
            }
            // Must match the launch exactly; it sizes the partial scratch.
            const bool tensor_core_decode_shape = attention_window <= 0 &&
                sink_tokens == 0 && q_heads == kv_heads * 6 && head_dim == 256;
            const int optimized_splits = qwen_gqa_decode_split_count(
                attended_positions, kv_heads, tensor_core_decode_shape);
            const size_t score_elements = optimized_decode
                ? static_cast<size_t>(q_heads) * optimized_splits *
                      static_cast<size_t>(head_dim + 2)
                : static_cast<size_t>(q_heads) * context_length;
            QwenDeviceTensor& scores = workspace_float(
                score_elements,
                optimized_decode
                    ? std::vector<uint64_t>{static_cast<uint64_t>(q_heads),
                                            static_cast<uint64_t>(optimized_splits),
                                            static_cast<uint64_t>(head_dim + 2)}
                    : std::vector<uint64_t>{static_cast<uint64_t>(q_heads),
                                             static_cast<uint64_t>(context_length)});
            if (cache_dtype == QwenKvCacheDType::Fp8) {
                require_launch(qwen_gqa_decode_attention_fp8_cuda(
                    q_norm.f16_data(),
                    layer.full.k_cache.fp8_data() + slot_offset,
                    layer.full.v_cache.fp8_data() + slot_offset,
                    layer.full.k_scale.f16_data() + slot_offset / kKvScaleBlock,
                    layer.full.v_scale.f16_data() + slot_offset / kKvScaleBlock,
                    attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    kKvScaleBlock, context_length, max_context),
                    "decode FP8-cache GQA");
            } else if (cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
                const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
                const size_t turboquant_slot_offset = static_cast<size_t>(slot_id) *
                    max_context * kv_heads * slot_bytes;
                require_launch(qwen_gqa_decode_attention_turboquant_k8v4_cuda(
                    q_norm.f16_data(),
                    layer.full.turboquant_cache.byte_data() + turboquant_slot_offset,
                    attention.f16_data(), scores.f32_data(), q_heads, kv_heads,
                    head_dim, context_length, max_context, attention_window,
                    sink_tokens), "decode TurboQuant K8V4 GQA");
            } else if (cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
                const size_t scale_offset = static_cast<size_t>(slot_id) * max_context * kv_heads;
                require_launch(qwen_gqa_decode_attention_int8_per_token_head_cuda(
                    q_norm.f16_data(),
                    layer.full.k_cache.int8_data() + slot_offset,
                    layer.full.v_cache.int8_data() + slot_offset,
                    layer.full.k_scale.f16_data() + scale_offset,
                    layer.full.v_scale.f16_data() + scale_offset,
                    attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    context_length, max_context, attention_window, sink_tokens),
                    "decode INT8 per-token-head GQA");
            } else if (optimized_decode) {
                require_launch(qwen_gqa_decode_attention_f16_fused_cuda(
                    q_norm.f16_data(),
                    k_read,
                    v_read,
                    attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    context_length, max_context, attention_window, sink_tokens),
                    "decode optimized FP16-cache GQA");
            } else {
                require_launch(qwen_gqa_decode_attention_f16(
                    q_norm.f16_data(),
                    k_read,
                    v_read,
                    attention.f16_data(),
                    scores.f32_data(), q_heads, kv_heads, head_dim,
                    context_length, max_context), "decode FP16-cache GQA");
            }
        } else if (cache_dtype == QwenKvCacheDType::Fp8) {
            const int context_length = position_offset + rows;
            // Dequantize the [0, context_length) range once into dense FP16
            // buffers, then call the tensor-core prefill kernel. O(ctx) dequant
            // + tensor core beats O(rows*ctx) inline dequant by the same factor
            // as TurboQuant (6-13×).
            QwenDeviceTensor& k_dense = workspace_half(
                static_cast<size_t>(context_length) * kv_heads * head_dim,
                {static_cast<uint64_t>(context_length),
                 static_cast<uint64_t>(kv_heads),
                 static_cast<uint64_t>(head_dim)});
            QwenDeviceTensor& v_dense = workspace_half(
                static_cast<size_t>(context_length) * kv_heads * head_dim,
                {static_cast<uint64_t>(context_length),
                 static_cast<uint64_t>(kv_heads),
                 static_cast<uint64_t>(head_dim)});
            require_launch(qwen_fp8_dequant_kv_cache_cuda(
                layer.full.k_cache.fp8_data() + slot_offset,
                layer.full.v_cache.fp8_data() + slot_offset,
                layer.full.k_scale.f16_data() + slot_offset / kKvScaleBlock,
                layer.full.v_scale.f16_data() + slot_offset / kKvScaleBlock,
                k_dense.f16_data(), v_dense.f16_data(), context_length,
                kv_heads, head_dim, kKvScaleBlock, max_context),
                "dequant FP8 cache to dense FP16");
            if (qwen_env_enabled_default("DSV4_QWEN_GQA_OPTIMIZED") ||
                attention_window > 0) {
                require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                    q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                    attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context, attention_window, sink_tokens),
                    "prefill FP8 via tensor-core tiled");
            } else {
                require_launch(qwen_gqa_prefill_attention_f16(
                    q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                    attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context),
                    "prefill FP8 via tensor-core exact");
            }
        } else if (cache_dtype == QwenKvCacheDType::Int8PerTokenHead) {
            const int context_length = position_offset + rows;
            // INT8 per-token-head uses the same dequant-once architecture as FP8
            QwenDeviceTensor& k_dense = workspace_half(
                static_cast<size_t>(context_length) * kv_heads * head_dim,
                {static_cast<uint64_t>(context_length),
                 static_cast<uint64_t>(kv_heads),
                 static_cast<uint64_t>(head_dim)});
            QwenDeviceTensor& v_dense = workspace_half(
                static_cast<size_t>(context_length) * kv_heads * head_dim,
                {static_cast<uint64_t>(context_length),
                 static_cast<uint64_t>(kv_heads),
                 static_cast<uint64_t>(head_dim)});
            const size_t scale_offset = static_cast<size_t>(slot_id) * max_context * kv_heads;
            require_launch(qwen_int8_dequant_kv_cache_cuda(
                layer.full.k_cache.int8_data() + slot_offset,
                layer.full.v_cache.int8_data() + slot_offset,
                layer.full.k_scale.f16_data() + scale_offset,
                layer.full.v_scale.f16_data() + scale_offset,
                k_dense.f16_data(), v_dense.f16_data(), context_length,
                kv_heads, head_dim, max_context),
                "dequant INT8 cache to dense FP16");
            if (qwen_env_enabled_default("DSV4_QWEN_GQA_OPTIMIZED") ||
                attention_window > 0) {
                require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                    q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                    attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context, attention_window, sink_tokens),
                    "prefill INT8 via tensor-core tiled");
            } else {
                require_launch(qwen_gqa_prefill_attention_f16(
                    q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                    attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context),
                    "prefill INT8 via tensor-core exact");
            }
        } else if (cache_dtype == QwenKvCacheDType::TurboQuantK8V4) {
            const int context_length = position_offset + rows;
            // Dequantize the [0, context_length) range once into dense FP16
            // buffers laid out like the FP16 cache, then call the tensor-core
            // prefill kernel on them. O(ctx) dequant + tensor core beats
            // O(rows*ctx) inline dequant by 6-13× and keeps prefill fast.
            QwenDeviceTensor& k_dense = workspace_half(
                static_cast<size_t>(context_length) * kv_heads * head_dim,
                {static_cast<uint64_t>(context_length),
                 static_cast<uint64_t>(kv_heads),
                 static_cast<uint64_t>(head_dim)});
            QwenDeviceTensor& v_dense = workspace_half(
                static_cast<size_t>(context_length) * kv_heads * head_dim,
                {static_cast<uint64_t>(context_length),
                 static_cast<uint64_t>(kv_heads),
                 static_cast<uint64_t>(head_dim)});
            const int slot_bytes = qwen_turboquant_k8v4_slot_bytes(head_dim);
            const size_t turboquant_slot_offset = static_cast<size_t>(slot_id) *
                max_context * kv_heads * slot_bytes;
            require_launch(qwen_turboquant_k8v4_dequant_kv_cuda(
                layer.full.turboquant_cache.byte_data() + turboquant_slot_offset, k_dense.f16_data(),
                v_dense.f16_data(), context_length, kv_heads, head_dim,
                max_context), "dequant TurboQuant cache to dense FP16");
            if (qwen_env_enabled_default("DSV4_QWEN_GQA_OPTIMIZED") ||
                attention_window > 0) {
                require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                    q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                    attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context, attention_window, sink_tokens),
                    "prefill TurboQuant via tensor-core tiled");
            } else {
                require_launch(qwen_gqa_prefill_attention_f16(
                    q_norm.f16_data(), k_dense.f16_data(), v_dense.f16_data(),
                    attention.f16_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context),
                    "prefill TurboQuant via tensor-core exact");
            }
        } else if (cache_dtype == QwenKvCacheDType::Fp16 && rows <= 8 && attention_window == 0) {
            const int context_length = position_offset + rows;
            // Tensor-core QK experiment for the real TP4 shape (one KV head per
            // rank). Keeps the existing FP32 softmax/PV order; opt-in until real
            // parity and latency are validated.
            if (qwen_env_enabled("QWEN_GQA_VERIFY_CUBLAS_QK") &&
                kv_heads == 1) {
                const size_t score_elements =
                    static_cast<size_t>(rows) * q_heads * context_length;
                QwenDeviceTensor& scores = workspace_float(
                    score_elements,
                    {static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(q_heads),
                     static_cast<uint64_t>(context_length)});
                PhaseScope sub(this, "full.attn_kernel");
                require_launch(qwen_gqa_verify_attention_f16_cublas_qk_cuda(
                    q_norm.f16_data(),
                    k_read,
                    v_read,
                    attention.f16_data(),
                    scores.f32_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context),
                    "verify cuBLAS-QK FP16-cache GQA");
            // The exact three-kernel path avoids split partials below 1K context;
            // the split-K path crosses over above that and remains faster at long
            // context. QWEN_GQA_VERIFY_SPLIT explicitly overrides the crossover.
            } else if ((std::getenv("QWEN_GQA_VERIFY_SPLIT") != nullptr
                            ? qwen_env_enabled("QWEN_GQA_VERIFY_SPLIT")
                            : context_length > 1024)) {
                // 32 splits stay slightly faster end-to-end at short context;
                // 64 cross over at 1K and remain best through 32K.
                const int default_splits = context_length >= 1024 ? 64 : 32;
                int target_splits = qwen_env_int(
                    "QWEN_GQA_VERIFY_SPLITS", default_splits);
                if (target_splits <= 0) target_splits = default_splits;
                const int verify_splits = std::max(
                    1, std::min(target_splits, context_length));
                const size_t partial_elements =
                    static_cast<size_t>(rows) * q_heads * verify_splits *
                    static_cast<size_t>(head_dim + 2);
                QwenDeviceTensor& partials = workspace_float(
                    partial_elements,
                    {static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(q_heads),
                     static_cast<uint64_t>(verify_splits),
                     static_cast<uint64_t>(head_dim + 2)});
                PhaseScope sub(this, "full.attn_kernel");
                require_launch(qwen_gqa_verify_attention_f16(
                    q_norm.f16_data(),
                    k_read,
                    v_read,
                    attention.f16_data(),
                    partials.f32_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context, verify_splits),
                    "verify split FP16-cache GQA");
            } else {
                const size_t score_elements =
                    static_cast<size_t>(rows) * q_heads * context_length;
                QwenDeviceTensor& scores = workspace_float(
                    score_elements,
                    {static_cast<uint64_t>(rows),
                     static_cast<uint64_t>(q_heads),
                     static_cast<uint64_t>(context_length)});
                { PhaseScope sub(this, "full.attn_kernel"); require_launch(qwen_gqa_verify_attention_f16_exact_cuda(
                    q_norm.f16_data(),
                    k_read,
                    v_read,
                    attention.f16_data(),
                    scores.f32_data(), rows, q_heads, kv_heads, head_dim,
                    position_offset, max_context),
                    "verify exact FP16-cache GQA"); }
            }
        // The tiled kernel shares each K/V element across a head group and a
        // pair of query rows, so it is the default for multi-row prefill. The
        // wider TP4 candidate is selected separately through LONG_TILE.
        } else if (qwen_env_enabled_default("DSV4_QWEN_GQA_OPTIMIZED") ||
                   attention_window > 0) {
            require_launch(qwen_gqa_prefill_attention_f16_tiled_cuda(
                q_norm.f16_data(),
                k_read,
                v_read,
                attention.f16_data(), rows,
                q_heads, kv_heads, head_dim, position_offset, max_context,
                attention_window, sink_tokens),
                "prefill optimized FP16-cache GQA");
        } else {
            require_launch(qwen_gqa_prefill_attention_f16(
                q_norm.f16_data(),
                k_read,
                v_read,
                attention.f16_data(), rows,
                q_heads, kv_heads, head_dim, position_offset, max_context),
                "prefill FP16-cache GQA");
        }
#endif
        require_launch(qwen_sigmoid_mul_f16(
            attention.f16_data(), gate.f16_data(), merged.f16_data(),
            rows * attention_dim), "FP16 full attention output gate");
        // Row-parallel output projection: each slice's rows are complete once its
        // GEMM lands, so the collective for slice i can run under slice i+1's GEMM.
        // This site was left serial when mlp.down and lin.out were pipelined, and
        // at 65K it was the single largest exposed collective in the profile
        // (ar.full.out 2.12 s against tp_all_reduce's 3.16 s total).
        if (!projection_all_reduce_overlapped(
                layer.full.out, merged.f16_data(), output, rows,
                static_cast<int>(config.hidden_size), "full.out", "full.out")) {
            { PhaseScope sub(this, "full.out_proj"); projection(layer.full.out, merged.f16_data(), output, rows, "full.out"); }
            all_reduce_half(output, rows * static_cast<int>(config.hidden_size), "full.out");
        }
    }

    void layer_forward(DeviceLayer& layer, const uint16_t* hidden,
                       uint16_t* output, int rows, int position_offset, int slot_id) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        begin_workspace();
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const size_t intermediate_elements = static_cast<size_t>(rows) *
            layer.gate.logical_shape[0];
        QwenDeviceTensor& normalized = workspace_half(
            hidden_elements, {static_cast<uint64_t>(rows),
                              static_cast<uint64_t>(hidden_size)});
        QwenDeviceTensor& attention = workspace_half(hidden_elements, normalized.shape);
        QwenDeviceTensor& post = workspace_half(hidden_elements, normalized.shape);
#ifdef POCKET_BACKEND_ASCEND
        // Every fused SwiGLU variant below is a quantized-weight kernel, and no
        // quantized kind loads on this backend. Keeping the flags as constants
        // lets the shared control flow read identically on both backends while
        // the dead arms are removed before codegen.
        constexpr bool fused_decode_swiglu = false;
        constexpr bool fused_small_batch_swiglu = false;
        constexpr bool fused_nvfp4_swiglu = false;
#else
        const bool compatible_fp8_swiglu =
            layer.gate.kind == QwenLinearKind::Fp8Block128 &&
            layer.up.kind == QwenLinearKind::Fp8Block128 &&
            layer.gate.logical_shape == layer.up.logical_shape &&
            layer.gate.scale.shape == layer.up.scale.shape;
        const bool fused_decode_swiglu = rows == 1 && compatible_fp8_swiglu;
        const bool fused_small_batch_swiglu = rows > 1 && rows <= 8 &&
            compatible_fp8_swiglu && hidden_size % 4 == 0;
        const char* nvfp4_mode = std::getenv("DSV4_QWEN_NVFP4");
        const bool nvfp4_wmma = nvfp4_mode == nullptr || *nvfp4_mode == '\0' ||
            std::strcmp(nvfp4_mode, "auto") == 0 ||
            std::strcmp(nvfp4_mode, "wmma") == 0;
        const char* fused_nvfp4 =
            std::getenv("DSV4_QWEN_NVFP4_FUSED_SWIGLU");
        const bool fused_nvfp4_swiglu = rows >= 8 && nvfp4_wmma &&
            fused_nvfp4 != nullptr && std::strcmp(fused_nvfp4, "1") == 0 &&
            layer.gate.kind == QwenLinearKind::NvFp4Group16 &&
            layer.up.kind == QwenLinearKind::NvFp4Group16 &&
            layer.gate.logical_shape == layer.up.logical_shape;
        const char* shared_nvfp4 =
            std::getenv("DSV4_QWEN_NVFP4_SHARED_Q8_SWIGLU");
        const bool shared_nvfp4_enabled = shared_nvfp4 == nullptr ||
            std::strcmp(shared_nvfp4, "0") != 0;
        const bool shared_nvfp4_swiglu = rows >= 8 && nvfp4_wmma &&
            shared_nvfp4_enabled && !fused_nvfp4_swiglu &&
            layer.gate.kind == QwenLinearKind::NvFp4Group16 &&
            layer.up.kind == QwenLinearKind::NvFp4Group16 &&
            layer.gate.logical_shape == layer.up.logical_shape;
#endif
        QwenDeviceTensor* gate = nullptr;
        QwenDeviceTensor* up = nullptr;
        if (!fused_decode_swiglu && !fused_small_batch_swiglu &&
            !fused_nvfp4_swiglu) {
            gate = &workspace_half(intermediate_elements,
                {static_cast<uint64_t>(rows), layer.gate.logical_shape[0]});
            up = &workspace_half(intermediate_elements, gate->shape);
        }
        QwenDeviceTensor& intermediate = workspace_half(
            intermediate_elements,
            {static_cast<uint64_t>(rows), layer.gate.logical_shape[0]});
        QwenDeviceTensor& mlp = workspace_half(hidden_elements, normalized.shape);

        norm(layer.input_norm, hidden, normalized.f16_data(), rows, hidden_size);
        if (layer.linear.qkv.weight.data != nullptr) {
            linear_attention(layer, normalized.f16_data(), attention.f16_data(),
                             rows, position_offset, slot_id);
        } else {
            full_attention(layer, normalized.f16_data(), attention.f16_data(),
                           rows, position_offset, slot_id);
        }
        // One fused pass replaces the residual copy, the add, and the norm. It is
        // the default: measured on the real 65K TP4 prefill it saves 1.51 s
        // (55.80 -> 54.28 s) with identical generated tokens. `=0` restores the
        // three separate passes for A/B.
        if (qwen_env_enabled_default("QWEN_FUSE_ATTN_RESID_NORM")) {
            PhaseScope scope(this, "attn_resid_norm");
            require_launch(
                qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
                    hidden, attention.f16_data(), layer.post_norm.f16_data(),
                    output, post.f16_data(), rows, hidden_size,
                    static_cast<float>(config.rms_norm_eps)),
                "Qwen fused attention residual RMSNorm");
        } else {
            {
                PhaseScope scope(this, "resid_copy");
                check_device(memcpy_d2d(output, hidden, hidden_elements * sizeof(uint16_t)),
                      "Qwen FP16 residual copy");
            }
            add(output, attention.f16_data(), rows * hidden_size);
            norm(layer.post_norm, output, post.f16_data(), rows, hidden_size);
        }
#ifdef POCKET_BACKEND_ASCEND
        {
            projection(layer.gate, post.f16_data(), gate->f16_data(), rows, "mlp.gate");
            projection(layer.up, post.f16_data(), up->f16_data(), rows, "mlp.up");
            require_launch(qwen_silu_mul_rows_f16(
                gate->f16_data(), up->f16_data(), intermediate.f16_data(), rows,
                static_cast<int>(layer.gate.logical_shape[0])), "FP16 SwiGLU");
        }
#else
        if (fused_decode_swiglu) {
            PhaseScope scope(this, "swiglu.d");
            require_launch(qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                post.f16_data(), layer.gate.weight.fp8_data(),
                layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
                layer.up.scale.f16_data(), intermediate.f16_data(),
                static_cast<int>(layer.gate.logical_shape[0]), hidden_size,
                hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
                "FP16 fused decode SwiGLU");
        } else if (fused_small_batch_swiglu) {
            PhaseScope scope(this, "swiglu.r");
            require_launch(
                qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
                    post.f16_data(), layer.gate.weight.fp8_data(),
                    layer.gate.scale.f16_data(), layer.up.weight.fp8_data(),
                    layer.up.scale.f16_data(), intermediate.f16_data(), rows,
                    static_cast<int>(layer.gate.logical_shape[0]), hidden_size,
                    hidden_size, static_cast<int>(layer.gate.logical_shape[0]),
                    hidden_size, static_cast<int>(layer.gate.scale.shape[1])),
                "FP16 fused small-batch SwiGLU");
        } else if (fused_nvfp4_swiglu) {
            PhaseScope scope(this, "projection_rows_nvfp4");
            require_launch(nvfp4_fused_swiglu_projection(
                layer.gate, layer.up, post.f16_data(),
                intermediate.f16_data(), rows),
                "NVFP4 fused Q8 WMMA SwiGLU projection");
        } else {
            if (shared_nvfp4_swiglu) {
                PhaseScope scope(this, "projection_rows_nvfp4");
                require_launch(nvfp4_shared_q8_swiglu_projection(
                    layer.gate, layer.up, post.f16_data(), gate->f16_data(),
                    up->f16_data(), rows),
                    "NVFP4 shared-Q8 WMMA gate/up projection");
            } else {
                projection(layer.gate, post.f16_data(), gate->f16_data(), rows, "mlp.gate");
                projection(layer.up, post.f16_data(), up->f16_data(), rows, "mlp.up");
            }
            require_launch(qwen_silu_mul_rows_f16(
                gate->f16_data(), up->f16_data(), intermediate.f16_data(), rows,
                static_cast<int>(layer.gate.logical_shape[0])), "FP16 SwiGLU");
        }
#endif
        // ar.mlp is the largest single collective (3.58 s of the 7.24 s
        // tp_all_reduce leaf at 32K), and down is a large GEMM, so this is the
        // best overlap candidate in the layer.
        if (!projection_all_reduce_overlapped(
                layer.down, intermediate.f16_data(), mlp.f16_data(), rows,
                hidden_size, "mlp.down", "mlp")) {
            projection(layer.down, intermediate.f16_data(), mlp.f16_data(), rows,
                       "mlp.down");
            all_reduce_half(mlp.f16_data(), rows * hidden_size, "mlp");
        }
        add(output, mlp.f16_data(), rows * hidden_size);
    }

    bool sampling_enabled() const { return options.temperature > 1.0e-5f; }

    // Draw one uniform per row on the host. Every rank seeds the same generator
    // and consumes it in the same order, so the rows match across the TP group
    // without an extra broadcast.
    const float* host_uniforms_for(int rows) {
        if (!sampling_rng_ready) {
            sampling_rng.seed(options.sampling_seed);
            sampling_rng_ready = true;
        }
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        host_uniform_scratch.resize(static_cast<size_t>(rows));
        for (int i = 0; i < rows; ++i) host_uniform_scratch[i] = dist(sampling_rng);
        return host_uniform_scratch.data();
    }

    // Device top-k -> temperature -> top-p sampling over a batch of logit rows.
    // Under TP the vocabulary is sharded, so each rank emits its local top-k as
    // global ids, NCCL merges the candidates, and the draw happens over the
    // merged set with a uniform every rank already agrees on.
    //
    // The Ascend build has no device sampler yet, so the whole routine is
    // compiled out and the constructor rejects a nonzero temperature up front.
#ifndef POCKET_BACKEND_ASCEND
    QwenVerifyBatch sample_tokens_for(QwenDeviceTensor& local_logits, int rows,
                                      int local_vocab, int vocab_start,
                                      int position_after) {
        int top_k = options.top_k;
        const int max_top_k = qwen_sampler_max_top_k();
        if (top_k <= 0 || top_k > max_top_k) top_k = max_top_k;
        if (top_k > local_vocab) top_k = local_vocab;

        allocate(sample_rng_states,
                 static_cast<size_t>(rows) * qwen_sampler_rng_state_size(),
                 {static_cast<uint64_t>(rows)}, SafeDType::I8);
        allocate_float(sample_uniforms, static_cast<size_t>(rows),
                       {static_cast<uint64_t>(rows)});
        allocate(argmax_token, static_cast<size_t>(rows) * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        allocate_float(argmax_logit, static_cast<size_t>(rows),
                       {static_cast<uint64_t>(rows)});

        const float* uniforms = host_uniforms_for(rows);
        check_device(memcpy_h2d(sample_uniforms.data, uniforms,
                                static_cast<size_t>(rows) * sizeof(float)),
                     "Qwen sampling uniform upload");

        QwenVerifyBatch result;
        result.top_tokens.resize(static_cast<size_t>(rows));
        result.top_logits.resize(static_cast<size_t>(rows));
        result.local_logits.resize(static_cast<size_t>(rows));
        result.position_after = position_after;

#ifdef DSV4_HAVE_TP_COMM
        if (options.tp_world > 1) {
            if (options.nccl_id_path.empty()) {
                throw std::runtime_error("Qwen TP requires --nccl-id-path");
            }
            // tp_global_topk_rows_device reduces the gathered candidates back
            // down to top_k per row, so the merged buffers are rows * top_k --
            // not rows * top_k * tp_world. The gathered staging lives inside
            // tp_comm's own workspace.
            const size_t cand = static_cast<size_t>(rows) * top_k;
            allocate(sample_cand_token, cand * sizeof(int),
                     {static_cast<uint64_t>(cand)}, SafeDType::I64);
            allocate_float(sample_cand_logit, cand, {static_cast<uint64_t>(cand)});
            allocate(sample_merged_token, cand * sizeof(int),
                     {static_cast<uint64_t>(cand)}, SafeDType::I64);
            allocate_float(sample_merged_logit, cand,
                           {static_cast<uint64_t>(cand)});

            require_launch(qwen_local_topk_candidates(
                local_logits.f32_data(),
                static_cast<int*>(sample_cand_token.data),
                sample_cand_logit.f32_data(), rows, local_vocab, vocab_start,
                top_k, nullptr),
                "Qwen sampling local top-k");

            tp_global_topk_rows_device(
                options.tp_world, options.tp_rank, options.device,
                options.nccl_id_path.c_str(),
                static_cast<const int*>(sample_cand_token.data),
                sample_cand_logit.f32_data(), rows, top_k,
                static_cast<int*>(sample_merged_token.data),
                sample_merged_logit.f32_data());

            require_launch(qwen_sample_from_candidates(
                static_cast<const int*>(sample_merged_token.data),
                sample_merged_logit.f32_data(),
                static_cast<int*>(argmax_token.data), argmax_logit.f32_data(),
                rows, top_k, options.temperature,
                options.top_p, top_k,
                static_cast<DeviceRngState*>(sample_rng_states.data),
                sample_uniforms.f32_data(), nullptr),
                "Qwen sampling from merged candidates");
        } else
#endif
        {
#ifndef DSV4_HAVE_TP_COMM
            if (options.tp_world > 1) {
                throw std::runtime_error(
                    "Qwen TP requires an NCCL-enabled build");
            }
#endif
            require_launch(qwen_sample_top_k_top_p_rows(
                local_logits.f32_data(),
                static_cast<int*>(argmax_token.data), argmax_logit.f32_data(),
                rows, local_vocab, vocab_start, options.temperature,
                options.top_p, top_k,
                static_cast<DeviceRngState*>(sample_rng_states.data),
                sample_uniforms.f32_data(), nullptr),
                "Qwen sampling single shard");
        }

        check_device(memcpy_d2h(result.top_tokens.data(), argmax_token.data,
                                static_cast<size_t>(rows) * sizeof(int)),
                     "Qwen sampled token copy");
        check_device(memcpy_d2h(result.top_logits.data(), argmax_logit.data,
                                static_cast<size_t>(rows) * sizeof(float)),
                     "Qwen sampled logit copy");
        result.local_logits = result.top_logits;
        return result;
    }
#endif

    QwenVerifyBatch top_tokens_for(const uint16_t* hidden, int rows,
                                   const QwenDeviceTensor* output_norm,
                                   int position_after) {
        if (rows <= 0) {
            throw std::runtime_error("Qwen logits require at least one row");
        }
        const int hidden_size = static_cast<int>(config.hidden_size);
        if (lm_head.logical_shape.size() != 2) {
            throw std::runtime_error("Qwen target head has invalid logical shape");
        }
        const int local_vocab = static_cast<int>(lm_head.logical_shape[0]);
        begin_workspace();
        QwenDeviceTensor& normalized = workspace_half(
            static_cast<size_t>(rows) * hidden_size,
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)});
        if (output_norm != nullptr) {
            norm(*output_norm, hidden, normalized.f16_data(), rows, hidden_size);
        } else {
            check_device(memcpy_d2d(normalized.data, hidden,
                                    static_cast<size_t>(rows) * hidden_size * sizeof(uint16_t)),
                         "Qwen normalized hidden copy");
        }
        QwenDeviceTensor& local_logits = workspace_float(
            static_cast<size_t>(rows) * local_vocab,
            {static_cast<uint64_t>(rows), static_cast<uint64_t>(local_vocab)});
        bool logits_ok = false;
#ifdef POCKET_BACKEND_ASCEND
        // The cuBLAS row variant is an alternative implementation of the same
        // GEMM, so the Ascend build has only the neutral launcher. Fp8Channel is
        // rejected for the same reason projection() rejects the quantized kinds.
        if (lm_head.kind != QwenLinearKind::DenseF16) {
            throw std::runtime_error(
                std::string("Qwen Ascend target head is not implemented for ") +
                qwen_linear_kind_name(lm_head.kind));
        }
        {
            PhaseScope scope(this, rows == 1 ? "lmhead.d" : "lmhead.r");
            logits_ok = qwen_fp16_matmul_rows_f16_f32(
                normalized.f16_data(), lm_head.weight.f16_data(),
                local_logits.f32_data(), rows, local_vocab, hidden_size,
                hidden_size, local_vocab, hidden_size);
        }
#else
        if (lm_head.kind == QwenLinearKind::DenseF16) {
            // Decode. The warp-per-row matvec reorders the reduction, so it is a
            // separate entry point rather than a retune of the reference path;
            // it is the default because the tokens it produces are identical to
            // that path for 128 greedy steps at 512/4096/8192/32768/65536 on the
            // real checkpoint, while decode is 2.0-5.2% faster.
            const bool matvec_logits = rows == 1 &&
                qwen_env_enabled_default("QWEN_FP16_LOGITS_MATVEC");
            const bool cublas_logits = rows > 1 &&
                qwen_env_enabled_default("QWEN_FP16_LOGITS_CUBLAS");
            PhaseScope scope(this, rows == 1 ? "lmhead.d" : "lmhead.r");
            logits_ok = matvec_logits
                ? qwen_fp16_matvec_rows_f16_f32_cuda(
                      normalized.f16_data(), lm_head.weight.f16_data(),
                      local_logits.f32_data(), rows, local_vocab, hidden_size,
                      hidden_size, local_vocab, hidden_size)
                : cublas_logits
                    ? qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                          normalized.f16_data(), lm_head.weight.f16_data(),
                          local_logits.f32_data(), rows, local_vocab, hidden_size,
                          hidden_size, local_vocab, hidden_size)
                    : qwen_fp16_matmul_rows_f16_f32(
                          normalized.f16_data(), lm_head.weight.f16_data(),
                          local_logits.f32_data(), rows, local_vocab, hidden_size,
                          hidden_size, local_vocab, hidden_size);
        } else if (lm_head.kind == QwenLinearKind::Fp8Channel) {
            PhaseScope scope(this, rows == 1 ? "lmhead.d" : "lmhead.r");
            logits_ok = qwen_fp8_e4m3_channel_matmul_rows_f16_f32_cuda(
                normalized.f16_data(), lm_head.weight.fp8_data(),
                lm_head.scale.f16_data(), local_logits.f32_data(), rows,
                local_vocab, hidden_size, hidden_size, local_vocab,
                hidden_size);
        } else {
            throw std::runtime_error(
                std::string("Qwen target-head CUDA path is not implemented for ") +
                qwen_linear_kind_name(lm_head.kind));
        }
#endif
        require_launch(logits_ok, "Qwen batched FP32 logits");
        allocate(argmax_token, static_cast<size_t>(rows) * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        allocate_float(argmax_logit, static_cast<size_t>(rows),
                       {static_cast<uint64_t>(rows)});
        const int vocab_start = static_cast<int>(weights_vocab_start());

        // Sampled path. Kept ahead of the argmax launch so a sampled run does
        // not pay for a top-1 reduction it discards; greedy falls through to
        // the original code unchanged.
#ifndef POCKET_BACKEND_ASCEND
        if (sampling_enabled()) {
            return sample_tokens_for(local_logits, rows, local_vocab,
                                     vocab_start, position_after);
        }
#endif

        {
            PhaseScope scope(this, "argmax");
            require_launch(qwen_argmax_fp32_rows(
                local_logits.f32_data(), static_cast<int*>(argmax_token.data),
                argmax_logit.f32_data(), rows, local_vocab, vocab_start),
                "Qwen batched local argmax");
        }
        QwenVerifyBatch result;
        result.top_tokens.resize(static_cast<size_t>(rows));
        result.top_logits.resize(static_cast<size_t>(rows));
        result.local_logits.resize(static_cast<size_t>(rows));
        result.position_after = position_after;
#ifdef DSV4_HAVE_TP_COMM
        if (options.tp_world > 1) {
            if (options.nccl_id_path.empty()) {
                throw std::runtime_error("Qwen TP requires --nccl-id-path");
            }
            const bool device_top1 = qwen_env_enabled("QWEN_VERIFY_DEVICE_TOP1");
            const bool packed_device_top1 = device_top1 &&
                qwen_env_enabled("QWEN_VERIFY_DEVICE_TOP1_PACKED");
            {
                PhaseScope scope(this, device_top1
                    ? (packed_device_top1 ? "top1_packed_merge"
                                          : "top1_device_merge")
                    : "top1_allreduce");
                if (device_top1) {
                    // Keep local argmax values intact for `local_logits`; the
                    // merged result has separate workspace buffers because the
                    // verifier reports both local and global TP logits.
                    QwenDeviceTensor& global_token = workspace.tensor(
                        static_cast<size_t>(rows),
                        {static_cast<uint64_t>(rows)}, SafeDType::I64);
                    QwenDeviceTensor& global_logit = workspace_float(
                        static_cast<size_t>(rows),
                        {static_cast<uint64_t>(rows)});
                    if (packed_device_top1) {
                        tp_global_top1_rows_packed_device(
                            options.tp_world, options.tp_rank, options.device,
                            options.nccl_id_path.c_str(),
                            static_cast<const int*>(argmax_token.data),
                            argmax_logit.f32_data(), rows,
                            static_cast<int*>(global_token.data),
                            global_logit.f32_data());
                    } else {
                        tp_global_top1_rows_device(
                            options.tp_world, options.tp_rank, options.device,
                            options.nccl_id_path.c_str(),
                            static_cast<const int*>(argmax_token.data),
                            argmax_logit.f32_data(), rows,
                            static_cast<int*>(global_token.data),
                            global_logit.f32_data());
                    }
                    check_device(memcpy_d2h(result.top_tokens.data(), global_token.data,
                                            static_cast<size_t>(rows) * sizeof(int)),
                                 "Qwen device top-1 token copy");
                    check_device(memcpy_d2h(result.top_logits.data(), global_logit.data,
                                            static_cast<size_t>(rows) * sizeof(float)),
                                 "Qwen device top-1 logit copy");
                } else {
                    tp_global_top1_rows(
                        options.tp_world, options.tp_rank, options.device,
                        options.nccl_id_path.c_str(),
                        static_cast<const int*>(argmax_token.data),
                        argmax_logit.f32_data(), rows, result.top_tokens.data(),
                        result.top_logits.data());
                }
            }
            check_device(memcpy_d2h(result.local_logits.data(), argmax_logit.data,
                                    static_cast<size_t>(rows) * sizeof(float)),
                         "Qwen batched local logit copy");
        } else
#endif
        {
#ifndef DSV4_HAVE_TP_COMM
            if (options.tp_world > 1) {
                throw std::runtime_error(
                    "Qwen TP requires an NCCL-enabled build");
            }
#endif
            check_device(memcpy_d2h(result.top_tokens.data(), argmax_token.data,
                                    static_cast<size_t>(rows) * sizeof(int)),
                         "Qwen batched argmax token copy");
            check_device(memcpy_d2h(result.top_logits.data(), argmax_logit.data,
                                    static_cast<size_t>(rows) * sizeof(float)),
                         "Qwen batched argmax logit copy");
            result.local_logits = result.top_logits;
        }
        return result;
    }

    QwenForwardResult logits_for(const uint16_t* hidden, int last_row,
                                 int position_after, int active_layers) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        QwenVerifyBatch batch = top_tokens_for(
            hidden + static_cast<size_t>(last_row) * hidden_size, 1,
            &final_norm, position_after);
        QwenForwardResult result;
        result.layers = active_layers;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.top_token = batch.top_tokens[0];
        result.top_logit = batch.top_logits[0];
        result.checksum = batch.local_logits[0];
        result.position = position_after;
        return result;
    }

    QwenVerifyBatch target_logits_for(const uint16_t* hidden, int rows,
                                      int position_after) {
        return top_tokens_for(hidden, rows, &final_norm, position_after);
    }

    QwenForwardResult mtp_logits_for(const uint16_t* normalized_hidden,
                                     int position_after) {
        const int hidden_size = static_cast<int>(config.hidden_size);
        QwenVerifyBatch batch = top_tokens_for(normalized_hidden, 1, nullptr,
                                               position_after);
        QwenForwardResult result;
        result.layers = 1;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.top_token = batch.top_tokens[0];
        result.top_logit = batch.top_logits[0];
        result.checksum = batch.local_logits[0];
        result.position = position_after;
        return result;
    }

    QwenForwardResult mtp_forward_rows(const std::vector<int>& tokens,
                                       const uint16_t* hidden, int position) {
        if (!mtp_enabled) {
            throw std::runtime_error("Qwen MTP rows requested while disabled");
        }
        const int rows = static_cast<int>(tokens.size());
        if (hidden == nullptr || rows <= 0 || position < 0 ||
            position + rows > max_context) {
            throw std::runtime_error("invalid Qwen MTP row extent");
        }
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const std::vector<uint64_t> hidden_shape = {
            static_cast<uint64_t>(rows), config.hidden_size};
        allocate(d_tokens, tokens.size() * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        check_device(memcpy_h2d(d_tokens.data, tokens.data(), tokens.size() * sizeof(int)),
                     "Qwen MTP token upload");
        allocate_half(mtp_embedding, hidden_elements, hidden_shape);
        require_launch(qwen_embedding_fp16_gather_f16(
            embed.f16_data(), static_cast<int*>(d_tokens.data),
            mtp_embedding.f16_data(), rows, hidden_size,
            static_cast<int>(weights_vocab_start()),
            static_cast<int>(embed.shape[0])), "Qwen MTP embedding lookup");
        all_reduce_half(mtp_embedding.f16_data(), rows * hidden_size, "mtp.emb");

        allocate_half(mtp_normalized_embedding, hidden_elements, hidden_shape);
        allocate_half(mtp_normalized_hidden, hidden_elements, hidden_shape);
        norm(mtp_pre_fc_norm_embedding, mtp_embedding.f16_data(),
             mtp_normalized_embedding.f16_data(), rows, hidden_size);
        norm(mtp_pre_fc_norm_hidden, hidden,
             mtp_normalized_hidden.f16_data(), rows, hidden_size);
        allocate_half(mtp_concat, hidden_elements * 2,
                      {static_cast<uint64_t>(rows),
                       static_cast<uint64_t>(2 * hidden_size)});
        require_launch(qwen_concat_rows_f16(
            mtp_normalized_embedding.f16_data(),
            mtp_normalized_hidden.f16_data(), mtp_concat.f16_data(), rows,
            hidden_size), "Qwen MTP fusion concat");
        allocate_half(mtp_fused, hidden_elements, hidden_shape);
        projection(mtp_fc, mtp_concat.f16_data(), mtp_fused.f16_data(), rows, "mtp.fc");
        allocate_half(mtp_next_hidden, hidden_elements, hidden_shape);
        layer_forward(mtp_layer, mtp_fused.f16_data(),
                      mtp_next_hidden.f16_data(), rows, position, 0);  // slot_id=0 for MTP
        allocate_half(mtp_normalized_output, hidden_elements, hidden_shape);
        norm(mtp_norm, mtp_next_hidden.f16_data(),
             mtp_normalized_output.f16_data(), rows, hidden_size);
        const uint16_t* last_hidden = mtp_normalized_output.f16_data() +
            static_cast<size_t>(rows - 1) * hidden_size;
        QwenForwardResult result = mtp_logits_for(last_hidden, position + rows);
        allocate_half(mtp_seed_hidden, hidden_size, {config.hidden_size});
        // Recursive Qwen3.5 MTP consumes the prior predictor's returned hidden,
        // and that return is after mtp.norm in both vLLM and SGLang.
        check_device(memcpy_d2d(mtp_seed_hidden.data, last_hidden,
                                static_cast<size_t>(hidden_size) * sizeof(uint16_t)),
                     "Qwen MTP seed hidden copy");
        mtp_position = position + rows;
        mtp_seed_input_token = tokens.back();
        mtp_next_token = result.top_token;
        mtp_next_logit = result.top_logit;
        mtp_next_checksum = result.checksum;
        mtp_seed_ready = true;
        return result;
    }

    QwenForwardResult mtp_forward_row(int token, const uint16_t* hidden,
                                      int position) {
        return mtp_forward_rows({token}, hidden, position);
    }

    QwenForwardResult prime_target_mtp(const std::vector<int>& shifted_tokens,
                                       int position) {
        const int rows = static_cast<int>(shifted_tokens.size());
        const int hidden_size = static_cast<int>(config.hidden_size);
        if (rows <= 0 || target_hidden_rows.data == nullptr ||
            target_hidden_rows.nbytes < static_cast<uint64_t>(rows) * hidden_size *
                sizeof(uint16_t)) {
            throw std::runtime_error(
                "Qwen MTP target prime requires matching target hidden rows");
        }
        return mtp_forward_rows(shifted_tokens, target_hidden_rows.f16_data(),
                                position);
    }

    QwenForwardResult seed_mtp(int input_token) {
        if (!has_target_last_hidden) {
            throw std::runtime_error(
                "Qwen MTP seed requires a committed target hidden");
        }
        return mtp_forward_row(input_token, target_last_hidden.f16_data(),
                               mtp_position - 1);
    }

    void rewrite_mtp_boundary(int input_token, int position) {
        if (!has_target_last_hidden || position < 0 || position >= max_context) {
            throw std::runtime_error(
                "Qwen MTP boundary rewrite requires a committed target hidden");
        }
        (void)mtp_forward_row(input_token, target_last_hidden.f16_data(), position);
    }

    std::vector<int> draft_tokens(int count, int input_token,
                                  QwenMtpStats* stats) {
        if (count <= 0) return {};
        const auto started = std::chrono::steady_clock::now();
        QwenForwardResult next;
        if (mtp_seed_ready && mtp_seed_input_token == input_token) {
            next.top_token = mtp_next_token;
            next.top_logit = mtp_next_logit;
            next.checksum = mtp_next_checksum;
        } else {
            next = seed_mtp(input_token);
        }
        std::vector<int> drafts;
        drafts.reserve(static_cast<size_t>(count));
        drafts.push_back(next.top_token);
        while (static_cast<int>(drafts.size()) < count) {
            next = mtp_forward_row(
                drafts.back(), mtp_seed_hidden.f16_data(), mtp_position);
            drafts.push_back(next.top_token);
        }
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            stats->proposed_drafts += static_cast<uint64_t>(drafts.size());
        }
        return drafts;
    }

    QwenForwardResult dflash2_speculative_step(int input_token,
                                               QwenMtpStats* stats) {
        if (!dflash2_enabled) {
            throw std::runtime_error("Qwen DFlash2 speculative step is disabled");
        }
        const int committed_position = dflash2->committed_position();
        prepare_transaction_state();
        const auto draft_started = std::chrono::steady_clock::now();
        const QwenDFlash2Proposal proposal = dflash2->propose(input_token);
        // The 48 gated-delta layers recur sequentially across verify rows, so a
        // verify block costs close to linearly in its width. When acceptance runs
        // well below the full seven drafts the tail rows are paid for and then
        // discarded, so allow the block to be capped. 0 keeps the full proposal.
        std::vector<int> draft_tokens_used = proposal.tokens;
        int width_limit = dflash2_draft_width;
        if (dflash2_adaptive_width) {
            // Verify one row past the accepted-count average, floored at two so a
            // recovering prompt can always re-earn width, and never above the
            // explicit cap when both are set.
            const int adaptive =
                dflash2_width_samples == 0
                    ? static_cast<int>(proposal.tokens.size())
                    : std::max(2, static_cast<int>(dflash2_accept_ewma + 1.5));
            width_limit = width_limit > 0 ? std::min(width_limit, adaptive) : adaptive;
        }
        if (width_limit > 0 &&
            static_cast<int>(draft_tokens_used.size()) > width_limit) {
            draft_tokens_used.resize(static_cast<size_t>(width_limit));
        }
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - draft_started).count();
            stats->proposed_drafts += draft_tokens_used.size();
        }
        std::vector<int> verify_inputs;
        verify_inputs.reserve(draft_tokens_used.size() + 1);
        verify_inputs.push_back(input_token);
        verify_inputs.insert(verify_inputs.end(), draft_tokens_used.begin(),
                             draft_tokens_used.end());
        QwenVerifyBatch verify;
        const auto verify_started = std::chrono::steady_clock::now();
        (void)run_chunk(verify_inputs, committed_position,
                        static_cast<int>(layers.size()), false, &verify);
        if (stats != nullptr) {
            stats->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_started).count();
            ++stats->verify_count;
        }
        int correct = 0;
        while (correct < static_cast<int>(draft_tokens_used.size()) &&
               verify.top_tokens[static_cast<size_t>(correct)] ==
                   draft_tokens_used[static_cast<size_t>(correct)]) {
            ++correct;
        }
        if (stats != nullptr) stats->correct_drafts += static_cast<uint64_t>(correct);
        if (dflash2_adaptive_width) {
            // A block that accepted every row it verified is censored: the true
            // acceptance may be higher, so credit it one extra row to let the
            // width grow back. Otherwise `correct` is the exact observation.
            const double observed =
                correct == static_cast<int>(draft_tokens_used.size())
                    ? static_cast<double>(correct) + 1.0
                    : static_cast<double>(correct);
            constexpr double kAlpha = 0.25;
            dflash2_accept_ewma = dflash2_width_samples == 0
                                      ? observed
                                      : dflash2_accept_ewma +
                                            kAlpha * (observed - dflash2_accept_ewma);
            ++dflash2_width_samples;
        }
        const size_t bonus_row = static_cast<size_t>(correct);
        const int bonus = verify.top_tokens[bonus_row];
        const float bonus_logit = verify.top_logits[bonus_row];
        const float bonus_checksum = verify.local_logits[bonus_row];
        if (correct != static_cast<int>(draft_tokens_used.size())) {
            if (stats != nullptr) {
                ++stats->rollback_count;
                stats->replay_tokens += static_cast<uint64_t>(correct + 1);
            }
            const auto replay_started = std::chrono::steady_clock::now();
            restore_transaction_state(committed_position);
            std::vector<int> replay(verify_inputs.begin(),
                                    verify_inputs.begin() + correct + 1);
            (void)run_chunk(replay, committed_position,
                            static_cast<int>(layers.size()), false);
            if (stats != nullptr) {
                stats->replay_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - replay_started).count();
            }
        }
        QwenForwardResult result;
        result.top_token = bonus;
        result.bonus_token = bonus;
        result.correct_drafts = correct;
        result.accept_tokens.assign(draft_tokens_used.begin(),
                                    draft_tokens_used.begin() + correct);
        result.accept_logits.assign(verify.top_logits.begin(),
                                    verify.top_logits.begin() + correct);
        result.accept_checksums.assign(verify.local_logits.begin(),
                                      verify.local_logits.begin() + correct);
        result.layers = static_cast<int>(layers.size());
        result.dim = static_cast<int>(config.hidden_size);
        result.logits = static_cast<int>(config.vocab_size);
        result.top_logit = bonus_logit;
        result.checksum = bonus_checksum;
        result.position = dflash2->committed_position();
        return result;
    }

    QwenForwardResult dspark_speculative_step(int input_token,
                                              QwenMtpStats* stats) {
        if (!dspark_enabled) {
            throw std::runtime_error("Qwen DSpark speculative step is disabled");
        }
        const int committed_position = dspark->committed_position();
        prepare_transaction_state();
        const auto draft_started = std::chrono::steady_clock::now();
        QwenDSparkProposal proposal;
        {
            RangeScope scope("qwen.dspark.draft");
            proposal = dspark->propose(input_token);
        }
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - draft_started).count();
            stats->proposed_drafts += proposal.tokens.size();
            for (float confidence : proposal.confidences) {
                if (!std::isfinite(confidence)) {
                    throw std::runtime_error(
                        "Qwen DSpark confidence is not finite");
                }
                if (stats->confidence_count == 0) {
                    stats->confidence_min = confidence;
                    stats->confidence_max = confidence;
                } else {
                    stats->confidence_min = std::min(
                        stats->confidence_min, confidence);
                    stats->confidence_max = std::max(
                        stats->confidence_max, confidence);
                }
                stats->confidence_sum += confidence;
                ++stats->confidence_count;
            }
        }
        std::vector<int> verify_inputs;
        verify_inputs.reserve(proposal.tokens.size() + 1);
        verify_inputs.push_back(input_token);
        verify_inputs.insert(verify_inputs.end(), proposal.tokens.begin(),
                             proposal.tokens.end());
        QwenVerifyBatch verify;
        const auto verify_started = std::chrono::steady_clock::now();
        {
            RangeScope scope("qwen.target.verify");
            (void)run_chunk(verify_inputs, committed_position,
                            static_cast<int>(layers.size()), false, &verify);
        }
        if (stats != nullptr) {
            stats->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_started).count();
            ++stats->verify_count;
        }
        int correct = 0;
        while (correct < static_cast<int>(proposal.tokens.size()) &&
               verify.top_tokens[static_cast<size_t>(correct)] ==
                   proposal.tokens[static_cast<size_t>(correct)]) {
            ++correct;
        }
        if (stats != nullptr) stats->correct_drafts += correct;
        const size_t bonus_row = static_cast<size_t>(correct);
        const int bonus = verify.top_tokens[bonus_row];
        const float bonus_logit = verify.top_logits[bonus_row];
        const float bonus_checksum = verify.local_logits[bonus_row];
        if (correct != static_cast<int>(proposal.tokens.size())) {
            if (stats != nullptr) {
                ++stats->rollback_count;
                stats->replay_tokens += static_cast<uint64_t>(correct + 1);
            }
            const auto replay_started = std::chrono::steady_clock::now();
            restore_transaction_state(committed_position);
            std::vector<int> replay(verify_inputs.begin(),
                                    verify_inputs.begin() + correct + 1);
            (void)run_chunk(replay, committed_position,
                            static_cast<int>(layers.size()), false);
            if (stats != nullptr) {
                stats->replay_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - replay_started).count();
            }
        }
        QwenForwardResult result;
        result.top_token = bonus;
        result.bonus_token = bonus;
        result.correct_drafts = correct;
        result.accept_tokens.assign(proposal.tokens.begin(),
                                    proposal.tokens.begin() + correct);
        result.accept_logits.assign(verify.top_logits.begin(),
                                    verify.top_logits.begin() + correct);
        result.accept_checksums.assign(verify.local_logits.begin(),
                                       verify.local_logits.begin() + correct);
        result.layers = static_cast<int>(layers.size());
        result.dim = static_cast<int>(config.hidden_size);
        result.logits = static_cast<int>(config.vocab_size);
        result.top_logit = bonus_logit;
        result.checksum = bonus_checksum;
        result.position = dspark->committed_position();
        return result;
    }

    QwenForwardResult speculative_step(int input_token, int draft_count,
                                       QwenMtpStats* stats) {
        if (!mtp_enabled || draft_count <= 0) {
            return run_chunk({input_token}, mtp_position,
                             static_cast<int>(layers.size()), true);
        }
        const int committed_position = mtp_position;
        prepare_transaction_state();
        const std::vector<int> drafts = draft_tokens(draft_count, input_token, stats);
        std::vector<int> verify_inputs;
        verify_inputs.reserve(drafts.size() + 1);
        verify_inputs.push_back(input_token);
        verify_inputs.insert(verify_inputs.end(), drafts.begin(), drafts.end());
        QwenVerifyBatch verify;
        const auto verify_started = std::chrono::steady_clock::now();
        (void)run_chunk(verify_inputs, committed_position,
                        static_cast<int>(layers.size()), false, &verify);
        if (stats != nullptr) {
            stats->verify_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verify_started).count();
        }
        int correct = 0;
        while (correct < draft_count &&
               verify.top_tokens[static_cast<size_t>(correct)] ==
                   drafts[static_cast<size_t>(correct)]) {
            ++correct;
        }
        if (stats != nullptr) {
            ++stats->verify_count;
            stats->correct_drafts += static_cast<uint64_t>(correct);
        }
        const size_t bonus_row = static_cast<size_t>(correct);
        const int bonus = verify.top_tokens[bonus_row];
        const float bonus_logit = verify.top_logits[bonus_row];
        const float bonus_checksum = verify.local_logits[bonus_row];
        const bool full_accept = correct == draft_count;
        if (!full_accept) {
            if (stats != nullptr) {
                ++stats->rollback_count;
                stats->replay_tokens += static_cast<uint64_t>(correct + 1);
            }
            const auto replay_started = std::chrono::steady_clock::now();
            restore_transaction_state(committed_position);
            std::vector<int> replay(verify_inputs.begin(),
                                    verify_inputs.begin() + correct + 1);
            (void)run_chunk(replay, committed_position,
                            static_cast<int>(layers.size()), false);
            if (stats != nullptr) {
                stats->replay_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - replay_started).count();
            }
        }
        // Rebind the last committed MTP row to the target bonus. On full accept
        // this fills the one row beyond the recursively proposed drafts; on
        // rejection it overwrites the stale speculative suffix after replay.
        const auto seed_started = std::chrono::steady_clock::now();
        rewrite_mtp_boundary(bonus, committed_position + correct);
        if (stats != nullptr) {
            stats->draft_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - seed_started).count();
        }
        // The target state now ends after the accepted input sequence. The
        // bonus is the next input and must not be consumed by target yet. The
        // MTP boundary row is already primed with that bonus for the next round.
        QwenForwardResult result;
        result.top_token = bonus;
        result.bonus_token = bonus;
        result.correct_drafts = correct;
        result.accept_tokens.assign(drafts.begin(), drafts.begin() + correct);
        result.accept_logits.assign(verify.top_logits.begin(),
                                    verify.top_logits.begin() + correct);
        result.accept_checksums.assign(verify.local_logits.begin(),
                                       verify.local_logits.begin() + correct);
        result.layers = static_cast<int>(layers.size());
        result.dim = static_cast<int>(config.hidden_size);
        result.logits = static_cast<int>(config.vocab_size);
        result.top_logit = bonus_logit;
        result.checksum = bonus_checksum;
        result.position = mtp_position;
        return result;
    }

    uint64_t weights_vocab_start() const {
        return static_cast<uint64_t>(options.tp_rank) *
               config.vocab_size / options.tp_world;
    }

    QwenForwardResult run_chunk(const std::vector<int>& token_ids,
                                int position_offset, int active_layers,
                                bool compute_logits,
                                QwenVerifyBatch* verify_batch = nullptr,
                                const std::vector<int>* mtp_shifted_tokens = nullptr,
                                int slot_id = 0) {
        if (token_ids.empty()) {
            throw std::runtime_error("Qwen forward requires at least one token");
        }
        if (position_offset < 0 ||
            position_offset + static_cast<int>(token_ids.size()) > max_context) {
            throw std::runtime_error("Qwen context length exceeded");
        }
        const int rows = static_cast<int>(token_ids.size());
        local_tokens = token_ids;
        allocate(d_tokens, local_tokens.size() * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        check_device(memcpy_h2d(d_tokens.data, local_tokens.data(), local_tokens.size() * sizeof(int)), "Qwen token upload");
        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const std::vector<uint64_t> hidden_shape = {
            static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)};
        allocate_half(hidden_a, hidden_elements, hidden_shape);
        allocate_half(hidden_b, hidden_elements, hidden_shape);
        require_launch(qwen_embedding_fp16_gather_f16(
            embed.f16_data(), static_cast<int*>(d_tokens.data),
            hidden_a.f16_data(), rows, hidden_size,
            static_cast<int>(weights_vocab_start()),
            static_cast<int>(embed.shape[0])), "Qwen FP16 embedding lookup");
        all_reduce_half(hidden_a.f16_data(), rows * hidden_size, "hidden_a");
        uint16_t* hidden = hidden_a.f16_data();
        uint16_t* output = hidden_b.f16_data();
        const int dspark_tap_count = dspark_enabled
            ? static_cast<int>(dspark_config->target_layer_ids.size()) : 0;
        const std::vector<int>* dflash2_tap_layers = dflash2_enabled
            ? &dflash2_config->target_layer_ids
            : (!dflash2_debug_target_layer_ids.empty()
                   ? &dflash2_debug_target_layer_ids : nullptr);
        const int dflash2_tap_count = dflash2_tap_layers != nullptr
            ? static_cast<int>(dflash2_tap_layers->size()) : 0;
        if (dspark_enabled) {
            allocate_half(
                dspark_target_taps,
                static_cast<size_t>(rows) * hidden_size * dspark_tap_count,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(hidden_size * dspark_tap_count)});
        }
        if (dflash2_tap_count != 0) {
            allocate_half(
                dflash2_target_taps,
                static_cast<size_t>(rows) * hidden_size * dflash2_tap_count,
                {static_cast<uint64_t>(rows),
                 static_cast<uint64_t>(hidden_size * dflash2_tap_count)});
        }
        int dspark_tap_index = 0;
        int dflash2_tap_index = 0;
        for (int layer_index = 0; layer_index < active_layers; ++layer_index) {
            {
                PhaseScope scope(this, rows == 1 ? "STACK.d" : "STACK.r");
                layer_forward(layers[static_cast<size_t>(layer_index)], hidden,
                              output, rows, position_offset, slot_id);
            }
            std::swap(hidden, output);
            if (dspark_enabled && dspark_tap_index < dspark_tap_count &&
                layer_index == dspark_config->target_layer_ids[
                    static_cast<size_t>(dspark_tap_index)]) {
                // Store [row, tap, hidden] as the row-major concat expected by
                // fc.weight [hidden, tap_count * hidden]. One pitched copy keeps
                // the exact layout without issuing one CUDA copy per target row.
                require_launch(qwen_copy_rows_strided_f16(
                    hidden, hidden_size,
                    dspark_target_taps.f16_data() +
                        static_cast<size_t>(dspark_tap_index) * hidden_size,
                    dspark_tap_count * hidden_size, rows, hidden_size),
                    "Qwen DSpark target tap copy");
                ++dspark_tap_index;
            }
            if (dflash2_tap_layers != nullptr &&
                dflash2_tap_index < dflash2_tap_count &&
                layer_index == (*dflash2_tap_layers)[
                    static_cast<size_t>(dflash2_tap_index)]) {
                require_launch(qwen_copy_rows_strided_f16(
                    hidden, hidden_size,
                    dflash2_target_taps.f16_data() +
                        static_cast<size_t>(dflash2_tap_index) * hidden_size,
                    dflash2_tap_count * hidden_size, rows, hidden_size),
                    "Qwen DFlash2 target tap copy");
                ++dflash2_tap_index;
            }
        }
        if (dspark_enabled) {
            if (dspark_tap_index != dspark_tap_count) {
                throw std::runtime_error("Qwen DSpark target taps were not captured");
            }
            if (dspark->committed_position() != position_offset) {
                if (position_offset <= dspark->committed_position()) {
                    dspark->crop_context(position_offset);
                } else {
                    throw std::runtime_error(
                        "Qwen DSpark target context has a position gap");
                }
            }
            dspark->append_target_taps(dspark_target_taps.f16_data(), rows,
                                       position_offset);
        }
        if (dflash2_tap_layers != nullptr) {
            if (dflash2_tap_index != dflash2_tap_count) {
                throw std::runtime_error("Qwen DFlash2 target taps were not captured");
            }
            if (dflash2_enabled) {
                if (dflash2->committed_position() != position_offset) {
                    if (position_offset <= dflash2->committed_position()) {
                        dflash2->crop_context(position_offset);
                    } else {
                        throw std::runtime_error(
                            "Qwen DFlash2 target context has a position gap");
                    }
                }
                dflash2->append_target_taps(dflash2_target_taps.f16_data(), rows,
                                            position_offset);
            } else if (dflash2_target_debug_callback) {
                QwenDFlash2DebugTensor tensor;
                tensor.name = "target_taps";
                tensor.dtype = QwenDFlash2DebugDType::F16;
                tensor.shape = dflash2_target_taps.shape;
                tensor.bytes.resize(static_cast<size_t>(dflash2_target_taps.nbytes));
                check_device(memcpy_d2h(tensor.bytes.data(), dflash2_target_taps.data,
                                        tensor.bytes.size()),
                             "Qwen DFlash2 target tap debug copy");
                dflash2_target_debug_callback(tensor);
            }
        }
        if (mtp_enabled) {
            allocate_half(target_hidden_rows, hidden_elements, hidden_shape);
            // Qwen3.5 MTP consumes the target model's returned hidden states,
            // which are after the target final RMSNorm (matching vLLM/SGLang).
            norm(final_norm, hidden, target_hidden_rows.f16_data(), rows,
                 hidden_size);
            allocate_half(target_last_hidden, hidden_size,
                          {config.hidden_size});
            check_device(memcpy_d2d(target_last_hidden.data,
                                    target_hidden_rows.f16_data() + static_cast<size_t>(rows - 1) * hidden_size,
                                    static_cast<size_t>(hidden_size) * sizeof(uint16_t)),
                         "Qwen target last hidden copy");
            has_target_last_hidden = true;
            mtp_seed_ready = false;
            mtp_position = position_offset + rows;
            if (mtp_shifted_tokens != nullptr) {
                if (mtp_shifted_tokens->size() != token_ids.size()) {
                    throw std::runtime_error(
                        "Qwen MTP shifted-token extent does not match target rows");
                }
                (void)prime_target_mtp(*mtp_shifted_tokens, position_offset);
            }
        }
        if (verify_batch != nullptr) {
            *verify_batch = target_logits_for(
                hidden, rows, position_offset + rows);
        }
        if (compute_logits) {
            if (verify_batch != nullptr) {
                QwenForwardResult result;
                result.layers = active_layers;
                result.dim = hidden_size;
                result.logits = static_cast<int>(config.vocab_size);
                result.top_token = verify_batch->top_tokens.back();
                result.top_logit = verify_batch->top_logits.back();
                result.checksum = verify_batch->local_logits.back();
                result.position = position_offset + rows;
                return result;
            }
            return logits_for(
                hidden, rows - 1, position_offset + rows, active_layers);
        }
        QwenForwardResult result;
        result.layers = active_layers;
        result.dim = hidden_size;
        result.logits = static_cast<int>(config.vocab_size);
        result.position = position_offset + rows;
        return result;
    }

    // One decode step for `rows` independent sequences in a single forward pass.
    //
    // This is not run_chunk with a wider row count: there, the rows are
    // consecutive tokens of one sequence sharing a position offset and a KV
    // slot. Here each row is its own sequence with its own position, its own KV
    // slot, and its own recurrent state, which is why the per-row metadata goes
    // to the device and the attention and recurrence kernels switch to their
    // batched variants. Everything between those points -- embedding, norms,
    // projections, SwiGLU, the LM head -- is already row-wise and is reused
    // unchanged. Amortizing the weight loads over the rows is the whole point:
    // decode is weight-bandwidth bound, so the batch is close to free.
    QwenVerifyBatch run_batched_decode(const std::vector<int>& tokens,
                                       const std::vector<int>& positions,
                                       const std::vector<int>& slot_ids,
                                       int active_layers) {
        const int rows = static_cast<int>(tokens.size());
        if (rows <= 0) {
            throw std::runtime_error("Qwen batched decode requires a row");
        }
        if (positions.size() != tokens.size() ||
            slot_ids.size() != tokens.size()) {
            throw std::runtime_error(
                "Qwen batched decode row metadata has mismatched extents");
        }
        // The speculative and MTP paths each keep a single-sequence context
        // (committed position, target taps, seed hidden) that a batch of
        // independent sequences would interleave into nonsense. They are
        // rejected rather than silently producing a wrong result.
        if (mtp_enabled || dspark_enabled || dflash2_enabled) {
            throw std::runtime_error(
                "Qwen batched decode is not compatible with MTP or speculative "
                "decoding");
        }
#ifdef POCKET_BACKEND_ASCEND
        throw std::runtime_error(
            "Qwen batched decode has no Ascend kernels yet");
#else
        // The batched append and attention kernels are FP16-cache only. A
        // quantized cache would have the FP16 append write raw halves into
        // packed codes, so it is refused rather than silently corrupted.
        if (options.kv_cache_dtype != QwenKvCacheDType::Fp16) {
            throw std::runtime_error(
                "Qwen batched decode requires an FP16 KV cache");
        }
        std::vector<int> context_lens(static_cast<size_t>(rows));
        int max_context_len = 0;
        for (int row = 0; row < rows; ++row) {
            const int position = positions[static_cast<size_t>(row)];
            const int slot = slot_ids[static_cast<size_t>(row)];
            if (position < 0 || position >= max_context) {
                throw std::runtime_error(
                    "Qwen batched decode position is out of range");
            }
            if (slot < 0 || slot >= max_batch_size) {
                throw std::runtime_error(
                    "Qwen batched decode slot is out of range");
            }
            context_lens[static_cast<size_t>(row)] = position + 1;
            max_context_len = std::max(max_context_len, position + 1);
        }

        const size_t row_bytes = static_cast<size_t>(rows) * sizeof(int);
        allocate(batch_positions, row_bytes, {static_cast<uint64_t>(rows)},
                 SafeDType::I64);
        allocate(batch_slot_ids, row_bytes, {static_cast<uint64_t>(rows)},
                 SafeDType::I64);
        allocate(batch_context_lens, row_bytes, {static_cast<uint64_t>(rows)},
                 SafeDType::I64);
        check_device(memcpy_h2d(batch_positions.data, positions.data(), row_bytes),
                     "Qwen batched decode position upload");
        check_device(memcpy_h2d(batch_slot_ids.data, slot_ids.data(), row_bytes),
                     "Qwen batched decode slot upload");
        check_device(memcpy_h2d(batch_context_lens.data, context_lens.data(),
                                row_bytes),
                     "Qwen batched decode context length upload");

        BatchRows metadata;
        metadata.positions = static_cast<const int*>(batch_positions.data);
        metadata.slot_ids = static_cast<const int*>(batch_slot_ids.data);
        metadata.context_lens = static_cast<const int*>(batch_context_lens.data);
        metadata.rows = rows;
        metadata.max_context_len = max_context_len;

        const int hidden_size = static_cast<int>(config.hidden_size);
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden_size;
        const std::vector<uint64_t> hidden_shape = {
            static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden_size)};
        local_tokens = tokens;
        allocate(d_tokens, local_tokens.size() * sizeof(int),
                 {static_cast<uint64_t>(rows)}, SafeDType::I64);
        check_device(memcpy_h2d(d_tokens.data, local_tokens.data(),
                                local_tokens.size() * sizeof(int)),
                     "Qwen batched decode token upload");
        allocate_half(hidden_a, hidden_elements, hidden_shape);
        allocate_half(hidden_b, hidden_elements, hidden_shape);
        require_launch(qwen_embedding_fp16_gather_f16(
            embed.f16_data(), static_cast<int*>(d_tokens.data),
            hidden_a.f16_data(), rows, hidden_size,
            static_cast<int>(weights_vocab_start()),
            static_cast<int>(embed.shape[0])),
            "Qwen batched decode embedding lookup");
        all_reduce_half(hidden_a.f16_data(), rows * hidden_size, "hidden_a");

        uint16_t* hidden = hidden_a.f16_data();
        uint16_t* output = hidden_b.f16_data();
        // Scoped so an exception inside a layer cannot leave the batched flag
        // set and silently redirect the next single-sequence forward.
        struct BatchScope {
            Impl* impl;
            explicit BatchScope(Impl* self, const BatchRows* rows) : impl(self) {
                impl->batch_rows = rows;
            }
            ~BatchScope() { impl->batch_rows = nullptr; }
        } scope(this, &metadata);
        for (int layer_index = 0; layer_index < active_layers; ++layer_index) {
            {
                PhaseScope phase(this, "STACK.b");
                // position_offset and slot_id are unused on the batched path:
                // both are per row and come from the uploaded metadata.
                layer_forward(layers[static_cast<size_t>(layer_index)], hidden,
                              output, rows, 0, 0);
            }
            std::swap(hidden, output);
        }
        // One LM head GEMM over all rows, which is the same amortization the
        // rest of the step gets. position_after is reported per row by the
        // caller, so the batch-wide value here is only the widest context.
        return target_logits_for(hidden, rows, max_context_len);
#endif
    }

    uint64_t activation_capacity_bytes() const {
        return hidden_a.capacity + hidden_b.capacity + d_tokens.capacity +
               argmax_token.capacity + argmax_logit.capacity +
               target_hidden_rows.capacity + target_last_hidden.capacity +
               mtp_seed_hidden.capacity + mtp_embedding.capacity +
               mtp_normalized_embedding.capacity +
               mtp_normalized_hidden.capacity + mtp_concat.capacity +
               mtp_fused.capacity + mtp_next_hidden.capacity +
               mtp_normalized_output.capacity + dspark_target_taps.capacity +
               dflash2_target_taps.capacity + workspace.capacity_bytes() +
               nvfp4_q8.capacity + nvfp4_q8_scale.capacity +
               (dspark != nullptr ? dspark->activation_workspace_bytes() : 0) +
               (dflash2 != nullptr ? dflash2->activation_workspace_bytes() : 0);
    }
};

QwenEngine::QwenEngine(const std::string& ckpt_dir,
                       const QwenEngineOptions& options, int layer_count,
                       int max_context)
    : ckpt_dir_(ckpt_dir), options_(options),
      config_(QwenConfig::from_hf_config(ckpt_dir)), index_(ckpt_dir),
      weights_(index_, config_, options.tp_world, options.tp_rank) {
    if (options_.device < 0) options_.device = options_.tp_rank;
    const int external_drafter_count =
        (!options_.dspark_checkpoint.empty() ? 1 : 0) +
        (!options_.dflash2_checkpoint.empty() ? 1 : 0);
    if (options_.mtp && external_drafter_count != 0) {
        throw std::runtime_error(
            "Qwen native MTP cannot be combined with an external drafter");
    }
    if (external_drafter_count > 1) {
        throw std::runtime_error(
            "Qwen DSpark and DFlash2 are mutually exclusive");
    }
    if ((options_.mtp || external_drafter_count != 0) && layer_count > 0 &&
        layer_count != static_cast<int>(config_.num_hidden_layers)) {
        throw std::runtime_error(
            "Qwen speculative decoding requires the complete target model; partial "
            "--smoke-layers would verify drafts against a different model");
    }
    if (options_.prefill_chunk_tokens <= 0) {
        throw std::runtime_error("Qwen prefill chunk size must be positive");
    }
    if (options_.attention_window < 0 || options_.attention_sink_tokens < 0) {
        throw std::runtime_error("Qwen attention window and sink must not be negative");
    }
    if (options_.attention_window == 0 && options_.attention_sink_tokens != 0) {
        throw std::runtime_error(
            "Qwen attention sink requires a nonzero attention window");
    }
    if (options_.kv_cache_dtype == QwenKvCacheDType::Fp8 &&
        options_.attention_window > 0) {
        throw std::runtime_error(
            "Qwen sparse attention currently requires an FP16 KV cache");
    }
    if (options_.kv_paged) {
        // FP16 only. The quantized caches carry their own scale and packed-slot
        // offset arithmetic, and paging them would mean reworking each one's
        // addressing with no coverage from the batched path, which is FP16
        // already. Refusing is what keeps the restriction visible instead of
        // silently degrading to a contiguous cache.
        if (options_.kv_cache_dtype != QwenKvCacheDType::Fp16) {
            throw std::runtime_error(
                std::string("Qwen paged KV cache requires an FP16 cache; got ") +
                qwen_kv_cache_dtype_name(options_.kv_cache_dtype));
        }
        if (options_.kv_block_size <= 0) {
            throw std::runtime_error(
                "Qwen paged KV block size must be positive");
        }
    }
#ifdef POCKET_BACKEND_ASCEND
    // full_attention only compiles the FP16 cache path on this backend, so the
    // rejection belongs here where the option is still reportable rather than
    // deep inside a layer.
    if (options_.kv_cache_dtype != QwenKvCacheDType::Fp16) {
        throw std::runtime_error(
            "Qwen Ascend backend currently supports only an FP16 KV cache");
    }
    if (options_.temperature > 1.0e-5f) {
        throw std::runtime_error(
            "Qwen sampling is not implemented on the Ascend backend; use greedy "
            "decoding (--temperature 0)");
    }
#endif
    if (!device_set(options_.device)) {
        throw std::runtime_error(
            std::string("failed to set Qwen ") + device_backend_name() +
            " device");
    }
    if (layer_count <= 0) {
        active_layers_ = static_cast<int>(config_.num_hidden_layers);
    } else if (layer_count <= static_cast<int>(config_.num_hidden_layers)) {
        active_layers_ = layer_count;
    } else {
        throw std::runtime_error("Qwen layer count exceeds model depth");
    }
    max_context_ = max_context > 0
        ? max_context : static_cast<int>(config_.max_position_embeddings);
    if (max_context_ <= 0 ||
        max_context_ > static_cast<int>(config_.max_position_embeddings)) {
        throw std::runtime_error("Qwen max context exceeds model configuration");
    }
    impl_ = new Impl(index_, config_, weights_, options_, max_context_,
                     active_layers_);
    resident_weight_bytes_ = impl_->uploaded_weight_bytes;
    resident_scale_bytes_ = impl_->uploaded_scale_bytes;
}

QwenEngine::~QwenEngine() {
    delete impl_;
    impl_ = nullptr;
}

uint64_t QwenEngine::verify_weight_bytes() const {
    return impl_->verify_weight_bytes;
}

uint64_t QwenEngine::activation_workspace_peak_bytes() const {
    return impl_->activation_capacity_bytes();
}

uint64_t QwenEngine::kv_cache_bytes() const {
    return impl_->cache_data_bytes;
}

uint64_t QwenEngine::kv_cache_scale_bytes() const {
    return impl_->cache_scale_bytes;
}

QwenRuntimeTelemetry QwenEngine::runtime_telemetry() const {
    return impl_->telemetry;
}

void QwenEngine::set_dflash2_debug_callback(
    QwenDFlash2DebugCallback callback) {
    if (!impl_->dflash2_enabled || impl_->dflash2 == nullptr) {
        throw std::runtime_error("Qwen DFlash2 debug callback requires --qwen-dflash2");
    }
    impl_->dflash2->set_debug_callback(std::move(callback));
}

QwenForwardResult QwenEngine::debug_prefill_dflash2(
    const std::vector<int>& token_ids,
    const std::vector<int>& target_layer_ids,
    QwenDFlash2DebugCallback callback) {
    if (!callback || target_layer_ids.empty()) {
        throw std::runtime_error("Qwen DFlash2 debug prefill requires target taps");
    }
    int previous = -1;
    for (int layer : target_layer_ids) {
        if (layer <= previous || layer < 0 || layer >= active_layers_) {
            throw std::runtime_error("invalid Qwen DFlash2 debug target layer IDs");
        }
        previous = layer;
    }
    clear_prefix_cache();
    impl_->dflash2_debug_target_layer_ids = target_layer_ids;
    impl_->dflash2_target_debug_callback = std::move(callback);
    try {
        QwenForwardResult result = prefill(token_ids);
        impl_->dflash2_debug_target_layer_ids.clear();
        impl_->dflash2_target_debug_callback = {};
        return result;
    } catch (...) {
        impl_->dflash2_debug_target_layer_ids.clear();
        impl_->dflash2_target_debug_callback = {};
        throw;
    }
}

void QwenEngine::warmup_tp() {
    std::optional<Impl::RangeScope> range;
    if (impl_->range_profile) range.emplace("qwen.warmup_tp");
    if (options_.tp_world == 1) return;

    // Initialize command channel for worker loop
    if (!impl_->cmd && !options_.nccl_id_path.empty()) {
        impl_->cmd = CmdChannel::create(options_.tp_world, options_.tp_rank,
                                        options_.nccl_id_path);
    }

    QwenDeviceTensor scratch;
    allocate_half(scratch, 1, {1});
    zero_tensor(scratch);
    impl_->all_reduce_half(scratch.f16_data(), 1, "scratch");
    check_device(device_synchronize(), "Qwen TP warmup synchronization");
}

void QwenEngine::reset() {
    clear_prefix_cache();
}

void QwenEngine::clear_prefix_cache() {
    position_ = 0;
    prefix_stats_ = QwenPrefixCacheStats{};
    // Every slot's reuse state, not just slot 0's: a stale entry on any slot
    // would survive the clear and be matched by the next request there.
    for (Impl::SlotPrefixState& slot : impl_->slot_prefix) {
        slot.cached_prompt.clear();
        slot.snapshots.clear();
        slot.cached_result = QwenForwardResult{};
        slot.has_cached_result = false;
    }
    std::fill(impl_->slot_positions.begin(), impl_->slot_positions.end(), 0);
    impl_->prefix_hits = 0;
    impl_->prefix_misses = 0;
    for (int slot = 0; slot < impl_->max_batch_size; ++slot) {
        impl_->zero_recurrent_state(slot);
    }
    // KV storage is overwritten before it is read. Avoid clearing several GiB
    // on every request; position_ bounds all attention reads.
    //
    // A paged cache is the exception: its blocks are a finite pool, so clearing
    // the positions without returning the blocks would leak the whole pool over
    // a few requests. The bytes still are not touched, only the bookkeeping.
    if (impl_->kv_paged()) {
        impl_->block_table->release_all();
        impl_->sync_block_table();
    }
}

// ========== Batch API implementation (Phase 3.1) ==========

bool QwenEngine::supports_batching() const {
    return true;  // Phase 3.1: always supported, opt-in via allocate_batch_slots
}

void QwenEngine::allocate_batch_slots(int max_batch_size) {
    if (max_batch_size < 1) {
        throw std::runtime_error("QwenEngine::allocate_batch_slots: max_batch_size must be >= 1");
    }
    // The KV arena was sized at construction from options.max_batch_size and
    // cannot grow here, so this call can only ever confirm that capacity or
    // ask for less of it. Growing past it would hand out slots whose offsets
    // run off the end of the cache.
    if (max_batch_size > impl_->max_batch_size) {
        throw std::runtime_error(
            "QwenEngine::allocate_batch_slots: requested " +
            std::to_string(max_batch_size) +
            " slots but the KV cache was built for " +
            std::to_string(impl_->max_batch_size) +
            "; set QwenEngineOptions::max_batch_size before construction");
    }
    // Phase 3.4: Reject a shrink that would orphan a live request rather than
    // silently stranding its KV cache.
    if (!impl_->request_to_slot.empty()) {
        throw std::runtime_error(
            "QwenEngine::allocate_batch_slots: cannot resize while " +
            std::to_string(impl_->request_to_slot.size()) +
            " request(s) still hold slots");
    }

    impl_->max_batch_size = max_batch_size;
    impl_->batch_mode_enabled = (max_batch_size > 1);

    // Phase 3.2: the multi-slot KV cache itself is allocated at construction
    // time; this only rebuilds the bookkeeping. Descending order makes back()
    // hand out slot 0 first.
    impl_->free_slots.clear();
    impl_->request_to_slot.clear();
    for (int slot = max_batch_size - 1; impl_->batch_mode_enabled && slot >= 0;
         --slot) {
        impl_->free_slots.push_back(slot);
    }
}

bool QwenEngine::kv_paged() const { return impl_->kv_paged(); }

int QwenEngine::kv_free_blocks() const {
    return impl_->kv_paged() ? impl_->block_pool->free_blocks() : 0;
}

int QwenEngine::kv_total_blocks() const {
    return impl_->kv_paged() ? impl_->block_pool->total_blocks() : 0;
}

int QwenEngine::kv_blocks_for_tokens(int tokens) const {
    if (!impl_->kv_paged() || tokens <= 0) return 0;
    const int block_size = impl_->paged_block_size();
    return (tokens + block_size - 1) / block_size;
}

int QwenEngine::allocate_slot(uint64_t request_id) {
    if (!impl_->batch_mode_enabled) {
        // Single session mode: always use slot 0
        return 0;
    }

    if (impl_->free_slots.empty()) {
        return -1;  // No slots available
    }

    int slot_id = impl_->free_slots.back();
    impl_->free_slots.pop_back();
    impl_->request_to_slot[request_id] = slot_id;
    return slot_id;
}

void QwenEngine::free_slot(uint64_t request_id) {
    if (!impl_->batch_mode_enabled) {
        return;  // Single session mode: nothing to free
    }

    auto it = impl_->request_to_slot.find(request_id);
    if (it == impl_->request_to_slot.end()) {
        return;  // Request not found
    }

    int slot_id = it->second;
    impl_->request_to_slot.erase(it);
    impl_->free_slots.push_back(slot_id);
    // Returning the slot returns its blocks. Under paging this is what makes
    // capacity dynamic: the next request draws from a pool the finished one has
    // already given back, rather than inheriting a fixed reservation.
    //
    // Under TP the workers hold their own copies of the block table, so they
    // have to release the same slot too or their pools leak one row per freed
    // slot and eventually run out. The command is a no-op unless a channel
    // exists and paging is on, which keeps the contiguous path unchanged.
    impl_->release_slot_paged_state(slot_id, position_);
    worker_command_free_slot(slot_id);
}

QwenBatchPrefillResult QwenEngine::batch_prefill(
    const std::vector<QwenBatchedRequest*>& requests, int token_budget) {

    // Requests still run one at a time: the saturation sweep
    // (bench_qwen_prefill_saturation) measured 1890 tok/s at a 4096-token chunk
    // against 1330 at 512 on TP4, so a single chunk of a few thousand rows
    // already feeds the GEMMs and merging prompts into one variable-length
    // forward would buy ~1.06x at 2048. What that sweep does not fix is a long
    // prompt holding the device: with `token_budget` set, each request advances
    // by at most that many tokens per call, so the scheduler regains control
    // between chunks and can interleave decode.
    QwenBatchPrefillResult result;
    result.results.reserve(requests.size());
    result.incomplete.reserve(requests.size());
    result.total_tokens = 0;

    auto start = std::chrono::steady_clock::now();

    for (QwenBatchedRequest* req : requests) {
        if (!req || req->prompt_tokens.empty()) {
            throw std::runtime_error("QwenEngine::batch_prefill: invalid request");
        }

        // Under TP the workers sit in run_worker_loop() waiting to be told which
        // collective to join; rank 0 must announce the op before running it, or
        // the group deadlocks with rank 0 computing alone.  No-op at world size 1.
        // The budget goes across too: the workers run the same bounded loop, so
        // every rank has to stop at the same token or the next collective pairs
        // mismatched shapes.
        worker_command_prefill(req->prompt_tokens, req->slot_id, token_budget);

        // Phase 3.3: Use req->slot_id to isolate KV cache
        QwenPartialPrefillResult step =
            prefill_bounded(req->prompt_tokens, req->slot_id,
                            std::max(0, token_budget));

        result.total_tokens += step.consumed_tokens - req->seq_len;
        req->seq_len = step.consumed_tokens;
        req->last_result = step.result;
        result.results.push_back(step.result);
        result.incomplete.push_back(!step.complete);

        if (step.complete) {
            // Only a finished prompt yields a token to generate from.
            req->last_token = step.result.top_token;
            // The prompt's first predicted token can itself be a stop token, so
            // the flag has to be set here too; a scheduler that only checked it
            // during decode would emit one spurious token past the stop.
            if (is_stop_token(req->sampling, req->last_token)) req->finished = true;
            // Freeing the prompt is safe only once nothing needs to resume from
            // it; a partial row is resumed by re-matching these same tokens.
            req->prompt_tokens.clear();
        }
    }

    auto end = std::chrono::steady_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();

    return result;
}

std::vector<QwenForwardResult> QwenEngine::batch_decode_tokens(
    const std::vector<int>& tokens, const std::vector<int>& slot_ids) {
    if (tokens.size() != slot_ids.size()) {
        throw std::runtime_error(
            "QwenEngine::batch_decode_tokens: token and slot extents differ");
    }
    const int rows = static_cast<int>(tokens.size());
    if (rows == 0) return {};
    std::optional<Impl::RangeScope> range;
    if (impl_->range_profile) range.emplace("qwen.batch_decode");

    // Two rows in the same slot would have the second read the KV the first
    // wrote at the same position, so the batch has to be over distinct slots.
    std::vector<int> positions(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const int slot = slot_ids[static_cast<size_t>(row)];
        for (int earlier = 0; earlier < row; ++earlier) {
            if (slot_ids[static_cast<size_t>(earlier)] == slot) {
                throw std::runtime_error(
                    "QwenEngine::batch_decode_tokens: slot " +
                    std::to_string(slot) + " appears twice in one batch");
            }
        }
        const int position = impl_->slot_position(slot, position_);
        if (position >= max_context_) {
            throw std::runtime_error("Qwen context length exceeded");
        }
        positions[static_cast<size_t>(row)] = position;
    }

    // Every row grows by one token. Reserving all of them before the forward
    // keeps the failure at a batch boundary: a mid-forward throw would leave
    // some rows appended and others not.
    for (int row = 0; row < rows; ++row) {
        impl_->reserve_paged_slot(slot_ids[static_cast<size_t>(row)],
                                  positions[static_cast<size_t>(row)] + 1);
    }
    impl_->sync_block_table();

    QwenVerifyBatch batch = impl_->run_batched_decode(
        tokens, positions, slot_ids, active_layers_);

    std::vector<QwenForwardResult> results(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        const size_t index = static_cast<size_t>(row);
        const int slot = slot_ids[index];
        const int position_after = positions[index] + 1;
        QwenForwardResult& forward = results[index];
        forward.layers = active_layers_;
        forward.dim = static_cast<int>(config_.hidden_size);
        forward.logits = static_cast<int>(config_.vocab_size);
        forward.top_token = batch.top_tokens[index];
        forward.top_logit = batch.top_logits[index];
        forward.checksum = batch.local_logits[index];
        forward.position = position_after;
        impl_->set_slot_position(slot, position_after, position_);
        if (options_.prefix_cache) {
            Impl::SlotPrefixState& prefix = impl_->prefix_for(slot);
            prefix.cached_prompt.push_back(tokens[index]);
            prefix.cached_result = forward;
            prefix.has_cached_result = true;
            if (impl_->is_periodic_snapshot_position(position_after)) {
                impl_->record_snapshot(position_after, &forward, true, slot);
            }
        }
    }
    return results;
}

bool QwenEngine::is_stop_token(const QwenBatchSamplingParams& sampling,
                               int token) const {
    if (sampling.ignore_eos) return false;
    const std::vector<int>& stops = sampling.stop_token_ids.empty()
        ? config_.eos_token_ids
        : sampling.stop_token_ids;
    return std::find(stops.begin(), stops.end(), token) != stops.end();
}

QwenBatchDecodeResult QwenEngine::batch_decode_step(
    const std::vector<QwenBatchedRequest*>& requests) {

    QwenBatchDecodeResult result;
    result.next_tokens.reserve(requests.size());
    result.finished.reserve(requests.size());
    result.hit_stop_token.reserve(requests.size());

    if (requests.empty()) {
        return result;
    }

    auto start = std::chrono::steady_clock::now();

    std::vector<int> tokens;
    std::vector<int> slots;
    tokens.reserve(requests.size());
    slots.reserve(requests.size());
    for (QwenBatchedRequest* req : requests) {
        if (!req) {
            throw std::runtime_error("QwenEngine::batch_decode_step: null request");
        }
        tokens.push_back(req->last_token);
        slots.push_back(req->slot_id);
    }

    // One forward for the whole batch. Under TP the workers sit in
    // run_worker_loop() waiting to be told which collective to join, so rank 0
    // announces the batch before running it or the group deadlocks with rank 0
    // computing alone. No-op at world size 1.
    worker_command_batch_decode(tokens, slots);
    std::vector<QwenForwardResult> forwards = batch_decode_tokens(tokens, slots);

    for (size_t index = 0; index < requests.size(); ++index) {
        QwenBatchedRequest* req = requests[index];
        const QwenForwardResult& forward = forwards[index];
        req->last_token = forward.top_token;
        req->last_result = forward;
        req->seq_len++;
        req->generated_tokens.push_back(forward.top_token);

        // A stop token is the last token of the output, so it is still appended
        // above and counted: truncating it here would make the returned
        // sequence disagree with the KV cache this slot now holds, and a
        // resumed request would then diverge. Callers strip it when detokenizing.
        const bool stopped = is_stop_token(req->sampling, forward.top_token);
        const bool length_capped = static_cast<int>(req->generated_tokens.size()) >=
                                   req->sampling.max_new_tokens;

        req->finished = stopped || length_capped;
        result.next_tokens.push_back(forward.top_token);
        result.finished.push_back(req->finished);
        result.hit_stop_token.push_back(stopped);
    }

    auto end = std::chrono::steady_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();

    return result;
}

QwenForwardResult QwenEngine::prefill(const std::vector<int>& token_ids, int slot_id) {
    // Budget 0 means "no bound", so this is the whole prompt in one call and the
    // result is always complete.
    return prefill_bounded(token_ids, slot_id, 0).result;
}

QwenPartialPrefillResult QwenEngine::prefill_partial(
    const std::vector<int>& token_ids, int slot_id, int max_tokens) {
    return prefill_bounded(token_ids, slot_id, std::max(0, max_tokens));
}

QwenPartialPrefillResult QwenEngine::prefill_bounded(
    const std::vector<int>& token_ids, int slot_id, int max_tokens) {
    std::optional<Impl::RangeScope> range;
    if (impl_->range_profile) range.emplace("qwen.prefill");
    if (token_ids.empty()) {
        throw std::runtime_error("Qwen prefill requires at least one token");
    }
    if (token_ids.size() > static_cast<size_t>(max_context_)) {
        throw std::runtime_error("Qwen context length exceeded");
    }

    if (slot_id < 0 || slot_id >= impl_->max_batch_size) {
        throw std::runtime_error("Qwen prefill slot " + std::to_string(slot_id) +
                                 " is outside the allocated batch");
    }

    prefix_stats_ = QwenPrefixCacheStats{};
    prefix_stats_.prompt_tokens = static_cast<int>(token_ids.size());
    impl_->phase_seconds.clear();
    impl_->phase_calls.clear();
    // Prefix reuse is scoped to this slot.  Each slot keeps its own cached
    // prompt, snapshot ring and cached result, so a batch can reuse on every
    // slot independently; sharing one copy would let a prompt match against
    // another sequence's tokens and resume from its recurrent state.
    Impl::SlotPrefixState& prefix = impl_->prefix_for(slot_id);
    const int slot_position = impl_->slot_position(slot_id, position_);
    const bool can_reuse = options_.prefix_cache &&
        !prefix.cached_prompt.empty() && slot_position ==
            static_cast<int>(prefix.cached_prompt.size());
    size_t common = 0;
    if (can_reuse) {
        const size_t limit = std::min(token_ids.size(), prefix.cached_prompt.size());
        while (common < limit && token_ids[common] == prefix.cached_prompt[common]) {
            ++common;
        }
    }
    prefix_stats_.matched_tokens = static_cast<int>(common);

    // Exact repeat: the cached final logits are already the requested result.
    if (can_reuse && common == token_ids.size() &&
        token_ids.size() == prefix.cached_prompt.size() && prefix.has_cached_result) {
        ++impl_->prefix_hits;
        prefix_stats_.hits = impl_->prefix_hits;
        prefix_stats_.misses = impl_->prefix_misses;
        prefix_stats_.reused_tokens = static_cast<int>(token_ids.size());
        prefix_stats_.resume_source = "live";
        prefix_stats_.snapshots = impl_->snapshot_count(slot_id);
        prefix_stats_.snapshot_bytes = impl_->snapshot_bytes();
        return {prefix.cached_result, static_cast<int>(token_ids.size()), true};
    }

    int start_position = 0;
    const QwenRecurrentSnapshot* resume_snapshot = nullptr;
    if (can_reuse && common == prefix.cached_prompt.size() &&
        common <= token_ids.size()) {
        // The current recurrent state is exactly the state after the common
        // prefix. This is the hot path for monotonically growing prompts.
        start_position = static_cast<int>(common);
        prefix_stats_.resume_source = "live";
        ++impl_->prefix_hits;
    } else if (options_.prefix_cache && common > 0) {
        // For a branch or a shorter prompt, restore the deepest safe snapshot.
        // A request-boundary snapshot also carries the final-token result, so
        // an exact shorter-prefix request can finish without recomputation.
        resume_snapshot = impl_->snapshot_at_or_before(
            static_cast<int>(common), slot_id);
        if (resume_snapshot != nullptr &&
            resume_snapshot->position == static_cast<int>(common) &&
            resume_snapshot->has_result && token_ids.size() == common) {
            impl_->restore_recurrent_state(*resume_snapshot, true, slot_id);
            impl_->set_slot_position(slot_id, static_cast<int>(common), position_);
            prefix.cached_prompt = token_ids;
            prefix.cached_result = resume_snapshot->result;
            prefix.has_cached_result = true;
            impl_->drop_snapshots_after(static_cast<int>(common), slot_id);
            prefix_stats_.resume_source = "snapshot";
            prefix_stats_.reused_tokens = static_cast<int>(common);
            prefix_stats_.snapshots = impl_->snapshot_count(slot_id);
            prefix_stats_.snapshot_bytes = impl_->snapshot_bytes();
            ++impl_->prefix_hits;
            prefix_stats_.hits = impl_->prefix_hits;
            prefix_stats_.misses = impl_->prefix_misses;
            return {prefix.cached_result, static_cast<int>(common), true};
        }
        if (resume_snapshot != nullptr &&
            resume_snapshot->position == static_cast<int>(token_ids.size())) {
            resume_snapshot = impl_->snapshot_strictly_before(
                static_cast<int>(token_ids.size()), slot_id);
        }
        if (resume_snapshot != nullptr) {
            impl_->restore_recurrent_state(*resume_snapshot, true, slot_id);
            start_position = resume_snapshot->position;
            prefix_stats_.resume_source = "snapshot";
            ++impl_->prefix_hits;
        } else {
            impl_->zero_recurrent_state(slot_id);
            prefix_stats_.resume_source = "empty";
            ++impl_->prefix_misses;
        }
        impl_->drop_snapshots_after(start_position, slot_id);
    } else {
        // Clears this slot's rows only; a concurrent request on another slot
        // keeps its state.
        impl_->zero_recurrent_state(slot_id);
        prefix_stats_.resume_source = "empty";
        if (can_reuse) ++impl_->prefix_misses;
        impl_->drop_snapshots_after(0, slot_id);
    }

    const int target_position = static_cast<int>(token_ids.size());
    if (options_.mtp && start_position > 0 && start_position < target_position) {
        // The cached MTP row at S-1 was paired with the prior request's token
        // x[S]. An append may reuse that token, while a branch or compression
        // may replace it. Rebind the shifted-token boundary before priming the
        // new suffix so every predictor row observes target h[S-1] + new x[S].
        impl_->rewrite_mtp_boundary(
            token_ids[static_cast<size_t>(start_position)],
            start_position - 1);
    }
    QwenForwardResult result;
    // A budget both caps how far this call advances and shrinks the chunk, since
    // a budget under prefill_chunk_tokens would otherwise be rounded up to a
    // whole chunk and silently consume the entire prompt. Stopping at an
    // arbitrary token is safe: the next call resumes through the growing-prompt
    // live path, which matches the cached prompt rather than needing a snapshot
    // at the boundary.
    const int chunk_size =
        max_tokens > 0 ? std::max(1, std::min(options_.prefill_chunk_tokens, max_tokens))
                       : std::max(1, options_.prefill_chunk_tokens);
    const int budget_limit =
        max_tokens > 0
            ? std::min(target_position, start_position + std::max(1, max_tokens))
            : target_position;
    // Blocks for everything this call will write, taken before the first chunk
    // so an exhausted pool surfaces here rather than part-way through a prompt
    // whose earlier chunks already advanced the recurrent state. The block table
    // is uploaded once for the whole prefill: it does not change again until the
    // next call, so the per-chunk forwards read the same rows.
    impl_->reserve_paged_slot(slot_id, budget_limit);
    impl_->sync_block_table();
    for (int offset = start_position; offset < budget_limit;) {
        int end = std::min(budget_limit, offset + chunk_size);
        // Split at exact snapshot boundaries even when a request begins at an
        // arbitrary position. Otherwise a 512-token append starting at 100 can
        // step over 4096 forever and never create the rollback point.
        if (options_.state_snapshot_interval_tokens > 0) {
            const int next_snapshot = impl_->next_periodic_snapshot_after(offset);
            if (next_snapshot > offset && next_snapshot < end) {
                end = next_snapshot;
            }
        }
        const bool periodic_snapshot =
            impl_->is_periodic_snapshot_position(end);
        // Periodic snapshots may later serve an exact shorter-prefix request.
        // Keep the final-row logits with those snapshots so restoring one does
        // not have to recompute the whole prefix just to recover its result.
        const bool snapshot_result = end == target_position || periodic_snapshot;
        std::vector<int> chunk(
            token_ids.begin() + static_cast<ptrdiff_t>(offset),
            token_ids.begin() + static_cast<ptrdiff_t>(end));
        if (options_.mtp) {

            std::vector<int> shifted;
            shifted.reserve(chunk.size());
            for (int position = offset; position < end; ++position) {
                shifted.push_back(
                    position + 1 < target_position
                        ? token_ids[static_cast<size_t>(position + 1)]
                        : -1);
            }
            if (shifted.back() < 0) {
                // The last shifted input is the target token predicted by this
                // final prompt row, so obtain its logits before priming MTP.
                result = impl_->run_chunk(chunk, offset, active_layers_, true,
                                          nullptr, nullptr, slot_id);
                shifted.back() = result.top_token;
                (void)impl_->prime_target_mtp(shifted, offset);
            } else {
                result = impl_->run_chunk(chunk, offset, active_layers_,
                                          snapshot_result, nullptr, &shifted, slot_id);
            }
        } else {
            result = impl_->run_chunk(chunk, offset, active_layers_,
                                      snapshot_result, nullptr, nullptr, slot_id);
        }
        if (periodic_snapshot || end == target_position) {
            impl_->record_snapshot(end, &result, periodic_snapshot, slot_id);
        }

        offset = end;
    }

    // A same-length prompt resumed from an interior snapshot always computes
    // at least one token; the only zero-work case returned above was exact hit.
    if (start_position == budget_limit) {
        throw std::runtime_error("Qwen prefix cache failed to produce logits");
    }
    // Commit only what was actually consumed. A partial call must leave the slot
    // position and the cached prompt at the boundary it truly reached, because
    // the next call finds its resume point by matching against exactly these
    // two: an over-long cached_prompt would make the remainder look already
    // computed and silently skip those tokens.
    const bool complete = budget_limit == target_position;
    impl_->set_slot_position(slot_id, budget_limit, position_);
    if (options_.prefix_cache) {
        prefix.cached_prompt.assign(
            token_ids.begin(),
            token_ids.begin() + static_cast<ptrdiff_t>(budget_limit));
        prefix.cached_result = result;
        // Interior logits predict the next prompt token, not the prompt's
        // continuation, so they must not satisfy a later exact-repeat hit.
        prefix.has_cached_result = complete;
    } else {
        prefix.cached_prompt.clear();
        prefix.has_cached_result = false;
    }
    prefix_stats_.reused_tokens = start_position;
    prefix_stats_.computed_tokens = budget_limit - start_position;
    prefix_stats_.snapshots = impl_->snapshot_count(slot_id);
    prefix_stats_.snapshot_bytes = impl_->snapshot_bytes();
    prefix_stats_.hits = impl_->prefix_hits;
    prefix_stats_.misses = impl_->prefix_misses;
    impl_->report_phase_profile("prefill");
    return {result, budget_limit, complete};
}

QwenForwardResult QwenEngine::decode_step(int token_id, int slot_id) {
    std::optional<Impl::RangeScope> range;
    if (impl_->range_profile) range.emplace("qwen.decode_step");

    // Phase 3.4: Use per-slot position when batch mode is enabled
    int current_position = position_;
    if (impl_->batch_mode_enabled && slot_id >= 0 &&
        slot_id < static_cast<int>(impl_->slot_positions.size())) {
        current_position = impl_->slot_positions[static_cast<size_t>(slot_id)];
    }

    if (current_position >= max_context_) {
        throw std::runtime_error("Qwen context length exceeded");
    }
    // One more token to hold. This is a no-op except on the step that crosses a
    // block boundary, and the table upload is skipped when nothing changed.
    impl_->reserve_paged_slot(slot_id, current_position + 1);
    impl_->sync_block_table();
    QwenForwardResult result = impl_->run_chunk(
        {token_id}, current_position, active_layers_, true, nullptr, nullptr, slot_id);

    ++current_position;
    impl_->set_slot_position(slot_id, current_position, position_);

    if (options_.prefix_cache) {
        // Appends to this slot's prompt only.  A shared buffer would splice the
        // slots' token streams together and corrupt the next prefix match.
        Impl::SlotPrefixState& prefix = impl_->prefix_for(slot_id);
        prefix.cached_prompt.push_back(token_id);
        prefix.cached_result = result;
        prefix.has_cached_result = true;
        if (impl_->is_periodic_snapshot_position(current_position)) {
            impl_->record_snapshot(current_position, &result, true, slot_id);
        }
    }
    return result;
}

std::vector<QwenForwardResult> QwenEngine::generate(
    const std::vector<int>& prompt_ids, int max_new_tokens, bool stop_at_eos) {
    // Checked against config_.eos_token_ids directly: this path has no
    // per-request sampling params, and an empty list makes this always false.
    const auto is_eos = [&](int token) {
        return stop_at_eos &&
               std::find(config_.eos_token_ids.begin(), config_.eos_token_ids.end(),
                         token) != config_.eos_token_ids.end();
    };
    if (max_new_tokens <= 0) return {};
    if (prompt_ids.empty()) {
        throw std::runtime_error("Qwen generation requires a non-empty prompt");
    }
    if (prompt_ids.size() + static_cast<size_t>(max_new_tokens) >
        static_cast<size_t>(max_context_)) {
        throw std::runtime_error("Qwen prompt plus generation exceeds context");
    }
    std::optional<Impl::RangeScope> range_generate;
    if (impl_->range_profile) range_generate.emplace("qwen.generate");
    mtp_stats_ = QwenMtpStats{};
    const auto prefill_started = std::chrono::steady_clock::now();
    QwenForwardResult next = prefill(prompt_ids);
    mtp_stats_.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_started).count();
    // Prefill already reported; reset so the decode/verify phases are attributed
    // on their own rather than buried under the prompt pass.
    impl_->phase_seconds.clear();
    impl_->phase_calls.clear();
    std::vector<QwenForwardResult> results;
    results.reserve(static_cast<size_t>(max_new_tokens));
    if (!options_.mtp && options_.dspark_checkpoint.empty() &&
        options_.dflash2_checkpoint.empty()) {
        std::optional<Impl::RangeScope> range_decode;
        if (impl_->range_profile) range_decode.emplace("qwen.decode_loop");
        for (int index = 0; index < max_new_tokens; ++index) {
            results.push_back(next);
            // The stop token stays in `results` as its last element, matching
            // the batched path.
            if (is_eos(next.top_token)) break;
            if (index + 1 < max_new_tokens) next = decode_step(next.top_token);
        }
        range_decode.reset();
        impl_->report_phase_profile("decode");
        return results;
    }

    // `prefill()` predicts the first output token without consuming it, just as
    // the plain path does. A speculative transaction consumes that token and
    // any correct drafts, then returns a bonus token which remains unconsumed
    // until the next transaction.
    results.push_back(next);
    if (is_eos(next.top_token)) {
        impl_->report_phase_profile("spec_decode");
        return results;
    }
    int current_token = next.top_token;
    const bool use_dspark = !options_.dspark_checkpoint.empty();
    const bool use_dflash2 = !options_.dflash2_checkpoint.empty();
    const bool use_external_drafter = use_dspark || use_dflash2;
    const int max_draft_tokens = use_external_drafter
        ? 7 : std::max(1, options_.mtp_speculative_tokens);
    int adaptive_draft_tokens = use_external_drafter
        ? 7 : (options_.mtp_adaptive ? 1 : max_draft_tokens);
    int full_accept_streak = 0;
    while (static_cast<int>(results.size()) < max_new_tokens) {
        const int remaining = max_new_tokens - static_cast<int>(results.size());
        if (remaining == 1) {
            next = decode_step(current_token);
            results.push_back(next);
            break;
        }
        if (use_external_drafter && remaining < 8) {
            // The published checkpoint is trained for a fixed seven-row block;
            // use exact plain decode for a short output tail rather than changing
            // its attention semantics or returning unnecessary tokens.
            while (static_cast<int>(results.size()) < max_new_tokens) {
                next = decode_step(current_token);
                results.push_back(next);
                if (is_eos(next.top_token)) break;
                current_token = next.top_token;
            }
            break;
        }
        const int proposed = std::min(adaptive_draft_tokens, remaining - 1);
        if (use_dspark) {
            next = impl_->dspark_speculative_step(current_token, &mtp_stats_);
        } else if (use_dflash2) {
            next = impl_->dflash2_speculative_step(current_token, &mtp_stats_);
        } else {
            next = impl_->speculative_step(current_token, proposed, &mtp_stats_);
        }
        const int correct = next.correct_drafts;
        if (!use_external_drafter && options_.mtp_adaptive) {
            if (correct == proposed) {
                ++full_accept_streak;
                if (full_accept_streak >= 1 &&
                    adaptive_draft_tokens < max_draft_tokens) {
                    adaptive_draft_tokens = std::min(
                        max_draft_tokens, adaptive_draft_tokens * 2);
                    full_accept_streak = 0;
                }
            } else {
                adaptive_draft_tokens = std::max(
                    1, std::min(adaptive_draft_tokens / 2, correct + 1));
                full_accept_streak = 0;
            }
        }
        const int consumed = 1 + correct;
        position_ += consumed;
        if (options_.prefix_cache) {
            // generate() is the single-session speculative path and has no slot
            // parameter, so it always drives slot 0.
            Impl::SlotPrefixState& prefix = impl_->prefix_for(0);
            prefix.cached_prompt.push_back(current_token);
            prefix.cached_prompt.insert(prefix.cached_prompt.end(),
                                        next.accept_tokens.begin(),
                                        next.accept_tokens.end());
            prefix.cached_result = next;
            prefix.has_cached_result = true;
        }

        bool stop_in_block = false;
        for (size_t index = 0; index < next.accept_tokens.size(); ++index) {
            if (static_cast<int>(results.size()) == max_new_tokens) break;
            QwenForwardResult token_result = next;
            token_result.top_token = next.accept_tokens[index];
            token_result.top_logit = next.accept_logits[index];
            token_result.checksum = next.accept_checksums[index];
            token_result.position = position_ - correct + static_cast<int>(index);
            const bool token_is_eos = is_eos(token_result.top_token);
            results.push_back(std::move(token_result));
            // A verified block can contain a stop token before its end. The rest
            // of the block is dropped from the output, but the KV cache and
            // position_ have already advanced over the whole block, so this
            // session must not be reused for further decoding after an early
            // stop here. generate() is a whole-generation call, so that only
            // matters to a caller that continues from the same session.
            if (token_is_eos) {
                stop_in_block = true;
                break;
            }
        }
        if (stop_in_block) break;
        if (static_cast<int>(results.size()) < max_new_tokens) {
            results.push_back(next);
            if (is_eos(next.top_token)) break;
        }
        current_token = next.bonus_token;
    }
    impl_->report_phase_profile("spec_decode");
    return results;
}

// TP worker loop implementation
void QwenEngine::run_worker_loop() {
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank == 0) {
        throw std::runtime_error("run_worker_loop: rank 0 must not enter worker loop");
    }
    if (!impl_->cmd) {
        throw std::runtime_error("run_worker_loop: command channel not initialized (call warmup_tp first)");
    }

    while (true) {
        int32_t header[4] = {0, 0, 0, 0};
        impl_->cmd->recv_from_root(header, 4);

        const auto cmd = static_cast<WorkerCommand>(header[0]);
        if (cmd == WorkerCommand::Shutdown) return;

        const int payload_len = header[3];
        std::vector<int> tokens;

        if (payload_len > 0) {
            std::vector<int32_t> payload(static_cast<size_t>(payload_len));
            impl_->cmd->recv_from_root(payload.data(),
                                       static_cast<std::size_t>(payload_len));
            tokens.assign(payload.begin(), payload.end());
        }

        // A failed collective leaves this rank out of step with rank 0, so the
        // loop must not continue past an error. Propagating exits the process
        // and lets the launcher tear the whole group down.
        // header[2] carries the KV slot rank 0 is computing into.  A worker that
        // ignored it would read and write slot 0 while rank 0 used another slot,
        // so batched decode would silently cross-contaminate sequences.
        const int slot_id = static_cast<int>(header[2]);

        switch (cmd) {
            case WorkerCommand::Prefill:
                // header[1] is the token budget rank 0 applied; 0 is unbounded.
                (void)prefill_bounded(tokens, slot_id,
                                      std::max(0, static_cast<int>(header[1])));
                break;
            case WorkerCommand::DecodeStep:
                (void)decode_step(header[1], slot_id);
                break;
            case WorkerCommand::BatchDecodeStep: {
                // The payload interleaves token and slot per row, because the
                // header carries only one slot field. header[1] is the row count.
                const int batch_rows = static_cast<int>(header[1]);
                if (batch_rows <= 0 ||
                    tokens.size() != static_cast<size_t>(batch_rows) * 2) {
                    throw std::runtime_error(
                        "run_worker_loop: malformed batched decode payload");
                }
                std::vector<int> batch_tokens(static_cast<size_t>(batch_rows));
                std::vector<int> batch_slots(static_cast<size_t>(batch_rows));
                for (int row = 0; row < batch_rows; ++row) {
                    batch_tokens[static_cast<size_t>(row)] =
                        tokens[static_cast<size_t>(row) * 2];
                    batch_slots[static_cast<size_t>(row)] =
                        tokens[static_cast<size_t>(row) * 2 + 1];
                }
                (void)batch_decode_tokens(batch_tokens, batch_slots);
                break;
            }
            case WorkerCommand::Reset:
                reset();
                break;
            case WorkerCommand::FreeSlot:
                // slot_id is header[2]. Releases this rank's copy of the slot's
                // blocks and clears its reuse state, mirroring rank 0's
                // free_slot, so the pool does not leak one row per freed slot.
                impl_->release_slot_paged_state(slot_id, position_);
                break;
            default:
                throw std::runtime_error(
                    "run_worker_loop: unknown command " + std::to_string(header[0]));
        }
    }
}

void QwenEngine::worker_command_prefill(const std::vector<int>& token_ids,
                                        int32_t slot_id, int32_t token_budget) {
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank != 0) {
        throw std::runtime_error("worker_command_prefill: rank 0 only");
    }
    if (!impl_->cmd) {
        throw std::runtime_error("worker_command_prefill: command channel not initialized (call warmup_tp first)");
    }

    // header[1] carries the prefill token budget, which was unused for this
    // command before bounded prefill existed. Every rank must apply the same
    // bound: a worker that ran the whole prompt while rank 0 stopped at a chunk
    // would arrive at the next collective with a different row count.
    int32_t header[4] = {
        static_cast<int32_t>(WorkerCommand::Prefill),
        token_budget,
        slot_id,
        static_cast<int32_t>(token_ids.size())
    };
    impl_->cmd->send_to_workers(header, 4);

    if (!token_ids.empty()) {
        std::vector<int32_t> payload(token_ids.begin(), token_ids.end());
        impl_->cmd->send_to_workers(payload.data(), payload.size());
    }
}

void QwenEngine::worker_command_decode(int32_t last_token, int32_t slot_id) {
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank != 0) {
        throw std::runtime_error("worker_command_decode: rank 0 only");
    }
    if (!impl_->cmd) {
        throw std::runtime_error("worker_command_decode: command channel not initialized (call warmup_tp first)");
    }

    int32_t header[4] = {
        static_cast<int32_t>(WorkerCommand::DecodeStep),
        last_token,
        slot_id,
        0
    };
    impl_->cmd->send_to_workers(header, 4);
}

void QwenEngine::worker_command_batch_decode(
    const std::vector<int>& tokens, const std::vector<int>& slot_ids) {
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank != 0) {
        throw std::runtime_error("worker_command_batch_decode: rank 0 only");
    }
    if (!impl_->cmd) {
        throw std::runtime_error("worker_command_batch_decode: command channel not initialized (call warmup_tp first)");
    }
    if (tokens.size() != slot_ids.size()) {
        throw std::runtime_error(
            "worker_command_batch_decode: token and slot extents differ");
    }
    if (tokens.empty()) return;

    const int32_t rows = static_cast<int32_t>(tokens.size());
    int32_t header[4] = {
        static_cast<int32_t>(WorkerCommand::BatchDecodeStep),
        rows,
        0,
        rows * 2
    };
    impl_->cmd->send_to_workers(header, 4);

    std::vector<int32_t> payload(static_cast<size_t>(rows) * 2);
    for (int32_t row = 0; row < rows; ++row) {
        payload[static_cast<size_t>(row) * 2] = tokens[static_cast<size_t>(row)];
        payload[static_cast<size_t>(row) * 2 + 1] =
            slot_ids[static_cast<size_t>(row)];
    }
    impl_->cmd->send_to_workers(payload.data(), payload.size());
}

void QwenEngine::worker_command_reset() {
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank != 0) {
        throw std::runtime_error("worker_command_reset: rank 0 only");
    }
    if (!impl_->cmd) {
        throw std::runtime_error("worker_command_reset: command channel not initialized (call warmup_tp first)");
    }

    int32_t header[4] = {
        static_cast<int32_t>(WorkerCommand::Reset),
        0,
        0,
        0
    };
    impl_->cmd->send_to_workers(header, 4);
}

void QwenEngine::worker_command_shutdown() {
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank != 0) {
        throw std::runtime_error("worker_command_shutdown: rank 0 only");
    }
    if (!impl_->cmd) {
        throw std::runtime_error("worker_command_shutdown: command channel not initialized (call warmup_tp first)");
    }

    int32_t header[4] = {
        static_cast<int32_t>(WorkerCommand::Shutdown),
        0,
        0,
        0
    };
    impl_->cmd->send_to_workers(header, 4);
}

void QwenEngine::worker_command_free_slot(int32_t slot_id) {
    // No-op at world size 1: the scheduler's free_slot already released this
    // rank's blocks, and there is no worker to keep in step.
    if (options_.tp_world <= 1) return;
    if (options_.tp_rank != 0) {
        throw std::runtime_error("worker_command_free_slot: rank 0 only");
    }
    // Contiguous arena: a slot already owns max_context implicitly, so there is
    // no per-slot block state for a worker to release. A worker's position and
    // prefix state are re-derived at the next prefill on that slot, so leaving
    // the command unsent keeps the non-paged path byte-for-byte unchanged.
    if (!impl_->kv_paged()) return;
    if (!impl_->cmd) {
        throw std::runtime_error("worker_command_free_slot: command channel not initialized (call warmup_tp first)");
    }

    int32_t header[4] = {
        static_cast<int32_t>(WorkerCommand::FreeSlot),
        0,
        slot_id,
        0
    };
    impl_->cmd->send_to_workers(header, 4);
}

}  // namespace dsv4
