// The engines this build contains, bound to the architectures they serve.
//
// Kept apart from core/model_registry.cpp on purpose: the registry itself has no
// engine dependency, so a checkpoint-inspection tool can link pocket_core alone.
// This translation unit is where naming a concrete engine is allowed.
//
// Registration is explicit rather than static-initializer-driven. In a static
// library a self-registering object is dropped by the linker unless something
// already references it, which makes "which models does this binary support"
// depend on link order; an explicit call cannot silently lose an engine.

#include "model_registry.hpp"

#include "persistent_engine_adapter.hpp"
#include "qwen_engine.hpp"

#include <memory>
#include <mutex>

namespace pocket {
namespace {

// DeepSeek-V4's full depth, and what the CLI's --serve path has always used when
// no explicit layer count was given.
constexpr int kDeepSeekV4Layers = 43;

std::unique_ptr<InferenceEngine> make_qwen_engine(const std::string& ckpt_dir,
                                                  const EngineOptions& options) {
    QwenEngineOptions qwen;
    qwen.tp_world = options.tp_world;
    qwen.tp_rank = options.tp_rank;
    qwen.device = options.device;
    qwen.nccl_id_path = options.nccl_id_path;
    qwen.max_batch_size = options.max_batch_size;
    qwen.kv_paged = options.kv_paged;
    qwen.kv_block_size = options.kv_block_size;
    qwen.kv_cache_bytes = options.kv_cache_bytes;
    // Everything else -- prefill chunk, KV dtype, prefix cache, snapshots, the
    // drafters -- keeps its tuned default. A caller that needs to move one of
    // those knows it is talking to Qwen and can construct QwenEngine directly.
    return std::make_unique<QwenEngine>(ckpt_dir, qwen, options.layer_count,
                                        options.max_context);
}

std::unique_ptr<InferenceEngine> make_deepseek_v4_engine(
    const std::string& ckpt_dir, const EngineOptions& options) {
    ForwardSmokeOptions opts;
    opts.tp_world = options.tp_world;
    opts.tp_rank = options.tp_rank;
    opts.device = options.device;
    opts.nccl_id_path = options.nccl_id_path;
    const int layers = options.layer_count > 0 ? options.layer_count : kDeepSeekV4Layers;
    return std::make_unique<PersistentEngineAdapter>(ckpt_dir, opts, layers,
                                                     options.max_context);
}

}  // namespace

void register_builtin_engines() {
    static std::once_flag once;
    std::call_once(once, [] {
        register_engine("qwen3_5", make_qwen_engine);
        register_engine("deepseek_v4", make_deepseek_v4_engine);
    });
}

}  // namespace pocket
