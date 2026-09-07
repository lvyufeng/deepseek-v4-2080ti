#pragma once

#include "inference_engine.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pocket {

// Construction parameters every registered engine understands.
//
// This is deliberately not the union of every engine's option struct. It carries
// what a *caller who does not know the model* can meaningfully supply: topology,
// device, depth, context, and the KV geometry admission reasons in. Model-specific
// tuning -- KV cache dtype, prefill chunk size, drafter checkpoints, attention
// windows -- stays on the concrete engine's own options struct, which a caller
// that has chosen a model can name directly.
//
// A factory maps these onto its native options and leaves the rest at their
// documented defaults, so going through the registry never silently changes a
// tuned default.
struct EngineOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    std::string nccl_id_path;
    // Layers to run. 0 means the checkpoint's full depth; smaller values are the
    // smoke-test path that loads a prefix of the network.
    int layer_count = 0;
    // Per-sequence context ceiling, prompt plus generation.
    int max_context = 8192;
    // Requested concurrent slots. An engine that declares max_slots == 1 may
    // reject anything larger rather than pretend to batch.
    int max_batch_size = 1;
    // Paged KV cache instead of a per-slot contiguous arena. Engines that cannot
    // page ignore this; ask caps() rather than assuming it took effect.
    bool kv_paged = false;
    int kv_block_size = 16;
    // Paged pool budget in bytes on this rank; 0 derives it from what the
    // contiguous arena would have reserved.
    uint64_t kv_cache_bytes = 0;
};

// Builds one engine for one checkpoint. Registered per architecture.
using EngineFactory = std::function<std::unique_ptr<InferenceEngine>(
    const std::string& ckpt_path, const EngineOptions& options)>;

// Bind an architecture string to its engine. Throws if the architecture is
// already registered: silently replacing a builtin would make which engine runs
// depend on link order.
void register_engine(const std::string& architecture, EngineFactory factory);

bool engine_registered(const std::string& architecture);

// Registered architectures in sorted order. This is what a server reports as the
// models it can serve, rather than a hand-maintained list that drifts.
std::vector<std::string> registered_architectures();

// Architecture a checkpoint declares about itself, normalized to a registry key.
//
// Reuses the readers that already exist rather than adding a parser: a path
// ending in `.gguf` is read through GGUFFile for `general.architecture`, and
// anything else is treated as a directory whose config.json declares
// `model_type` (falling back to `text_config.model_type` for the multimodal
// wrappers, which is where Qwen3.5 hides it).
//
// Throws when neither is readable, so a mistyped checkpoint path fails here with
// the path in the message instead of deep inside a loader. Returns "" when the
// checkpoint parses but declares no architecture at all.
std::string detect_architecture(const std::string& ckpt_path);

// Detect, look up, construct. Throws naming the registered architectures when
// the checkpoint's architecture has no engine.
std::unique_ptr<InferenceEngine> create_engine(const std::string& ckpt_path,
                                               const EngineOptions& options);

// Register the engines this build contains. Idempotent, so a process that links
// both the CLI and the Python module can call it from either entry point.
//
// Defined alongside the engines in engine/engine_registry_builtin.cpp, not here:
// the registry itself must stay free of engine dependencies so a tool that links
// only pocket_core can still parse and inspect a checkpoint.
void register_builtin_engines();

}  // namespace pocket
