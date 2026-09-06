// Definitions that stand in for the CUDA-only engines on the Ascend backend.
//
// Three subsystems are still CUDA-only: the DeepSeek-V4 forward engine
// (engine/deepseek_v4_engine.cpp, which also owns PersistentEngine), the DSpark draft
// engine (engine/dspark_engine.cpp), and the two Qwen external drafters
// (engine/qwen_dspark.cpp, engine/qwen_dflash2.cpp). Together they name ~120
// CUDA kernels directly, so they are excluded from the Ascend target rather than
// guarded line by line.
//
// A runtime guard would not be enough: a CUDA symbol referenced from any object
// in the link has to resolve even when its branch is unreachable. Excluding the
// sources removes those references, and this translation unit supplies the
// handful of definitions the surviving callers still need -- main.cpp and
// core/openai_server.cpp for the DeepSeek-V4 entry points, and engine/qwen_engine.cpp
// for the drafter runtime methods its shared bookkeeping calls.
//
// Every entry point throws. None of them is reachable from the Qwen FP16 path
// that this backend does implement: QwenEngine's constructor rejects a drafter
// checkpoint before any runtime is built, and the DeepSeek-V4 model is a different
// architecture altogether.

#include "deepseek_v4_engine.hpp"
#include "persistent_engine.hpp"
#include "qwen_dflash2.hpp"
#include "qwen_dspark.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace pocket {
namespace {

[[noreturn]] void unimplemented(const char* what) {
    throw std::runtime_error(
        std::string(what) + " is not implemented on the Ascend backend");
}

}  // namespace

// --- DeepSeek-V4 forward entry points ---------------------------------------
//
// DeepSeekV4Engine itself only parses GGUF metadata on the host, so it keeps its real
// behaviour; the forward entry points below are what need the device.

DeepSeekV4Engine::DeepSeekV4Engine(const std::string& model_path)
    : gguf_(model_path), config_(ModelConfig::from_gguf(gguf_)) {}

ForwardSmokeResult run_safetensors_min_layer_smoke(const std::string&) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

ForwardSmokeResult run_safetensors_layer_loop_smoke(const std::string&, int) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

ForwardSmokeResult run_safetensors_token_forward(const std::string&, int, int) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

ForwardSmokeResult run_safetensors_token_forward_at_position(
    const std::string&, int, int, int) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

ForwardSmokeResult run_safetensors_token_forward_with_options(
    const std::string&, int, int, int, const ForwardSmokeOptions&) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

ForwardSmokeResult run_safetensors_prompt_forward(
    const std::string&, const std::vector<int>&, int) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

ForwardSmokeResult run_safetensors_prompt_forward_with_options(
    const std::string&, const std::vector<int>&, int,
    const ForwardSmokeOptions&) {
    unimplemented("DeepSeek-V4 safetensors forward");
}

std::vector<ForwardSmokeResult> run_safetensors_generate_tokens(
    const std::string&, const std::vector<int>&, int, int) {
    unimplemented("DeepSeek-V4 safetensors generate");
}

std::vector<ForwardSmokeResult> run_safetensors_generate_tokens_with_options(
    const std::string&, const std::vector<int>&, int, int,
    const ForwardSmokeOptions&) {
    unimplemented("DeepSeek-V4 safetensors generate");
}

GenerateSmokeResult run_safetensors_generate_tokens_timed_with_options(
    const std::string&, const std::vector<int>&, int, int,
    const ForwardSmokeOptions&) {
    unimplemented("DeepSeek-V4 safetensors generate");
}

// --- PersistentEngine -------------------------------------------------------
//
// State is declared but never defined in the header, so the empty definition
// here is what lets the unique_ptr member's destructor instantiate.

struct PersistentEngine::State {};

PersistentEngine::PersistentEngine(const std::string&,
                                   const ForwardSmokeOptions&, int, int) {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

PersistentEngine::~PersistentEngine() = default;

void PersistentEngine::reset_session() {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

int PersistentEngine::prefill(const std::vector<int>&, const SamplingParams&) {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

int PersistentEngine::decode_step(int, int, const SamplingParams&) {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

int PersistentEngine::eos_id() const {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

int PersistentEngine::max_context() const {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

const Tokenizer& PersistentEngine::tokenizer() const {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

void PersistentEngine::run_worker_loop() {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

void PersistentEngine::warmup_tp() {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

void PersistentEngine::worker_command_prefill(const std::vector<int>&) {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

void PersistentEngine::worker_command_decode(int32_t, int32_t) {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

void PersistentEngine::worker_command_reset() {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

void PersistentEngine::worker_command_shutdown() {
    unimplemented("DeepSeek-V4 PersistentEngine");
}

// --- Qwen external drafters -------------------------------------------------
//
// QwenEngine keeps the drafter members and their bookkeeping on every backend so
// that run_chunk and the speculative steps stay backend-neutral. Construction is
// rejected in QwenEngine's constructor, which is why none of these can be
// entered: the owning unique_ptr is always null on this backend.

struct QwenDSparkRuntime::Impl {};

QwenDSparkRuntime::QwenDSparkRuntime(const std::string&,
                                    const QwenDSparkConfig&,
                                    const QwenDSparkWeightMap&,
                                    const QwenDeviceTensor&,
                                    QwenTargetHeadAdapter, int, int, int,
                                    std::string, int) {
    unimplemented("Qwen DSpark drafter");
}

QwenDSparkRuntime::~QwenDSparkRuntime() = default;

void QwenDSparkRuntime::reset() { unimplemented("Qwen DSpark drafter"); }

int QwenDSparkRuntime::committed_position() const {
    unimplemented("Qwen DSpark drafter");
}

uint64_t QwenDSparkRuntime::resident_weight_bytes() const {
    unimplemented("Qwen DSpark drafter");
}

uint64_t QwenDSparkRuntime::context_cache_bytes() const {
    unimplemented("Qwen DSpark drafter");
}

uint64_t QwenDSparkRuntime::activation_workspace_bytes() const {
    unimplemented("Qwen DSpark drafter");
}

void QwenDSparkRuntime::append_target_taps(const uint16_t*, int, int) {
    unimplemented("Qwen DSpark drafter");
}

void QwenDSparkRuntime::crop_context(int) {
    unimplemented("Qwen DSpark drafter");
}

QwenDSparkProposal QwenDSparkRuntime::propose(int) {
    unimplemented("Qwen DSpark drafter");
}

struct QwenDFlash2Runtime::Impl {};

QwenDFlash2Runtime::QwenDFlash2Runtime(const std::string&,
                                      const QwenDFlash2Config&,
                                      const QwenDFlash2WeightMap&,
                                      const QwenDeviceTensor&,
                                      QwenTargetHeadAdapter, int, int, int,
                                      std::string, int) {
    unimplemented("Qwen DFlash2 drafter");
}

QwenDFlash2Runtime::~QwenDFlash2Runtime() = default;

void QwenDFlash2Runtime::reset() { unimplemented("Qwen DFlash2 drafter"); }

int QwenDFlash2Runtime::committed_position() const {
    unimplemented("Qwen DFlash2 drafter");
}

uint64_t QwenDFlash2Runtime::resident_weight_bytes() const {
    unimplemented("Qwen DFlash2 drafter");
}

uint64_t QwenDFlash2Runtime::context_cache_bytes() const {
    unimplemented("Qwen DFlash2 drafter");
}

uint64_t QwenDFlash2Runtime::activation_workspace_bytes() const {
    unimplemented("Qwen DFlash2 drafter");
}

void QwenDFlash2Runtime::set_debug_callback(QwenDFlash2DebugCallback) {
    unimplemented("Qwen DFlash2 drafter");
}

void QwenDFlash2Runtime::debug_load_target_taps(
    const std::vector<uint16_t>&, int, int) {
    unimplemented("Qwen DFlash2 drafter");
}

QwenDFlash2Proposal QwenDFlash2Runtime::debug_propose_from_host(
    const std::vector<uint16_t>&, int, int, int) {
    unimplemented("Qwen DFlash2 drafter");
}

void QwenDFlash2Runtime::append_target_taps(const uint16_t*, int, int) {
    unimplemented("Qwen DFlash2 drafter");
}

void QwenDFlash2Runtime::crop_context(int) {
    unimplemented("Qwen DFlash2 drafter");
}

QwenDFlash2Proposal QwenDFlash2Runtime::propose(int) {
    unimplemented("Qwen DFlash2 drafter");
}

}  // namespace pocket
