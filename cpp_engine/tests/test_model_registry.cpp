// The model-selection seam: architecture detection and the engine registry.
//
// What this pins is that choosing an engine is driven by what a checkpoint
// declares about itself, and that the scheduler is driven by what an engine
// declares about itself. Both used to be inferred -- `is_qwen3_5_checkpoint()`
// in main.cpp, an `engine_kind` string in the Python backend, and
// `kv_total_blocks() == 0` in the scheduler -- and an inference that happens to
// be right for the one model that exists is exactly what a second model breaks.
//
// No checkpoint and no device are needed here: the detection fixtures are a few
// bytes of JSON and a header-only GGUF, and the engines are stubs. That is the
// point -- the seam has to be testable without the 27B weights.

#include "batch_scheduler.hpp"
#include "model_registry.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "  " << message << ": PASS\n";
    } else {
        std::cout << "  " << message << ": FAIL\n";
        ++failures;
    }
}

void check_eq(const std::string& actual, const std::string& expected,
              const std::string& message) {
    check(actual == expected, message + " (got '" + actual + "', expected '" +
                                  expected + "')");
}

std::string fixture_root() {
    const char* base = std::getenv("CLAUDE_JOB_DIR");
    const std::string root = base != nullptr ? std::string(base) + "/tmp"
                                             : std::string("/tmp");
    return root + "/model_registry_fixture";
}

void make_dir(const std::string& path) {
    const std::string cmd = "mkdir -p '" + path + "'";
    if (std::system(cmd.c_str()) != 0) {
        throw std::runtime_error("could not create fixture dir " + path);
    }
}

void write_file(const std::string& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not write fixture " + path);
    out << contents;
    if (!out) throw std::runtime_error("short write to fixture " + path);
}

// A checkpoint directory whose config.json is exactly `body`.
std::string config_dir(const std::string& name, const std::string& body) {
    const std::string dir = fixture_root() + "/" + name;
    make_dir(dir);
    write_file(dir + "/config.json", body);
    return dir;
}

// --- Minimal GGUF writer ----------------------------------------------------
//
// Header plus one metadata string, zero tensors. Enough for the architecture
// probe, which is all detect_architecture reads out of a GGUF.

void put_u32(std::string& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

void put_u64(std::string& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

void put_string(std::string& out, const std::string& value) {
    put_u64(out, value.size());
    out += value;
}

std::string gguf_with_architecture(const std::string& name,
                                   const std::string& architecture) {
    std::string bytes;
    bytes += "GGUF";
    put_u32(bytes, 3);   // version
    put_u64(bytes, 0);   // tensor count
    put_u64(bytes, 1);   // metadata count
    put_string(bytes, "general.architecture");
    put_u32(bytes, 8);   // GGUF string type
    put_string(bytes, architecture);

    const std::string path = fixture_root() + "/" + name + ".gguf";
    write_file(path, bytes);
    return path;
}

// --- Stub engines -----------------------------------------------------------

// Declares whatever capabilities the test hands it and records what the
// scheduler asked of it. Runs no forward pass; every assertion here is about
// negotiation, which happens before the first token.
class StubEngine : public pocket::InferenceEngine {
public:
    explicit StubEngine(pocket::Capabilities caps) : caps_(caps) {}

    pocket::Capabilities caps() const override { return caps_; }
    int max_context() const override { return 4096; }
    int device() const override { return 0; }

    void allocate_batch_slots(int max_batch_size) override {
        requested_slots_ = max_batch_size;
    }
    int allocate_slot(uint64_t) override { return -1; }
    void free_slot(uint64_t) override {}

    bool kv_paged() const override { return caps_.paged_kv; }
    // Zero even when paged, which is the shape the old inference could not tell
    // apart from "this engine has no paged cache at all".
    int kv_free_blocks() const override { return 0; }
    int kv_total_blocks() const override { return 0; }
    int kv_blocks_for_tokens(int) const override { return 0; }

    pocket::BatchPrefillResult batch_prefill(
        const std::vector<pocket::BatchedRequest*>&, int) override {
        return {};
    }
    pocket::BatchDecodeResult batch_decode_step(
        const std::vector<pocket::BatchedRequest*>&) override {
        return {};
    }

    int requested_slots() const { return requested_slots_; }

private:
    pocket::Capabilities caps_;
    int requested_slots_ = -1;
};

pocket::Capabilities single_session_caps() {
    pocket::Capabilities caps;
    caps.paged_kv = false;
    caps.continuous_batching = false;
    caps.chunked_prefill = false;
    caps.max_slots = 1;
    return caps;
}

pocket::Capabilities paged_batched_caps(int slots) {
    pocket::Capabilities caps;
    caps.paged_kv = true;
    caps.continuous_batching = true;
    caps.chunked_prefill = true;
    caps.max_slots = slots;
    return caps;
}

// ============================================================================

void test_architecture_detection() {
    std::cout << "detect_architecture reads what the checkpoint declares\n";

    const std::string deepseek = config_dir(
        "deepseek", R"({"model_type": "deepseek_v4", "num_hidden_layers": 43})");
    check_eq(pocket::detect_architecture(deepseek), "deepseek_v4",
             "root model_type is the architecture");

    const std::string qwen = config_dir("qwen", R"({"model_type": "qwen3_5"})");
    check_eq(pocket::detect_architecture(qwen), "qwen3_5",
             "qwen3_5 is its own key");

    // The multimodal wrapper hides the runtime's type one level down, and the
    // text-only export spells it differently. Both are the same engine, so both
    // have to fold onto one key or a checkpoint reaches no factory at all.
    const std::string nested = config_dir(
        "qwen_nested", R"({"text_config": {"model_type": "qwen3_5_text"}})");
    check_eq(pocket::detect_architecture(nested), "qwen3_5",
             "text_config.model_type is used when the root declares none");

    const std::string cased = config_dir("cased", R"({"model_type": "Qwen3_5_Text"})");
    check_eq(pocket::detect_architecture(cased), "qwen3_5",
             "architecture keys are case-insensitive");

    const std::string silent = config_dir("silent", R"({"hidden_size": 4096})");
    check_eq(pocket::detect_architecture(silent), "",
             "a config that declares no model_type detects as unknown");

    const std::string gguf = gguf_with_architecture("minimax", "minimax-m2");
    check_eq(pocket::detect_architecture(gguf), "minimax-m2",
             "a .gguf path is read through general.architecture");

    bool threw = false;
    try {
        pocket::detect_architecture(fixture_root() + "/does_not_exist");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a path that is neither a config dir nor a .gguf is an error");
}

void test_registry_dispatch() {
    std::cout << "the registry dispatches on the detected architecture\n";

    const std::string dir = config_dir("stub_arch", R"({"model_type": "stub_arch"})");

    std::string seen_ckpt;
    int seen_slots = 0;
    pocket::register_engine("stub_arch", [&](const std::string& ckpt,
                                             const pocket::EngineOptions& options) {
        seen_ckpt = ckpt;
        seen_slots = options.max_batch_size;
        return std::unique_ptr<pocket::InferenceEngine>(
            new StubEngine(paged_batched_caps(options.max_batch_size)));
    });

    check(pocket::engine_registered("stub_arch"), "a registered engine is visible");

    pocket::EngineOptions options;
    options.max_batch_size = 3;
    std::unique_ptr<pocket::InferenceEngine> engine =
        pocket::create_engine(dir, options);
    check(engine != nullptr, "create_engine builds the registered engine");
    check_eq(seen_ckpt, dir, "the factory receives the checkpoint path");
    check(seen_slots == 3, "the factory receives the caller's options");
    check(engine->caps().max_slots == 3, "the built engine's caps reach the caller");

    // Silently replacing a registration would make which engine runs depend on
    // link order, which is exactly the failure a registry is supposed to remove.
    bool duplicate_threw = false;
    try {
        pocket::register_engine("stub_arch", [](const std::string&,
                                                const pocket::EngineOptions&) {
            return std::unique_ptr<pocket::InferenceEngine>();
        });
    } catch (const std::exception&) {
        duplicate_threw = true;
    }
    check(duplicate_threw, "re-registering an architecture is rejected");

    const std::string unknown = config_dir("unknown", R"({"model_type": "not_a_model"})");
    std::string message;
    try {
        pocket::create_engine(unknown, options);
    } catch (const std::exception& e) {
        message = e.what();
    }
    check(message.find("not_a_model") != std::string::npos &&
              message.find("stub_arch") != std::string::npos,
          "an unregistered architecture is reported with what is registered");
}

void test_builtin_registrations() {
    std::cout << "the build's engines are registered by name\n";

    pocket::register_builtin_engines();
    // Idempotent: the CLI and the Python module are separate entry points into
    // the same process image, and either may be first.
    pocket::register_builtin_engines();

    check(pocket::engine_registered("qwen3_5"), "qwen3_5 has an engine");
    check(pocket::engine_registered("deepseek_v4"), "deepseek_v4 has an engine");

    const std::vector<std::string> known = pocket::registered_architectures();
    bool sorted = true;
    for (size_t i = 1; i < known.size(); ++i) {
        if (known[i - 1] > known[i]) sorted = false;
    }
    check(sorted, "registered_architectures is ordered, so a server's model list is stable");
}

void test_scheduler_honours_declared_capabilities() {
    std::cout << "the scheduler runs at the width the engine declares\n";

    // A single-session engine used to be indistinguishable from a wide one with
    // an empty block pool: both answer kv_total_blocks() == 0. The scheduler
    // would have opened 8 slots on it and fed it a chunked prefill budget.
    StubEngine serial(single_session_caps());
    {
        pocket::BatchScheduler scheduler(&serial, 8);
        check(serial.requested_slots() == 1,
              "a max_slots=1 engine is never asked for more than one slot");
        check(scheduler.max_batch_size() == 1,
              "the scheduler's own width follows the declaration");
        check(scheduler.prefill_token_budget() == 0,
              "no chunked prefill means no token budget");
        scheduler.set_prefill_token_budget(4096);
        check(scheduler.prefill_token_budget() == 0,
              "a budget cannot be set on an engine that ignores it");
        scheduler.stop();
    }

    StubEngine wide(paged_batched_caps(4));
    {
        pocket::BatchScheduler scheduler(&wide, 8);
        check(wide.requested_slots() == 4,
              "a request wider than max_slots is clamped, not rejected");
        check(scheduler.prefill_token_budget() > 0,
              "an engine that declares chunked prefill keeps its budget");
        scheduler.stop();
    }

    StubEngine exact(paged_batched_caps(8));
    {
        pocket::BatchScheduler scheduler(&exact, 4);
        check(exact.requested_slots() == 4,
              "a request inside max_slots is passed through unchanged");
        scheduler.stop();
    }
}

}  // namespace

int main() {
    try {
        make_dir(fixture_root());
        test_architecture_detection();
        test_registry_dispatch();
        test_builtin_registrations();
        test_scheduler_honours_declared_capabilities();
    } catch (const std::exception& e) {
        std::cout << "[FAIL] " << e.what() << "\n";
        return 1;
    }

    if (failures != 0) {
        std::cout << "[FAIL] " << failures << " model registry checks failed\n";
        return 1;
    }
    std::cout << "[PASS] model registry\n";
    return 0;
}
