// Architecture detection and the engine registry.
//
// This is the model-selection seam: everything above it -- the CLI, the server,
// the Python backend -- names a checkpoint, and this decides which runtime opens
// it. Before this existed the answer was an `is_qwen3_5_checkpoint()` if/else in
// main.cpp and an `engine_kind` string in the Python backend, so adding a model
// meant editing every dispatch site.
//
// Deliberately free of engine dependencies: pocket_core links this, and a tool
// that inspects a checkpoint without any accelerator toolkit present must keep
// working. The engines register themselves from engine/engine_registry_builtin.cpp.

#include "model_registry.hpp"

#include "gguf_reader.hpp"
#include "json_lite.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace pocket {
namespace {

std::mutex& registry_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::string, EngineFactory>& registry() {
    static std::map<std::string, EngineFactory> factories;
    return factories;
}

bool try_read_file(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    *out = buffer.str();
    return true;
}

bool has_suffix(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a)) ==
                                 std::tolower(static_cast<unsigned char>(b));
                      });
}

std::string lowered(const std::string& value) {
    std::string out = value;
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Fold the spellings a checkpoint may use onto the key an engine registers.
//
// Qwen3.5 publishes `qwen3_5` at the config root and `qwen3_5_text` inside
// text_config, and either can be the only one present depending on whether the
// checkpoint is the multimodal wrapper or the text-only export. Both are the same
// runtime, so both have to reach the same factory.
std::string canonical_architecture(const std::string& raw) {
    const std::string arch = lowered(raw);
    if (arch == "qwen3_5_text") return "qwen3_5";
    return arch;
}

std::string architecture_from_json_config(const std::string& ckpt_dir) {
    std::string text;
    if (!try_read_file(ckpt_dir + "/config.json", &text)) return {};
    const JsonValue root_value = parse_json(text);
    if (!root_value.is_object()) return {};
    const JsonObject& root = root_value.object();

    const JsonValue* model_type = object_get(root, "model_type");
    if (model_type != nullptr && model_type->is_string() && !model_type->string().empty()) {
        return model_type->string();
    }
    // A multimodal wrapper may declare only the nested text model's type.
    const JsonValue* nested = object_get(root, "text_config");
    if (nested == nullptr || !nested->is_object()) return {};
    const JsonValue* nested_type = object_get(nested->object(), "model_type");
    if (nested_type == nullptr || !nested_type->is_string()) return {};
    return nested_type->string();
}

}  // namespace

void register_engine(const std::string& architecture, EngineFactory factory) {
    if (architecture.empty()) {
        throw std::invalid_argument("register_engine: architecture must not be empty");
    }
    if (!factory) {
        throw std::invalid_argument("register_engine: factory must not be empty for architecture '" +
                                    architecture + "'");
    }
    const std::string key = canonical_architecture(architecture);
    std::lock_guard<std::mutex> lock(registry_mutex());
    if (registry().count(key) != 0) {
        throw std::runtime_error("register_engine: architecture '" + key +
                                 "' is already registered");
    }
    registry().emplace(key, std::move(factory));
}

bool engine_registered(const std::string& architecture) {
    const std::string key = canonical_architecture(architecture);
    std::lock_guard<std::mutex> lock(registry_mutex());
    return registry().count(key) != 0;
}

std::vector<std::string> registered_architectures() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    std::vector<std::string> names;
    names.reserve(registry().size());
    for (const auto& entry : registry()) names.push_back(entry.first);
    return names;  // std::map already orders these.
}

std::string detect_architecture(const std::string& ckpt_path) {
    if (ckpt_path.empty()) {
        throw std::runtime_error("detect_architecture: empty checkpoint path");
    }
    if (has_suffix(ckpt_path, ".gguf")) {
        const GGUFFile file(ckpt_path);
        return canonical_architecture(
            file.metadata_string("general.architecture").value_or(std::string()));
    }
    std::string probe;
    if (!try_read_file(ckpt_path + "/config.json", &probe)) {
        throw std::runtime_error(
            "detect_architecture: no config.json under '" + ckpt_path +
            "' and the path is not a .gguf file");
    }
    return canonical_architecture(architecture_from_json_config(ckpt_path));
}

std::unique_ptr<InferenceEngine> create_engine(const std::string& ckpt_path,
                                               const EngineOptions& options) {
    const std::string architecture = detect_architecture(ckpt_path);

    EngineFactory factory;
    std::vector<std::string> known;
    {
        std::lock_guard<std::mutex> lock(registry_mutex());
        auto it = registry().find(architecture);
        if (it != registry().end()) factory = it->second;
        for (const auto& entry : registry()) known.push_back(entry.first);
    }

    if (!factory) {
        std::ostringstream message;
        message << "no engine registered for architecture '"
                << (architecture.empty() ? std::string("<undeclared>") : architecture)
                << "' declared by " << ckpt_path << "; registered: ";
        if (known.empty()) {
            // The usual cause is a caller that never called
            // register_builtin_engines(), which is a much more useful thing to
            // say than "unknown architecture".
            message << "<none; call register_builtin_engines() first>";
        } else {
            for (size_t i = 0; i < known.size(); ++i) {
                if (i != 0) message << ", ";
                message << known[i];
            }
        }
        throw std::runtime_error(message.str());
    }

    return factory(ckpt_path, options);
}

}  // namespace pocket
