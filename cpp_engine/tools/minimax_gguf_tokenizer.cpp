#include "gguf_reader.hpp"
#include "model_config.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <dirent.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr const char* kDefaultSystemPrompt = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.";

struct Args {
    std::string gguf;
    std::string prompt;
    std::string ids_csv;
    bool chat = false;
    bool thinking = false;
    bool context_info = false;
    bool json = false;
    bool skip_special = false;
};

bool is_dir(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string first_gguf_path(const std::string& path) {
    if (!is_dir(path)) return path;
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) throw std::runtime_error("failed to open directory: " + path);
    std::vector<std::string> files;
    while (dirent* ent = readdir(dir)) {
        std::string name = ent->d_name;
        if (ends_with(name, ".gguf")) files.push_back(name);
    }
    closedir(dir);
    if (files.empty()) throw std::runtime_error("no .gguf files found in directory: " + path);
    std::sort(files.begin(), files.end());
    return path + "/" + files.front();
}

std::string csv(const std::vector<int>& ids) {
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) oss << ',';
        oss << ids[i];
    }
    return oss.str();
}

std::vector<int> parse_ids_csv(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stoi(item));
    }
    return out;
}

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    oss << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    oss << '"';
    return oss.str();
}

std::string render_minimax_chat_prompt(const std::string& user_text, bool thinking) {
    std::string prompt;
    prompt += "]~!b[]~b]system\n";
    prompt += kDefaultSystemPrompt;
    prompt += "[e~[\n";
    prompt += "]~b]user\n";
    prompt += user_text;
    prompt += "[e~[\n";
    prompt += "]~b]ai\n";
    if (thinking) prompt += "<think>\n";
    return prompt;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--gguf" || arg == "--model") && i + 1 < argc) {
            args.gguf = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--ids" && i + 1 < argc) {
            args.ids_csv = argv[++i];
        } else if (arg == "--chat") {
            args.chat = true;
        } else if (arg == "--thinking") {
            args.thinking = true;
        } else if (arg == "--context-info") {
            args.context_info = true;
        } else if (arg == "--json") {
            args.json = true;
        } else if (arg == "--skip-special") {
            args.skip_special = true;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (args.gguf.empty()) throw std::runtime_error("--gguf is required");
    return args;
}

void print_context_info(const pocket::GGUFFile& gguf, const pocket::ModelConfig& cfg, bool as_json) {
    const uint64_t kv_bytes_per_token = cfg.n_layers * cfg.kv_heads * cfg.head_dim * 2ull * 2ull;
    const double kv_mib = static_cast<double>(kv_bytes_per_token) * static_cast<double>(cfg.context_length) / 1024.0 / 1024.0;
    if (as_json) {
        std::cout << "{"
                  << "\"architecture\":" << json_escape(cfg.architecture) << ','
                  << "\"context_length\":" << cfg.context_length << ','
                  << "\"n_layers\":" << cfg.n_layers << ','
                  << "\"n_kv_heads\":" << cfg.kv_heads << ','
                  << "\"head_dim\":" << cfg.head_dim << ','
                  << "\"kv_cache_bytes_per_token_fp16\":" << kv_bytes_per_token << ','
                  << "\"kv_cache_mib_at_context_fp16\":" << kv_mib;
        if (auto eos = gguf.metadata_u64("tokenizer.ggml.eos_token_id")) std::cout << ",\"eos_token_id\":" << *eos;
        if (auto bos = gguf.metadata_u64("tokenizer.ggml.bos_token_id")) std::cout << ",\"bos_token_id\":" << *bos;
        std::cout << "}\n";
    } else {
        std::cout << "architecture=" << cfg.architecture << '\n';
        std::cout << "context_length=" << cfg.context_length << '\n';
        std::cout << "n_layers=" << cfg.n_layers << '\n';
        std::cout << "n_kv_heads=" << cfg.kv_heads << '\n';
        std::cout << "head_dim=" << cfg.head_dim << '\n';
        std::cout << "kv_cache_bytes_per_token_fp16=" << kv_bytes_per_token << '\n';
        std::cout << "kv_cache_mib_at_context_fp16=" << kv_mib << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const std::string gguf_path = first_gguf_path(args.gguf);
        pocket::GGUFFile gguf(gguf_path);
        pocket::Tokenizer tokenizer = pocket::Tokenizer::from_gguf(gguf);
        pocket::ModelConfig cfg = pocket::ModelConfig::from_gguf(gguf);

        if (args.context_info) print_context_info(gguf, cfg, args.json);

        if (!args.prompt.empty()) {
            const std::string prompt_text = args.chat ? render_minimax_chat_prompt(args.prompt, args.thinking) : args.prompt;
            const auto ids = tokenizer.encode_basic(prompt_text, false);
            if (args.json) {
                std::cout << "{\"prompt_text\":" << json_escape(prompt_text)
                          << ",\"prompt_csv\":" << json_escape(csv(ids))
                          << ",\"prompt_ids\":[";
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << ids[i];
                }
                std::cout << "],\"context_length\":" << cfg.context_length << "}\n";
            } else {
                std::cout << csv(ids) << '\n';
            }
        }

        if (!args.ids_csv.empty()) {
            std::cout << tokenizer.decode_tokens(parse_ids_csv(args.ids_csv), args.skip_special) << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
