#include "gguf_reader.hpp"
#include "model_config.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <dirent.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr const char* kDefaultSystemPrompt = "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.";

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

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

std::string render_minimax_chat_prompt(const std::string& user_text) {
    std::string prompt;
    prompt += "]~!b[]~b]system\n";
    prompt += kDefaultSystemPrompt;
    prompt += "[e~[\n";
    prompt += "]~b]user\n";
    prompt += user_text;
    prompt += "[e~[\n";
    prompt += "]~b]ai\n";
    return prompt;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_tokenizer <gguf_file_or_dir>\n";
        return 2;
    }
    try {
        pocket::GGUFFile gguf(first_gguf_path(argv[1]));
        pocket::Tokenizer tok = pocket::Tokenizer::from_gguf(gguf);
        pocket::ModelConfig cfg = pocket::ModelConfig::from_gguf(gguf);

        require(cfg.architecture == "minimax-m2", "expected minimax-m2 architecture");
        require(cfg.context_length == 196608, "bad MiniMax context length");
        require(cfg.n_layers == 62, "bad MiniMax layer count");
        require(cfg.kv_heads == 8, "bad MiniMax kv head count");
        require(cfg.head_dim == 128, "bad MiniMax head dim");
        require(tok.vocab_size() == 200064, "bad MiniMax vocab size");
        require(tok.is_special_token(200034), "BOS should be special");
        require(tok.is_special_token(200020), "EOS should be special");

        const std::string prompt = render_minimax_chat_prompt("请用一句话介绍你自己。");
        const std::vector<int> ids = tok.encode_basic(prompt, false);
        const std::vector<int> expected_prefix = {
            200034, 200019, 28463, 10, 2985, 457, 258, 12473, 23413,
            46, 5324, 1925, 355, 35353, 12973, 45, 77, 50, 46, 55,
        };
        require(ids.size() >= expected_prefix.size(), "encoded prompt too short");
        for (size_t i = 0; i < expected_prefix.size(); ++i) {
            if (ids[i] != expected_prefix[i]) {
                std::ostringstream oss;
                oss << "encoded prompt prefix mismatch at " << i << " got=" << ids[i]
                    << " want=" << expected_prefix[i];
                throw std::runtime_error(oss.str());
            }
        }
        require(tok.decode_tokens({758, 3100, 20886, 58, 494, 4088, 829, 49864, 10201, 60103}, false)
                    .find("请用一句话介绍你自己") != std::string::npos,
                "bad MiniMax decode smoke");
        require(tok.decode_tokens({200034, 200019, 758}, true).find("]~") == std::string::npos,
                "special tokens were not skipped");

        const uint64_t kv_bytes_per_token = cfg.n_layers * cfg.kv_heads * cfg.head_dim * 2ull * 2ull;
        require(kv_bytes_per_token == 253952, "bad KV bytes/token estimate");
        std::cout << "[PASS] gguf tokenizer vocab=" << tok.vocab_size()
                  << " context=" << cfg.context_length
                  << " kv_bytes_per_token=" << kv_bytes_per_token << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
