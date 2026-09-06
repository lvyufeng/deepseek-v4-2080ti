#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace pocket {

class GGUFFile;

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(const std::string& ckpt_dir);
    static Tokenizer from_gguf(const GGUFFile& gguf);

    std::string token(int id) const;
    std::string decode_piece(int id) const;
    std::string decode_tokens(const std::vector<int>& ids, bool skip_special_tokens = true) const;
    std::vector<int> encode_basic(const std::string& text, bool add_bos = true) const;
    size_t vocab_size() const { return id_to_token_.size(); }
    // -1 when the exact token string is not in the vocabulary.
    int token_id(const std::string& token) const;
    bool is_special_token(int id) const;

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::map<std::pair<std::string, std::string>, int> merge_rank_;
    // Tokens hidden from decoded text when skip_special_tokens is set: BOS/EOS,
    // padding, placeholders. Deliberately narrower than atomic_token_id_list_ —
    // markup such as <think>, </think> and ｜DSML｜ must survive decoding because
    // the chat-completion parser consumes it.
    std::vector<uint8_t> special_token_ids_;
    // Tokens that must be matched verbatim while encoding instead of being run
    // through BPE. Every added token belongs here, special or not.
    std::vector<int> atomic_token_id_list_;
};

}  // namespace pocket
