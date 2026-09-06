#include "tokenizer.hpp"

#include "gguf_reader.hpp"
#include "json_lite.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace pocket {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t json_u64(const JsonValue& value) {
    const double n = value.number();
    if (n < 0) throw std::runtime_error("negative tokenizer id");
    return static_cast<uint64_t>(n + 0.5);
}

void set_token(std::vector<std::string>& id_to_token, uint64_t id, const std::string& token) {
    if (id >= id_to_token.size()) id_to_token.resize(static_cast<size_t>(id) + 1);
    id_to_token[static_cast<size_t>(id)] = token;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string utf8_codepoint(uint32_t cp) {
    std::string out;
    if (cp <= 0x7f) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

std::vector<std::string> byte_alphabet() {
    std::vector<uint32_t> bs;
    for (uint32_t c = '!'; c <= '~'; ++c) bs.push_back(c);
    for (uint32_t c = 0xA1; c <= 0xAC; ++c) bs.push_back(c);
    for (uint32_t c = 0xAE; c <= 0xFF; ++c) bs.push_back(c);
    std::vector<uint32_t> cs = bs;
    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    std::vector<std::string> out(256);
    for (size_t i = 0; i < bs.size(); ++i) out[bs[i]] = utf8_codepoint(cs[i]);
    return out;
}

std::pair<std::string, std::string> split_merge(const std::string& merge) {
    const size_t pos = merge.find(' ');
    if (pos == std::string::npos) throw std::runtime_error("bad BPE merge: " + merge);
    return {merge.substr(0, pos), merge.substr(pos + 1)};
}

std::unordered_map<std::string, unsigned char> byte_decoder() {
    std::unordered_map<std::string, unsigned char> out;
    const auto alphabet = byte_alphabet();
    for (size_t i = 0; i < alphabet.size(); ++i) out[alphabet[i]] = static_cast<unsigned char>(i);
    return out;
}

bool starts_with_at(const std::string& s, size_t pos, const std::string& prefix) {
    return pos + prefix.size() <= s.size() && s.compare(pos, prefix.size(), prefix) == 0;
}

enum class ByteTokenClass { Space, Whitespace, Letter, Number, Other };

size_t utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

ByteTokenClass byte_token_class(const std::string& s, size_t pos) {
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c == ' ') return ByteTokenClass::Space;
    if (c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f') return ByteTokenClass::Whitespace;
    if (c < 128) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return ByteTokenClass::Letter;
        if (c >= '0' && c <= '9') return ByteTokenClass::Number;
        return ByteTokenClass::Other;
    }
    // Approximation for GPT-2 ByteLevel's Unicode letter grouping. This covers
    // CJK prompts well enough for smoke text generation without adding a Unicode
    // property dependency to cpp_engine.
    return ByteTokenClass::Letter;
}

std::vector<std::string> bytelevel_pretokens(const std::string& text) {
    std::vector<std::string> out;
    std::string pending_space;
    for (size_t pos = 0; pos < text.size();) {
        ByteTokenClass cls = byte_token_class(text, pos);
        if (cls == ByteTokenClass::Space) {
            size_t start = pos;
            while (pos < text.size() && text[pos] == ' ') ++pos;
            const size_t count = pos - start;
            if (pos < text.size()) {
                if (count > 1) out.push_back(std::string(count - 1, ' '));
                pending_space = " ";
            } else {
                out.push_back(std::string(count, ' '));
            }
            continue;
        }
        if (cls == ByteTokenClass::Whitespace) {
            size_t start = pos;
            while (pos < text.size() && byte_token_class(text, pos) == ByteTokenClass::Whitespace) {
                ++pos;
            }
            out.push_back(text.substr(start, pos - start));
            continue;
        }
        const size_t start = pos;
        while (pos < text.size() && byte_token_class(text, pos) == cls) {
            pos += utf8_char_len(static_cast<unsigned char>(text[pos]));
        }
        out.push_back(pending_space + text.substr(start, pos - start));
        pending_space.clear();
    }
    return out;
}

const std::vector<std::string>& metadata_strings(const GGUFFile& gguf, const std::string& key) {
    auto it = gguf.metadata().find(key);
    if (it == gguf.metadata().end()) throw std::runtime_error("missing GGUF metadata array: " + key);
    const auto* values = std::get_if<std::vector<std::string>>(&it->second);
    if (values == nullptr) throw std::runtime_error("GGUF metadata key is not string array: " + key);
    return *values;
}

std::vector<int64_t> metadata_int_array_as_i64(const GGUFFile& gguf, const std::string& key) {
    auto it = gguf.metadata().find(key);
    if (it == gguf.metadata().end()) throw std::runtime_error("missing GGUF metadata array: " + key);
    if (const auto* values = std::get_if<std::vector<int64_t>>(&it->second)) return *values;
    if (const auto* values = std::get_if<std::vector<uint64_t>>(&it->second)) {
        std::vector<int64_t> out;
        out.reserve(values->size());
        for (uint64_t v : *values) out.push_back(static_cast<int64_t>(v));
        return out;
    }
    throw std::runtime_error("GGUF metadata key is not integer array: " + key);
}

}  // namespace

Tokenizer::Tokenizer(const std::string& ckpt_dir) {
    JsonValue root_value = parse_json(read_file(ckpt_dir + "/tokenizer.json"));
    const JsonObject& root = root_value.object();
    const JsonObject& model = object_get(root, "model")->object();
    const JsonObject& vocab = object_get(model, "vocab")->object();
    for (const auto& [tok, id_value] : vocab) {
        const uint64_t id = json_u64(id_value);
        set_token(id_to_token_, id, tok);
        token_to_id_[tok] = static_cast<int>(id);
    }
    if (const JsonValue* merges = object_get(model, "merges")) {
        int rank = 0;
        for (const JsonValue& item : merges->array()) {
            merge_rank_[split_merge(item.string())] = rank++;
        }
    }
    if (const JsonValue* added = object_get(root, "added_tokens")) {
        std::vector<uint8_t> seen;
        for (const JsonValue& item : added->array()) {
            const JsonObject& obj = item.object();
            const JsonValue* id = object_get(obj, "id");
            const JsonValue* content = object_get(obj, "content");
            if (id == nullptr || content == nullptr) continue;
            const uint64_t token_id = json_u64(*id);
            set_token(id_to_token_, token_id, content->string());
            token_to_id_[content->string()] = static_cast<int>(token_id);

            // Every added token is atomic for encoding purposes.
            if (token_id >= seen.size()) seen.resize(static_cast<size_t>(token_id) + 1, 0);
            if (seen[static_cast<size_t>(token_id)] == 0) {
                seen[static_cast<size_t>(token_id)] = 1;
                atomic_token_id_list_.push_back(static_cast<int>(token_id));
            }

            // Only tokens HF itself marks special are stripped from decoded text.
            // <think>, </think> and ｜DSML｜ carry "special": false precisely
            // because downstream parsing needs them.
            const JsonValue* special = object_get(obj, "special");
            const bool is_special = special != nullptr && special->is_bool() && special->boolean();
            if (!is_special) continue;
            if (token_id >= special_token_ids_.size()) special_token_ids_.resize(static_cast<size_t>(token_id) + 1, 0);
            special_token_ids_[static_cast<size_t>(token_id)] = 1;
        }
    }
}

Tokenizer Tokenizer::from_gguf(const GGUFFile& gguf) {
    Tokenizer tok;
    const auto& tokens = metadata_strings(gguf, "tokenizer.ggml.tokens");
    const auto& merges = metadata_strings(gguf, "tokenizer.ggml.merges");
    const auto token_types = metadata_int_array_as_i64(gguf, "tokenizer.ggml.token_type");
    if (tokens.size() != token_types.size()) throw std::runtime_error("GGUF tokenizer tokens/token_type length mismatch");
    tok.id_to_token_ = tokens;
    tok.special_token_ids_.assign(tokens.size(), 0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tok.token_to_id_[tokens[i]] = static_cast<int>(i);
        if (token_types[i] != 1) {
            tok.special_token_ids_[i] = 1;
            tok.atomic_token_id_list_.push_back(static_cast<int>(i));
        }
    }
    int rank = 0;
    for (const std::string& merge : merges) {
        tok.merge_rank_[split_merge(merge)] = rank++;
    }
    return tok;
}

std::string Tokenizer::token(int id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return "<invalid>";
    const std::string& tok = id_to_token_[static_cast<size_t>(id)];
    return tok.empty() ? "<empty>" : tok;
}

std::string Tokenizer::decode_piece(int id) const {
    std::string s = token(id);
    s = replace_all(s, "Ġ", " ");
    s = replace_all(s, "▁", " ");
    s = replace_all(s, "Ċ", "\n");
    return s;
}

int Tokenizer::token_id(const std::string& token) const {
    const auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool Tokenizer::is_special_token(int id) const {
    return id >= 0 && static_cast<size_t>(id) < special_token_ids_.size() && special_token_ids_[static_cast<size_t>(id)] != 0;
}

std::string Tokenizer::decode_tokens(const std::vector<int>& ids, bool skip_special_tokens) const {
    static const auto decoder = byte_decoder();
    std::string out;
    for (int id : ids) {
        std::string piece = token(id);
        if (skip_special_tokens && is_special_token(id)) continue;
        for (size_t i = 0; i < piece.size();) {
            bool matched = false;
            for (const auto& [encoded, byte] : decoder) {
                if (starts_with_at(piece, i, encoded)) {
                    out.push_back(static_cast<char>(byte));
                    i += encoded.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) out.push_back(piece[i++]);
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode_basic(const std::string& text, bool add_bos) const {
    static const std::vector<std::string> bytes = byte_alphabet();
    std::vector<int> ids;
    if (add_bos) ids.push_back(0);

    auto append_bpe_pretoken = [&](const std::string& pretoken) {
        std::vector<std::string> pieces;
        pieces.reserve(pretoken.size());
        for (unsigned char c : pretoken) pieces.push_back(bytes[c]);
        while (pieces.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_pos = pieces.size();
            for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                auto it = merge_rank_.find({pieces[i], pieces[i + 1]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }
            if (best_pos == pieces.size()) break;
            pieces[best_pos] += pieces[best_pos + 1];
            pieces.erase(pieces.begin() + static_cast<long>(best_pos + 1));
        }
        for (const auto& piece : pieces) {
            auto it = token_to_id_.find(piece);
            if (it == token_to_id_.end()) throw std::runtime_error("BPE piece missing from vocab: " + piece);
            ids.push_back(it->second);
        }
    };

    auto append_bpe_segment = [&](const std::string& segment) {
        for (const std::string& pretoken : bytelevel_pretokens(segment)) {
            append_bpe_pretoken(pretoken);
        }
    };

    auto longest_atomic_at = [&](size_t pos, int& token_id, size_t& len) -> bool {
        token_id = -1;
        len = 0;
        for (int id : atomic_token_id_list_) {
            if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
            const std::string& tok = id_to_token_[static_cast<size_t>(id)];
            if (tok.empty()) continue;
            if (starts_with_at(text, pos, tok) && tok.size() > len) {
                token_id = id;
                len = tok.size();
            }
        }
        return token_id >= 0;
    };

    std::string segment;
    for (size_t pos = 0; pos < text.size();) {
        int special_id = -1;
        size_t special_len = 0;
        if (longest_atomic_at(pos, special_id, special_len)) {
            if (!segment.empty()) {
                append_bpe_segment(segment);
                segment.clear();
            }
            ids.push_back(special_id);
            pos += special_len;
        } else {
            segment.push_back(text[pos++]);
        }
    }
    if (!segment.empty()) append_bpe_segment(segment);
    return ids;
}

}  // namespace pocket
