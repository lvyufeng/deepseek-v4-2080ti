#include "tokenizer.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void check_encode(const pocket::Tokenizer& tok, const std::string& text, const std::vector<int>& expected) {
    auto ids = tok.encode_basic(text, false);
    if (ids != expected) {
        std::string got;
        for (int id : ids) got += std::to_string(id) + " ";
        std::string want;
        for (int id : expected) want += std::to_string(id) + " ";
        throw std::runtime_error("encode mismatch for [" + text + "] got=[" + got + "] want=[" + want + "]");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_tokenizer <ckpt_dir>\n";
        return 2;
    }
    try {
        pocket::Tokenizer tok(argv[1]);
        require(tok.vocab_size() >= 129280, "bad vocab size");
        require(tok.token(0) == "<｜begin▁of▁sentence｜>", "bad bos token");
        require(tok.token(1) == "<｜end▁of▁sentence｜>", "bad eos token");
        require(!tok.token(107590).empty(), "missing generated token");
        check_encode(tok, "hello", {33310});
        check_encode(tok, " hello", {44388});
        check_encode(tok, "two", {23315});
        check_encode(tok, " two", {1234});
        check_encode(tok, "Hello world", {19923, 2058});
        check_encode(tok, "123", {6895});
        auto bos = tok.encode_basic("hello", true);
        require(bos.size() == 2 && bos[0] == 0 && bos[1] == 33310, "bad add_bos encode");
        require(tok.decode_tokens(tok.encode_basic("Hello world", true)) == "Hello world", "bad decode roundtrip");
        require(tok.decode_tokens({55262}) == "我们必须", "bad byte-level decode");

        // Chat markup carries "special": false in tokenizer.json and must survive
        // skip_special_tokens decoding — the completion parser splits reasoning
        // from content on </think>, and tool calls on ｜DSML｜.
        const int think_open = tok.token_id("<think>");
        const int think_close = tok.token_id("</think>");
        const int dsml = tok.token_id("｜DSML｜");
        require(think_open == 128821 && think_close == 128822 && dsml == 128825,
                "unexpected chat markup token ids");
        for (int id : {think_open, think_close, dsml, tok.token_id("<｜User｜>"),
                       tok.token_id("<｜Assistant｜>")}) {
            require(!tok.is_special_token(id),
                    "chat markup must not be flagged special: id " + std::to_string(id));
        }
        require(tok.decode_tokens({think_close}) == "</think>", "</think> was stripped");
        require(tok.decode_tokens({dsml}) == "｜DSML｜", "｜DSML｜ was stripped");
        require(tok.decode_tokens({19923, think_close, 2058}) == "Hello</think> world",
                "bad markup-preserving decode");
        // Markup is still atomic when encoding.
        check_encode(tok, "</think>", {think_close});
        check_encode(tok, "Hello</think>", {19923, think_close});

        // BOS/EOS/padding stay hidden, and explicitly asking for them works.
        for (int id : {0, 1, 2}) {
            require(tok.is_special_token(id),
                    "bos/eos/pad must be special: id " + std::to_string(id));
        }
        require(tok.decode_tokens({0, 19923, 1}) == "Hello", "bos/eos leaked into text");
        require(tok.decode_tokens({0, 19923, 1}, false) ==
                    "<｜begin▁of▁sentence｜>Hello<｜end▁of▁sentence｜>",
                "bad decode with skip_special_tokens=false");

        std::cout << "[PASS] tokenizer vocab=" << tok.vocab_size()
                  << " token107590=" << tok.decode_piece(107590) << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
