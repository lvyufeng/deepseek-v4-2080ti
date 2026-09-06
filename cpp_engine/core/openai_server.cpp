#include "openai_server.hpp"

#include "json_lite.hpp"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace pocket {

namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

// Returns the JSON array text for `messages` from the request, preserving the
// caller-provided structure verbatim. json_lite doesn't ship a serializer, so
// we extract the raw substring from the original body using offsets we infer
// by re-rendering the value.
std::string extract_messages_json(const std::string& body) {
    // Trivially locate `"messages"` and bracket-match to find the array.
    size_t key = body.find("\"messages\"");
    if (key == std::string::npos) return "[]";
    size_t pos = body.find('[', key);
    if (pos == std::string::npos) return "[]";
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = pos; i < body.size(); ++i) {
        char c = body[i];
        if (esc) { esc = false; continue; }
        if (c == '\\' && in_str) { esc = true; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '[' || c == '{') ++depth;
        else if (c == ']' || c == '}') {
            --depth;
            if (depth == 0 && c == ']') return body.substr(pos, i - pos + 1);
        }
    }
    return "[]";
}

// Same as extract_messages_json but for the top-level `tools` array. Returns
// an empty string when no `tools` field is present.
std::string extract_tools_json(const std::string& body) {
    size_t key = body.find("\"tools\"");
    if (key == std::string::npos) return "";
    size_t pos = body.find('[', key);
    if (pos == std::string::npos) return "";
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = pos; i < body.size(); ++i) {
        char c = body[i];
        if (esc) { esc = false; continue; }
        if (c == '\\' && in_str) { esc = true; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '[' || c == '{') ++depth;
        else if (c == ']' || c == '}') {
            --depth;
            if (depth == 0 && c == ']') return body.substr(pos, i - pos + 1);
        }
    }
    return "";
}

double get_number(const JsonObject& obj, const std::string& key, double fallback) {
    const JsonValue* v = object_get(obj, key);
    if (v != nullptr && v->is_number()) return v->number();
    return fallback;
}

bool get_bool(const JsonObject& obj, const std::string& key, bool fallback) {
    const JsonValue* v = object_get(obj, key);
    if (v != nullptr && v->is_bool()) return v->boolean();
    return fallback;
}

std::string get_string(const JsonObject& obj, const std::string& key, const std::string& fallback = "") {
    const JsonValue* v = object_get(obj, key);
    if (v != nullptr && v->is_string()) return v->string();
    return fallback;
}

// Truncate `s` to its longest prefix that is a complete UTF-8 sequence.
// Returns {complete_prefix, leftover_bytes}. Walks back at most 3 bytes from
// the end looking for a start byte; if the trailing sequence is complete the
// full string is returned, if incomplete the partial sequence is withheld.
std::pair<std::string, std::string> split_utf8_complete(const std::string& s) {
    if (s.empty()) return {"", ""};
    size_t i = s.size();
    for (int back = 0; back < 4 && i > 0; ++back) {
        unsigned char c = static_cast<unsigned char>(s[i - 1]);
        if ((c & 0x80) == 0) {
            // ASCII byte at i-1; everything up to s.size() is complete.
            return { s, "" };
        }
        if ((c & 0xC0) == 0xC0) {
            // Start byte at i-1.
            size_t expected = 1;
            if ((c & 0xE0) == 0xC0) expected = 2;
            else if ((c & 0xF0) == 0xE0) expected = 3;
            else if ((c & 0xF8) == 0xF0) expected = 4;
            size_t available = s.size() - (i - 1);
            if (available >= expected) {
                // Trailing sequence is complete.
                return { s, "" };
            }
            // Incomplete; withhold the partial sequence starting at i-1.
            return { s.substr(0, i - 1), s.substr(i - 1) };
        }
        // Continuation byte; keep walking back.
        --i;
    }
    // No start byte found within the last 4 bytes; emit everything as-is.
    return { s, "" };
}

std::string make_request_id() {
    using clock = std::chrono::steady_clock;
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
    std::ostringstream os;
    os << "chatcmpl-cpp-" << std::hex << t;
    return os.str();
}

std::string render_choice_message(const std::string& content, const std::string& reasoning, const std::string& tool_calls_json) {
    std::ostringstream os;
    os << "{\"role\":\"assistant\"";
    os << ",\"content\":\"" << json_escape(content) << "\"";
    if (!reasoning.empty()) {
        os << ",\"reasoning_content\":\"" << json_escape(reasoning) << "\"";
    }
    if (!tool_calls_json.empty() && tool_calls_json != "[]") {
        os << ",\"tool_calls\":" << tool_calls_json;
    }
    os << "}";
    return os.str();
}

}  // namespace

struct OpenAIServer::Impl {
    PersistentEngine& engine;
    PythonSidecar& sidecar;
    OpenAIServerConfig cfg;
    httplib::Server svr;
    std::mutex inflight;
    std::atomic<bool> running{false};

    Impl(PersistentEngine& e, PythonSidecar& s, const OpenAIServerConfig& c) : engine(e), sidecar(s), cfg(c) {}

    void handle_health(httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    }

    void handle_models(httplib::Response& res) {
        std::ostringstream os;
        os << "{\"object\":\"list\",\"data\":[{\"id\":\"deepseek-v4-flash\",\"object\":\"model\",\"owned_by\":\"local\"}]}";
        res.set_content(os.str(), "application/json");
    }

    bool encode_request(const std::string& body, const JsonObject& obj,
                        EncodeReply& out, std::string& thinking_mode_out,
                        int& max_tokens_out, bool& stream_out,
                        SamplingParams& sp_out, std::string& err_out) {
        EncodeRequest enc;
        enc.messages_json = extract_messages_json(body);
        enc.tools_json = extract_tools_json(body);
        thinking_mode_out = get_string(obj, "thinking_mode", cfg.default_thinking_mode);
        enc.thinking_mode = thinking_mode_out;
        enc.reasoning_effort = get_string(obj, "reasoning_effort", "");
        enc.add_generation_prompt = get_bool(obj, "add_generation_prompt", true);
        enc.drop_thinking = get_bool(obj, "drop_thinking", true);

        max_tokens_out = static_cast<int>(get_number(obj, "max_tokens", cfg.default_max_tokens));
        if (max_tokens_out <= 0) max_tokens_out = cfg.default_max_tokens;
        stream_out = get_bool(obj, "stream", false);
        sp_out.temperature = static_cast<float>(get_number(obj, "temperature", 1.0));
        sp_out.top_p = static_cast<float>(get_number(obj, "top_p", 1.0));
        sp_out.seed = static_cast<uint64_t>(get_number(obj, "seed", 0));
        // Treat near-zero temperature as greedy.
        sp_out.greedy = sp_out.temperature <= 1.0e-5f;
        out = sidecar.encode(enc);
        if (!out.ok) {
            err_out = out.err;
            return false;
        }
        if (out.token_ids.empty()) {
            err_out = "encode produced no tokens";
            return false;
        }
        return true;
    }

    void emit_error(httplib::Response& res, int status, const std::string& msg) {
        std::ostringstream os;
        os << "{\"error\":{\"message\":\"" << json_escape(msg)
           << "\",\"type\":\"server_error\"}}";
        res.status = status;
        res.set_content(os.str(), "application/json");
    }

    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
        const std::string& body = req.body;
        JsonValue jv;
        try { jv = parse_json(body); } catch (const std::exception& ex) {
            emit_error(res, 400, std::string("invalid JSON: ") + ex.what());
            return;
        }
        if (!jv.is_object()) {
            emit_error(res, 400, "request body must be JSON object");
            return;
        }
        const auto& obj = jv.object();
        EncodeReply enc_reply;
        std::string thinking_mode;
        int max_tokens = 0;
        bool stream = false;
        SamplingParams sp;
        std::string err;
        if (!encode_request(body, obj, enc_reply, thinking_mode, max_tokens, stream, sp, err)) {
            emit_error(res, 400, err);
            return;
        }

        if (cfg.log_requests) {
            std::cerr << "[server] request prompt_tokens=" << enc_reply.token_ids.size()
                      << " max_tokens=" << max_tokens << " stream=" << (stream ? 1 : 0)
                      << " thinking_mode=" << thinking_mode << "\n";
        }

        std::lock_guard<std::mutex> inflight_lk(inflight);
        if (static_cast<int>(enc_reply.token_ids.size()) + max_tokens > engine.max_context()) {
            emit_error(res, 400, "prompt + max_tokens exceeds max_context");
            return;
        }

        if (stream) {
            handle_stream(req, res, enc_reply, max_tokens, sp, thinking_mode);
        } else {
            handle_nonstream(res, enc_reply, max_tokens, sp, thinking_mode);
        }
    }

    void handle_nonstream(httplib::Response& res, const EncodeReply& enc, int max_tokens,
                          const SamplingParams& sp, const std::string& thinking_mode) {
        engine.worker_command_reset();
        engine.reset_session();
        engine.worker_command_prefill(enc.token_ids);
        int token = engine.prefill(enc.token_ids, sp);

        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(max_tokens));
        const int eos_id = engine.eos_id();
        bool hit_eos = (token == eos_id);
        if (!hit_eos) generated.push_back(token);
        int position = static_cast<int>(enc.token_ids.size());
        for (int step = 1; step < max_tokens && !hit_eos; ++step) {
            engine.worker_command_decode(token, position + step - 1);
            token = engine.decode_step(token, position + step - 1, sp);
            if (token == eos_id) { hit_eos = true; break; }
            generated.push_back(token);
        }

        const std::string text = engine.tokenizer().decode_tokens(generated);
        const ParsedMessage parsed = sidecar.parse(text, thinking_mode);
        std::string content = parsed.ok ? parsed.content : text;
        std::string reasoning = parsed.ok ? parsed.reasoning : std::string();
        std::string tool_calls_json = parsed.ok ? parsed.tool_calls_json : "[]";

        std::ostringstream os;
        os << "{\"id\":\"" << make_request_id() << "\""
           << ",\"object\":\"chat.completion\""
           << ",\"created\":" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()
           << ",\"model\":\"deepseek-v4-flash\""
           << ",\"choices\":[{\"index\":0,\"finish_reason\":\"" << (hit_eos ? "stop" : "length") << "\""
           << ",\"message\":" << render_choice_message(content, reasoning, tool_calls_json) << "}]"
           << ",\"usage\":{\"prompt_tokens\":" << enc.token_ids.size()
           << ",\"completion_tokens\":" << generated.size()
           << ",\"total_tokens\":" << (enc.token_ids.size() + generated.size()) << "}}";
        res.set_content(os.str(), "application/json");
    }

    void handle_stream(const httplib::Request& /*req*/, httplib::Response& res,
                       const EncodeReply& enc, int max_tokens,
                       const SamplingParams& sp, const std::string& thinking_mode) {
        const std::string id = make_request_id();
        const long long created = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string model = "deepseek-v4-flash";
        const int eos_id = engine.eos_id();

        // Sliding-window prefix used to compute deltas + UTF-8 boundary buffer.
        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(max_tokens));
        size_t sent_offset = 0;  // bytes of decode_tokens(generated) already sent

        // In thinking mode the prompt ends with <think>, so the completion starts
        // inside the reasoning block and switches to the answer at the first
        // </think>. Splitting on the token id rather than the decoded text keeps
        // each stream's UTF-8 decoding self-contained and cannot be fooled by a
        // model that writes the literal characters.
        const int think_end_id = engine.tokenizer().token_id("</think>");
        bool in_reasoning = (thinking_mode == "thinking");

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [this, enc, max_tokens, sp, id, created, model, eos_id, think_end_id,
             generated, sent_offset, in_reasoning]
            (size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {

            engine.worker_command_reset();
            engine.reset_session();

            auto send_chunk = [&](const std::string& delta, const char* field,
                                  const char* finish_reason = nullptr) {
                std::ostringstream os;
                os << "{\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\""
                   << ",\"created\":" << created << ",\"model\":\"" << model << "\""
                   << ",\"choices\":[{\"index\":0,\"delta\":{";
                if (!delta.empty()) {
                    os << "\"" << field << "\":\"" << json_escape(delta) << "\"";
                }
                os << "}";
                if (finish_reason != nullptr) {
                    os << ",\"finish_reason\":\"" << finish_reason << "\"";
                } else {
                    os << ",\"finish_reason\":null";
                }
                os << "}]}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            };

            // First chunk: role marker.
            {
                std::ostringstream os;
                os << "{\"id\":\"" << id << "\",\"object\":\"chat.completion.chunk\""
                   << ",\"created\":" << created << ",\"model\":\"" << model << "\""
                   << ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
                std::string line = "data: " + os.str() + "\n\n";
                sink.write(line.data(), line.size());
            }

            engine.worker_command_prefill(enc.token_ids);
            int token = engine.prefill(enc.token_ids, sp);
            bool hit_eos = (token == eos_id);
            int position = static_cast<int>(enc.token_ids.size());
            int decoded_count = 0;

            auto emit_delta = [&]() {
                const std::string full = engine.tokenizer().decode_tokens(generated);
                if (full.size() <= sent_offset) return;
                std::string candidate = full.substr(sent_offset);
                auto [complete, leftover] = split_utf8_complete(candidate);
                if (!complete.empty()) {
                    sent_offset += complete.size();
                    send_chunk(complete, in_reasoning ? "reasoning_content" : "content");
                }
            };

            // Closing the reasoning block restarts the delta window: the marker
            // itself belongs to neither stream, and dropping the tokens before it
            // keeps decode_tokens() cheap on long generations.
            auto accept = [&](int tok) {
                if (in_reasoning && tok == think_end_id) {
                    emit_delta();
                    generated.clear();
                    sent_offset = 0;
                    in_reasoning = false;
                    return;
                }
                generated.push_back(tok);
                emit_delta();
            };

            if (!hit_eos) {
                accept(token);
                ++decoded_count;
            }

            for (int step = 1; step < max_tokens && !hit_eos; ++step) {
                engine.worker_command_decode(token, position + step - 1);
                token = engine.decode_step(token, position + step - 1, sp);
                if (token == eos_id) { hit_eos = true; break; }
                accept(token);
                ++decoded_count;
            }

            // Flush any remaining tail bytes (e.g. an isolated partial sequence
            // at EOS). In practice decode_tokens at terminal state produces
            // valid UTF-8, so this is usually empty.
            {
                const std::string full = engine.tokenizer().decode_tokens(generated);
                if (full.size() > sent_offset) {
                    std::string tail = full.substr(sent_offset);
                    sent_offset += tail.size();
                    send_chunk(tail, in_reasoning ? "reasoning_content" : "content");
                }
            }

            // Final chunk with finish_reason.
            send_chunk("", "content", hit_eos ? "stop" : "length");
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    }

    void run() {
        running = true;
        svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) { handle_health(res); });
        svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) { handle_models(res); });
        svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            try { handle_chat_completions(req, res); }
            catch (const std::exception& ex) { emit_error(res, 500, ex.what()); }
        });
        std::cerr << "[server] listening on " << cfg.host << ":" << cfg.port << "\n";
        svr.listen(cfg.host.c_str(), cfg.port);
        running = false;
    }

    void stop() {
        if (running) svr.stop();
    }
};

OpenAIServer::OpenAIServer(PersistentEngine& engine, PythonSidecar& sidecar, const OpenAIServerConfig& cfg)
    : impl_(std::make_unique<Impl>(engine, sidecar, cfg)) {}
OpenAIServer::~OpenAIServer() = default;
void OpenAIServer::run() { impl_->run(); }
void OpenAIServer::stop() { impl_->stop(); }

}  // namespace pocket
