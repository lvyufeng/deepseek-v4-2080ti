#include "python_sidecar.hpp"

#include "json_lite.hpp"

#include <fcntl.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

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

std::string read_line(int fd) {
    std::string out;
    char ch = 0;
    while (true) {
        ssize_t n = ::read(fd, &ch, 1);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("PythonSidecar read failed: ") + std::strerror(errno));
        }
        if (n == 0) {
            if (out.empty()) throw std::runtime_error("PythonSidecar pipe closed unexpectedly");
            return out;
        }
        if (ch == '\n') return out;
        out += ch;
    }
}

void write_all(int fd, const std::string& s) {
    const char* data = s.data();
    size_t left = s.size();
    while (left > 0) {
        ssize_t n = ::write(fd, data, left);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("PythonSidecar write failed: ") + std::strerror(errno));
        }
        data += n;
        left -= static_cast<size_t>(n);
    }
}

}  // namespace

PythonSidecar::PythonSidecar(const std::string& python_bin,
                             const std::string& script_path,
                             const std::string& ckpt_dir) {
    int parent_to_child[2] = {-1, -1};
    int child_to_parent[2] = {-1, -1};
    if (::pipe(parent_to_child) != 0) throw std::runtime_error("pipe1 failed");
    if (::pipe(child_to_parent) != 0) {
        ::close(parent_to_child[0]); ::close(parent_to_child[1]);
        throw std::runtime_error("pipe2 failed");
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(parent_to_child[0]); ::close(parent_to_child[1]);
        ::close(child_to_parent[0]); ::close(child_to_parent[1]);
        throw std::runtime_error("fork failed");
    }
    if (pid == 0) {
        // Child: stdin <- parent_to_child[0], stdout -> child_to_parent[1]
        ::dup2(parent_to_child[0], STDIN_FILENO);
        ::dup2(child_to_parent[1], STDOUT_FILENO);
        ::close(parent_to_child[0]); ::close(parent_to_child[1]);
        ::close(child_to_parent[0]); ::close(child_to_parent[1]);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(python_bin.c_str()));
        argv.push_back(const_cast<char*>("-u"));
        argv.push_back(const_cast<char*>(script_path.c_str()));
        argv.push_back(const_cast<char*>("--ckpt"));
        argv.push_back(const_cast<char*>(ckpt_dir.c_str()));
        argv.push_back(nullptr);
        ::execvp(python_bin.c_str(), argv.data());
        std::cerr << "execvp python failed: " << std::strerror(errno) << "\n";
        _exit(127);
    }
    // Parent
    child_pid_ = pid;
    ::close(parent_to_child[0]);
    ::close(child_to_parent[1]);
    write_fd_ = parent_to_child[1];
    read_fd_ = child_to_parent[0];

    // Read banner. Sidecar emits {"ok":true,"ready":true,"eos_token_id":N}.
    std::string banner = read_line(read_fd_);
    JsonValue v = parse_json(banner);
    if (!v.is_object()) throw std::runtime_error("sidecar banner not an object: " + banner);
    const auto& obj = v.object();
    const JsonValue* ok = object_get(obj, "ok");
    if (ok == nullptr || !ok->is_bool() || !ok->boolean()) {
        throw std::runtime_error("sidecar banner not ok: " + banner);
    }
    const JsonValue* eos = object_get(obj, "eos_token_id");
    if (eos != nullptr && eos->is_number()) {
        eos_token_id_ = static_cast<int>(eos->number());
    }
}

PythonSidecar::~PythonSidecar() {
    try { shutdown(); } catch (...) {}
    if (write_fd_ >= 0) ::close(write_fd_);
    if (read_fd_ >= 0) ::close(read_fd_);
    if (child_pid_ > 0) {
        int status = 0;
        ::kill(child_pid_, SIGTERM);
        ::waitpid(child_pid_, &status, 0);
    }
}

void PythonSidecar::shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    if (write_fd_ < 0) return;
    try {
        write_all(write_fd_, "{\"op\":\"shutdown\"}\n");
        (void)read_line(read_fd_);
    } catch (...) {}
}

std::string PythonSidecar::send_request(const std::string& json_line) {
    std::lock_guard<std::mutex> lk(mu_);
    write_all(write_fd_, json_line);
    return read_line(read_fd_);
}

EncodeReply PythonSidecar::encode(const EncodeRequest& req) {
    EncodeReply reply;
    // Sidecar uses newline-delimited JSON on stdin. Raw newlines/tabs inside
    // our embedded `messages` / `tools` arrays would break the line framing,
    // so flatten any whitespace outside of JSON string literals. Inside JSON
    // strings, raw newlines are illegal, so this is always safe.
    auto flatten = [](const std::string& s) {
        std::string out;
        out.reserve(s.size());
        bool in_str = false;
        bool esc = false;
        for (char c : s) {
            if (esc) { out.push_back(c); esc = false; continue; }
            if (c == '\\' && in_str) { out.push_back(c); esc = true; continue; }
            if (c == '"') { in_str = !in_str; out.push_back(c); continue; }
            if (!in_str && (c == '\n' || c == '\r' || c == '\t')) { out.push_back(' '); continue; }
            out.push_back(c);
        }
        return out;
    };
    const std::string messages_inline = req.messages_json.empty() ? std::string("[]") : flatten(req.messages_json);
    const std::string tools_inline = req.tools_json.empty() ? std::string() : flatten(req.tools_json);
    std::ostringstream os;
    os << "{\"op\":\"encode\""
       << ",\"messages\":" << messages_inline
       << ",\"thinking_mode\":\"" << json_escape(req.thinking_mode) << "\""
       << ",\"add_generation_prompt\":" << (req.add_generation_prompt ? "true" : "false")
       << ",\"drop_thinking\":" << (req.drop_thinking ? "true" : "false");
    if (!req.reasoning_effort.empty()) {
        os << ",\"reasoning_effort\":\"" << json_escape(req.reasoning_effort) << "\"";
    }
    if (!tools_inline.empty()) {
        os << ",\"tools\":" << tools_inline;
    }
    os << "}\n";

    std::string resp = send_request(os.str());
    JsonValue v;
    try { v = parse_json(resp); } catch (const std::exception& ex) {
        reply.err = std::string("parse_json failed: ") + ex.what() + " raw=" + resp;
        return reply;
    }
    if (!v.is_object()) { reply.err = "encode response not object: " + resp; return reply; }
    const auto& obj = v.object();
    const JsonValue* ok = object_get(obj, "ok");
    if (ok == nullptr || !ok->is_bool() || !ok->boolean()) {
        const JsonValue* err = object_get(obj, "err");
        reply.err = err != nullptr && err->is_string() ? err->string() : resp;
        return reply;
    }
    const JsonValue* pt = object_get(obj, "prompt_text");
    if (pt != nullptr && pt->is_string()) reply.prompt_text = pt->string();
    const JsonValue* ids = object_get(obj, "token_ids");
    if (ids != nullptr && ids->is_array()) {
        const auto& arr = ids->array();
        reply.token_ids.reserve(arr.size());
        for (const auto& item : arr) {
            if (!item.is_number()) { reply.err = "token_ids item not number"; return reply; }
            reply.token_ids.push_back(static_cast<int>(item.number()));
        }
    }
    reply.ok = true;
    return reply;
}

ParsedMessage PythonSidecar::parse(const std::string& text, const std::string& thinking_mode) {
    ParsedMessage parsed;
    std::ostringstream os;
    os << "{\"op\":\"parse\",\"text\":\"" << json_escape(text)
       << "\",\"thinking_mode\":\"" << json_escape(thinking_mode) << "\"}\n";
    std::string resp = send_request(os.str());
    JsonValue v;
    try { v = parse_json(resp); } catch (const std::exception& ex) {
        parsed.err = std::string("parse_json failed: ") + ex.what() + " raw=" + resp;
        return parsed;
    }
    if (!v.is_object()) { parsed.err = "parse response not object: " + resp; return parsed; }
    const auto& obj = v.object();
    const JsonValue* ok = object_get(obj, "ok");
    if (ok == nullptr || !ok->is_bool() || !ok->boolean()) {
        const JsonValue* err = object_get(obj, "err");
        parsed.err = err != nullptr && err->is_string() ? err->string() : resp;
        return parsed;
    }
    const JsonValue* content = object_get(obj, "content");
    if (content != nullptr && content->is_string()) parsed.content = content->string();
    const JsonValue* reasoning = object_get(obj, "reasoning_content");
    if (reasoning != nullptr && reasoning->is_string()) parsed.reasoning = reasoning->string();
    // tool_calls round-trip as a raw JSON string for the server to embed.
    const JsonValue* tools = object_get(obj, "tool_calls");
    if (tools != nullptr && tools->is_array()) {
        // Serialize back to JSON ourselves (lightweight; rare path).
        std::ostringstream tos;
        tos << "[";
        bool first = true;
        for (const auto& tc : tools->array()) {
            if (!first) tos << ",";
            first = false;
            // We just embed the raw payload as-is. Since JsonValue has no
            // serializer we re-extract known fields. tool_call shape from
            // deepseek_v4.parse_message_from_completion_text is
            // {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}.
            tos << "{";
            if (tc.is_object()) {
                bool firstk = true;
                for (const auto& [k, val] : tc.object()) {
                    if (!firstk) tos << ",";
                    firstk = false;
                    tos << "\"" << json_escape(k) << "\":";
                    if (val.is_string()) {
                        tos << "\"" << json_escape(val.string()) << "\"";
                    } else if (val.is_object()) {
                        tos << "{";
                        bool firstj = true;
                        for (const auto& [jk, jv] : val.object()) {
                            if (!firstj) tos << ",";
                            firstj = false;
                            tos << "\"" << json_escape(jk) << "\":";
                            if (jv.is_string()) tos << "\"" << json_escape(jv.string()) << "\"";
                            else if (jv.is_number()) tos << jv.number();
                            else if (jv.is_bool()) tos << (jv.boolean() ? "true" : "false");
                            else tos << "null";
                        }
                        tos << "}";
                    } else if (val.is_number()) {
                        tos << val.number();
                    } else if (val.is_bool()) {
                        tos << (val.boolean() ? "true" : "false");
                    } else {
                        tos << "null";
                    }
                }
            }
            tos << "}";
        }
        tos << "]";
        parsed.tool_calls_json = tos.str();
    } else {
        parsed.tool_calls_json = "[]";
    }
    parsed.ok = true;
    return parsed;
}

}  // namespace pocket
