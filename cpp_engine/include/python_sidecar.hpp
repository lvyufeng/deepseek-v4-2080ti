#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace pocket {

struct EncodeRequest {
    std::string messages_json;        // JSON array of {role, content, ...}
    std::string thinking_mode = "chat";
    std::string reasoning_effort;     // empty = none
    std::string tools_json;           // JSON array of tools or empty
    bool add_generation_prompt = true;
    bool drop_thinking = true;
};

struct EncodeReply {
    bool ok = false;
    std::string err;
    std::string prompt_text;
    std::vector<int> token_ids;
};

struct ParsedMessage {
    bool ok = false;
    std::string err;
    std::string content;
    std::string reasoning;
    std::string tool_calls_json;      // raw JSON array string, may be "[]"
};

// Thin C++ wrapper around a long-running python helper. The helper renders
// DeepSeek-V4 chat templates and parses generated text into structured fields.
// One sidecar instance per server process; calls are serialised via the
// internal mutex.
class PythonSidecar {
public:
    PythonSidecar(const std::string& python_bin,
                  const std::string& script_path,
                  const std::string& ckpt_dir);
    ~PythonSidecar();

    PythonSidecar(const PythonSidecar&) = delete;
    PythonSidecar& operator=(const PythonSidecar&) = delete;

    int eos_token_id() const { return eos_token_id_; }

    EncodeReply encode(const EncodeRequest& req);
    ParsedMessage parse(const std::string& text, const std::string& thinking_mode);

private:
    std::string send_request(const std::string& json_line);
    void shutdown();

    int child_pid_ = -1;
    int write_fd_ = -1;
    int read_fd_ = -1;
    int eos_token_id_ = 1;
    std::mutex mu_;
};

}  // namespace pocket
