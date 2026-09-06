#pragma once

#include "persistent_engine.hpp"
#include "python_sidecar.hpp"

#include <atomic>
#include <mutex>
#include <string>

namespace pocket {

struct OpenAIServerConfig {
    int port = 8000;
    std::string host = "0.0.0.0";
    std::string default_thinking_mode = "chat";
    int default_max_tokens = 256;
    bool log_requests = true;
};

// Single-tenant OpenAI-compatible HTTP server backed by PersistentEngine.
// Requests are serialised: while one chat completion is generating, others
// queue on the inflight mutex. Stream and non-stream paths are both supported.
class OpenAIServer {
public:
    OpenAIServer(PersistentEngine& engine, PythonSidecar& sidecar, const OpenAIServerConfig& cfg);
    ~OpenAIServer();

    // Blocks until stop() is called or SIGINT/SIGTERM arrives.
    void run();
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pocket
