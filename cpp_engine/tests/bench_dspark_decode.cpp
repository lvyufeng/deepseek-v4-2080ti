// TP end-to-end benchmark for plain and DSpark speculative decode.
//
// POCKETLLM_BENCH_MODE selects one isolated path per process:
//   plain_nodspark: plain decode without loading DSpark or hidden capture
//   plain_dspark:   plain decode after loading DSpark (capture overhead)
//   spec:           speculative decode; POCKETLLM_CPP_BATCHED_VERIFY selects verify
// Fixture format: one case per line, <name>\t<comma-separated token ids>.
// Rank 0 prints one RESULT_JSON object per case; other output is diagnostic.

#include "persistent_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pocket;

namespace {

struct Fixture {
    std::string name;
    std::vector<int> prompt;
};

struct DecodeResult {
    std::vector<int> tokens;
    double seconds = 0.0;
    int rounds = 0;
    int accepted = 0;
};

void cuda_sync() {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaDeviceSynchronize: ") +
                                 cudaGetErrorString(err));
    }
}

std::vector<int> parse_ids(const std::string& text) {
    std::vector<int> ids;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) ids.push_back(std::stoi(item));
    }
    return ids;
}

std::vector<Fixture> load_fixtures(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fixture file: " + path);
    std::vector<Fixture> fixtures;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("fixture line needs name<TAB>ids: " + line);
        }
        Fixture f{line.substr(0, tab), parse_ids(line.substr(tab + 1))};
        if (f.name.empty() || f.prompt.empty()) {
            throw std::runtime_error("empty fixture name or prompt: " + line);
        }
        fixtures.push_back(std::move(f));
    }
    if (fixtures.empty()) throw std::runtime_error("fixture file is empty: " + path);
    return fixtures;
}

std::string json_tokens(const std::vector<int>& tokens) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) out << ',';
        out << tokens[i];
    }
    out << ']';
    return out.str();
}

DecodeResult run_plain(PersistentEngine& engine, const Fixture& fixture,
                       int token_count, const SamplingParams& sp) {
    engine.worker_command_reset();
    engine.reset_session();
    engine.worker_command_prefill(fixture.prompt);
    int token = engine.prefill(fixture.prompt, sp);
    int position = static_cast<int>(fixture.prompt.size());

    cuda_sync();
    const auto start = std::chrono::steady_clock::now();
    DecodeResult result;
    result.tokens.reserve(token_count);
    for (int i = 0; i < token_count; ++i) {
        engine.worker_command_decode(token, position);
        token = engine.decode_step(token, position++, sp);
        result.tokens.push_back(token);
    }
    cuda_sync();
    result.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    result.rounds = token_count;
    return result;
}

DecodeResult run_speculative(PersistentEngine& engine, const Fixture& fixture,
                             int token_count, const SamplingParams& sp) {
    engine.worker_command_reset();
    engine.reset_session();
    engine.worker_command_prefill(fixture.prompt);
    int token = engine.prefill(fixture.prompt, sp);
    int position = static_cast<int>(fixture.prompt.size());

    cuda_sync();
    const auto start = std::chrono::steady_clock::now();
    DecodeResult result;
    result.tokens.reserve(token_count);
    while (static_cast<int>(result.tokens.size()) < token_count) {
        const std::vector<int> generated = engine.speculative_step(token, position, sp);
        if (generated.empty()) throw std::runtime_error("speculative_step returned no tokens");
        ++result.rounds;
        result.accepted += static_cast<int>(generated.size()) - 1;
        for (int t : generated) {
            if (static_cast<int>(result.tokens.size()) < token_count) {
                result.tokens.push_back(t);
            }
        }
        position += static_cast<int>(generated.size());
        token = generated.back();
    }
    cuda_sync();
    result.seconds = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    return result;
}

// Emit the prefill top-k for one fixture. Same schema as
// tests/debug_prefill_topk.py so a first-token mismatch can be read as a
// near-tie or a real divergence rather than inferred from the argmax alone.
void emit_prefill_topk(PersistentEngine& engine, const Fixture& fixture,
                       const SamplingParams& sp) {
    engine.worker_command_reset();
    engine.reset_session();
    engine.worker_command_prefill(fixture.prompt);
    (void)engine.prefill(fixture.prompt, sp);
    const std::vector<int>& tokens = engine.last_topk_tokens();
    const std::vector<float>& values = engine.last_topk_logits();
    if (tokens.empty()) return;
    std::cout << "TOPK_JSON {\"runtime\":\"cpp\",\"case\":\"" << fixture.name
              << "\",\"prompt_tokens\":" << fixture.prompt.size()
              << ",\"top_tokens\":" << json_tokens(tokens) << ",\"top_logits\":[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << "],\"margin\":";
    if (values.size() > 1) {
        std::cout << (values[0] - values[1]);
    } else {
        std::cout << "null";
    }
    std::cout << "}\n" << std::flush;
}

int first_mismatch(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t common = std::min(a.size(), b.size());
    for (size_t i = 0; i < common; ++i) {
        if (a[i] != b[i]) return static_cast<int>(i);
    }
    return a.size() == b.size() ? -1 : static_cast<int>(common);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> <fixture.tsv> [tokens=32] [repeats=3] [layers=43]"
                     " [tp_world=1] [tp_rank=0] [nccl_id_path]\n";
        return 2;
    }
    try {
        const std::string ckpt_dir = argv[1];
        std::cerr << "[bench_dspark_decode] loading fixtures from " << argv[2] << std::endl;
        const std::vector<Fixture> fixtures = load_fixtures(argv[2]);
        std::cerr << "[bench_dspark_decode] loaded " << fixtures.size() << " fixtures" << std::endl;
        const int token_count = argc > 3 ? std::atoi(argv[3]) : 32;
        const int repeats = argc > 4 ? std::atoi(argv[4]) : 3;
        const int layer_count = argc > 5 ? std::atoi(argv[5]) : 43;
        const std::string mode = std::getenv("POCKETLLM_BENCH_MODE") != nullptr
                                     ? std::getenv("POCKETLLM_BENCH_MODE")
                                     : "spec";
        if (mode != "plain_nodspark" && mode != "plain_dspark" && mode != "spec" &&
            mode != "dspark_suite") {
            throw std::runtime_error(
                "POCKETLLM_BENCH_MODE must be plain_nodspark, plain_dspark, spec, or dspark_suite");
        }
        const bool use_dspark = mode != "plain_nodspark";
        const bool use_speculative = mode == "spec";
        const bool batched_verify =
            std::getenv("POCKETLLM_CPP_BATCHED_VERIFY") != nullptr &&
            std::atoi(std::getenv("POCKETLLM_CPP_BATCHED_VERIFY")) != 0;
        const std::string path = use_speculative
                                     ? (batched_verify ? "spec_batched" : "spec_sequential")
                                     : mode;

        ForwardSmokeOptions opts;
        opts.tp_world = argc > 6 ? std::atoi(argv[6]) : 1;
        opts.tp_rank = argc > 7 ? std::atoi(argv[7]) : 0;
        opts.device = opts.tp_rank;
        if (argc > 8) opts.nccl_id_path = argv[8];
        std::cerr << "[bench_dspark_decode] rank " << opts.tp_rank << " setting device " << opts.device << std::endl;
        if (token_count <= 0 || repeats <= 0) throw std::runtime_error("tokens/repeats must be positive");
        if (opts.tp_world > 1 && opts.nccl_id_path.empty()) {
            throw std::runtime_error("tp_world > 1 needs nccl_id_path");
        }
        if (cudaSetDevice(opts.device) != cudaSuccess) {
            throw std::runtime_error("failed to set CUDA device " + std::to_string(opts.device));
        }
        std::cerr << "[bench_dspark_decode] rank " << opts.tp_rank << " device set OK, loading checkpoint" << std::endl;

        int max_prompt = 0;
        for (const Fixture& f : fixtures) {
            max_prompt = std::max(max_prompt, static_cast<int>(f.prompt.size()));
        }
        PersistentEngine engine(ckpt_dir, opts, layer_count,
                                std::max(2048, max_prompt + token_count + 16));
        if (use_dspark) engine.load_dspark(ckpt_dir);
        engine.warmup_tp();
        if (opts.tp_rank != 0) {
            engine.run_worker_loop();
            return 0;
        }

        SamplingParams sp;
        sp.greedy = true;
        sp.temperature = 1.0f;
        sp.seed = 12345;

        std::cout << std::setprecision(9);
        auto run_path = [&](const std::string& result_path, bool speculative) {
            // Warm the exact path before reporting it. Each measured repeat resets
            // and prefills the engine, but model load remains shared by the suite.
            if (speculative) {
                (void)run_speculative(engine, fixtures.front(), 1, sp);
            } else {
                (void)run_plain(engine, fixtures.front(), 1, sp);
            }
            for (const Fixture& fixture : fixtures) {
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    const DecodeResult result = speculative
                                                    ? run_speculative(engine, fixture, token_count, sp)
                                                    : run_plain(engine, fixture, token_count, sp);
                    const double tps = token_count / result.seconds;
                    std::cout << "RESULT_JSON {\"runtime\":\"cpp\",\"path\":\""
                              << result_path << "\",\"case\":\"" << fixture.name
                              << "\",\"repeat\":" << repeat
                              << ",\"prompt_tokens\":" << fixture.prompt.size()
                              << ",\"decode_tokens\":" << token_count
                              << ",\"seconds\":" << result.seconds
                              << ",\"tps\":" << tps
                              << ",\"rounds\":" << result.rounds
                              << ",\"accepted_tokens\":" << result.accepted
                              << ",\"accepted_per_round\":"
                              << static_cast<double>(result.accepted) / result.rounds
                              << ",\"tokens\":" << json_tokens(result.tokens) << "}\n"
                              << std::flush;
                }
            }
        };

        if (std::getenv("POCKETLLM_CPP_TOPK_DIAG") != nullptr &&
            std::atoi(std::getenv("POCKETLLM_CPP_TOPK_DIAG")) > 0) {
            for (const Fixture& fixture : fixtures) emit_prefill_topk(engine, fixture, sp);
        }

        if (mode == "dspark_suite") {
            run_path("plain_dspark", false);
            setenv("POCKETLLM_CPP_BATCHED_VERIFY", "0", 1);
            run_path("spec_sequential", true);
            setenv("POCKETLLM_CPP_BATCHED_VERIFY", "1", 1);
            run_path("spec_batched", true);
        } else {
            run_path(path, use_speculative);
        }
        engine.worker_command_shutdown();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "bench_dspark_decode: " << e.what() << "\n";
        return 1;
    }
}
