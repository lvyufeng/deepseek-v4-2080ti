// Phase 3 step: end-to-end greedy decode smoke. Loads the model once and runs
// a per-step forward over [seed_tokens] + [max_new_tokens] positions, with
// KV cache sized to the full sequence. Validates that:
//   - the run completes without OOM/throw,
//   - each step produces a valid in-range token,
//   - per-step wall time is reported for a coarse decode-tps signal.
//
// TP mode (optional, all ranks must agree):
//   --tp-world W --tp-rank R --nccl-id-path PATH [--device D]
//
// Positional args are: <model.gguf> [max_new_tokens] [seed1 seed2 ...]

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void append_seed_file(const std::string& path, std::vector<int>& seeds) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open --seed-file: " + path);
    long long value = 0;
    while (in >> value) {
        if (value < 0 || value > 2147483647LL)
            throw std::runtime_error("seed token out of int range in --seed-file");
        seeds.push_back(static_cast<int>(value));
        if (in.peek() == ',') in.get();
    }
    if (!in.eof()) throw std::runtime_error("failed to parse integer token ids from --seed-file: " + path);
}

}  // namespace

int main(int argc, char** argv) {
    pocket::ForwardSmokeOptions opts;
    std::string ckpt;
    int max_new = 4;
    std::vector<int> seeds;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (a == "--tp-world") opts.tp_world = std::stoi(need("--tp-world"));
        else if (a == "--tp-rank") opts.tp_rank = std::stoi(need("--tp-rank"));
        else if (a == "--device") opts.device = std::stoi(need("--device"));
        else if (a == "--nccl-id-path") opts.nccl_id_path = need("--nccl-id-path");
        else if (a == "--seed-file") append_seed_file(need("--seed-file"), seeds);
        else positional.push_back(a);
    }
    if (positional.empty()) {
        std::cerr << "usage: test_gguf_generate <model.gguf> [max_new_tokens] [seed1 seed2 ...] "
                     "[--seed-file PATH] [--tp-world W --tp-rank R --nccl-id-path PATH --device D]\n";
        return 2;
    }
    ckpt = positional[0];
    if (positional.size() >= 2) max_new = std::atoi(positional[1].c_str());
    for (size_t i = 2; i < positional.size(); ++i) seeds.push_back(std::atoi(positional[i].c_str()));
    if (seeds.empty()) seeds = {1234};

    try {
        auto r = pocket::run_gguf_generate_smoke(ckpt, seeds, max_new, opts);
        const bool is_root = (opts.tp_rank == 0);
        if (is_root) {
            auto print_int_vector = [](const char* label, const std::vector<int>& values) {
                std::printf("%s = [", label);
                const size_t n = values.size();
                if (n <= 32) {
                    for (size_t i = 0; i < n; ++i) std::printf("%s%d", i ? ", " : "", values[i]);
                } else {
                    for (size_t i = 0; i < 16; ++i) std::printf("%s%d", i ? ", " : "", values[i]);
                    std::printf(", ...");
                    for (size_t i = n - 16; i < n; ++i) std::printf(", %d", values[i]);
                }
                std::printf("] (count=%zu)\n", n);
            };
            auto print_float_vector = [](const char* label, const std::vector<float>& values) {
                std::printf("%s = ", label);
                const size_t n = values.size();
                if (n <= 32) {
                    for (size_t i = 0; i < n; ++i) std::printf("%s%.3f", i ? ", " : "", values[i]);
                } else {
                    for (size_t i = 0; i < 16; ++i) std::printf("%s%.3f", i ? ", " : "", values[i]);
                    std::printf(", ...");
                    for (size_t i = n - 16; i < n; ++i) std::printf(", %.3f", values[i]);
                }
                std::printf(" (count=%zu)\n", n);
            };
            std::printf("n_layers=%d dim=%d vocab=%d prompt_tokens=%d decode_tokens=%d\n",
                        r.n_layers, r.dim, r.vocab, r.prompt_tokens, r.decode_tokens);
            std::printf("load_seconds   = %.3f\n", r.load_seconds);
            std::printf("forward_seconds= %.3f (%d positions; %.1f ms/step)\n",
                        r.forward_seconds,
                        static_cast<int>(r.top_logits.size()),
                        r.top_logits.empty() ? 0.0
                                             : 1000.0 * r.forward_seconds /
                                                   static_cast<double>(r.top_logits.size()));
            if (r.prefill_seconds > 0.0) {
                std::printf("prefill_seconds= %.3f (%d tokens; %.2f tok/s)\n",
                            r.prefill_seconds,
                            r.prompt_tokens,
                            static_cast<double>(r.prompt_tokens) / r.prefill_seconds);
            }
            if (r.decode_tokens > 0) {
                const double decode_den = r.decode_seconds > 0.0 ? r.decode_seconds : r.forward_seconds;
                const double tps = static_cast<double>(r.decode_tokens) / decode_den;
                std::printf("decode_seconds = %.3f (%d tokens; %.2f tok/s; %.1f ms/token)\n",
                            r.decode_seconds,
                            r.decode_tokens,
                            tps,
                            1000.0 * decode_den / static_cast<double>(r.decode_tokens));
            }
            print_int_vector("seeds", seeds);
            print_int_vector("generated", r.generated_tokens);
            print_float_vector("top_logits[first..last]", r.top_logits);
        }

        if (static_cast<int>(r.generated_tokens.size()) != max_new) {
            std::cerr << "[FAIL] generated_tokens size mismatch\n"; return 1;
        }
        for (int t : r.generated_tokens) {
            if (t < 0 || t >= r.vocab) {
                std::cerr << "[FAIL] generated token out of vocab\n"; return 1;
            }
        }
        if (is_root) std::cout << "[PASS] gguf greedy decode smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
