// Batched decode throughput benchmark: measure concurrent request throughput
// against the baseline sequential processing and the theoretical 3.88× ceiling.
//
// Usage: ./bench_qwen_batched_throughput <checkpoint_dir> [--batch-size N]
//
// Spawns N concurrent requests at various context lengths and measures:
// - Sequential processing baseline (one at a time, for comparison)
// - Batched processing throughput (requests/second)
// - GPU utilization (batched kernel amortization)

#include "qwen_config.hpp"
#include "qwen_engine.hpp"
#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

struct BenchOptions {
    std::string checkpoint_dir;
    int batch_size = 4;
    int warmup_steps = 5;
    int measured_steps = 20;
    int context_len = 4096;
    int decode_tokens = 32;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;  // -1 means "use tp_rank"
    std::string nccl_id_path = "/tmp/qwen_bench_nccl_id";
};

void usage(const char* prog) {
    std::cout << "Usage: " << prog << " <checkpoint_dir> [options]\n"
              << "Options:\n"
              << "  --batch-size N     Number of concurrent requests (default 4)\n"
              << "  --context N        Initial context length (default 4096)\n"
              << "  --decode N         Tokens to decode per request (default 32)\n"
              << "  --warmup N         Warmup steps (default 5)\n"
              << "  --measured N       Measured steps (default 20)\n"
              << "  --tp-world N       TP world size (default 1)\n"
              << "  --tp-rank N        TP rank (default 0)\n"
              << "  --device N         CUDA device ordinal (default: same as tp-rank)\n"
              << "  --nccl-id PATH     NCCL ID file path (default /tmp/qwen_bench_nccl_id)\n";
}

BenchOptions parse_args(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        std::exit(1);
    }
    BenchOptions opts;
    opts.checkpoint_dir = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--batch-size" && i + 1 < argc) {
            opts.batch_size = std::atoi(argv[++i]);
        } else if (arg == "--context" && i + 1 < argc) {
            opts.context_len = std::atoi(argv[++i]);
        } else if (arg == "--decode" && i + 1 < argc) {
            opts.decode_tokens = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            opts.warmup_steps = std::atoi(argv[++i]);
        } else if (arg == "--measured" && i + 1 < argc) {
            opts.measured_steps = std::atoi(argv[++i]);
        } else if (arg == "--tp-world" && i + 1 < argc) {
            opts.tp_world = std::atoi(argv[++i]);
        } else if (arg == "--tp-rank" && i + 1 < argc) {
            opts.tp_rank = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            opts.device = std::atoi(argv[++i]);
        } else if (arg == "--nccl-id" && i + 1 < argc) {
            opts.nccl_id_path = argv[++i];
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            usage(argv[0]);
            std::exit(1);
        }
    }
    return opts;
}

struct Request {
    uint64_t request_id;
    int slot_id;
    int context_len;
    int tokens_decoded;
    int target_tokens;
    std::vector<int> prompt;
    std::vector<int> outputs;
    bool finished;
};

double bench_sequential_baseline(pocket::QwenEngine& engine,
                                 const std::vector<Request>& requests,
                                 int measured_steps) {
    engine.reset();
    std::vector<Request> active = requests;

    // Prefill all requests
    for (auto& req : active) {
        engine.prefill(req.prompt, req.slot_id);
    }

    // Warmup
    for (int step = 0; step < 5; ++step) {
        for (auto& req : active) {
            if (!req.finished) {
                const int token = (step * 7 + req.request_id) % 64;
                engine.decode_step(token, req.slot_id);
            }
        }
    }

    // Measure sequential decode: one request at a time
    const auto t0 = Clock::now();
    for (int step = 0; step < measured_steps; ++step) {
        for (auto& req : active) {
            if (!req.finished) {
                const int token = (step * 7 + req.request_id) % 64;
                const auto result = engine.decode_step(token, req.slot_id);
                req.outputs.push_back(result.top_token);
                req.tokens_decoded++;
                if (req.tokens_decoded >= req.target_tokens) {
                    req.finished = true;
                }
            }
        }
    }
    const auto t1 = Clock::now();
    const double elapsed = Seconds(t1 - t0).count();
    const int total_steps = static_cast<int>(active.size()) * measured_steps;
    return elapsed / total_steps;
}

double bench_batched_throughput(pocket::QwenEngine& engine,
                                const std::vector<Request>& requests,
                                int measured_steps) {
    engine.reset();
    std::vector<Request> active = requests;

    // Prefill all requests
    for (auto& req : active) {
        engine.prefill(req.prompt, req.slot_id);
    }

    // Warmup
    std::vector<int> batch_tokens(active.size());
    std::vector<int> batch_slots(active.size());
    for (int step = 0; step < 5; ++step) {
        for (size_t i = 0; i < active.size(); ++i) {
            batch_tokens[i] = (step * 7 + active[i].request_id) % 64;
            batch_slots[i] = active[i].slot_id;
        }
        engine.batch_decode_tokens(batch_tokens, batch_slots);
    }

    // Measure batched decode: all requests in one forward
    const auto t0 = Clock::now();
    for (int step = 0; step < measured_steps; ++step) {
        for (size_t i = 0; i < active.size(); ++i) {
            batch_tokens[i] = (step * 7 + active[i].request_id) % 64;
            batch_slots[i] = active[i].slot_id;
        }
        const auto results = engine.batch_decode_tokens(batch_tokens, batch_slots);
        for (size_t i = 0; i < active.size(); ++i) {
            active[i].outputs.push_back(results[i].top_token);
            active[i].tokens_decoded++;
            if (active[i].tokens_decoded >= active[i].target_tokens) {
                active[i].finished = true;
            }
        }
    }
    const auto t1 = Clock::now();
    const double elapsed = Seconds(t1 - t0).count();
    const int total_steps = static_cast<int>(active.size()) * measured_steps;
    return elapsed / total_steps;
}

}  // namespace

int main(int argc, char** argv) {
    if (!pocket::cuda_runtime_available()) {
        std::cerr << "CUDA runtime not available\n";
        return 1;
    }

    try {
        const BenchOptions opts = parse_args(argc, argv);

        std::cout << "Qwen batched decode throughput benchmark\n"
                  << "=========================================\n"
                  << "Checkpoint: " << opts.checkpoint_dir << "\n"
                  << "Batch size: " << opts.batch_size << "\n"
                  << "Context: " << opts.context_len << " tokens\n"
                  << "Decode: " << opts.decode_tokens << " tokens/request\n"
                  << "Measured steps: " << opts.measured_steps << "\n"
                  << "TP: " << opts.tp_world << " (rank " << opts.tp_rank << ")\n\n";

        // Create engine with batching enabled
        pocket::QwenEngineOptions engine_opts;
        engine_opts.tp_world = opts.tp_world;
        engine_opts.tp_rank = opts.tp_rank;
        engine_opts.device = opts.device >= 0 ? opts.device : opts.tp_rank;
        engine_opts.nccl_id_path = opts.nccl_id_path;
        engine_opts.kv_cache_dtype = pocket::QwenKvCacheDType::Fp16;
        engine_opts.prefix_cache = false;
        engine_opts.max_batch_size = opts.batch_size * 2;  // Extra headroom

        pocket::QwenEngine engine(opts.checkpoint_dir, engine_opts);
        engine.allocate_batch_slots(engine_opts.max_batch_size);

        // Create test requests at staggered context lengths
        std::vector<Request> requests;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> context_dist(
            opts.context_len - 512, opts.context_len + 512);

        for (int i = 0; i < opts.batch_size; ++i) {
            Request req;
            req.request_id = static_cast<uint64_t>(i);
            req.slot_id = i;
            req.context_len = std::max(1024, context_dist(rng));
            req.tokens_decoded = 0;
            req.target_tokens = opts.decode_tokens;
            req.finished = false;
            req.prompt.resize(static_cast<size_t>(req.context_len));
            for (int j = 0; j < req.context_len; ++j) {
                req.prompt[static_cast<size_t>(j)] = (i * 7 + j) % 64;
            }
            requests.push_back(req);
        }

        // Benchmark sequential baseline
        std::cout << "Sequential baseline (one request at a time)...\n";
        const double seq_per_step = bench_sequential_baseline(
            engine, requests, opts.measured_steps);
        const double seq_req_per_sec = 1.0 / seq_per_step;
        std::cout << "  " << std::fixed << std::setprecision(3)
                  << seq_per_step * 1000.0 << " ms/step\n"
                  << "  " << std::setprecision(2) << seq_req_per_sec
                  << " requests/sec (effective)\n\n";

        // Benchmark batched path
        std::cout << "Batched decode (all requests in one forward)...\n";
        const double batch_per_step = bench_batched_throughput(
            engine, requests, opts.measured_steps);
        const double batch_req_per_sec = 1.0 / batch_per_step;
        std::cout << "  " << std::fixed << std::setprecision(3)
                  << batch_per_step * 1000.0 << " ms/step\n"
                  << "  " << std::setprecision(2) << batch_req_per_sec
                  << " requests/sec (effective)\n\n";

        const double speedup = seq_per_step / batch_per_step;
        std::cout << "Summary\n"
                  << "-------\n"
                  << "Speedup: " << std::setprecision(2) << speedup << "×\n"
                  << "Throughput improvement: " << std::setprecision(1)
                  << (speedup - 1.0) * 100.0 << "%\n"
                  << "Effective concurrent requests/sec: "
                  << std::setprecision(2) << batch_req_per_sec << "\n";

        if (speedup < 1.5) {
            std::cout << "\n⚠️  WARNING: Speedup below 1.5× suggests batching "
                      << "overhead or kernel fallback.\n";
        } else if (speedup >= 3.0) {
            std::cout << "\n✓ Speedup >= 3.0× approaching theoretical ceiling "
                      << "(measured 3.88× on GEMM).\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}