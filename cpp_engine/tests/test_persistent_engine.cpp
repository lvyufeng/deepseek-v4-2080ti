#include "persistent_engine.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: test_persistent_engine <ckpt_dir> [layer_count] [max_new_tokens]\n";
            return 2;
        }
        const std::string ckpt = argv[1];
        const int layer_count = argc >= 3 ? std::stoi(argv[2]) : 43;
        const int max_new = argc >= 4 ? std::stoi(argv[3]) : 4;

        cudaSetDevice(0);

        pocket::ForwardSmokeOptions opts;
        opts.tp_world = 1;
        opts.tp_rank = 0;
        opts.device = 0;

        std::vector<int> prompt = {0, 17665, 31114, 12, 526, 318, 264, 4017, 30};

        pocket::PersistentEngine engine(ckpt, opts, layer_count, /*max_context=*/2048);

        auto run_once = [&](const std::vector<int>& seed, std::vector<int>& out) {
            engine.reset_session();
            pocket::SamplingParams sp;
            sp.greedy = true;
            int tok = engine.prefill(seed, sp);
            out.clear();
            out.push_back(tok);
            int position = static_cast<int>(seed.size());
            for (int step = 1; step < max_new; ++step) {
                tok = engine.decode_step(tok, position + step - 1, sp);
                out.push_back(tok);
            }
        };

        std::vector<int> a, b;
        run_once(prompt, a);
        run_once(prompt, b);

        std::cout << "round1:";
        for (int t : a) std::cout << " " << t;
        std::cout << "\nround2:";
        for (int t : b) std::cout << " " << t;
        std::cout << "\n";

        if (a != b) {
            std::cerr << "[FAIL] decode differs between sessions (reset is leaking state)\n";
            return 1;
        }
        if (a.empty()) {
            std::cerr << "[FAIL] no tokens generated\n";
            return 1;
        }
        std::cout << "[PASS] persistent_engine round-trip stable across reset_session\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
