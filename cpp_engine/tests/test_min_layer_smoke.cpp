#include "deepseek_v4_engine.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: test_min_layer_smoke <ckpt_dir>\n";
            return 2;
        }
        pocket::ForwardSmokeResult result = pocket::run_safetensors_min_layer_smoke(argv[1]);
        std::cout << "[PASS] min_layer_smoke token=" << result.token
                  << " dim=" << result.dim
                  << " inter=" << result.inter
                  << " logits=" << result.logits
                  << " top_token=" << result.top_token
                  << " top_logit=" << result.top_logit
                  << " checksum=" << result.checksum << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
