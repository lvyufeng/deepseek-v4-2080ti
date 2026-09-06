#include "model_config.hpp"
#include "safetensors_model.hpp"
#include "safetensors_reader.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_safetensors_model <ckpt_dir>\n";
        return 2;
    }
    try {
        std::string ckpt = argv[1];
        pocket::SafeTensorsIndex index(ckpt);
        pocket::ModelConfig config = pocket::ModelConfig::from_hf_config(ckpt);
        pocket::SafeTensorsModelMap map(index, config);

        require(map.embed().name == "embed.weight", "bad embed tensor");
        require(map.embed().shape == std::vector<uint64_t>({129280, 4096}), "bad embed shape");
        require(map.layers().size() == 43, "bad layer count");
        require(map.layers()[0].ffn.experts.size() == 256, "bad expert count");
        require(map.layers()[0].attn.wq_a.shape == std::vector<uint64_t>({1024, 4096}), "bad wq_a shape");
        require(map.layers()[0].attn.wkv.shape == std::vector<uint64_t>({512, 4096}), "bad wkv shape");
        require(map.layers()[0].attn.attn_sink.dtype == pocket::SafeDType::F32, "bad attn_sink dtype");
        require(map.layers()[0].ffn.gate_weight.shape == std::vector<uint64_t>({256, 4096}), "bad gate weight shape");
        require(!map.layers()[0].ffn.gate_tid2eid.name.empty(), "missing layer 0 tid2eid");
        require(map.layers()[3].ffn.gate_tid2eid.name.empty(), "unexpected layer 3 tid2eid");
        const auto& w1 = map.layers()[0].ffn.experts[0].w1;
        require(w1.weight_name == "layers.0.ffn.experts.0.w1.weight", "bad expert w1 name");
        require(w1.rows == 2048 && w1.cols == 4096 && w1.scale_cols == 128, "bad expert w1 shape");
        const auto& last = map.layers()[42].ffn.experts[255].w3;
        require(last.rows == 2048 && last.cols == 4096 && last.scale_cols == 128, "bad last expert shape");

        std::cout << "[PASS] safetensors_model layers=" << map.layers().size()
                  << " experts=" << map.layers()[0].ffn.experts.size()
                  << " embed_shape=" << map.embed().shape[0] << "x" << map.embed().shape[1] << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
