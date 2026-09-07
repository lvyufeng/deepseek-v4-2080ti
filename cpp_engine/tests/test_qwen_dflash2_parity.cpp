// Runs the real native DFlash2 draft from target taps captured by QwenEngine and
// writes every opt-in intermediate stage to a versioned tensor file. The Python
// exporter reads the capture and produces matching BF16 and SM75-emulated
// references; compare_qwen_dflash2_parity.py locates the first divergent stage.

#include "qwen_config.hpp"
#include "qwen_dflash2.hpp"
#include "qwen_engine.hpp"
#include "qwen_weights.hpp"
#include "safetensors_reader.hpp"

#include "qwen_dflash2_tensor_file.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<int> read_tokens(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open token fixture: " + path);
    std::vector<int> tokens;
    std::string text((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
    for (char& character : text) {
        if (character == ',') character = ' ';
    }
    std::stringstream stream(text);
    int token = 0;
    while (stream >> token) tokens.push_back(token);
    if (tokens.empty()) throw std::runtime_error("token fixture is empty");
    return tokens;
}

void append_unique(pocket_test::DFlash2TensorFile& file,
                   const pocket::QwenDFlash2DebugTensor& tensor) {
    for (const auto& existing : file.tensors) {
        if (existing.name == tensor.name) {
            throw std::runtime_error("duplicate debug stage: " + tensor.name);
        }
    }
    file.tensors.push_back(tensor);
}

std::vector<uint16_t> target_taps_from(
    const pocket::QwenDFlash2DebugTensor& tensor) {
    if (tensor.dtype != pocket::QwenDFlash2DebugDType::F16 ||
        tensor.shape.size() != 2 || tensor.bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("target_taps has invalid dtype or shape");
    }
    std::vector<uint16_t> taps(tensor.bytes.size() / sizeof(uint16_t));
    std::memcpy(taps.data(), tensor.bytes.data(), tensor.bytes.size());
    return taps;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6 || argc > 10) {
        std::cerr << "usage: test_qwen_dflash2_parity <target_ckpt> <draft_ckpt> "
                     "<tokens.txt> <capture.qdf2> <output.qdf2> "
                     "[tp_rank] [tp_world] [device] [nccl_id_path]\n";
        return 2;
    }
    try {
        const std::string target_checkpoint = argv[1];
        const std::string draft_checkpoint = argv[2];
        const std::vector<int> tokens = read_tokens(argv[3]);
        const std::string capture_path = argv[4];
        const std::string output_path = argv[5];
        const int tp_rank = argc > 6 ? std::atoi(argv[6]) : 0;
        const int tp_world = argc > 7 ? std::atoi(argv[7]) : 1;
        const int device = argc > 8 ? std::atoi(argv[8]) : tp_rank;
        const std::string nccl_path = argc > 9 ? argv[9] : std::string();
        if (tp_world <= 0 || tp_rank < 0 || tp_rank >= tp_world || device < 0) {
            throw std::runtime_error("invalid TP parity geometry");
        }
        if (cudaSetDevice(device) != cudaSuccess) {
            throw std::runtime_error("cannot select CUDA device");
        }

        const pocket::QwenDFlash2Config draft_config =
            pocket::QwenDFlash2Config::from_directory(draft_checkpoint);
        const pocket::QwenConfig target_config =
            pocket::QwenConfig::from_hf_config(target_checkpoint);
        draft_config.validate_for_target(target_config.hidden_size,
                                         target_config.vocab_size,
                                         target_config.num_hidden_layers);

        pocket::QwenEngineOptions options;
        options.tp_world = tp_world;
        options.tp_rank = tp_rank;
        options.device = device;
        options.prefill_chunk_tokens = static_cast<int>(tokens.size());
        options.prefix_cache = false;
        options.nccl_id_path = nccl_path;
        pocket_test::DFlash2TensorFile capture;
        capture.position_offset = 0;
        capture.anchor_token = -1;
        {
            auto target = std::make_unique<pocket::QwenEngine>(
                target_checkpoint, options, 0,
                static_cast<int>(tokens.size()) + draft_config.block_size + 1);
            target->warmup_tp();
            const pocket::ForwardResult target_result =
                target->debug_prefill_dflash2(
                    tokens, draft_config.target_layer_ids,
                    [&](const auto& tensor) {
                        if (tensor.name == "target_taps") append_unique(capture, tensor);
                    });
            capture.anchor_token = target_result.top_token;
            pocket_test::write_tensor_file(capture_path, capture);
        }

        const auto& captured_taps =
            pocket_test::require_tensor(capture, "target_taps");
        const std::vector<uint16_t> taps = target_taps_from(captured_taps);
        const int context_rows = static_cast<int>(captured_taps.shape[0]);

        // The 64-layer target is gone before loading the replicated 3.85 GB
        // drafter, keeping TP=1 parity usable on a 22 GiB card as well.
        const pocket::SafeTensorsIndex target_index(target_checkpoint);
        const pocket::QwenWeightMap target_weights(
            target_index, target_config, tp_world, tp_rank);
        pocket::QwenDeviceTensor embedding = pocket::qwen_upload_tensor(
            target_index, target_weights.embed_tokens());
        const pocket::QwenLinearRef& head_ref = target_weights.lm_head();
        pocket::QwenDeviceTensor lm_head = pocket::qwen_upload_tensor(
            target_index, head_ref.weight);
        pocket::QwenDeviceTensor lm_head_scale;
        if (head_ref.has_scale) {
            lm_head_scale = pocket::qwen_upload_tensor(
                target_index, head_ref.scale);
        }
        const pocket::QwenTargetHeadAdapter target_head{
            head_ref.kind, &lm_head,
            head_ref.has_scale ? &lm_head_scale : nullptr,
            static_cast<int>(head_ref.logical_local_shape.at(0)),
            static_cast<int>(head_ref.logical_local_shape.at(1)),
            static_cast<uint64_t>(tp_rank) * target_config.vocab_size /
                tp_world};
        const pocket::SafeTensorsIndex draft_index =
            pocket::SafeTensorsIndex::from_single_file(draft_checkpoint);
        const pocket::QwenDFlash2WeightMap draft_weights(
            draft_index, draft_config, tp_world, tp_rank);
        pocket::QwenDFlash2Runtime draft(
            draft_checkpoint, draft_config, draft_weights, embedding,
            target_head, tp_world, tp_rank, device, nccl_path,
            context_rows + draft_config.block_size + 1);

        pocket_test::DFlash2TensorFile output;
        output.position_offset = 0;
        output.anchor_token = capture.anchor_token;
        draft.set_debug_callback([&](const auto& tensor) {
            append_unique(output, tensor);
        });
        const pocket::QwenDFlash2Proposal proposal = draft.debug_propose_from_host(
            taps, context_rows, 0, capture.anchor_token);
        pocket_test::write_tensor_file(output_path, output);

        std::cout << "[PASS] qwen_dflash2_parity rank=" << tp_rank
                  << " context_rows=" << context_rows
                  << " anchor=" << capture.anchor_token << " drafted=";
        for (size_t index = 0; index < proposal.tokens.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << proposal.tokens[index];
        }
        std::cout << " stages=" << output.tensors.size() << "\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[FAIL] " << exception.what() << "\n";
        return 1;
    }
}
