#include "cmd_channel.hpp"
#include "cuda_ops.hpp"
#include "device_runtime.hpp"
#include "deepseek_v4_engine.hpp"
#include "model_config.hpp"
#include "openai_server.hpp"
#include "persistent_engine.hpp"
#include "qwen_config.hpp"
#include "qwen_engine.hpp"
#include "python_sidecar.hpp"
#include "safetensors_reader.hpp"
#include "tokenizer.hpp"
#include "tp_comm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {

struct Args {
    std::string model;
    std::string ckpt;
    std::string inspect_tensor;
    std::string prompt;
    std::string token_ids_csv;
    std::string token_ids_file;
    int smoke_layers = 1;
    int forward_token = -1;
    int position = 0;
    int max_new_tokens = 1;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;
    std::string nccl_id_path;
    bool generate_token = false;
    bool resident_bench = false;
    bool qwen_audit = false;
    bool qwen_audit_strict = false;
    bool dump_config = false;
    bool inspect = false;
    bool smoke_forward = false;
    bool qwen_persistent_stdin = false;
    bool use_persistent = false;
    int max_context = 0;
    // 0 means "leave the engine's own default alone" rather than duplicating the
    // constant here, so raising QwenEngineOptions::prefill_chunk_tokens does not
    // need a matching edit in the argument parser.
    int prefill_chunk_tokens = 0;
    std::string kv_cache_dtype = "fp16";
    int qwen_attention_window = 0;
    int qwen_attention_sink_tokens = 0;
    bool qwen_prefix_cache = true;
    int qwen_snapshot_interval = 4096;
    int qwen_max_snapshots = 82;
    bool qwen_mtp = false;
    int qwen_mtp_tokens = 1;
    // Greedy by default so existing results stay reproducible.
    float qwen_temperature = 0.0f;
    float qwen_top_p = 1.0f;
    int qwen_top_k = 20;
    unsigned long long qwen_seed = 0;
    bool qwen_mtp_adaptive = false;
    std::string qwen_dspark_checkpoint;
    std::string qwen_dflash2_checkpoint;
    bool serve = false;
    int port = 8000;
    std::string host = "0.0.0.0";
    std::string python_bin = "python";
    std::string sidecar_script;
};

bool path_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

// True when every requested action only reads checkpoint metadata on the host.
// Those modes must not bind an accelerator: auditing a 52 GiB checkpoint has to
// work on a machine with no vendor runtime, and with --tp-world 4 on a host that
// does not have four visible devices.
bool is_host_only_mode(const Args& args) {
    return !args.serve && !args.smoke_forward && !args.generate_token &&
           !args.resident_bench && !args.use_persistent &&
           !args.qwen_persistent_stdin;
}

bool is_dir(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--ckpt" && i + 1 < argc) {
            args.ckpt = argv[++i];
        } else if (arg == "--inspect-tensor" && i + 1 < argc) {
            args.inspect_tensor = argv[++i];
        } else if (arg == "--qwen-audit") {
            args.qwen_audit = true;
        } else if (arg == "--qwen-audit-strict") {
            args.qwen_audit = true;
            args.qwen_audit_strict = true;
        } else if (arg == "--dump-config") {
            args.dump_config = true;
        } else if (arg == "--inspect") {
            args.inspect = true;
        } else if (arg == "--smoke-forward") {
            args.smoke_forward = true;
        } else if (arg == "--qwen-persistent-stdin") {
            args.smoke_forward = true;
            args.qwen_persistent_stdin = true;
        } else if (arg == "--smoke-layers" && i + 1 < argc) {
            args.smoke_forward = true;
            args.smoke_layers = std::stoi(argv[++i]);
        } else if (arg == "--forward-token" && i + 1 < argc) {
            args.smoke_forward = true;
            args.forward_token = std::stoi(argv[++i]);
        } else if (arg == "--position" && i + 1 < argc) {
            args.position = std::stoi(argv[++i]);
        } else if (arg == "--generate-token" && i + 1 < argc) {
            args.smoke_forward = true;
            args.generate_token = true;
            args.forward_token = std::stoi(argv[++i]);
        } else if (arg == "--resident-bench") {
            args.smoke_forward = true;
            args.generate_token = true;
            args.resident_bench = true;
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.smoke_forward = true;
            args.prompt = argv[++i];
        } else if (arg == "--token-ids" && i + 1 < argc) {
            args.smoke_forward = true;
            args.token_ids_csv = argv[++i];
        } else if (arg == "--token-ids-file" && i + 1 < argc) {
            args.smoke_forward = true;
            args.token_ids_file = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            ++i;
        } else if (arg == "--max-new-tokens" && i + 1 < argc) {
            args.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--tp-world" && i + 1 < argc) {
            args.tp_world = std::stoi(argv[++i]);
        } else if (arg == "--tp-rank" && i + 1 < argc) {
            args.tp_rank = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            args.device = std::stoi(argv[++i]);
        } else if (arg == "--nccl-id-path" && i + 1 < argc) {
            args.nccl_id_path = argv[++i];
        } else if (arg == "--use-persistent") {
            args.use_persistent = true;
        } else if (arg == "--max-context" && i + 1 < argc) {
            args.max_context = std::stoi(argv[++i]);
        } else if (arg == "--prefill-chunk-tokens" && i + 1 < argc) {
            args.prefill_chunk_tokens = std::stoi(argv[++i]);
            // An explicit value must still be positive. The 0 sentinel means
            // "unspecified", so it is only valid when the flag is absent.
            if (args.prefill_chunk_tokens <= 0) {
                throw std::runtime_error("--prefill-chunk-tokens must be positive");
            }
        } else if (arg == "--kv-cache-dtype" && i + 1 < argc) {
            args.kv_cache_dtype = argv[++i];
        } else if (arg == "--qwen-attention-window" && i + 1 < argc) {
            args.qwen_attention_window = std::stoi(argv[++i]);
        } else if (arg == "--qwen-attention-sink-tokens" && i + 1 < argc) {
            args.qwen_attention_sink_tokens = std::stoi(argv[++i]);
        } else if (arg == "--qwen-no-prefix-cache") {
            args.qwen_prefix_cache = false;
        } else if (arg == "--qwen-snapshot-interval" && i + 1 < argc) {
            args.qwen_snapshot_interval = std::stoi(argv[++i]);
        } else if (arg == "--qwen-max-snapshots" && i + 1 < argc) {
            args.qwen_max_snapshots = std::stoi(argv[++i]);
        } else if (arg == "--qwen-mtp") {
            args.qwen_mtp = true;
        } else if (arg == "--qwen-mtp-tokens" && i + 1 < argc) {
            args.qwen_mtp = true;
            args.qwen_mtp_tokens = std::stoi(argv[++i]);
        } else if (arg == "--qwen-mtp-adaptive") {
            args.qwen_mtp = true;
            args.qwen_mtp_adaptive = true;
        } else if (arg == "--qwen-temperature" && i + 1 < argc) {
            args.qwen_temperature = std::stof(argv[++i]);
        } else if (arg == "--qwen-top-p" && i + 1 < argc) {
            args.qwen_top_p = std::stof(argv[++i]);
        } else if (arg == "--qwen-top-k" && i + 1 < argc) {
            args.qwen_top_k = std::stoi(argv[++i]);
        } else if (arg == "--qwen-seed" && i + 1 < argc) {
            args.qwen_seed = std::stoull(argv[++i]);
        } else if (arg == "--qwen-dspark" && i + 1 < argc) {
            args.qwen_dspark_checkpoint = argv[++i];
        } else if (arg == "--qwen-dflash2" && i + 1 < argc) {
            args.qwen_dflash2_checkpoint = argv[++i];
        } else if (arg == "--serve") {
            args.serve = true;
        } else if (arg == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--python" && i + 1 < argc) {
            args.python_bin = argv[++i];
        } else if (arg == "--sidecar" && i + 1 < argc) {
            args.sidecar_script = argv[++i];
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (args.ckpt.empty() && !args.model.empty() && is_dir(args.model) && path_exists(args.model + "/model.safetensors.index.json")) {
        args.ckpt = args.model;
        args.model.clear();
    }
    if (args.tp_world <= 0) throw std::runtime_error("--tp-world must be positive");
    if (args.tp_rank < 0 || args.tp_rank >= args.tp_world) throw std::runtime_error("--tp-rank must be in [0, tp_world)");
    if (args.prefill_chunk_tokens < 0) throw std::runtime_error("--prefill-chunk-tokens must be positive");
    if (args.qwen_snapshot_interval < 0 || args.qwen_max_snapshots < 0) {
        throw std::runtime_error("Qwen snapshot settings must not be negative");
    }
    if (args.qwen_mtp_tokens <= 0) {
        throw std::runtime_error("--qwen-mtp-tokens must be positive");
    }
    const int external_drafter_count =
        (!args.qwen_dspark_checkpoint.empty() ? 1 : 0) +
        (!args.qwen_dflash2_checkpoint.empty() ? 1 : 0);
    if (args.qwen_mtp && external_drafter_count != 0) {
        throw std::runtime_error(
            "external Qwen drafters cannot be combined with native Qwen MTP");
    }
    if (external_drafter_count > 1) {
        throw std::runtime_error(
            "--qwen-dspark and --qwen-dflash2 are mutually exclusive");
    }
    if (!args.qwen_dspark_checkpoint.empty() &&
        (!is_dir(args.qwen_dspark_checkpoint) ||
         !path_exists(args.qwen_dspark_checkpoint + "/config.json") ||
         !path_exists(args.qwen_dspark_checkpoint + "/model.safetensors"))) {
        throw std::runtime_error(
            "--qwen-dspark must name a complete DSpark checkpoint directory");
    }
    if (!args.qwen_dflash2_checkpoint.empty() &&
        (!is_dir(args.qwen_dflash2_checkpoint) ||
         !path_exists(args.qwen_dflash2_checkpoint + "/config.json") ||
         !path_exists(args.qwen_dflash2_checkpoint + "/model.safetensors"))) {
        throw std::runtime_error(
            "--qwen-dflash2 must name a complete DFlash2 checkpoint directory");
    }
    if (args.kv_cache_dtype != "fp16" && args.kv_cache_dtype != "fp8" &&
        args.kv_cache_dtype != "turboquant_k8v4" &&
        args.kv_cache_dtype != "int8_per_token_head") {
        throw std::runtime_error("--kv-cache-dtype must be fp16, fp8, turboquant_k8v4, or int8_per_token_head");
    }
    if (args.qwen_attention_window < 0) {
        throw std::runtime_error("--qwen-attention-window must not be negative");
    }
    if (args.qwen_attention_sink_tokens < 0) {
        throw std::runtime_error("--qwen-attention-sink-tokens must not be negative");
    }
    if (args.qwen_attention_window == 0 && args.qwen_attention_sink_tokens != 0) {
        throw std::runtime_error(
            "--qwen-attention-sink-tokens requires --qwen-attention-window");
    }
    if (args.model.empty() && args.ckpt.empty()) {
        throw std::runtime_error("--model or --ckpt is required");
    }
    return args;
}


std::vector<int> parse_token_ids_csv(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stoi(item));
    }
    return out;
}

std::vector<int> load_token_ids(const Args& args) {
    if (!args.token_ids_csv.empty()) return parse_token_ids_csv(args.token_ids_csv);
    if (!args.token_ids_file.empty()) {
        std::ifstream in(args.token_ids_file);
        if (!in) throw std::runtime_error("failed to open token ids file: " + args.token_ids_file);
        std::stringstream buf;
        buf << in.rdbuf();
        return parse_token_ids_csv(buf.str());
    }
    return {};
}

constexpr int kQwenPersistentHeaderInts = 4;
enum class QwenPersistentCommand : int32_t {
    Prefill = 0,
    Decode = 1,
    Reset = 2,
    Shutdown = 3,
    Generate = 4,
};

void qwen_send_command(pocket::CmdChannel& channel, QwenPersistentCommand command,
                       int32_t arg0 = 0, int32_t arg1 = 0,
                       const std::vector<int>* payload = nullptr) {
    const int32_t count = payload == nullptr
        ? 0 : static_cast<int32_t>(payload->size());
    const int32_t header[kQwenPersistentHeaderInts] = {
        static_cast<int32_t>(command), arg0, arg1, count};
    channel.send_to_workers(header, kQwenPersistentHeaderInts);
    if (payload != nullptr && !payload->empty()) {
        std::vector<int32_t> packed(payload->begin(), payload->end());
        channel.send_to_workers(packed.data(), packed.size());
    }
}

void run_qwen_persistent_worker(pocket::QwenEngine& qwen,
                                pocket::CmdChannel& channel) {
    while (true) {
        int32_t header[kQwenPersistentHeaderInts] = {0, 0, 0, 0};
        channel.recv_from_root(header, kQwenPersistentHeaderInts);
        const auto command = static_cast<QwenPersistentCommand>(header[0]);
        if (command == QwenPersistentCommand::Shutdown) return;
        if (command == QwenPersistentCommand::Reset) {
            qwen.clear_prefix_cache();
            continue;
        }
        std::vector<int> payload;
        if (header[3] > 0) {
            std::vector<int32_t> packed(static_cast<size_t>(header[3]));
            channel.recv_from_root(packed.data(), packed.size());
            payload.assign(packed.begin(), packed.end());
        }
        if (command == QwenPersistentCommand::Prefill) {
            (void)qwen.prefill(payload);
        } else if (command == QwenPersistentCommand::Decode) {
            (void)qwen.decode_step(header[1]);
        } else if (command == QwenPersistentCommand::Generate) {
            const std::vector<pocket::ForwardResult> outputs =
                qwen.generate(payload, header[1]);
            std::cout << "qwen_persistent_worker_result=1 tp_rank="
                      << qwen.options().tp_rank << " tokens=";
            for (size_t index = 0; index < outputs.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << outputs[index].top_token;
            }
            std::cout << '\n';
            std::cout.flush();
        } else {
            throw std::runtime_error("unknown Qwen persistent worker command");
        }
    }
}

void print_safe_tensor(const pocket::SafeTensorInfo& info, const std::string& shard) {
    std::cout << info.name << " dtype=" << pocket::safe_dtype_name(info.dtype) << " shape=[";
    for (size_t i = 0; i < info.shape.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << info.shape[i];
    }
    std::cout << "] begin=" << info.data_begin << " abs=" << info.absolute_begin << " bytes=" << info.nbytes << " shard=" << shard << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const auto process_started = std::chrono::steady_clock::now();
    try {
        Args args = parse_args(argc, argv);
        const bool host_only = is_host_only_mode(args);
        if (!host_only) {
            if (args.device >= 0) {
                if (!pocket::device_set(args.device)) throw std::runtime_error("failed to set device");
            } else if (args.tp_world > 1) {
                if (!pocket::device_set(args.tp_rank)) throw std::runtime_error("failed to set device for tp rank");
            }
        }
        if (args.tp_world > 1) {
            std::cout << "tp_world=" << args.tp_world << " tp_rank=" << args.tp_rank
                      << " device=" << (args.device >= 0 ? args.device : args.tp_rank) << "\n";
        }
        if (args.serve) {
            if (args.ckpt.empty()) throw std::runtime_error("--serve requires --ckpt");
            if (pocket::is_qwen3_5_checkpoint(args.ckpt)) {
                throw std::runtime_error("Qwen checkpoint is not supported by the DeepSeek-V4 OpenAI server yet; use --smoke-forward or --generate-token");
            }
            const int layer_count = args.smoke_layers > 0 ? args.smoke_layers : 43;
            const int max_context = args.max_context > 0 ? args.max_context : 8192;
            pocket::ForwardSmokeOptions opts;
            opts.tp_world = args.tp_world;
            opts.tp_rank = args.tp_rank;
            opts.device = args.device >= 0 ? args.device : args.tp_rank;
            opts.nccl_id_path = args.nccl_id_path;
            const pocket::ModelConfig model_cfg = pocket::ModelConfig::from_hf_config(args.ckpt);
            pocket::PersistentEngine engine(args.ckpt, opts, layer_count, max_context);
            std::cout << "server_max_context=" << max_context << "\n";
            std::cout << "model_context_length=" << model_cfg.context_length << "\n";
            engine.warmup_tp();
            if (args.tp_rank > 0) {
                // Worker rank: park on the NCCL command channel until rank 0
                // sends SHUTDOWN.
                engine.run_worker_loop();
                return 0;
            }
            const std::string sidecar_script = args.sidecar_script.empty()
                ? std::string("src/server/cpp_sidecar.py")
                : args.sidecar_script;
            pocket::PythonSidecar sidecar(args.python_bin, sidecar_script, args.ckpt);
            pocket::OpenAIServerConfig cfg;
            cfg.port = args.port;
            cfg.host = args.host;
            pocket::OpenAIServer server(engine, sidecar, cfg);
            server.run();
            engine.worker_command_shutdown();
            return 0;
        }
        if (!args.ckpt.empty()) {
            const bool qwen_checkpoint = pocket::is_qwen3_5_checkpoint(args.ckpt);
            pocket::SafeTensorsIndex index(args.ckpt);
            std::cout << "pocketllm_engine opened " << args.ckpt << "\n";
            std::cout << "format=safetensors tensors=" << index.tensor_count()
                      << " shards=" << index.shard_count()
                      << " total_size=" << index.total_size()
                      << " backend=" << pocket::device_backend_name() << " device_runtime=" << (pocket::device_runtime_available() ? "yes" : "no") << "\n";
            if (args.dump_config) {
                if (qwen_checkpoint) {
                    const pocket::QwenConfig qwen_config = pocket::QwenConfig::from_hf_config(args.ckpt);
                    std::cout << qwen_config.to_string();
                } else {
                    std::cout << pocket::ModelConfig::from_hf_config(args.ckpt).to_string();
                }
            }
            if (args.qwen_audit) {
                if (!qwen_checkpoint) throw std::runtime_error("--qwen-audit requires a Qwen checkpoint");
                const pocket::QwenConfig qwen_config = pocket::QwenConfig::from_hf_config(args.ckpt);
                const pocket::QwenWeightMap map(index, qwen_config, args.tp_world, args.tp_rank);
                const pocket::QwenLinearKindCounts kinds = map.checkpoint_linear_kind_counts();
                const pocket::QwenCoverage cover = map.coverage();
                const double gib = 1024.0 * 1024.0 * 1024.0;
                std::cout << "qwen_audit tp_world=" << args.tp_world
                          << " tp_rank=" << args.tp_rank
                          << " tensors=" << map.tensor_count()
                          << " weight_bytes=" << map.local_weight_bytes()
                          << " scale_bytes=" << map.local_scale_bytes()
                          << " weight_GiB=" << (static_cast<double>(map.local_weight_bytes()) / gib)
                          << " scale_GiB=" << (static_cast<double>(map.local_scale_bytes()) / gib)
                          << " total_GiB="
                          << (static_cast<double>(map.local_weight_bytes() + map.local_scale_bytes()) / gib)
                          << " host_global_metadata_bytes="
                          << map.host_global_metadata_bytes()
                          << " checkpoint_dense_f16_linears=" << kinds.dense_f16
                          << " checkpoint_fp8_block128_linears=" << kinds.fp8_block128
                          << " checkpoint_fp8_channel_linears=" << kinds.fp8_channel
                          << " checkpoint_nvfp4_group16_linears=" << kinds.nvfp4_group16
                          << "\n";
                // Coverage is what tells an official multimodal checkpoint apart
                // from an unrecognized one: the text runtime must claim every
                // index entry that is not a vision tensor.
                std::cout << "qwen_coverage index_tensors=" << cover.index_tensors
                          << " mapped=" << cover.mapped_tensors
                          << " visual_ignored=" << cover.visual_tensors
                          << " unexpected=" << cover.unexpected_tensors
                          << " checkpoint_text_bytes=" << cover.checkpoint_text_bytes
                          << " checkpoint_text_GiB="
                          << (static_cast<double>(cover.checkpoint_text_bytes) / gib)
                          << " sharded_GiB="
                          << (static_cast<double>(cover.sharded_local_bytes) / gib)
                          << " replicated_GiB="
                          << (static_cast<double>(cover.replicated_local_bytes) / gib)
                          << "\n";
                for (const std::string& name : cover.unexpected_examples) {
                    std::cout << "qwen_coverage_unexpected " << name << "\n";
                }
                const pocket::QwenTensorRef& embed = map.embed_tokens();
                const pocket::QwenLinearRef& head = map.lm_head();
                const pocket::QwenMlpWeights& mlp = map.layers().front().mlp;
                std::cout << "qwen_tp_local storage_dtype="
                          << pocket::safe_dtype_name(embed.dtype)
                          << " device_dtype=" << pocket::safe_dtype_name(embed.device_dtype)
                          << " embed_rows=" << embed.local_shape.at(0)
                          << " head_rows=" << head.logical_local_shape.at(0)
                          << " mlp_intermediate_rows="
                          << mlp.gate_proj.logical_local_shape.at(0)
                          << " down_proj_k=" << mlp.down_proj.logical_local_shape.at(1)
                          << " linear_attention_layers="
                          << qwen_config.linear_attention_layers()
                          << " full_attention_layers="
                          << qwen_config.full_attention_layers()
                          << " mtp=" << (map.mtp().found ? 1 : 0)
                          << "\n";
                if (args.qwen_audit_strict) {
                    // Throws unless the whole index is accounted for, so a CI run
                    // fails on an unknown checkpoint variant instead of silently
                    // loading a partial model.
                    map.require_full_coverage();
                    std::cout << "qwen_audit_strict=ok\n";
                }
            }
            if (!args.inspect_tensor.empty()) {
                const std::string* shard_name = index.shard_for_tensor(args.inspect_tensor);
                if (shard_name == nullptr) throw std::runtime_error("tensor not found: " + args.inspect_tensor);
                pocket::SafeTensorsShard shard(index.shard_path(*shard_name));
                const auto* info = shard.find_tensor(args.inspect_tensor);
                if (info == nullptr) throw std::runtime_error("tensor missing in shard header: " + args.inspect_tensor);
                print_safe_tensor(*info, *shard_name);
            }
            if (args.smoke_forward) {
                if (qwen_checkpoint) {
                    std::unique_ptr<pocket::Tokenizer> tokenizer;
                    if (!args.prompt.empty()) {
                        tokenizer = std::make_unique<pocket::Tokenizer>(args.ckpt);
                    }
                    std::vector<int> prompt_ids = load_token_ids(args);
                    if (!args.prompt.empty()) prompt_ids = tokenizer->encode_basic(args.prompt, false);
                    if (prompt_ids.empty() && !args.qwen_persistent_stdin) {
                        if (args.forward_token < 0) throw std::runtime_error("Qwen smoke requires --prompt, --token-ids, or --generate-token");
                        prompt_ids.push_back(args.forward_token);
                    }
                    if (args.qwen_persistent_stdin && args.max_context <= 0) {
                        throw std::runtime_error("--qwen-persistent-stdin requires --max-context");
                    }
                    pocket::QwenEngineOptions qwen_opts;
                    qwen_opts.tp_world = args.tp_world;
                    qwen_opts.tp_rank = args.tp_rank;
                    qwen_opts.device = args.device >= 0 ? args.device : args.tp_rank;
                    if (args.prefill_chunk_tokens > 0) {
                        qwen_opts.prefill_chunk_tokens = args.prefill_chunk_tokens;
                    }
                    qwen_opts.kv_cache_dtype = pocket::parse_qwen_kv_cache_dtype(args.kv_cache_dtype);
                    qwen_opts.attention_window = args.qwen_attention_window;
                    qwen_opts.attention_sink_tokens = args.qwen_attention_sink_tokens;
                    // Prefix snapshots are useful only while this engine stays
                    // alive across requests. Keep one-shot runs free of their
                    // extra device allocations; --qwen-no-prefix-cache still
                    // overrides persistent mode.
                    qwen_opts.prefix_cache = args.qwen_prefix_cache &&
                        args.qwen_persistent_stdin;
                    qwen_opts.state_snapshot_interval_tokens = args.qwen_snapshot_interval;
                    qwen_opts.max_state_snapshots = args.qwen_max_snapshots;
                    qwen_opts.mtp = args.qwen_mtp;
                    qwen_opts.mtp_speculative_tokens = args.qwen_mtp_tokens;
                    qwen_opts.mtp_adaptive = args.qwen_mtp_adaptive;
                    qwen_opts.dspark_checkpoint = args.qwen_dspark_checkpoint;
                    qwen_opts.dflash2_checkpoint = args.qwen_dflash2_checkpoint;
                    qwen_opts.nccl_id_path = args.nccl_id_path;
                    qwen_opts.temperature = args.qwen_temperature;
                    qwen_opts.top_p = args.qwen_top_p;
                    qwen_opts.top_k = args.qwen_top_k;
                    qwen_opts.sampling_seed = args.qwen_seed;
                    const int qwen_context = args.max_context > 0
                        ? args.max_context
                        : static_cast<int>(prompt_ids.size()) + std::max(1, args.max_new_tokens);
                    if (!args.qwen_persistent_stdin &&
                        static_cast<uint64_t>(prompt_ids.size()) +
                            static_cast<uint64_t>(std::max(1, args.max_new_tokens)) >
                        static_cast<uint64_t>(qwen_context)) {
                        throw std::runtime_error("Qwen prompt plus generation exceeds --max-context");
                    }
                    const auto model_load_started =
                        std::chrono::steady_clock::now();
                    pocket::QwenEngine qwen(args.ckpt, qwen_opts, args.smoke_layers, qwen_context);
                    const double model_load_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - model_load_started).count();
                    pocket::QwenRuntimeTelemetry qwen_telemetry =
                        qwen.runtime_telemetry();
                    std::cout << "qwen_startup=1 layers=" << (args.smoke_layers > 0 ? args.smoke_layers : 64)
                              << " persistent_stdin=" << (args.qwen_persistent_stdin ? 1 : 0)
                              << " prefill_chunk_tokens=" << qwen_opts.prefill_chunk_tokens
                              << " kv_cache_dtype=" << pocket::qwen_kv_cache_dtype_name(qwen_opts.kv_cache_dtype)
                              << " attention_window=" << qwen_opts.attention_window
                              << " attention_sink_tokens=" << qwen_opts.attention_sink_tokens
                              << " prefix_cache=" << (qwen_opts.prefix_cache ? 1 : 0)
                              << " mtp=" << (qwen_opts.mtp ? 1 : 0)
                              << " mtp_tokens=" << qwen_opts.mtp_speculative_tokens
                              << " mtp_adaptive=" << (qwen_opts.mtp_adaptive ? 1 : 0)
                              << " dspark=" << (!qwen_opts.dspark_checkpoint.empty() ? 1 : 0)
                              << " dflash2=" << (!qwen_opts.dflash2_checkpoint.empty() ? 1 : 0)
                              << " dspark_checkpoint=" << (qwen_opts.dspark_checkpoint.empty()
                                  ? "-" : qwen_opts.dspark_checkpoint)
                              << " dflash2_checkpoint=" << (qwen_opts.dflash2_checkpoint.empty()
                                  ? "-" : qwen_opts.dflash2_checkpoint)
                              << " snapshot_interval=" << qwen_opts.state_snapshot_interval_tokens
                              << " max_snapshots=" << qwen_opts.max_state_snapshots
                              << " prompt_tokens=" << prompt_ids.size()
                              << " max_context=" << qwen_context
                              << " model_load_seconds=" << model_load_seconds
                              << " process_elapsed_seconds="
                              << std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() -
                                     process_started).count()
                              << " host_global_metadata_bytes="
                              << qwen_telemetry.host_global_metadata_bytes
                              << " checkpoint_dense_f16_linears="
                              << qwen_telemetry.checkpoint_linear_kinds.dense_f16
                              << " checkpoint_fp8_block128_linears="
                              << qwen_telemetry.checkpoint_linear_kinds.fp8_block128
                              << " checkpoint_fp8_channel_linears="
                              << qwen_telemetry.checkpoint_linear_kinds.fp8_channel
                              << " checkpoint_nvfp4_group16_linears="
                              << qwen_telemetry.checkpoint_linear_kinds.nvfp4_group16
                              << " active_dense_f16_linears="
                              << qwen_telemetry.active_linear_kinds.dense_f16
                              << " active_fp8_block128_linears="
                              << qwen_telemetry.active_linear_kinds.fp8_block128
                              << " active_fp8_channel_linears="
                              << qwen_telemetry.active_linear_kinds.fp8_channel
                              << " active_nvfp4_group16_linears="
                              << qwen_telemetry.active_linear_kinds.nvfp4_group16
                              << " nvfp4_q8_workspace_peak_bytes="
                              << qwen_telemetry.nvfp4_q8_workspace_peak_bytes
                              << " nvfp4_decode_path="
                              << qwen_telemetry.nvfp4_decode_path
                              << " nvfp4_prefill_path="
                              << qwen_telemetry.nvfp4_prefill_path
                              << " fp8_channel_decode_path="
                              << qwen_telemetry.fp8_channel_decode_path
                              << " fp8_channel_prefill_path="
                              << qwen_telemetry.fp8_channel_prefill_path
                              << " target_head_path="
                              << qwen_telemetry.target_head_path << "\n";
                    if (args.qwen_persistent_stdin) {
                        if (args.tp_world > 1 && args.nccl_id_path.empty()) {
                            throw std::runtime_error(
                                "Qwen persistent TP requires --nccl-id-path");
                        }
                        const std::string channel_path = args.nccl_id_path.empty()
                            ? std::string("/tmp/pocketllm_qwen_persistent")
                            : args.nccl_id_path + ".qwen";
                        std::unique_ptr<pocket::CmdChannel> channel =
                            pocket::CmdChannel::create(args.tp_world, args.tp_rank,
                                                     channel_path);
                        qwen.warmup_tp();
                        if (args.tp_rank > 0) {
                            run_qwen_persistent_worker(qwen, *channel);
                            return 0;
                        }
                        bool shutdown_sent = false;
                        const auto shutdown_workers = [&]() {
                            if (shutdown_sent) return;
                            try {
                                qwen_send_command(*channel, QwenPersistentCommand::Shutdown);
                                shutdown_sent = true;
                            } catch (...) {
                                // Preserve the original exception. The parent
                                // process will terminate workers in its cleanup.
                            }
                        };
                        try {
                            std::string line;
                            while (std::getline(std::cin, line)) {
                                if (line.empty()) continue;
                                std::stringstream request(line);
                                int max_new_tokens = 0;
                                request >> max_new_tokens;
                                if (max_new_tokens <= 0) {
                                    throw std::runtime_error(
                                        "persistent Qwen input must start with max_new_tokens");
                                }
                                std::vector<int> ids;
                                int token = 0;
                                while (request >> token) ids.push_back(token);
                                if (ids.empty() || static_cast<int>(ids.size()) + max_new_tokens >
                                                       qwen.max_context()) {
                                    throw std::runtime_error(
                                        "persistent Qwen prompt is empty or exceeds max_context");
                                }
                                const auto t0 = std::chrono::steady_clock::now();
                                double plain_prefill_seconds = 0.0;
                                std::vector<int> generated;
                                generated.reserve(static_cast<size_t>(max_new_tokens));
                                if (qwen.options().mtp ||
                                    !qwen.options().dspark_checkpoint.empty() ||
                                    !qwen.options().dflash2_checkpoint.empty()) {
                                    qwen_send_command(*channel, QwenPersistentCommand::Generate,
                                                      max_new_tokens, 0, &ids);
                                    const std::vector<pocket::ForwardResult> outputs =
                                        qwen.generate(ids, max_new_tokens);
                                    for (const pocket::ForwardResult& output : outputs) {
                                        generated.push_back(output.top_token);
                                    }
                                } else {
                                    qwen_send_command(*channel, QwenPersistentCommand::Prefill,
                                                      0, 0, &ids);
                                    const auto prefill_started =
                                        std::chrono::steady_clock::now();
                                    pocket::ForwardResult next = qwen.prefill(ids);
                                    plain_prefill_seconds = std::chrono::duration<double>(
                                        std::chrono::steady_clock::now() - prefill_started).count();
                                    generated.push_back(next.top_token);
                                    for (int step = 1; step < max_new_tokens; ++step) {
                                        qwen_send_command(*channel, QwenPersistentCommand::Decode,
                                                          next.top_token,
                                                          static_cast<int>(ids.size()) + step - 1);
                                        next = qwen.decode_step(next.top_token);
                                        generated.push_back(next.top_token);
                                    }
                                }
                                const auto t1 = std::chrono::steady_clock::now();
                                const auto stats = qwen.prefix_cache_stats();
                                const double wall_seconds =
                                    std::chrono::duration<double>(t1 - t0).count();
                                const bool speculative = qwen.options().mtp ||
                                    !qwen.options().dspark_checkpoint.empty() ||
                                    !qwen.options().dflash2_checkpoint.empty();
                                const double prefill_seconds = speculative
                                    ? qwen.mtp_stats().prefill_seconds
                                    : plain_prefill_seconds;
                                const double decode_seconds = std::max(
                                    0.0, wall_seconds - prefill_seconds);
                                std::cout << "qwen_persistent_result=1 prompt_tokens="
                                          << ids.size() << " generated_tokens="
                                          << generated.size() << " tokens=";
                                for (size_t i = 0; i < generated.size(); ++i) {
                                    if (i) std::cout << ',';
                                    std::cout << generated[i];
                                }
                                std::cout << " wall=" << wall_seconds
                                          << " prefill_seconds=" << prefill_seconds
                                          << " decode_seconds=" << decode_seconds
                                          << " decode_tokens_per_s="
                                          << (decode_seconds > 0.0 && generated.size() > 1
                                                  ? (generated.size() - 1) / decode_seconds
                                                  : 0.0)
                                          << " prefill_tokens_per_s="
                                          << (prefill_seconds > 0.0 ? ids.size() / prefill_seconds : 0.0)
                                          << " prefill_computed_tokens_per_s="
                                          << (prefill_seconds > 0.0
                                                  ? stats.computed_tokens / prefill_seconds
                                                  : 0.0)
                                          << " prefix_reused_tokens=" << stats.reused_tokens
                                          << " prefix_computed_tokens=" << stats.computed_tokens
                                          << " prefix_matched_tokens=" << stats.matched_tokens
                                          << " prefix_resume_source=" << stats.resume_source
                                          << " prefix_snapshots=" << stats.snapshots
                                          << " prefix_snapshot_bytes=" << stats.snapshot_bytes
                                          << " mtp=" << (qwen.options().mtp ? 1 : 0)
                                          << " dspark=" << (!qwen.options().dspark_checkpoint.empty() ? 1 : 0)
                                          << " dflash2=" << (!qwen.options().dflash2_checkpoint.empty() ? 1 : 0)
                                          << " mtp_tokens=" << qwen.options().mtp_speculative_tokens
                                          << " mtp_adaptive=" << (qwen.options().mtp_adaptive ? 1 : 0)
                                          << " mtp_accept_rate=" << qwen.mtp_stats().accept_rate()
                                          << " spec_accept_length=" << qwen.mtp_stats().accept_length()
                                          << " mtp_verify_count=" << qwen.mtp_stats().verify_count
                                          << " mtp_proposed_drafts=" << qwen.mtp_stats().proposed_drafts
                                          << " mtp_correct_drafts=" << qwen.mtp_stats().correct_drafts
                                          << " mtp_rollback_count=" << qwen.mtp_stats().rollback_count
                                          << " mtp_replay_tokens=" << qwen.mtp_stats().replay_tokens
                                          << " dspark_confidence_count=" << qwen.mtp_stats().confidence_count
                                          << " dspark_confidence_mean=" << qwen.mtp_stats().mean_confidence()
                                          << " dspark_confidence_min=" << qwen.mtp_stats().confidence_min
                                          << " dspark_confidence_max=" << qwen.mtp_stats().confidence_max
                                          << " mtp_prefill_seconds=" << qwen.mtp_stats().prefill_seconds
                                          << " mtp_draft_seconds=" << qwen.mtp_stats().draft_seconds
                                          << " mtp_verify_seconds=" << qwen.mtp_stats().verify_seconds
                                          << " mtp_replay_seconds=" << qwen.mtp_stats().replay_seconds << "\n";
                                std::cout.flush();
                            }
                        } catch (...) {
                            shutdown_workers();
                            throw;
                        }
                        shutdown_workers();
                        return 0;
                    }
                    if (args.generate_token) {
                        qwen.warmup_tp();
                        using Clock = std::chrono::steady_clock;
                        const auto t_total0 = Clock::now();
                        const auto t_prefill0 = Clock::now();
                        std::vector<pocket::ForwardResult> generated;
                        generated.reserve(static_cast<size_t>(args.max_new_tokens));
                        pocket::QwenPrefixCacheStats prefix_stats;
                        Clock::time_point t_prefill1;
                        Clock::time_point t_decode0;
                        if (qwen.options().mtp ||
                            !qwen.options().dspark_checkpoint.empty() ||
                            !qwen.options().dflash2_checkpoint.empty()) {
                            generated = qwen.generate(prompt_ids, args.max_new_tokens);
                            prefix_stats = qwen.prefix_cache_stats();
                            t_prefill1 = t_prefill0 + std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(qwen.mtp_stats().prefill_seconds));
                            t_decode0 = t_prefill1;
                        } else {
                            pocket::ForwardResult next = qwen.prefill(prompt_ids);
                            prefix_stats = qwen.prefix_cache_stats();
                            t_prefill1 = Clock::now();
                            t_decode0 = t_prefill1;
                            generated.push_back(next);
                            for (int step = 1; step < args.max_new_tokens; ++step) {
                                next = qwen.decode_step(next.top_token);
                                generated.push_back(next);
                            }
                        }
                        const auto t_decode1 = Clock::now();
                        for (size_t step = 0; step < generated.size(); ++step) {
                            std::cout << "generate_step=" << step
                                      << " token=" << generated[step].top_token
                                      << " token_text=" << (tokenizer != nullptr
                                          ? tokenizer->decode_tokens({generated[step].top_token})
                                          : std::string())
                                      << " top_token=" << generated[step].top_token
                                      << " top_logit=" << generated[step].top_logit
                                      << " checksum=" << generated[step].checksum << "\n";
                        }
                        qwen_telemetry = qwen.runtime_telemetry();
                        size_t free_bytes = 0;
                        size_t total_bytes = 0;
                        if (!pocket::device_mem_info(&free_bytes, &total_bytes)) {
                            free_bytes = 0;
                            total_bytes = 0;
                        }
                        std::cout << "qwen_runtime=1 layers=" << generated.front().layers
                                  << " resident_weight_bytes=" << qwen.resident_weight_bytes()
                                  << " resident_scale_bytes=" << qwen.resident_scale_bytes()
                                  << " verify_weight_bytes=" << qwen.verify_weight_bytes()
                                  << " activation_workspace_peak_bytes=" << qwen.activation_workspace_peak_bytes()
                                  << " kv_cache_bytes=" << qwen.kv_cache_bytes()
                                  << " kv_cache_scale_bytes=" << qwen.kv_cache_scale_bytes()
                                  << " model_load_seconds=" << model_load_seconds
                                  << " process_elapsed_seconds="
                                  << std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() -
                                         process_started).count()
                                  << " host_global_metadata_bytes="
                                  << qwen_telemetry.host_global_metadata_bytes
                                  << " checkpoint_dense_f16_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.dense_f16
                                  << " checkpoint_fp8_block128_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.fp8_block128
                                  << " checkpoint_fp8_channel_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.fp8_channel
                                  << " checkpoint_nvfp4_group16_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.nvfp4_group16
                                  << " active_dense_f16_linears="
                                  << qwen_telemetry.active_linear_kinds.dense_f16
                                  << " active_fp8_block128_linears="
                                  << qwen_telemetry.active_linear_kinds.fp8_block128
                                  << " active_fp8_channel_linears="
                                  << qwen_telemetry.active_linear_kinds.fp8_channel
                                  << " active_nvfp4_group16_linears="
                                  << qwen_telemetry.active_linear_kinds.nvfp4_group16
                                  << " nvfp4_q8_workspace_peak_bytes="
                                  << qwen_telemetry.nvfp4_q8_workspace_peak_bytes
                                  << " nvfp4_decode_path="
                                  << qwen_telemetry.nvfp4_decode_path
                                  << " nvfp4_prefill_path="
                                  << qwen_telemetry.nvfp4_prefill_path
                                  << " fp8_channel_decode_path="
                                  << qwen_telemetry.fp8_channel_decode_path
                                  << " fp8_channel_prefill_path="
                                  << qwen_telemetry.fp8_channel_prefill_path
                                  << " target_head_path="
                                  << qwen_telemetry.target_head_path
                                  << " prefill_chunk_tokens=" << qwen.options().prefill_chunk_tokens
                                  << " kv_cache_dtype=" << pocket::qwen_kv_cache_dtype_name(qwen.options().kv_cache_dtype)
                                  << " attention_window=" << qwen.options().attention_window
                                  << " attention_sink_tokens=" << qwen.options().attention_sink_tokens
                                  << " prefix_cache=" << (qwen.options().prefix_cache ? 1 : 0)
                                  << " snapshot_interval=" << qwen.options().state_snapshot_interval_tokens
                                  << " max_snapshots=" << qwen.options().max_state_snapshots
                                  << " max_context=" << qwen.max_context()
                                  << " prompt_tokens=" << prompt_ids.size()
                                  << " generated_tokens=" << generated.size()
                                  << " prefix_reused_tokens=" << prefix_stats.reused_tokens
                                  << " prefix_computed_tokens=" << prefix_stats.computed_tokens
                                  << " prefix_matched_tokens=" << prefix_stats.matched_tokens
                                  << " prefix_resume_source=" << prefix_stats.resume_source
                                  << " prefix_snapshots=" << prefix_stats.snapshots
                                  << " prefix_snapshot_bytes=" << prefix_stats.snapshot_bytes
                                  << " prefix_hits=" << prefix_stats.hits
                                  << " prefix_misses=" << prefix_stats.misses
                                  << " mtp=" << (qwen.options().mtp ? 1 : 0)
                                  << " dspark=" << (!qwen.options().dspark_checkpoint.empty() ? 1 : 0)
                                  << " dflash2=" << (!qwen.options().dflash2_checkpoint.empty() ? 1 : 0)
                                  << " mtp_tokens=" << qwen.options().mtp_speculative_tokens
                                  << " mtp_adaptive=" << (qwen.options().mtp_adaptive ? 1 : 0)
                                  << " mtp_accept_rate=" << qwen.mtp_stats().accept_rate()
                                  << " spec_accept_length=" << qwen.mtp_stats().accept_length()
                                  << " mtp_verify_count=" << qwen.mtp_stats().verify_count
                                  << " mtp_proposed_drafts=" << qwen.mtp_stats().proposed_drafts
                                  << " mtp_correct_drafts=" << qwen.mtp_stats().correct_drafts
                                  << " mtp_rollback_count=" << qwen.mtp_stats().rollback_count
                                  << " mtp_replay_tokens=" << qwen.mtp_stats().replay_tokens
                                  << " dspark_confidence_count=" << qwen.mtp_stats().confidence_count
                                  << " dspark_confidence_mean=" << qwen.mtp_stats().mean_confidence()
                                  << " dspark_confidence_min=" << qwen.mtp_stats().confidence_min
                                  << " dspark_confidence_max=" << qwen.mtp_stats().confidence_max
                                  << " mtp_prefill_seconds=" << qwen.mtp_stats().prefill_seconds
                                  << " mtp_draft_seconds=" << qwen.mtp_stats().draft_seconds
                                  << " mtp_verify_seconds=" << qwen.mtp_stats().verify_seconds
                                  << " mtp_replay_seconds=" << qwen.mtp_stats().replay_seconds;
                        if (args.resident_bench) {
                            const auto seconds = [](auto begin, auto end) {
                                return std::chrono::duration<double>(end - begin).count();
                            };
                            const double total = seconds(t_total0, t_decode1);
                            const double prefill = seconds(t_prefill0, t_prefill1);
                            const double decode = seconds(t_decode0, t_decode1);
                            const int decode_tokens = generated.size() > 1
                                ? static_cast<int>(generated.size()) - 1 : 0;
                            std::cout << " wall=" << total
                                      << " prefill_seconds=" << prefill
                                      << " prefill_tokens_per_s="
                                      << (prefill > 0.0 ? prompt_ids.size() / prefill : 0.0)
                                      << " decode_seconds=" << decode
                                      << " decode_tokens_per_s="
                                      << (decode > 0.0 ? decode_tokens / decode : 0.0)
                                      << " decode_token_count=" << decode_tokens;
                        }
                        if (total_bytes != 0) {
                            std::cout << " gpu_memory_used_bytes=" << (total_bytes - free_bytes)
                                      << " gpu_memory_total_bytes=" << total_bytes;
                        }
                        std::cout << " process_elapsed_final_seconds="
                                  << std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() -
                                         process_started).count()
                                  << "\n";
                    } else {
                        pocket::ForwardResult result = qwen.prefill(prompt_ids);
                        qwen_telemetry = qwen.runtime_telemetry();
                        std::cout << "smoke_forward=1 qwen_runtime=1 token=" << prompt_ids.back()
                                  << " layers=" << result.layers
                                  << " dim=" << result.dim
                                  << " logits=" << result.logits
                                  << " top_token=" << result.top_token
                                  << " top_logit=" << result.top_logit
                                  << " checksum=" << result.checksum
                                  << " position=" << result.position
                                  << " resident_weight_bytes=" << qwen.resident_weight_bytes()
                                  << " resident_scale_bytes=" << qwen.resident_scale_bytes()
                                  << " verify_weight_bytes=" << qwen.verify_weight_bytes()
                                  << " activation_workspace_peak_bytes=" << qwen.activation_workspace_peak_bytes()
                                  << " kv_cache_bytes=" << qwen.kv_cache_bytes()
                                  << " kv_cache_scale_bytes=" << qwen.kv_cache_scale_bytes()
                                  << " model_load_seconds=" << model_load_seconds
                                  << " process_elapsed_seconds="
                                  << std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() -
                                         process_started).count()
                                  << " host_global_metadata_bytes="
                                  << qwen_telemetry.host_global_metadata_bytes
                                  << " checkpoint_dense_f16_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.dense_f16
                                  << " checkpoint_fp8_block128_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.fp8_block128
                                  << " checkpoint_fp8_channel_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.fp8_channel
                                  << " checkpoint_nvfp4_group16_linears="
                                  << qwen_telemetry.checkpoint_linear_kinds.nvfp4_group16
                                  << " active_dense_f16_linears="
                                  << qwen_telemetry.active_linear_kinds.dense_f16
                                  << " active_fp8_block128_linears="
                                  << qwen_telemetry.active_linear_kinds.fp8_block128
                                  << " active_fp8_channel_linears="
                                  << qwen_telemetry.active_linear_kinds.fp8_channel
                                  << " active_nvfp4_group16_linears="
                                  << qwen_telemetry.active_linear_kinds.nvfp4_group16
                                  << " nvfp4_q8_workspace_peak_bytes="
                                  << qwen_telemetry.nvfp4_q8_workspace_peak_bytes
                                  << " nvfp4_decode_path="
                                  << qwen_telemetry.nvfp4_decode_path
                                  << " nvfp4_prefill_path="
                                  << qwen_telemetry.nvfp4_prefill_path
                                  << " fp8_channel_decode_path="
                                  << qwen_telemetry.fp8_channel_decode_path
                                  << " fp8_channel_prefill_path="
                                  << qwen_telemetry.fp8_channel_prefill_path
                                  << " target_head_path="
                                  << qwen_telemetry.target_head_path
                                  << " prefill_chunk_tokens=" << qwen.options().prefill_chunk_tokens
                                  << " kv_cache_dtype=" << pocket::qwen_kv_cache_dtype_name(qwen.options().kv_cache_dtype)
                                  << " attention_window=" << qwen.options().attention_window
                                  << " attention_sink_tokens=" << qwen.options().attention_sink_tokens
                                  << " prefix_cache=" << (qwen.options().prefix_cache ? 1 : 0)
                                  << " snapshot_interval=" << qwen.options().state_snapshot_interval_tokens
                                  << " max_snapshots=" << qwen.options().max_state_snapshots
                                  << " max_context=" << qwen.max_context()
                                  << " process_elapsed_final_seconds="
                                  << std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() -
                                         process_started).count()
                                  << "\n";
                    }
                    return 0;
                }
                pocket::Tokenizer tokenizer(args.ckpt);
                std::vector<int> prompt_ids = load_token_ids(args);
                if (!prompt_ids.empty()) {
                    args.forward_token = prompt_ids.back();
                    args.position = static_cast<int>(prompt_ids.size()) - 1;
                    std::cout << "prompt_tokens=" << prompt_ids.size()
                              << " last_token=" << args.forward_token
                              << " position=" << args.position
                              << " token_ids_source=" << (args.token_ids_csv.empty() ? "file" : "csv") << "\n";
                }
                if (!args.prompt.empty()) {
                    prompt_ids = tokenizer.encode_basic(args.prompt, false);
                    if (prompt_ids.empty()) throw std::runtime_error("prompt encoded to no tokens");
                    args.forward_token = prompt_ids.back();
                    args.position = static_cast<int>(prompt_ids.size()) - 1;
                    std::cout << "prompt_tokens=" << prompt_ids.size()
                              << " last_token=" << args.forward_token
                              << " position=" << args.position
                              << " last_text=" << tokenizer.decode_piece(args.forward_token) << "\n";
                    if (!args.generate_token) {
                        pocket::ForwardSmokeOptions opts;
                        opts.tp_world = args.tp_world;
                        opts.tp_rank = args.tp_rank;
                        opts.device = args.device >= 0 ? args.device : args.tp_rank;
                        opts.nccl_id_path = args.nccl_id_path;
                        pocket::ForwardSmokeResult result = pocket::run_safetensors_prompt_forward_with_options(args.ckpt, prompt_ids, args.smoke_layers, opts);
                        int top_token = result.top_token;
                        float top_logit = result.top_logit;
#ifdef POCKET_HAVE_TP_COMM
                        if (args.tp_world > 1 && !args.nccl_id_path.empty()) {
                            pocket::TpTopResult global = pocket::tp_global_top1(
                                args.tp_world,
                                args.tp_rank,
                                args.device >= 0 ? args.device : args.tp_rank,
                                args.nccl_id_path.c_str(),
                                result.top_token,
                                result.top_logit);
                            top_token = global.token;
                            top_logit = global.logit;
                        }
#endif
                        std::cout << "smoke_forward=1 token=" << result.token
                                  << " layers=" << result.layers
                                  << " dim=" << result.dim
                                  << " inter=" << result.inter
                                  << " logits=" << result.logits
                                  << " tp_world=" << args.tp_world
                                  << " tp_rank=" << args.tp_rank
                                  << " local_top_token=" << result.top_token
                                  << " local_top_logit=" << result.top_logit
                                  << " top_token=" << top_token
                                  << " top_logit=" << top_logit
                                  << " checksum=" << result.checksum << "\n";
                        return 0;
                    }
                }
                if (args.generate_token) {
                    if (prompt_ids.empty()) {
                        if (args.forward_token < 0) throw std::runtime_error("--generate-token or --prompt is required for generation");
                        prompt_ids.push_back(args.forward_token);
                    }
                    pocket::ForwardSmokeOptions opts;
                    opts.tp_world = args.tp_world;
                    opts.tp_rank = args.tp_rank;
                    opts.device = args.device >= 0 ? args.device : args.tp_rank;
                    opts.nccl_id_path = args.nccl_id_path;
                    if (args.use_persistent) {
                        const int max_context = args.max_context > 0
                            ? args.max_context
                            : static_cast<int>(prompt_ids.size()) + args.max_new_tokens;
                        pocket::PersistentEngine engine(args.ckpt, opts, args.smoke_layers, max_context);
                        engine.warmup_tp();
                        if (args.tp_rank > 0) {
                            engine.run_worker_loop();
                            return 0;
                        }
                        pocket::SamplingParams sp;
                        sp.greedy = true;
                        std::vector<int> generated_ids;
                        generated_ids.reserve(static_cast<size_t>(args.max_new_tokens));
                        using Clock = std::chrono::steady_clock;
                        const auto t_total0 = Clock::now();
                        engine.worker_command_reset();
                        engine.reset_session();
                        const auto t_prefill0 = Clock::now();
                        engine.worker_command_prefill(prompt_ids);
                        int token = engine.prefill(prompt_ids, sp);
                        const auto t_prefill1 = Clock::now();
                        generated_ids.push_back(token);
                        int position = static_cast<int>(prompt_ids.size());
                        const auto t_decode0 = Clock::now();
                        for (int step = 1; step < args.max_new_tokens; ++step) {
                            engine.worker_command_decode(token, position + step - 1);
                            token = engine.decode_step(token, position + step - 1, sp);
                            generated_ids.push_back(token);
                        }
                        const auto t_decode1 = Clock::now();
                        engine.worker_command_shutdown();
                        auto sec = [](auto a, auto b) {
                            return std::chrono::duration<double>(b - a).count();
                        };
                        for (size_t step = 0; step < generated_ids.size(); ++step) {
                            std::cout << "generate_step=" << step
                                      << " token=" << generated_ids[step]
                                      << " token_text=" << tokenizer.decode_tokens({generated_ids[step]})
                                      << " decoded=" << tokenizer.decode_tokens(std::vector<int>(generated_ids.begin(), generated_ids.begin() + step + 1)) << "\n";
                        }
                        if (args.resident_bench) {
                            const double wall = sec(t_total0, t_decode1);
                            const double prefill = sec(t_prefill0, t_prefill1);
                            const double decode = sec(t_decode0, t_decode1);
                            const double prefill_tps = prefill > 0.0 ? static_cast<double>(prompt_ids.size()) / prefill : 0.0;
                            const int decoded = args.max_new_tokens > 1 ? args.max_new_tokens - 1 : 0;
                            const double decode_tps = decode > 0.0 ? static_cast<double>(decoded) / decode : 0.0;
                            const double total_tokens = static_cast<double>(prompt_ids.size() + generated_ids.size());
                            const double tps = wall > 0.0 ? total_tokens / wall : 0.0;
                            std::cout << "resident_bench=1 prompt_tokens=" << prompt_ids.size()
                                      << " generated_tokens=" << generated_ids.size()
                                      << " wall=" << wall
                                      << " tokens_per_s=" << tps
                                      << " prefill_seconds=" << prefill
                                      << " prefill_tokens_per_s=" << prefill_tps
                                      << " decode_seconds=" << decode
                                      << " decode_tokens_per_s=" << decode_tps
                                      << " decode_token_count=" << decoded
                                      << " tp_world=" << args.tp_world
                                      << " tp_rank=" << args.tp_rank
                                      << " use_persistent=1\n";
                        }
                        return 0;
                    }
                    auto timed = pocket::run_safetensors_generate_tokens_timed_with_options(args.ckpt, prompt_ids, args.smoke_layers, args.max_new_tokens, opts);
                    auto& results = timed.tokens;
                    std::vector<int> generated_ids;
                    generated_ids.reserve(results.size());
                    for (size_t step = 0; step < results.size(); ++step) {
                        const pocket::ForwardSmokeResult& result = results[step];
                        generated_ids.push_back(result.token);
                        std::cout << "generate_step=" << step
                                  << " token=" << result.token
                                  << " token_text=" << tokenizer.decode_tokens({result.token})
                                  << " top_token=" << result.top_token
                                  << " top_text=" << tokenizer.decode_tokens({result.top_token})
                                  << " decoded=" << tokenizer.decode_tokens(generated_ids)
                                  << " top_logit=" << result.top_logit
                                  << " checksum=" << result.checksum << "\n";
                    }
                    if (args.resident_bench) {
                        const double wall = timed.wall_seconds;
                        const double prefill = timed.prefill_seconds;
                        const double decode = timed.decode_seconds;
                        const double prefill_tps = prefill > 0.0 ? static_cast<double>(timed.prompt_tokens) / prefill : 0.0;
                        const double decode_tps = decode > 0.0 ? static_cast<double>(timed.decode_tokens) / decode : 0.0;
                        const double tokens = static_cast<double>(prompt_ids.size() + results.size());
                        const double tps = wall > 0.0 ? tokens / wall : 0.0;
                        std::cout << "resident_bench=1 prompt_tokens=" << prompt_ids.size()
                                  << " generated_tokens=" << results.size()
                                  << " wall=" << wall
                                  << " tokens_per_s=" << tps
                                  << " prefill_seconds=" << prefill
                                  << " prefill_tokens_per_s=" << prefill_tps
                                  << " decode_seconds=" << decode
                                  << " decode_tokens_per_s=" << decode_tps
                                  << " decode_token_count=" << timed.decode_tokens
                                  << " tp_world=" << args.tp_world
                                  << " tp_rank=" << args.tp_rank << "\n";
                    }
                } else {
                    pocket::ForwardSmokeResult result = args.forward_token >= 0
                        ? pocket::run_safetensors_token_forward(args.ckpt, args.forward_token, args.smoke_layers)
                        : pocket::run_safetensors_layer_loop_smoke(args.ckpt, args.smoke_layers);
                    std::cout << "smoke_forward=1 token=" << result.token
                              << " layers=" << result.layers
                              << " dim=" << result.dim
                              << " inter=" << result.inter
                              << " logits=" << result.logits
                              << " top_token=" << result.top_token
                              << " top_logit=" << result.top_logit
                              << " checksum=" << result.checksum << "\n";
                }
            }
            if (!args.dump_config && args.inspect_tensor.empty() && !args.inspect &&
                !args.smoke_forward && !args.qwen_audit) {
                std::cout << "inference_not_implemented=1\n";
            }
            return 0;
        }
        pocket::DeepSeekV4Engine engine(args.model);
        std::cout << "pocketllm_engine opened " << args.model << "\n";
        std::cout << "format=gguf gguf_version=" << engine.gguf().version()
                  << " tensors=" << engine.gguf().tensor_count()
                  << " metadata=" << engine.gguf().metadata_count()
                  << " alignment=" << engine.gguf().alignment()
                  << " backend=" << pocket::device_backend_name() << " device_runtime=" << (pocket::device_runtime_available() ? "yes" : "no") << "\n";
        if (args.dump_config) {
            std::cout << engine.config().to_string();
        }
        if (!args.dump_config && !args.inspect) {
            std::cout << "inference_not_implemented=1\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
