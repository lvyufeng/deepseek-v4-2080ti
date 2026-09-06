# PocketLLM

[English](README.md) | 中文

PocketLLM 是一个面向消费级多卡系统的大模型推理工程栈，包含 C++/CUDA 与 PyTorch runtime。它结合模型专用 kernel、低 bit 格式、tensor/expert parallel、CPU/GPU placement，以及面向单请求的可复现实测 benchmark。

项目最初来自在 4×RTX 2080 Ti 上运行 DeepSeek-V4 的工程实践，目前已经包含 DeepSeek-V4、MiniMax-M2.7、GLM-5.2 和 Qwen3.8-27B-FP8 的已验证 runtime。PocketLLM 不是一个“所有模型共用同一后端”的框架：不同模型使用与其架构和 checkpoint 格式匹配的执行路径。

> **项目状态：** 研究和工程软件。下面的数字来自特定 checkpoint、硬件和测试口径，不代表通用性能保证。

## PocketLLM 提供什么

- **模型专用推理路径：** 支持 hybrid attention、MLA、GQA、Gated DeltaNet、dense MLP 和 routed MoE 层。
- **避免不必要的低 bit 展开：** 在支持的热路径中直接消费 FP4、FP8 E4M3、GGUF Q4/Q5/Q8、IQ1/IQ2/IQ3、Q2 等量化 block。
- **消费级 GPU 并行：** 支持 PCIe 多卡上的 TP4/NCCL；对放不进显存的 checkpoint，支持 CPU/NUMA expert placement。
- **Prefill/decode 分离：** 大 batch kernel 与单 token latency 路径独立调度、独立优化。
- **原生 C++/CUDA runtime：** `cpp_engine/` 当前支持 DeepSeek-V4 GGUF/Safetensors 路径，以及 Qwen3.8 FP8 Safetensors 文本生成。
- **检查和验证工具：** GGUF 架构/spec 报告、Safetensors audit、tensor shape 检查、数值 parity 测试和真实 checkpoint benchmark。

## 支持模型一览

| 模型 | Checkpoint / 格式 | Runtime 状态 | 已验证路径 | 4×RTX 2080 Ti 代表结果 |
| --- | --- | --- | --- | --- |
| [DeepSeek-V4-Flash](docs/models/deepseek-v4.md) | Safetensors FP4/FP8；GGUF Q2/IQ2/IQ1 | **已验证 generation** | PyTorch 异构、C++/CUDA、GGUF TP4 | C++ FP4：32K–64K prefill 约 401 tok/s；decode 约 3.7 tok/s |
| [MiniMax-M2.7](docs/models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | **已验证 TP4 generation** | Raw-block CUDA、GGUF TP4 | Full-model 256-token prefill 约 104.9–107 tok/s；43-layer decode benchmark 10.32 tok/s |
| [GLM-5.2](docs/models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | **已验证文本生成** | Raw-block CUDA、GGUF TP4 | prefill 约 0.79 tok/s；decode 约 0.66 tok/s |
| [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | **已验证 C++ 文本 runtime** | C++/CUDA TP4、GPU-resident FP8 | 512-token prompt：prefill 416.48 tok/s；decode 35.87 tok/s |

模型页面会把“模型架构规格”和“PocketLLM 当前实际实现能力”分开。`inspect`、`smoke` 和 benchmark 也不自动等于 production serving 保证。

## 性能摘要

本节数字除非特别说明，都来自同一台基线机器上的真实 checkpoint：4× NVIDIA RTX 2080 Ti、每卡 22 GiB、PCIe Gen3、无 NVLink、单请求执行、适用时使用 TP4。比较前请先阅读 [Benchmark 口径](docs/benchmarking.md)。

### Qwen3.8-27B-FP8 C++ runtime

- 64-token prompt：prefill 138.61–138.69 tok/s，decode 36.82 tok/s。
- 512-token prompt：prefill 416.48 tok/s，decode 35.87 tok/s。
- 实测每 rank 约使用 8.0–8.6 GiB；本地 FP8 权重和 scale 常驻 GPU。
- 四个 TP rank 的生成 token 序列一致。当前路径是 text-only，尚未接入 OpenAI server。

### DeepSeek-V4 C++ FP4 runtime

- 32K prompt：prefill 约 402 tok/s，约 11.2 GiB/rank。
- 64K prompt：prefill 约 401 tok/s，约 14.5 GiB/rank。
- Decode：该 4×RTX 2080 Ti 配置下约 3.7 tok/s。

### MiniMax-M2.7 与 GLM-5.2 GGUF runtime

- MiniMax-M2.7 在 Q4/Q5 MMA 和 IQ2 DP4A 路径后，full-model 256-token prefill 约 104.9–107 tok/s；另一个 fused RMSNorm 的 43-layer decode benchmark 达到 10.32 tok/s。
- GLM-5.2 已通过 raw-block GGUF 路径实现 generation。由于模型规模、expert staging 和逐层同步，其当前 decode floor 明显更低；resident-cache、routed-TP 和 fused-RMSNorm 等实验开关默认不启用。

这些是模型专属结果，不能合并成一个 PocketLLM 总分。

## 架构概览

PocketLLM 包含两类互补执行方式：

1. **GPU-resident 与低 bit 执行：** 在总显存预算允许时，让本地权重或 expert block 常驻 GPU。
2. **异构执行：** 将 routed experts 放在 CPU/NUMA 内存，只把当前 token 或 prefill chunk 激活的量化 block 搬到 GPU。

Runtime 是模型专用的：DeepSeek-V4 使用 MLA/indexing 和 routed-expert 调度；MiniMax-M2.7、GLM-5.2 使用 GGUF raw-block 路径；Qwen3.8 使用 Safetensors FP8 online unpacking 加 hybrid linear/full attention。设计上的热路径不会将完整量化权重展开成 FP32 副本。

## 快速开始

### 构建 Python extensions

```bash
python -m pip install -r requirements.txt
python setup.py build_ext
```

Python package metadata 现在使用 `pocketllm`；为了兼容性，现有 `src.*` Python import namespace 不变。

### 构建 C++/CUDA engine

```bash
cmake -S cpp_engine -B build/cpp_engine -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp_engine -j
```

可执行文件为 `pocketllm_engine`：

```text
build/cpp_engine/pocketllm_engine
```

它原名 `dsv4_cpp_engine`。该改名与 `pocket::` 命名空间、`POCKETLLM_*` 环境变量一起构成破坏性变更，
详见[迁移说明](docs/migration/dsv4-to-pocket-rename.md)。

### 运行 DeepSeek-V4 C++ TP4 serving

```bash
CKPT=/path/to/DeepSeek-V4-Flash \
PORT=8000 \
MAX_CONTEXT=8192 \
PYTHON=python \
bash scripts/run_cpp_serve_tp4.sh
```

该命令让 rank 0 运行 OpenAI 兼容服务，rank 1–3 运行 NCCL worker。

### 通过共享 raw-block CLI 运行 GGUF 模型

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_gguf \
  --gguf-path /path/to/model.gguf \
  --seed-file /path/to/prompt_tokens.bin \
  --max-new-tokens 32 \
  --prewarm
```

GLM-5.2 文本 prompt：

```bash
PYTHONPATH=$PWD torchrun --standalone --nproc-per-node=4 \
  -m src.cli.generate_glm \
  --gguf-path /path/to/GLM-5.2-GGUF/UD-Q2_K_XL \
  --prompt "你好" \
  --chat \
  --max-new-tokens 32 \
  --prewarm
```

### 检查 GGUF checkpoint

```bash
PYTHONPATH=$PWD python -m src.cli.inspect_gguf \
  --gguf-path /path/to/model.gguf \
  --architecture auto \
  --spec-summary \
  --validate-spec \
  --capability-report \
  --placement-report
```

### 运行 Qwen3.8-27B-FP8 C++ smoke/benchmark

Qwen 路径支持 text prompt 或 token IDs，并通过 NCCL ID 文件启动 TP4 rank：

```bash
rm -f /tmp/pocketllm_qwen_nccl.id
for rank in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$rank \
  build/cpp_engine/pocketllm_engine \
    --ckpt /path/to/Qwen3.8-27B-FP8 \
    --tp-world 4 --tp-rank $rank --device 0 \
    --nccl-id-path /tmp/pocketllm_qwen_nccl.id \
    --prompt "请用一段话解释 tensor parallelism。" \
    --generate-token 123 --max-new-tokens 32 --smoke-layers 0 --resident-bench \
    > /tmp/pocketllm_qwen_rank${rank}.log 2>&1 &
done
wait
```

正常运行时使用相同的 Qwen smoke 参数；rank 0 会输出 `prefill_tokens_per_s`、`decode_tokens_per_s`、resident weight bytes 和 GPU memory。Qwen OpenAI server adapter 尚未实现。

对于单并发客户端，如果后续请求会追加或压缩上一次请求，使用长期存活的 TP4 token-ID worker。rank 0 读取 `<max_new_tokens> token0 token1 ...`，并输出 exact prefix 统计；追加请求复用 live state，分叉请求从 GPU snapshot 恢复：

```bash
python scripts/bench_qwen_prefix_cache.py \\
  --ckpt /path/to/Qwen3.8-27B-FP8 \\
  --token-ids-file /path/to/prompt_ids.csv \\
  --max-context 32768 \\
  --max-new-tokens 4 \\
  --compression-prefix-tokens 4096
```

benchmark 会启动 rank 1–3 command worker，让 rank 0 在多轮请求间保持 engine。使用 `--disable-prefix-cache` 做 cold parity A/B。一次性 Qwen 命令只处理单个请求，因此默认不创建 prefix snapshot；`--qwen-persistent-stdin` 开启缓存，`--qwen-no-prefix-cache` 可显式关闭。

## 文档

- [文档总览](docs/README.md)
- [模型支持矩阵](docs/models/README.md)
- [Benchmark 口径](docs/benchmarking.md)
- [DeepSeek-V4](docs/models/deepseek-v4.md)
- [MiniMax-M2.7](docs/models/minimax-m2.7.md)
- [GLM-5.2](docs/models/glm-5.2.md)
- [Qwen3.8-27B-FP8](docs/models/qwen3.8-27b-fp8.md)
- [DSpark speculative decoding](docs/dspark.md)
- [FlashMemory 1M context](docs/FLASHMEMORY_1M_CONTEXT.md)
- [MiniMax decode bottleneck 分析](docs/minimax_decode_bottleneck_analysis.md)
- [历史 2080 Ti 报告](docs/reports/dsv4_2080ti_report.pdf)

## Roadmap

- [x] DeepSeek-V4 FP4/FP8 与 GGUF Q2/IQ2/IQ1 generation 路径。
- [x] MiniMax-M2.7 与 GLM-5.2 GGUF raw-block generation 路径。
- [x] Qwen3.8-27B-FP8 C++ TP4 文本 runtime。
- [ ] 在不破坏现有脚本的前提下，统一 C++ model dispatch 和 binary 命名。
- [ ] Qwen OpenAI 兼容 serving adapter。
- [ ] 在实测有收益时接入 CUDA Graph 和 persistent decode dispatch。
- [ ] 增加更多模型 benchmark fixture 和自动化 regression dashboard。

## 已知限制

- 性能高度依赖 GPU 型号、PCIe 拓扑、NUMA placement、驱动/runtime 版本和 checkpoint 变体。
- PCIe 系统上的 GGUF expert staging 可能主导 decode；prefill TPS 高不代表 decode TPS 高。
- DSpark 当前 C++ verify path 是 sequential，不应宣称为加速路径；multi-token verify 有独立的数值漂移策略。
- Qwen runtime 当前只支持 text checkpoint 路径，视觉输入和 OpenAI 兼容 Qwen serving 尚未接入。
- 部分实验优化在真实端到端测试出现回归后被保留为 opt-in 或关闭；具体见模型页和历史分析文档。

## License

PocketLLM 代码使用 [PolyForm Noncommercial License 1.0.0](LICENSE)。

个人使用、学术研究、教育、非商业 benchmark 和非商业部署属于允许用途。商业使用需要版权持有者单独书面许可。

模型权重、tokenizer、CUDA、PyTorch、GGUF 资源和其他第三方组件分别受其自身许可证约束。PocketLLM 代码许可证不授予任何第三方模型资产的额外权利。

## 致谢

PocketLLM 基于 CUDA、PyTorch、safetensors、GGUF、Transformers、NCCL 和 llama.cpp 量化研究。仓库中的模型专用 runtime 与 benchmark，是面向消费级硬件可复现本地推理的工程实践。
