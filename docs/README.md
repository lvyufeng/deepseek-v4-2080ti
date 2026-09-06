# PocketLLM documentation

This directory contains model-specific support notes, reproducible benchmark definitions, and engineering analyses. Start with the model matrix when you need to know whether a checkpoint is inspectable, runnable, or benchmarked end to end.

## Getting started

- [Project home](../README.md) · [中文首页](../README_CN.md)
- [Benchmarking and reporting rules](benchmarking.md)
- [C++/CUDA engine notes](../cpp_engine/README.md)
- [Migration: `dsv4` → `pocket`](migration/dsv4-to-pocket-rename.md) — breaking rename of the
  namespace, build targets, executable, and every `DSV4_*` environment variable

## Model guides

- [Model support matrix](models/README.md)
- [DeepSeek-V4](models/deepseek-v4.md)
- [DeepSeek-V4 GGUF Q2 single-GPU history](models/deepseek-v4-gguf-q2-single-gpu.md)
- [MiniMax-M2.7](models/minimax-m2.7.md)
- [GLM-5.2](models/glm-5.2.md)
- [Qwen3.8-27B-FP8](models/qwen3.8-27b-fp8.md)

## Architecture and optimization notes

- [DSpark speculative decoding](dspark.md)
- [FlashMemory 1M context](FLASHMEMORY_1M_CONTEXT.md)
- [MiniMax decode bottleneck analysis](minimax_decode_bottleneck_analysis.md)
- [Qwen4-Exp performance analysis](qwen4_exp_performance.md)
- [Qwen quantized KV cache at 65K (TG512)](qwen_kv_cache_65k_tg512.md)
- [Phase 2 validation results](PHASE2_VALIDATION_RESULTS.md)

## Historical reports

- [DeepSeek-V4 on 4×RTX 2080 Ti](reports/dsv4_2080ti_report.pdf)

Historical notes retain the measurement context and conclusions from the experiment that produced them. Current support status and the latest comparable figures belong in the model pages above.
