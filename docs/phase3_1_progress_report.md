# Phase 3.1 Progress Report - cpp_engine Batching Implementation

**日期**: 2026-09-02  
**状态**: 进行中 (编译测试阶段)

## 已完成工作

### 1. 架构设计和规划 ✅

**完成文档**:
- `docs/cpp_engine_batching_phase3_1.md` - 完整的实施计划
- `docs/vllm_sglang_comparison.md` - 与 vLLM/SGLang 的对比分析

**设计亮点**:
- 向后兼容：默认单 session 模式，零性能开销
- 渐进式启用：用户显式调用 `allocate_batch_slots(N)` 
- 固定 slot 策略：避免 paged attention 的复杂性
- FCFS 调度：先实现正确性，后优化公平性

### 2. Per-Request State 设计 ✅ (Task #2)

**文件**: `cpp_engine/include/qwen_engine.hpp`

**新增类型**:
```cpp
struct QwenBatchSamplingParams {
    float temperature, top_p;
    int top_k, max_new_tokens;
    unsigned long long seed;
};

struct QwenBatchedRequest {
    uint64_t request_id;
    std::vector<int> prompt_tokens;
    int seq_len;                    // 当前序列长度
    int slot_id;                    // KV cache slot (-1 = 未分配)
    int cached_prefix_len;          // prefix cache 命中
    QwenBatchSamplingParams sampling;
    bool finished;
    int last_token;
    std::vector<int> generated_tokens;
    QwenForwardResult last_result;
};

struct QwenBatchPrefillResult { ... };
struct QwenBatchDecodeResult { ... };
```

### 3. Batch API 声明 ✅

**新增方法** (在 `QwenEngine` 类中):
```cpp
bool supports_batching() const;
void allocate_batch_slots(int max_batch_size);
int allocate_slot(uint64_t request_id);
void free_slot(uint64_t request_id);
QwenBatchPrefillResult batch_prefill(const vector<QwenBatchedRequest*>&);
QwenBatchDecodeResult batch_decode_step(const vector<QwenBatchedRequest*>&);
```

### 4. Slot 管理实现 ✅ (Task #3 部分完成)

**文件**: `cpp_engine/engine/qwen_engine.cpp`

**添加到 `QwenEngine::Impl`**:
```cpp
int max_batch_size = 1;
bool batch_mode_enabled = false;
std::unordered_map<uint64_t, int> request_to_slot;
std::vector<int> free_slots;
```

### 5. Batch API 初步实现 ✅ (Task #4 部分完成)

**已实现函数**:

1. **`supports_batching()`**:
   - 返回 `true`（Phase 3.1 始终支持，opt-in 启用）

2. **`allocate_batch_slots(max_batch_size)`**:
   - 验证参数
   - 设置 `batch_mode_enabled` 标志
   - 单 session 模式 (max_batch_size=1) 无需重新分配
   - 多 slot 模式：抛出异常提示 KV cache 重分配待实现

3. **`allocate_slot(request_id)`**:
   - 单 session 模式返回 slot 0
   - 多 slot 模式从 `free_slots` 分配

4. **`free_slot(request_id)`**:
   - 释放 slot 回 `free_slots`

5. **`batch_prefill(requests)`**:
   - **当前实现**: FCFS 串行处理每个请求
   - 调用现有 `prefill()` 逐个处理
   - 记录每个请求的 seq_len 和 last_token
   - 返回聚合结果

6. **`batch_decode_step(requests)`**:
   - **当前实现**: 串行处理每个 decode step
   - 调用现有 `decode_step()` 逐个处理
   - 检查完成条件 (max_new_tokens)
   - 返回 next_tokens 和 finished 标志

## 当前状态

### 编译测试中 🔄

**命令**: `make -j8 pocketllm_engine`  
**状态**: 后台运行中（timeout → background task）

**预期**:
- ✅ 语法正确（头文件、类型定义无误）
- ⚠️ 可能需要小调整（模板实例化、链接）

### Task 状态

| Task | 状态 | 进度 |
|------|------|------|
| #1 总体规划 | ✅ 完成 | 100% |
| #2 Per-request state | ✅ 完成 | 100% |
| #3 KV cache 重构 | 🔄 进行中 | 30% |
| #4 Batch API | 🔄 进行中 | 60% |
| #5 调度器 | ⏸️ 待开始 | 0% |
| #6 CppBackend 集成 | ⏸️ 待开始 | 0% |
| #7 测试 | ⏸️ 待开始 | 0% |
| #8 性能验证 | ⏸️ 待开始 | 0% |

## 待办事项

### 短期 (本次 session)

1. **完成编译验证** ✅
   - 检查编译输出
   - 修复任何编译错误

2. **实现 KV cache multi-slot 重分配** (Task #3 核心)
   - 修改 `allocate_batch_slots()` 中的 TODO
   - 重新分配所有层的 k_cache/v_cache 为 `[max_batch_size, max_seq_len, ...]`
   - 修改 attention kernel 调用添加 slot offset

3. **完善 batch API** (Task #4)
   - 在 `batch_prefill()` 中使用 slot_id
   - 在 `batch_decode_step()` 中使用 slot_id
   - 添加 EOS token 检查

### 中期 (下次 session)

4. **实现 FCFS 调度器** (Task #5)
   - 创建 `cpp_engine/core/simple_scheduler.hpp`
   - 实现非阻塞 `submit()` 和阻塞 `step()`
   - 线程安全的队列管理

5. **Python bindings** (Task #6)
   - 在 pybind11 中暴露 batch API
   - CppBackend 检测 `supports_batching()`
   - 启动后台调度器线程

6. **基础测试** (Task #7)
   - 单请求 parity 测试
   - 2/4 并发正确性测试

### 长期 (Phase 3 完成)

7. **性能优化**
   - 真正的 batched decode kernel（而非串行）
   - Prefix cache 跨请求共享
   - 内存池优化

8. **完整测试套件** (Task #7)
   - 8 并发压力测试
   - Token-by-token 一致性验证
   - 内存泄漏检查

9. **性能基准** (Task #8)
   - 1/2/4/8 并发吞吐测量
   - 与 baseline 对比
   - 验收标准检查

## 技术决策记录

### 1. 为什么 `batch_prefill()` 和 `batch_decode_step()` 当前是串行的？

**原因**:
- **正确性优先**: 先确保单请求路径不受影响
- **渐进式**: Phase 3.1 先建立框架，Phase 3.2 再优化 kernel
- **降低风险**: 避免同时改动 API 和 kernel 导致难以调试

**未来优化**:
- `batch_decode_step()`: 真正的 batched GQA attention kernel
- `batch_prefill()`: chunked mixed-length prefill

### 2. 为什么 `allocate_batch_slots()` 抛出异常？

**原因**:
- KV cache 重分配是复杂且风险高的操作
- 需要修改所有 64 层的 tensor shape
- 需要修改所有 attention kernel 调用点
- 先完成 API 框架和单 session 验证，再逐步实现

**实施计划**:
- Step 1: 在构造函数中支持 `max_batch_size` 参数（从环境变量读取）
- Step 2: 一次性分配 multi-slot KV cache
- Step 3: 修改 attention kernel 添加 slot offset

### 3. 单 session 零开销如何保证？

**实现**:
- `max_batch_size = 1` (默认) → 不触发重分配
- `batch_mode_enabled = false` → 跳过 slot 管理
- `allocate_slot()` 直接返回 0
- 现有 `prefill()`/`decode_step()` 内部调用保持不变

## 风险和缓解

### 风险 1: KV cache 重分配破坏现有功能

**当前状态**: 未实施，风险未触发  
**缓解措施**:
- 默认单 session 不重分配
- 先在 TP1 环境测试
- Token parity 验证

### 风险 2: 编译错误

**当前状态**: 编译中  
**预期问题**:
- 模板参数推导
- 头文件依赖顺序

**缓解措施**:
- 逐步编译测试
- 修复一个错误后重新编译

### 风险 3: 性能回归

**当前状态**: 未测试  
**缓解措施**:
- 单请求路径代码路径不变
- 性能门控：单请求 ≤1.05× baseline

## 验收标准

### Phase 3.1 最小可行产品 (MVP)

- [ ] **编译通过**: 无错误无警告
- [ ] **单 session 模式**: 默认行为与现有版本完全一致
- [ ] **单请求性能**: ≤1.05× baseline (8K prompt, TG128)
- [ ] **2 并发正确性**: 两个请求独立生成正确 tokens
- [ ] **API 完整性**: 所有声明的函数可调用

### Phase 3.1 完整目标

- [ ] **Multi-slot KV cache**: 支持 max_batch_size=8
- [ ] **Slot 管理**: allocate/free 正确无泄漏
- [ ] **Batch prefill**: 串行处理多个请求
- [ ] **Batch decode**: 串行处理多个 decode step
- [ ] **8 并发正确性**: Token-by-token parity
- [ ] **吞吐提升**: 2 并发 ≥1.8×, 4 并发 ≥3×

## 下一步行动

### 立即 (编译完成后)

1. ✅ 检查编译输出
2. 🔧 修复任何错误
3. ✅ 测试单 session 模式 (默认行为)
4. ✅ 验证 API 可调用性

### 今天内

5. 🔨 实现 KV cache multi-slot 重分配
6. 🧪 单请求 parity 测试
7. 📊 单请求性能基准

### 本周内

8. 🔄 实现调度器
9. 🐍 Python bindings
10. ✅ 2/4 并发测试

## 参考

- **实施文档**: `docs/cpp_engine_batching_phase3_1.md`
- **架构对比**: `docs/vllm_sglang_comparison.md`
- **性能基线**: TP2 8K: 1434 tok/s prefill, 30.9 tok/s decode
- **内存预算**: 22 GB/rank (2080 Ti)
- **目标硬件**: 4×2080 Ti, TP2/TP4

---

**更新**: 编译测试中，等待结果...
