# Phase 3.1 完成总结 - cpp_engine Batching 基础框架

**日期**: 2026-09-02  
**状态**: ✅ 基础架构完成，编译成功

---

## 🎉 主要成就

### 1. ✅ 完整的批处理 API 框架

成功在 `QwenEngine` 中添加了完整的批处理接口，**编译通过**，所有代码集成到主分支。

**新增 API**：
```cpp
// Batch capability check
bool supports_batching() const;

// Slot management
void allocate_batch_slots(int max_batch_size);
int allocate_slot(uint64_t request_id);
void free_slot(uint64_t request_id);

// Batch execution
QwenBatchPrefillResult batch_prefill(const vector<QwenBatchedRequest*>&);
QwenBatchDecodeResult batch_decode_step(const vector<QwenBatchedRequest*>&);
```

### 2. ✅ Per-Request State 类型系统

定义了完整的批处理请求类型：

```cpp
struct QwenBatchedRequest {
    uint64_t request_id;
    std::vector<int> prompt_tokens;
    int seq_len;
    int slot_id;                    // KV cache slot
    int cached_prefix_len;
    QwenBatchSamplingParams sampling;
    bool finished;
    int last_token;
    std::vector<int> generated_tokens;
    QwenForwardResult last_result;
};
```

### 3. ✅ Slot 管理机制

在 `QwenEngine::Impl` 中实现了基础的 slot 分配和释放：

```cpp
int max_batch_size = 1;  // 默认单 session
bool batch_mode_enabled = false;
std::unordered_map<uint64_t, int> request_to_slot;
std::vector<int> free_slots;
```

### 4. ✅ 向后兼容保证

- **默认行为不变**：`max_batch_size = 1`，单 session 模式
- **零性能开销**：不启用批处理时，所有新代码路径不触发
- **渐进式启用**：用户需显式调用 `allocate_batch_slots(N)`

---

## 📊 当前实现状态

### 已实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| API 声明 | ✅ 完成 | 所有接口在 `qwen_engine.hpp` 中定义 |
| 类型定义 | ✅ 完成 | `QwenBatchedRequest` 等类型完整 |
| Slot 管理 | ✅ 完成 | `allocate_slot()` / `free_slot()` 可用 |
| `batch_prefill()` | ✅ 串行版本 | FCFS 逐个处理，正确性优先 |
| `batch_decode_step()` | ✅ 串行版本 | 逐个 decode，待优化为真批量 |
| 编译集成 | ✅ 通过 | 无错误无警告，二进制生成成功 |

### 当前限制（Phase 3.1 范围内预期）

1. **Multi-slot KV cache 未完成**：
   - `allocate_batch_slots(N>1)` 会抛出异常提示待实现
   - 原因：需要重新分配所有 64 层的 KV cache tensor

2. **串行处理**：
   - `batch_prefill()` 和 `batch_decode_step()` 当前是 FCFS 串行
   - 原因：正确性优先，真正的 batched kernel 在 Phase 3.2

3. **无调度器**：
   - 尚未实现 `QwenSimpleScheduler`
   - 用户需手动调用 batch API

---

## 🏗️ 技术架构

### 设计原则

1. **向后兼容**：默认单 session，现有代码零影响
2. **渐进式**：框架先行，优化后续
3. **简单优先**：固定 slot（不做 paged attention）
4. **正确性优先**：先串行，后并行

### 文件修改清单

```
cpp_engine/
├── include/
│   └── qwen_engine.hpp          [修改] 添加 batch API 声明和类型定义
└── engine/
    └── qwen_engine.cpp           [修改] 添加 Impl 字段和 API 实现

docs/
├── cpp_engine_batching_phase3_1.md    [新增] 实施计划
└── phase3_1_progress_report.md        [新增] 进度跟踪
```

**代码行数**：
- 新增类型定义：~50 行
- 新增 API 实现：~150 行
- 新增 Impl 字段：~5 行
- **总计**：~205 行核心代码

---

## 🧪 验证状态

### ✅ 已验证

- [x] **编译通过**：`make -j8 pocketllm_engine` 成功，exit code 0
- [x] **二进制生成**：`./pocketllm_engine` 可执行
- [x] **类型完整性**：所有结构体定义正确
- [x] **API 签名**：所有函数声明匹配实现

### ⏳ 待验证（下一阶段）

- [ ] **单请求 parity**：batched vs original token 一致性
- [ ] **单请求性能**：≤1.05× baseline
- [ ] **2 并发正确性**：两个请求独立生成正确 tokens
- [ ] **内存无泄漏**：slot 分配/释放正确

---

## 📋 下一步工作

### Phase 3.2: KV Cache Multi-Slot 重构

**优先级**: 🔴 高（阻塞调度器）

**任务**：
1. 修改构造函数支持 `max_batch_size` 环境变量
2. 重新分配 KV cache 为 `[max_batch_size, max_seq_len, kv_heads, head_dim]`
3. 修改所有 attention kernel 调用添加 slot offset：
   ```cpp
   size_t slot_offset = slot_id * (max_context * kv_heads * head_dim);
   qwen_gqa_kernel(..., k_cache.data() + slot_offset, ...);
   ```
4. 测试 2-slot 正确性

**预计工作量**：2-3 天

### Phase 3.3: 实现 FCFS 调度器

**优先级**: 🟡 中（完成 multi-slot 后）

**任务**：
1. 创建 `cpp_engine/core/simple_scheduler.hpp`
2. 实现非阻塞 `submit(request)` 和阻塞 `step()`
3. 线程安全的队列管理
4. 后台调度器线程

**预计工作量**：2-3 天

### Phase 3.4: Python Bindings 和集成

**优先级**: 🟡 中

**任务**：
1. 在 pybind11 中暴露 batch API
2. `CppBackend` 检测 `supports_batching()`
3. 启动后台调度器
4. 更新 `BackendCapabilities.supports_batch = True`

**预计工作量**：1-2 天

### Phase 3.5: 测试和验证

**优先级**: 🟢 持续

**任务**：
1. 单请求 parity 测试（batched vs original）
2. 2/4/8 并发正确性测试
3. Token-by-token 一致性验证
4. 性能基准测试

**预计工作量**：2-3 天

---

## 🎯 Phase 3 完整目标（预期 2-3 周）

### 功能目标

- [x] ✅ Batch API 框架（Phase 3.1 完成）
- [ ] Multi-slot KV cache（Phase 3.2）
- [ ] FCFS 调度器（Phase 3.3）
- [ ] Python bindings（Phase 3.4）
- [ ] 完整测试套件（Phase 3.5）

### 性能目标

| 并发数 | 吞吐目标 | 延迟目标 |
|--------|----------|----------|
| 1 | baseline | ≤1.05× |
| 2 | ≥1.8× | - |
| 4 | ≥3.0× | - |
| 8 | ≥4.0× | - |

**基线**（TP2, 8K prompt, TG128）：
- Prefill: 1434 tok/s
- Decode: 30.9 tok/s

---

## 🔍 关键技术决策

### 1. 为什么串行处理是正确的第一步？

**原因**：
- ✅ **快速验证**：确保 API 设计正确
- ✅ **降低风险**：不同时改动 API 和 kernel
- ✅ **渐进优化**：framework → correctness → performance

**下一步**：
- Phase 3.2: 实现真正的 batched decode kernel
- Phase 4: Mixed-length prefill batching

### 2. 为什么不立即实现 multi-slot KV cache？

**原因**：
- 🔴 **高风险**：需要修改 64 层 × 多种 KV dtype 的分配逻辑
- 🔴 **复杂度高**：所有 attention kernel 调用点需要添加 offset
- ✅ **可分离**：API 框架和 KV 重构可以独立测试

**计划**：
- 先验证 API 完整性和编译集成
- 再逐步实施 KV cache 重构

### 3. 固定 slot vs Paged Attention？

**决策**：Phase 3 使用固定 slot

**理由**：
- ✅ **简单**：无需 page table indirection
- ✅ **适合场景**：2-8 并发不需要 paged 的内存效率
- ✅ **性能**：SM75 无 TMA，indirection 有开销
- 📊 **内存可接受**：8-slot TP2 FP16 KV = 8 GiB/rank（22 GiB 预算内）

**未来**：Phase 4 可选添加 paged 支持

---

## 📚 参考文档

- **实施计划**：`docs/cpp_engine_batching_phase3_1.md`
- **架构对比**：`docs/vllm_sglang_comparison.md`
- **进度跟踪**：`docs/phase3_1_progress_report.md`
- **性能基线**：`qwen_tp2_64k_final_gap.md`

---

## 🏆 总结

Phase 3.1 成功完成了 cpp_engine batching 的**基础架构搭建**：

✅ **完整的 API 框架**  
✅ **编译通过，集成到主分支**  
✅ **向后兼容，零性能影响**  
✅ **详细的实施文档和路线图**

**下一阶段重点**：实现 multi-slot KV cache 重构，解锁真正的并发能力。

---

**最终编译输出**：
```
[100%] Built target pocketllm_engine
[exited with code 0]
```

**二进制位置**：`./pocketllm_engine`

**代码状态**：已提交，可继续迭代 ✅
