# Phase 3.3 完成总结 - Slot ID Threading and Batch API Integration

**日期**: 2026-09-02  
**状态**: ✅ 完成并编译通过

---

## 🎉 主要成就

### 1. ✅ 消除全局状态

成功将 `slot_id` 从全局变量 (`current_slot_id`) 改为显式参数传递，KV cache 寻址不再依赖共享可变状态。
（引擎其余单 session 状态尚未 slot 化，见文末「已知限制」。）

**Before (Phase 3.2)**:
```cpp
struct QwenEngine::Impl {
    int current_slot_id = 0;  // Global state - not thread-safe!
};

void full_attention(...) {
    const int slot_id = current_slot_id;  // Read from global
    const size_t slot_offset = kv_slot_offset_elements(slot_id, ...);
}
```

**After (Phase 3.3)**:
```cpp
struct QwenEngine::Impl {
    // REMOVED: int current_slot_id = 0;
};

void full_attention(..., int slot_id) {  // Explicit parameter
    const size_t slot_offset = kv_slot_offset_elements(slot_id, ...);
}
```

### 2. ✅ 完整的参数传递链

实现了从顶层 API 到底层 kernel 的完整 `slot_id` 传递链：

```
batch_prefill/batch_decode_step (req->slot_id)
    ↓
prefill/decode_step (slot_id parameter, default 0)
    ↓
run_chunk (slot_id parameter, default 0)
    ↓
layer_forward (slot_id parameter)
    ↓
full_attention (slot_id parameter)
    ↓
KV cache kernels (slot_offset = f(slot_id))
```

### 3. ✅ 向后兼容的默认参数

使用默认参数保持单 session API 的向后兼容性：

```cpp
// Public API (qwen_engine.hpp)
QwenForwardResult prefill(const std::vector<int>& tokens, int slot_id = 0);
QwenForwardResult decode_step(int token, int slot_id = 0);

// Usage: Single-session mode
engine.prefill(tokens);           // slot_id=0 (implicit)
engine.decode_step(token);        // slot_id=0 (implicit)

// Usage: Multi-slot mode
engine.prefill(tokens, req->slot_id);      // explicit slot
engine.decode_step(token, req->slot_id);   // explicit slot
```

### 4. ✅ Batch API 集成

成功将 batch API 与 slot 管理机制集成：

```cpp
QwenBatchPrefillResult QwenEngine::batch_prefill(
    const std::vector<QwenBatchedRequest*>& requests) {
    
    for (QwenBatchedRequest* req : requests) {
        // Phase 3.3: Use req->slot_id to isolate KV cache
        QwenForwardResult fwd_result = prefill(req->prompt_tokens, req->slot_id);
        // ...
    }
}

QwenBatchDecodeResult QwenEngine::batch_decode_step(
    const std::vector<QwenBatchedRequest*>& requests) {
    
    for (QwenBatchedRequest* req : requests) {
        // Phase 3.3: Use req->slot_id to isolate KV cache
        QwenForwardResult fwd_result = decode_step(req->last_token, req->slot_id);
        // ...
    }
}
```

---

## 📊 实施统计

### 代码修改

| 文件 | 函数 | 修改类型 |
|------|------|---------|
| `qwen_engine.hpp` | `prefill()` | 添加 `slot_id` 参数 (default 0) |
| `qwen_engine.hpp` | `decode_step()` | 添加 `slot_id` 参数 (default 0) |
| `qwen_engine.cpp` | `prefill()` | 传递 `slot_id` 到 `run_chunk()` |
| `qwen_engine.cpp` | `decode_step()` | 传递 `slot_id` 到 `run_chunk()` |
| `qwen_engine.cpp` | `run_chunk()` | 添加 `slot_id` 参数 (default 0) |
| `qwen_engine.cpp` | `run_chunk()` | 传递 `slot_id` 到 `layer_forward()` |
| `qwen_engine.cpp` | `layer_forward()` | 添加 `slot_id` 参数 |
| `qwen_engine.cpp` | `layer_forward()` | 传递 `slot_id` 到 `full_attention()` |
| `qwen_engine.cpp` | `full_attention()` | 添加 `slot_id` 参数 |
| `qwen_engine.cpp` | `full_attention()` | 使用 `slot_id` 计算 offset |
| `qwen_engine.cpp` | `batch_prefill()` | 使用 `req->slot_id` |
| `qwen_engine.cpp` | `batch_decode_step()` | 使用 `req->slot_id` |
| `qwen_engine.cpp` | `Impl` | 移除 `current_slot_id` 全局状态 |
| `qwen_engine.cpp` | `mtp_forward_rows()` | 传递 `slot_id=0` |
| **总计** | **14 处修改** | |

### 参数传递链

| 层级 | 函数 | slot_id 来源 |
|------|------|-------------|
| 1 | `batch_prefill()` | `req->slot_id` |
| 1 | `batch_decode_step()` | `req->slot_id` |
| 2 | `prefill()` | 参数 (default 0) |
| 2 | `decode_step()` | 参数 (default 0) |
| 3 | `run_chunk()` | 参数 (default 0) |
| 4 | `layer_forward()` | 参数 |
| 5 | `full_attention()` | 参数 |
| 6 | KV cache kernels | `kv_slot_offset_elements(slot_id, ...)` |

---

## 🔍 设计决策

### 1. 显式参数传递 vs Thread-Local Storage

**选择**: 显式参数传递

**理由**:
- ✅ 调用链清晰，易于追踪和调试
- ✅ 线程安全，无隐藏状态
- ✅ 测试友好，每个调用显式指定 slot
- ✅ 符合 C++ 最佳实践（显式优于隐式）

**Trade-off**:
- ⚠️ 需要修改多个函数签名（一次性工作）
- ⚠️ 参数列表变长（可接受，只增加 1 个参数）

### 2. 默认参数 slot_id=0

**选择**: 在公共 API 使用默认参数 `slot_id=0`

**理由**:
- ✅ 保持向后兼容性（单 session 代码无需修改）
- ✅ 简化单 session 用法（不需要显式传递 0）
- ✅ 清晰的语义（默认使用 slot 0）

**适用范围**:
- `prefill(tokens, slot_id=0)`
- `decode_step(token, slot_id=0)`
- `run_chunk(..., slot_id=0)`

**不适用**:
- `layer_forward()` - 内部函数，必须显式传递
- `full_attention()` - 内部函数，必须显式传递

### 3. MTP 使用 slot_id=0

**选择**: MTP (Medusa-style Token Prediction) 始终使用 `slot_id=0`

**理由**:
- MTP 是单 session 特性（不支持 batch）
- 简化实现，避免引入不必要的复杂性

```cpp
// mtp_forward_rows()
layer_forward(mtp_layer, mtp_fused.f16_data(),
              mtp_next_hidden.f16_data(), rows, position, 0);  // slot_id=0
```

---

## 🧪 编译验证

### ✅ 编译成功

```bash
cd build && make -j8 pocketllm_engine
[100%] Built target pocketllm_engine
```

**结果**:
- ✅ 无编译错误
- ✅ 无警告
- ✅ 二进制生成成功

### 关键修复过程

**问题 1**: `full_attention()` 中 `slot_id` 未定义  
**修复**: 添加 `slot_id` 参数，从 `current_slot_id` 改为参数传递

**问题 2**: `layer_forward()` 调用缺少 `slot_id` 参数  
**修复**: 添加 `slot_id` 参数到函数签名，并更新所有调用点

**问题 3**: `mtp_forward_rows()` 调用缺少 `slot_id` 参数  
**修复**: 传递 `slot_id=0`（MTP 不支持 batch）

---

## 🎯 验收标准

### Phase 3.3 最小可行产品 (MVP)

- [x] ✅ **编译通过**: 无错误无警告
- [x] ✅ **全局状态移除**: `current_slot_id` 已删除
- [x] ✅ **参数传递链完整**: 从 batch API 到 kernel
- [x] ✅ **向后兼容**: 单 session API 无需修改
- [x] ✅ **Batch API 集成**: 使用 `req->slot_id`
- [ ] ⏳ **功能测试**: 单 session parity（下一步）
- [ ] ⏳ **Slot 隔离测试**: 不同 slot 互不干扰（下一步）

### Phase 3.3 完整目标

- [x] ✅ **代码完成**: 所有实现完成
- [x] ✅ **编译验证**: 成功通过
- [x] ✅ **KV cache 寻址无全局状态**: `slot_id` 全程显式
- [ ] ⏳ **单 session parity**: 与 Phase 3.2 一致
- [ ] ⏳ **2-slot 正确性**: 独立生成验证
- [ ] ⏳ **性能零回归**: ≤1.05× baseline

---

## ⚠️ 已知限制（Phase 3.3 未覆盖）

本阶段只做了 **KV cache 寻址** 的 slot 化，引擎仍有其他单 session 状态没有按 slot 拆分。
换句话说：**slot_id 让不同请求写到 KV cache 的不同区域，但还不足以让两个请求真正并行推进。**

| 共享状态 | 位置 | 影响 |
|---------|------|------|
| `position_` | `QwenEngine` 成员 | 所有 slot 共用一个位置计数器；交替 decode 两个请求会串位 |
| `cached_prompt` / `cached_result_` | `Impl` / `QwenEngine` | prefix cache 是单份，跨 slot 会互相污染 |
| `snapshots` | `Impl` | 快照不带 slot 维度 |
| workspace / `begin_workspace()` | `Impl` | 单份临时缓冲，串行执行下安全，真并发不安全 |

因此当前 `batch_prefill` / `batch_decode_step` **仍必须串行执行**（现有实现正是逐请求 for 循环），
"线程安全" 只对 `slot_id` 这一路参数成立，不代表整个引擎可并发调用。

Phase 3.4 之后需要把 `position_` 与 prefix cache 状态也做成 per-slot（例如 `std::vector<SlotState>`），
才能声明多请求并发正确。测试时对每个 slot 独立跑完整序列，不要交替 decode，否则失败来自
`position_` 而不是 KV cache 寻址。

---

## 📋 下一步工作

### Phase 3.4: 功能测试和验证

#### Test 1: 单 Session Parity

```python
def test_single_session_parity_phase33():
    """验证 Phase 3.3 与 Phase 3.2 结果一致"""
    engine = QwenEngine(config, options)
    
    prompt = [1, 2, 3, 4, 5]
    
    # 不传 slot_id（使用默认 0）
    result1 = engine.prefill(prompt)
    tokens1 = [result1.next_token]
    for _ in range(10):
        result1 = engine.decode_step(tokens1[-1])
        tokens1.append(result1.next_token)
    
    # 显式传 slot_id=0
    engine.reset()
    result2 = engine.prefill(prompt, 0)
    tokens2 = [result2.next_token]
    for _ in range(10):
        result2 = engine.decode_step(tokens2[-1], 0)
        tokens2.append(result2.next_token)
    
    # 应该完全一致
    assert tokens1 == tokens2
```

#### Test 2: Slot 隔离性

```python
def test_slot_isolation_phase33():
    """验证不同 slot 互不干扰"""
    engine = QwenEngine(config, options)
    engine.allocate_batch_slots(2)
    
    prompt_a = [1, 2, 3]
    prompt_b = [4, 5, 6]
    
    # 在 slot 0 中预填充
    result_a = engine.prefill(prompt_a, slot_id=0)
    
    # 在 slot 1 中预填充
    result_b = engine.prefill(prompt_b, slot_id=1)
    
    # 在 slot 0 中解码
    tokens_a = [result_a.next_token]
    for _ in range(5):
        result = engine.decode_step(tokens_a[-1], slot_id=0)
        tokens_a.append(result.next_token)
    
    # 在 slot 1 中解码
    tokens_b = [result_b.next_token]
    for _ in range(5):
        result = engine.decode_step(tokens_b[-1], slot_id=1)
        tokens_b.append(result.next_token)
    
    # slot 0 和 slot 1 应该生成不同的序列
    assert tokens_a != tokens_b
    
    # 验证 slot 0 的状态未被 slot 1 影响
    # 继续在 slot 0 中解码，应该延续之前的序列
    result = engine.decode_step(tokens_a[-1], slot_id=0)
    # ... 进一步验证
```

#### Test 3: Batch API 正确性

```python
def test_batch_api_phase33():
    """验证 batch API 使用正确的 slot_id"""
    engine = QwenEngine(config, options)
    engine.allocate_batch_slots(2)
    
    req1 = QwenBatchedRequest(
        request_id=1,
        prompt_tokens=[1, 2, 3],
        slot_id=engine.allocate_slot(1),
        sampling=default_sampling
    )
    
    req2 = QwenBatchedRequest(
        request_id=2,
        prompt_tokens=[4, 5, 6],
        slot_id=engine.allocate_slot(2),
        sampling=default_sampling
    )
    
    # Batch prefill
    result = engine.batch_prefill([req1, req2])
    assert len(result.results) == 2
    
    # Batch decode
    for _ in range(10):
        result = engine.batch_decode_step([req1, req2])
        assert len(result.next_tokens) == 2
    
    # 验证两个请求生成了不同的序列
    assert req1.generated_tokens != req2.generated_tokens
```

#### Test 4: 性能回归测试

```python
def test_performance_regression():
    """验证 Phase 3.3 性能无回归"""
    engine = QwenEngine(config, options)
    
    prompt = list(range(100))
    
    # Warmup
    for _ in range(5):
        engine.reset()
        engine.prefill(prompt)
    
    # Benchmark
    times = []
    for _ in range(20):
        engine.reset()
        start = time.time()
        engine.prefill(prompt)
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    
    # 应该 ≤ Phase 3.2 baseline 的 1.05×
    # (实际应该完全一致，因为默认参数没有运行时开销)
    assert avg_time <= baseline_time * 1.05
```

---

## 💡 技术亮点

### 1. 零开销的默认参数

```cpp
// 编译器优化
QwenForwardResult prefill(const std::vector<int>& tokens, int slot_id = 0);

// 调用：engine.prefill(tokens)
// 等价于：engine.prefill(tokens, 0)
// 编译后：直接传递常量 0，无额外开销
```

### 2. 清晰的调用链

显式参数传递使调用链一目了然：

```cpp
// 从 batch API 到 kernel，slot_id 清晰可见
batch_prefill(req) 
  → prefill(req->prompt_tokens, req->slot_id)
    → run_chunk(..., slot_id)
      → layer_forward(..., slot_id)
        → full_attention(..., slot_id)
          → kernel(..., ptr + kv_slot_offset_elements(slot_id, ...))
```

### 3. slot 寻址不再依赖共享状态

每个调用携带自己的 `slot_id`，KV cache 寻址路径上没有可变共享状态：

```cpp
// Phase 3.2 (slot 靠全局变量传递)
impl_->current_slot_id = 0;  // Global write
full_attention(...);          // Read from global

// Phase 3.3 (slot 显式传参)
full_attention(..., slot_id);  // Explicit parameter, no shared state
```

注意这只解决了 `slot_id` 这一路。`position_`、prefix cache 和 workspace 仍是单份共享状态，
所以引擎整体还不能并发调用（见「已知限制」）。

---

## 🔗 相关文档

- **实施计划**: `docs/phase3_3_implementation_plan.md`
- **Phase 3.2 总结**: `docs/phase3_2_completion_summary.md`
- **Phase 3.1 总结**: `docs/phase3_1_completion_summary.md`
- **总体架构**: `docs/cpp_engine_batching_phase3_1.md`

---

## ✅ 总结

Phase 3.3 成功实现了 **Slot ID Threading and Batch API Integration**：

1. ✅ **消除 slot 全局状态** - `current_slot_id` 已移除，寻址路径无共享可变状态
2. ✅ **完整参数传递链** - 从 batch API 到 kernel 清晰可追踪
3. ✅ **向后兼容** - 单 session API 无需修改，默认参数 `slot_id=0`
4. ✅ **Batch API 集成** - 使用 `req->slot_id` 隔离 KV cache
5. ✅ **编译成功** - 无错误，可继续迭代

**下一步**: Phase 3.4 实施功能测试，验证正确性和性能。

---

**编译输出**:
```
[100%] Built target pocketllm_engine
```

**代码状态**: 已完成，待测试 ✅
