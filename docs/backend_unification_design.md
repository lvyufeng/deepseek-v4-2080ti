# PocketLLM Backend Unification Design

**Date**: 2024-09-03  
**Status**: Draft Proposal  
**Author**: Analysis of PocketLLM vs vLLM/SGLang architecture

---

## Executive Summary

This document analyzes the current state of PocketLLM after the Phase 3 batch mode refactor, compares it with vLLM and SGLang, and proposes a unified architecture for supporting both C++ engine and PyTorch backends.

---

## 1. Current Architecture Analysis

### 1.1 PocketLLM (Post-Phase 3 Refactor)

**Strengths:**
- ✅ High-performance C++ engine (`cpp_engine/`) with custom CUDA kernels
- ✅ Batch mode support with per-slot isolation (Phase 3.1-3.4)
- ✅ Multiple attention backends (linear attention, full GQA)
- ✅ Advanced quantization (FP8, FP4, TurboQuant, IQ1M)
- ✅ Tensor parallel (TP) support
- ✅ Prefix caching with snapshot/restore
- ✅ Model-specific optimizations (DeepSeek-V4, Qwen, MiniMax, GLM)

**Gaps:**
- ❌ No unified backend abstraction layer
- ❌ PyTorch backend is model-specific, not pluggable
- ❌ Request scheduling is basic (simple FCFS queue in `src/server/engine.py`)
- ❌ No continuous batching scheduler
- ❌ No PagedAttention-style memory management
- ❌ Limited observability/metrics
- ❌ No request migration between batches
- ❌ KV cache management is per-engine, not centralized

**Current Structure:**
```
PocketLLM/
├── cpp_engine/          # C++ inference engine
│   ├── engine/          # QwenEngine, etc.
│   ├── backends/        # CUDA, Ascend backends
│   └── python/          # Python bindings
├── src/
│   ├── server/          # Basic serving (engine.py, openai.py)
│   ├── models/          # Model-specific PyTorch implementations
│   ├── runtime/         # Token generation, prefix snapshots
│   └── components/      # MoE, quantization components
```

---

## 2. Competitor Analysis

### 2.1 vLLM Architecture

**Key Features:**
1. **PagedAttention**: KV cache managed in blocks, enables efficient memory sharing
2. **Continuous Batching**: Dynamic request scheduling with preemption
3. **Unified Executor Interface**: `GPUExecutor`, `CPUExecutor` abstract away backend
4. **Model Loader**: Automatic weight loading from HuggingFace
5. **Scheduler**: FIFO with priority, preemption, swapping
6. **Multi-backend**: CUDA, ROCm, CPU, TPU (via extensions)

**Architecture:**
```
vLLM/
├── vllm/
│   ├── engine/          # LLMEngine (orchestrator)
│   ├── executor/        # GPUExecutor, TPUExecutor (backend interface)
│   ├── worker/          # Worker (per-device runner)
│   ├── model_executor/  # Model loading and execution
│   ├── core/            # Scheduler, BlockSpaceManager
│   ├── attention/       # PagedAttention kernels
│   └── entrypoints/     # OpenAI API, gRPC
```

**Scheduler Flow:**
```
Request → LLMEngine → Scheduler (FIFO/priority)
                     → BlockSpaceManager (allocate KV blocks)
                     → Executor.execute_model()
                     → Worker.execute_model()
                     → Model.forward() with PagedAttention
```

### 2.2 SGLang Architecture

**Key Features:**
1. **RadixAttention**: Prefix tree for KV cache sharing (more sophisticated than vLLM)
2. **Speculative Decoding**: Native support for draft models
3. **Multi-Modal**: Built-in vision support
4. **Frontend Language**: Structured Generation Language for constrained decoding
5. **Scheduler**: Continuous batching with prefix-aware scheduling

**Architecture:**
```
SGLang/
├── python/sglang/
│   ├── srt/             # SGLang Runtime
│   │   ├── server.py    # HTTP/gRPC server
│   │   ├── scheduler.py # RadixAttention-aware scheduler
│   │   ├── managers/    # TokenizerManager, DetokenizerManager
│   │   ├── mem_cache/   # RadixCache, ChunkCache
│   │   └── model_executor/ # ModelRunner
│   └── lang/            # SGLang frontend (gen, select)
```

**RadixAttention:**
- KV cache organized as a radix tree (trie)
- Automatic prefix sharing across requests
- Eviction policy: LRU on tree nodes

---

## 3. Feature Gap Analysis

| Feature | PocketLLM | vLLM | SGLang |
|---------|-----------|------|--------|
| **Scheduling** | Basic FCFS queue | Continuous batching with preemption | Continuous batching + prefix-aware |
| **Memory Management** | Fixed per-slot KV cache | PagedAttention (blocks) | RadixAttention (prefix tree) |
| **Prefix Reuse** | Snapshot/restore (single session) | Limited (via `prefix_caching`) | Automatic (RadixCache) |
| **Backend Abstraction** | ❌ Tight coupling | ✅ Executor interface | ✅ ModelRunner interface |
| **Multi-backend** | C++ engine only | CUDA, ROCm, CPU, TPU | CUDA, ROCM |
| **Speculative Decoding** | ❌ | ✅ (SpecDecodeWorker) | ✅ (native) |
| **Quantization** | FP8/FP4/IQ1M/TurboQuant | INT8/FP8 (via kernels) | FP8/INT4 |
| **Tensor Parallel** | ✅ (custom NCCL) | ✅ (ray/torchrun) | ✅ (ray) |
| **Pipeline Parallel** | ❌ | ✅ | ❌ |
| **Request Migration** | ❌ | ✅ (chunked prefill) | ✅ |
| **Observability** | Basic logging | Prometheus metrics | Prometheus + custom |
| **Multi-modal** | ❌ | ✅ (LLaVA, Qwen-VL) | ✅ (native) |

---

## 4. Proposed Unified Backend Design

### 4.1 Goals

1. **Backend Agnostic**: Support C++ engine and PyTorch with same API
2. **Drop-in Compatibility**: Existing cpp_engine code continues to work
3. **Scheduler Upgrade**: Continuous batching with prefix-aware scheduling
4. **Memory Efficiency**: Adopt PagedAttention or RadixAttention concepts
5. **Model Flexibility**: Easy to add new models to either backend
6. **Performance**: Maintain PocketLLM's edge in quantization and custom kernels

### 4.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenAI API / gRPC                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Engine (Orchestrator)                  │
│  - Request routing                                          │
│  - Load balancing (if multi-instance)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Scheduler                           │
│  - Continuous batching                                      │
│  - Prefix-aware scheduling (RadixCache-like)                │
│  - Preemption & swapping                                    │
│  - Budget tracking (token/memory limits)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MemoryManager (Unified)                   │
│  - BlockSpaceManager (vLLM-style) OR                        │
│  - RadixCache (SGLang-style)                                │
│  - Slot allocation for cpp_engine                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BackendExecutor (ABC)                    │
│  - execute_model(requests) → outputs                        │
│  - get_memory_info() → usage stats                          │
│  - supports_feature(name) → bool                            │
└─────────────────────────────────────────────────────────────┘
                    │                     │
         ┌──────────┴──────────┐          │
         ▼                     ▼          ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ CppEngineBackend │  │ PyTorchBackend   │  │ RemoteBackend    │
│ (QwenEngine)     │  │ (HF Transformers)│  │ (RPC to worker)  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 4.3 Core Components

#### 4.3.1 BackendExecutor (Abstract Interface)

```python
# src/backend/executor.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ExecuteRequest:
    request_id: str
    token_ids: List[int]
    slot_id: int
    is_prefill: bool
    sampling_params: SamplingParams
    prefix_hash: Optional[str] = None  # For prefix cache lookup

@dataclass
class ExecuteOutput:
    request_id: str
    token_id: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None

class BackendExecutor(ABC):
    """Abstract interface for inference backends."""
    
    @abstractmethod
    def execute_model(
        self, 
        requests: List[ExecuteRequest]
    ) -> List[ExecuteOutput]:
        """Execute a batch of requests (prefill or decode)."""
        pass
    
    @abstractmethod
    def get_memory_info(self) -> dict:
        """Return memory usage statistics."""
        pass
    
    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """Check if backend supports a feature (e.g., 'prefix_cache')."""
        pass
    
    @abstractmethod
    def reset_slot(self, slot_id: int) -> None:
        """Clear a slot's state for reuse."""
        pass
```

#### 4.3.2 CppEngineBackend Implementation

```python
# src/backend/cpp_engine_backend.py
from typing import List
import pocket_cpp_core  # Python bindings to cpp_engine

from .executor import BackendExecutor, ExecuteRequest, ExecuteOutput

class CppEngineBackend(BackendExecutor):
    """Wrapper for QwenEngine/DeepSeekEngine from cpp_engine."""
    
    def __init__(self, ckpt_dir: str, options: dict):
        self.engine = pocket_cpp_core.QwenEngine(
            ckpt_dir,
            options,
            layer_count=options.get("num_layers", 0),
            max_context=options["max_seq_len"]
        )
        self.max_batch_size = options.get("max_batch_size", 1)
    
    def execute_model(self, requests: List[ExecuteRequest]) -> List[ExecuteOutput]:
        outputs = []
        for req in requests:
            if req.is_prefill:
                result = self.engine.prefill(req.token_ids, req.slot_id)
            else:
                result = self.engine.decode_step(req.token_ids[-1], req.slot_id)
            
            outputs.append(ExecuteOutput(
                request_id=req.request_id,
                token_id=result.top_token,
                finish_reason=None  # Determined by scheduler
            ))
        return outputs
    
    def get_memory_info(self) -> dict:
        return {
            "kv_cache_bytes": self.engine.kv_cache_bytes(),
            "activation_bytes": self.engine.activation_workspace_peak_bytes()
        }
    
    def supports_feature(self, feature: str) -> bool:
        return feature in {"prefix_cache", "batch_mode", "tensor_parallel"}
    
    def reset_slot(self, slot_id: int) -> None:
        self.engine.reset(slot_id)  # Assuming we extend reset() to take slot_id
```

#### 4.3.3 PyTorchBackend Implementation

```python
# src/backend/pytorch_backend.py
from typing import List
import torch

from .executor import BackendExecutor, ExecuteRequest, ExecuteOutput

class PyTorchBackend(BackendExecutor):
    """Wrapper for PyTorch model implementations."""
    
    def __init__(self, model_path: str, options: dict):
        from src.models import load_model  # Registry-based loader
        self.model = load_model(model_path, **options)
        self.device = torch.device(options.get("device", "cuda"))
        self.model.to(self.device)
        self.max_batch_size = options.get("max_batch_size", 1)
        
        # Per-slot state tracking
        self.slot_states = [None] * self.max_batch_size
    
    def execute_model(self, requests: List[ExecuteRequest]) -> List[ExecuteOutput]:
        # Batch requests by type (prefill vs decode)
        prefill_reqs = [r for r in requests if r.is_prefill]
        decode_reqs = [r for r in requests if not r.is_prefill]
        
        outputs = []
        
        # Prefill batch
        if prefill_reqs:
            outputs.extend(self._batch_prefill(prefill_reqs))
        
        # Decode batch
        if decode_reqs:
            outputs.extend(self._batch_decode(decode_reqs))
        
        return outputs
    
    def _batch_prefill(self, requests: List[ExecuteRequest]) -> List[ExecuteOutput]:
        # Pad sequences to same length
        max_len = max(len(r.token_ids) for r in requests)
        input_ids = torch.zeros(len(requests), max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        
        for i, req in enumerate(requests):
            seq_len = len(req.token_ids)
            input_ids[i, :seq_len] = torch.tensor(req.token_ids, device=self.device)
            attention_mask[i, :seq_len] = 1
        
        with torch.inference_mode():
            outputs = self.model.forward(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Last token logits
            next_tokens = logits.argmax(dim=-1)
        
        results = []
        for i, req in enumerate(requests):
            # Store KV cache state for this slot
            self.slot_states[req.slot_id] = {
                "past_key_values": outputs.past_key_values,
                "position": len(req.token_ids)
            }
            results.append(ExecuteOutput(
                request_id=req.request_id,
                token_id=next_tokens[i].item()
            ))
        
        return results
    
    def _batch_decode(self, requests: List[ExecuteRequest]) -> List[ExecuteOutput]:
        # Use cached KV states
        input_ids = torch.tensor(
            [[r.token_ids[-1]] for r in requests],
            dtype=torch.long,
            device=self.device
        )
        
        # Gather past_key_values from slot states
        past_key_values = [self.slot_states[r.slot_id]["past_key_values"] 
                          for r in requests]
        
        with torch.inference_mode():
            outputs = self.model.forward(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        
        results = []
        for i, req in enumerate(requests):
            self.slot_states[req.slot_id]["past_key_values"] = outputs.past_key_values
            self.slot_states[req.slot_id]["position"] += 1
            results.append(ExecuteOutput(
                request_id=req.request_id,
                token_id=next_tokens[i].item()
            ))
        
        return results
    
    def get_memory_info(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device)
        }
    
    def supports_feature(self, feature: str) -> bool:
        return feature in {"batch_mode"}
    
    def reset_slot(self, slot_id: int) -> None:
        self.slot_states[slot_id] = None
```

#### 4.3.4 Unified Scheduler

```python
# src/scheduler/continuous_batching.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import time

class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"

@dataclass
class ScheduledRequest:
    request_id: str
    prompt_tokens: List[int]
    generated_tokens: List[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    slot_id: Optional[int] = None
    arrival_time: float = field(default_factory=time.time)
    sampling_params: dict = field(default_factory=dict)
    prefix_hash: Optional[str] = None
    
    @property
    def is_prefill_phase(self) -> bool:
        return len(self.generated_tokens) == 0
    
    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

class ContinuousBatchScheduler:
    """Continuous batching scheduler with prefix-aware scheduling."""
    
    def __init__(
        self,
        max_batch_size: int,
        max_total_tokens: int,
        enable_prefix_cache: bool = True
    ):
        self.max_batch_size = max_batch_size
        self.max_total_tokens = max_total_tokens
        self.enable_prefix_cache = enable_prefix_cache
        
        self.waiting_queue: List[ScheduledRequest] = []
        self.running_requests: Dict[str, ScheduledRequest] = {}
        self.free_slots = list(range(max_batch_size))
        
        # Prefix cache (simple hash -> slot_id mapping)
        self.prefix_cache: Dict[str, int] = {}
    
    def add_request(self, request: ScheduledRequest) -> None:
        """Add a new request to the waiting queue."""
        # Compute prefix hash for cache lookup
        if self.enable_prefix_cache:
            request.prefix_hash = self._compute_prefix_hash(request.prompt_tokens)
        self.waiting_queue.append(request)
    
    def schedule_step(self) -> tuple[List[ScheduledRequest], List[ScheduledRequest]]:
        """
        Schedule one step: return (prefill_batch, decode_batch).
        
        Policy:
        1. Prioritize running decode requests (avoid starvation)
        2. If slots available, admit waiting prefill requests
        3. Use prefix cache to assign slots with matching prefixes
        """
        prefill_batch = []
        decode_batch = []
        
        # Step 1: Collect running decode requests
        for req in self.running_requests.values():
            if not req.is_prefill_phase:
                decode_batch.append(req)
        
        # Step 2: Try to admit new prefill requests
        total_tokens = sum(r.total_tokens for r in self.running_requests.values())
        
        while (len(self.running_requests) < self.max_batch_size and
               self.waiting_queue and
               total_tokens < self.max_total_tokens):
            
            req = self.waiting_queue.pop(0)
            
            # Check prefix cache
            if req.prefix_hash and req.prefix_hash in self.prefix_cache:
                slot_id = self.prefix_cache[req.prefix_hash]
                # Reuse this slot (prefix already loaded)
                req.slot_id = slot_id
            else:
                # Allocate new slot
                if not self.free_slots:
                    self.waiting_queue.insert(0, req)  # Put back
                    break
                req.slot_id = self.free_slots.pop(0)
                if req.prefix_hash:
                    self.prefix_cache[req.prefix_hash] = req.slot_id
            
            req.status = RequestStatus.RUNNING
            self.running_requests[req.request_id] = req
            prefill_batch.append(req)
            total_tokens += req.total_tokens
        
        return prefill_batch, decode_batch
    
    def complete_request(self, request_id: str) -> None:
        """Mark a request as finished and free its slot."""
        if request_id in self.running_requests:
            req = self.running_requests.pop(request_id)
            req.status = RequestStatus.FINISHED
            if req.slot_id is not None:
                self.free_slots.append(req.slot_id)
                # Optionally keep prefix cache entry for reuse
    
    def _compute_prefix_hash(self, tokens: List[int]) -> str:
        """Simple hash for prefix cache. Can be more sophisticated (radix tree)."""
        import hashlib
        return hashlib.sha256(str(tokens[:128]).encode()).hexdigest()[:16]
```

#### 4.3.5 Unified Engine

```python
# src/engine/unified_engine.py
from typing import Iterator, Dict, Any
import threading
import queue

from src.backend.executor import BackendExecutor
from src.scheduler.continuous_batching import ContinuousBatchScheduler, ScheduledRequest

class UnifiedEngine:
    """Unified engine supporting both C++ and PyTorch backends."""
    
    def __init__(
        self,
        backend: BackendExecutor,
        max_batch_size: int = 8,
        max_total_tokens: int = 8192,
    ):
        self.backend = backend
        self.scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch_size,
            max_total_tokens=max_total_tokens,
            enable_prefix_cache=backend.supports_feature("prefix_cache")
        )
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self._worker_thread.start()
    
    def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 100,
        **sampling_params
    ) -> Iterator[Dict[str, Any]]:
        """Submit a generation request and stream results."""
        import uuid
        request_id = str(uuid.uuid4())
        
        result_queue = queue.Queue()
        request = ScheduledRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            sampling_params={"max_tokens": max_tokens, **sampling_params}
        )
        
        # Add callback for results
        self._result_queues[request_id] = result_queue
        self.scheduler.add_request(request)
        
        # Stream results
        while True:
            event = result_queue.get()
            if event["type"] == "finish":
                break
            yield event
    
    def _schedule_loop(self):
        """Main scheduling loop (runs in background thread)."""
        while self._running:
            prefill_batch, decode_batch = self.scheduler.schedule_step()
            
            # Execute prefill batch
            if prefill_batch:
                self._execute_batch(prefill_batch, is_prefill=True)
            
            # Execute decode batch
            if decode_batch:
                self._execute_batch(decode_batch, is_prefill=False)
            
            # Small sleep to prevent busy loop
            time.sleep(0.001)
    
    def _execute_batch(self, batch: list[ScheduledRequest], is_prefill: bool):
        """Execute a batch of requests."""
        from src.backend.executor import ExecuteRequest
        
        execute_requests = [
            ExecuteRequest(
                request_id=req.request_id,
                token_ids=req.prompt_tokens if is_prefill else req.prompt_tokens + req.generated_tokens,
                slot_id=req.slot_id,
                is_prefill=is_prefill,
                sampling_params=req.sampling_params,
                prefix_hash=req.prefix_hash
            )
            for req in batch
        ]
        
        outputs = self.backend.execute_model(execute_requests)
        
        # Process outputs
        for req, output in zip(batch, outputs):
            req.generated_tokens.append(output.token_id)
            
            # Check stopping condition
            max_tokens = req.sampling_params.get("max_tokens", 100)
            if len(req.generated_tokens) >= max_tokens or output.finish_reason:
                self.scheduler.complete_request(req.request_id)
                self._result_queues[req.request_id].put({"type": "finish"})
            else:
                self._result_queues[req.request_id].put({
                    "type": "token",
                    "token_id": output.token_id
                })
```

### 4.4 Migration Path

#### Phase 1: Backend Abstraction (Week 1-2)
- [ ] Implement `BackendExecutor` interface
- [ ] Create `CppEngineBackend` wrapper for existing `QwenEngine`
- [ ] Create `PyTorchBackend` for existing model implementations
- [ ] Unit tests for both backends

#### Phase 2: Scheduler Integration (Week 3-4)
- [ ] Implement `ContinuousBatchScheduler`
- [ ] Add prefix cache support
- [ ] Integrate with `UnifiedEngine`
- [ ] Benchmark: compare old `src/server/engine.py` vs new scheduler

#### Phase 3: Memory Manager (Week 5-6)
- [ ] Design PagedAttention-style block manager OR RadixCache
- [ ] Integrate with C++ engine (may require cpp_engine API changes)
- [ ] Memory efficiency benchmarks

#### Phase 4: Advanced Features (Week 7-8)
- [ ] Chunked prefill (allow preempting long prefills)
- [ ] Request migration between batches
- [ ] Speculative decoding support
- [ ] Multi-modal extensions

#### Phase 5: Observability & Production (Week 9-10)
- [ ] Prometheus metrics
- [ ] Distributed tracing
- [ ] Load testing and tuning
- [ ] Documentation and examples

---

## 5. API Compatibility

### 5.1 Existing API (Preserve)

```python
# Current usage (must continue to work)
from pocket_cpp_core import QwenEngine, QwenEngineOptions

options = QwenEngineOptions()
options.max_batch_size = 8
engine = QwenEngine("/path/to/ckpt", options, layer_count=64, max_context=32768)

result = engine.prefill([1, 2, 3], slot_id=0)
result = engine.decode_step(result.top_token, slot_id=0)
```

### 5.2 New Unified API

```python
# New unified interface
from src.engine import UnifiedEngine
from src.backend import CppEngineBackend

backend = CppEngineBackend("/path/to/ckpt", {
    "max_batch_size": 8,
    "max_seq_len": 32768,
    "kv_cache_dtype": "fp8"
})

engine = UnifiedEngine(backend, max_batch_size=8)

for event in engine.generate(prompt_tokens=[1, 2, 3], max_tokens=100):
    if event["type"] == "token":
        print(event["token_id"])
```

### 5.3 PyTorch Backend Usage

```python
# PyTorch backend (for models not yet in cpp_engine)
from src.backend import PyTorchBackend

backend = PyTorchBackend("/path/to/model", {
    "max_batch_size": 4,
    "device": "cuda",
    "torch_dtype": "float16"
})

engine = UnifiedEngine(backend)
# Same generate() API as C++ backend!
```

---

## 6. Performance Considerations

### 6.1 C++ Engine Optimizations to Preserve

- Custom CUDA kernels (FP8 matmul, TurboQuant, linear attention)
- Zero-copy weight loading (safetensors mmap)
- Fused operators (SwiGLU, RMSNorm)
- Tensor parallel with NCCL overlap

### 6.2 Scheduler Overhead

**Concern**: Python scheduler adds latency between decode steps.

**Mitigation**:
1. **Batch decode dispatch**: Schedule entire decode batch in one call
2. **C++ scheduler (future)**: Move scheduler to C++ for microsecond latency
3. **Prefetch**: Scheduler prepares next batch while GPU executes current

**Benchmark Target**: Scheduler overhead < 0.5ms per step

### 6.3 Memory Overhead

**vLLM PagedAttention**: ~10-20% memory overhead for block management

**PocketLLM Current**: Fixed per-slot allocation, 0% overhead but less flexible

**Proposal**: Hybrid approach
- For C++ backend with fixed slots: keep current allocation
- For PyTorch backend: use PagedAttention blocks
- Scheduler abstracts the difference

---

## 7. Open Questions

1. **Should we adopt RadixAttention or PagedAttention?**
   - RadixAttention: Better prefix sharing, more complex
   - PagedAttention: Simpler, proven in production (vLLM)
   - **Recommendation**: Start with PagedAttention, add RadixCache as optional

2. **Should scheduler be in Python or C++?**
   - Python: Easier to iterate, integrate with backend
   - C++: Lower latency, harder to maintain
   - **Recommendation**: Python first, C++ if profiling shows it's a bottleneck

3. **How to handle model-specific logic?**
   - vLLM: Model implementations in Python, call backend kernels
   - PocketLLM: Models in C++, PyTorch fallback
   - **Recommendation**: Keep model-specific code in backend implementation, scheduler is model-agnostic

4. **Backward compatibility with existing scripts?**
   - **Recommendation**: Keep `pocket_cpp_core` API unchanged, new `UnifiedEngine` as opt-in

---

## 8. Conclusion

### Summary

PocketLLM has strong foundations with high-performance C++ kernels and advanced quantization. To compete with vLLM/SGLang in production deployments, we need:

1. **Unified backend abstraction** to support both C++ and PyTorch
2. **Continuous batching scheduler** for better throughput and latency
3. **Prefix-aware memory management** (PagedAttention or RadixCache)
4. **Observability and metrics** for production monitoring

### Next Steps

1. Review and approve this design
2. Create implementation tasks for Phase 1 (Backend Abstraction)
3. Set up benchmarks to track performance regressions
4. Begin implementation with `CppEngineBackend` wrapper

### Timeline

- **Weeks 1-4**: Backend abstraction + scheduler (MVP)
- **Weeks 5-6**: Memory manager integration
- **Weeks 7-8**: Advanced features (chunked prefill, speculative)
- **Weeks 9-10**: Production readiness (metrics, docs, benchmarks)

**Total**: ~10 weeks to feature parity with vLLM in scheduling/batching, while maintaining PocketLLM's performance edge in quantization and custom kernels.

---

**End of Document**
