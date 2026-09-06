from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence
import os
import threading
import time

import torch
from torch.autograd.profiler import record_function

from src.kernels.cuda_loader import load_cuda_kernel
from src.kernels.ops import Packed4BitWeightAlongK, _dequant_fp4_weight_torch, _quantize_int8_weight_torch


_EXT_DIR = Path(__file__).resolve().parents[3] / "build" / "extensions"
_EXT_PATH = _EXT_DIR / "deepseek_cpu_moe_ext.so"
_NATIVE_MOD = None
_RUNTIME_OMP_THREADS: int | None = None
_WORKER_EXECUTOR: ThreadPoolExecutor | None = None
_PERSISTENT_WORKER = None
_WORKER_LOCK = threading.Lock()
_FP4_BLOCK_SIZE = 32
_INPROC_SERVER_LOCK = threading.Lock()
_INPROC_SERVER = None


@dataclass
class _InProcessCPUMoEServer:
    shm_name: str
    shm: object
    thread: threading.Thread

    def stop(self) -> None:
        try:
            req, resp, layer_id, _stop = self.shm.read_header()
            self.shm.write_header(req, resp, layer_id, 1)
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        self.thread.join(timeout=1.0)
        try:
            self.shm.close(unlink=True)
        except Exception:
            pass


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _env_enabled_default_on(name: str) -> bool:
    """Like _env_enabled but defaults to True; returns False only if the env is
    explicitly set to a falsy value ("0", "false", "no", "off"). Used for
    optimizations that have been validated as safe and should ship enabled."""
    val = os.getenv(name)
    if val is None:
        return True
    return val.lower() not in {"0", "false", "no", "off"}


def _native_gguf_type_id(type_name: str) -> int | None:
    if type_name == "iq2_xxs":
        return 0
    if type_name == "q2_k":
        return 1
    return None


def _layer_imatrix_capture_enabled(layer_idx: int | None) -> bool:
    if layer_idx is None or not _env_enabled("DEEPSEEK_IMATRIX_CAPTURE"):
        return False
    try:
        from src.loader.gguf.imatrix import collector

        c = collector()
        layer = int(layer_idx)
        return (
            c.should_capture(layer, "ffn_gate_exps")
            or c.should_capture(layer, "ffn_up_exps")
            or c.should_capture(layer, "ffn_down_exps")
        )
    except Exception:
        return False


def _apply_native_runtime_config(module) -> None:
    if module is None or _RUNTIME_OMP_THREADS is None:
        return
    setter = getattr(module, "set_omp_num_threads", None)
    if setter is not None:
        setter(int(_RUNTIME_OMP_THREADS))


def configure_cpu_routed_runtime(omp_threads: int | None = None) -> None:
    global _RUNTIME_OMP_THREADS
    if omp_threads is not None:
        _RUNTIME_OMP_THREADS = int(omp_threads) if omp_threads > 0 else None
        if _RUNTIME_OMP_THREADS is not None:
            os.environ["OMP_NUM_THREADS"] = str(_RUNTIME_OMP_THREADS)
    _apply_native_runtime_config(_NATIVE_MOD)


def _get_worker_executor() -> ThreadPoolExecutor:
    global _WORKER_EXECUTOR
    if _WORKER_EXECUTOR is None:
        with _WORKER_LOCK:
            if _WORKER_EXECUTOR is None:
                _WORKER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-cpu-moe")
    return _WORKER_EXECUTOR


def _find_extension_path() -> Path | None:
    import sysconfig

    search_dirs = (_EXT_DIR, _EXT_DIR.parent.parent)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix:
        for directory in search_dirs:
            abi_path = directory / f"deepseek_cpu_moe_ext{ext_suffix}"
            if abi_path.exists():
                return abi_path
    for directory in search_dirs:
        path = directory / "deepseek_cpu_moe_ext.so"
        if path.exists():
            return path
    matches = []
    for directory in search_dirs:
        matches.extend(directory.glob("deepseek_cpu_moe_ext*.so"))
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def _load_native_mod():
    global _NATIVE_MOD
    if _NATIVE_MOD is not None:
        return _NATIVE_MOD
    ext_path = _find_extension_path()
    if ext_path is None:
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("deepseek_cpu_moe_ext", ext_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NATIVE_MOD = module
        _apply_native_runtime_config(module)
        return module
    except Exception:
        return None


def run_native_int8_loop(
    shm_name: str,
    w1_layers: torch.Tensor,
    w2_layers: torch.Tensor,
    w3_layers: torch.Tensor,
    s1_layers: torch.Tensor,
    s2_layers: torch.Tensor,
    s3_layers: torch.Tensor,
    n_layers: int,
    dim: int,
    topk: int,
    inter_dim: int,
    n_routed_experts: int,
    output_slots: int,
    swiglu_limit: float,
    use_v2: bool = True,
) -> None:
    """Drive the persistent native int8 MoE server loop against an existing shared memory segment."""
    native_mod = _load_native_mod()
    loop_name = "cpu_moe_server_loop_int8_v2" if use_v2 else "cpu_moe_server_loop_int8"
    if native_mod is None or not hasattr(native_mod, loop_name):
        raise RuntimeError(f"native {loop_name} is unavailable")
    getattr(native_mod, loop_name)(
        shm_name,
        w1_layers.data_ptr(),
        w2_layers.data_ptr(),
        w3_layers.data_ptr(),
        s1_layers.data_ptr(),
        s2_layers.data_ptr(),
        s3_layers.data_ptr(),
        int(n_layers),
        int(dim),
        int(topk),
        int(inter_dim),
        int(n_routed_experts),
        int(output_slots),
        float(swiglu_limit),
    )


class _PersistentWorkerTask:
    def __init__(self, ready_event, runner, input_cpu, expert_ids_cpu, weights_cpu, output_cpu):
        self.ready_event = ready_event
        self.runner = runner
        self.input_cpu = input_cpu
        self.expert_ids_cpu = expert_ids_cpu
        self.weights_cpu = weights_cpu
        self.output_cpu = output_cpu
        self.done = False
        self.error = None


class _PersistentCPUWorker:
    def __init__(self):
        self._cond = threading.Condition()
        self._task: _PersistentWorkerTask | None = None
        self._thread = threading.Thread(target=self._loop, name="deepseek-cpu-moe-persistent", daemon=True)
        self._thread.start()

    def submit(self, ready_event, runner, input_cpu, expert_ids_cpu, weights_cpu, output_cpu) -> _PersistentWorkerTask:
        task = _PersistentWorkerTask(ready_event, runner, input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
        with self._cond:
            while self._task is not None:
                self._cond.wait()
            self._task = task
            self._cond.notify()
        return task

    def wait(self, task: _PersistentWorkerTask) -> None:
        with self._cond:
            while not task.done:
                self._cond.wait()
            if task.error is not None:
                raise task.error

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._task is None:
                    self._cond.wait()
                task = self._task
            try:
                _run_cpu_task(
                    task.ready_event,
                    task.runner,
                    task.input_cpu,
                    task.expert_ids_cpu,
                    task.weights_cpu,
                    task.output_cpu,
                )
            except BaseException as exc:
                task.error = exc
            with self._cond:
                task.done = True
                self._task = None
                self._cond.notify_all()


def _get_persistent_worker() -> _PersistentCPUWorker:
    global _PERSISTENT_WORKER
    if _PERSISTENT_WORKER is None:
        with _WORKER_LOCK:
            if _PERSISTENT_WORKER is None:
                _PERSISTENT_WORKER = _PersistentCPUWorker()
    return _PERSISTENT_WORKER


@dataclass
class _PendingTask:
    slot: int
    batch_size: int
    device: torch.device
    future: Future | None
    worker_task: _PersistentWorkerTask | None = None


def _run_cpu_task(
    ready_event: torch.cuda.Event | None,
    runner,
    input_cpu: torch.Tensor,
    expert_ids_cpu: torch.Tensor,
    weights_cpu: torch.Tensor,
    output_cpu: torch.Tensor,
) -> None:
    with torch.inference_mode():
        if ready_event is not None:
            ready_event.synchronize()
        runner(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)


class CPURoutedExpertsBuffer:
    capture_bs = []
    capture_buffers: Dict[int, tuple] = {}
    temp_bs = 0
    temp_buffer: tuple = tuple()
    buffer_depth = 2

    @classmethod
    def get_buffer(cls, hidden_states: torch.Tensor, num_experts_per_tok: int, output_dim: int):
        batch_size, hidden_size = hidden_states.shape
        if batch_size in cls.capture_buffers:
            return cls.capture_buffers[batch_size]
        if batch_size == cls.temp_bs:
            return cls.temp_buffer

        pin_memory = True
        input_cpu = [
            torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        expert_ids_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", pin_memory=pin_memory, dtype=torch.long)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((batch_size, num_experts_per_tok), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((batch_size, output_dim), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((batch_size, output_dim), device=hidden_states.device, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, output_gpu)
        if batch_size in cls.capture_bs:
            cls.capture_buffers[batch_size] = cur_buffer
        cls.temp_bs = batch_size
        cls.temp_buffer = cur_buffer
        return cur_buffer


class CPURoutedItemsBuffer:
    capture_keys = []
    capture_buffers: Dict[tuple[int, int, int], tuple] = {}
    temp_key: tuple[int, int, int] | None = None
    temp_buffer: tuple = tuple()
    buffer_depth = 2

    @classmethod
    def get_buffer(cls, item_count: int, hidden_size: int, output_dim: int, device: torch.device):
        key = (item_count, hidden_size, output_dim)
        if key in cls.capture_buffers:
            return cls.capture_buffers[key]
        if key == cls.temp_key:
            return cls.temp_buffer

        pin_memory = True
        input_cpu = [
            torch.zeros((item_count, hidden_size), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        expert_ids_cpu = [
            torch.zeros((item_count, 1), device="cpu", pin_memory=pin_memory, dtype=torch.long)
            for _ in range(cls.buffer_depth)
        ]
        weights_cpu = [
            torch.zeros((item_count, 1), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        output_cpu = [
            torch.zeros((item_count, output_dim), device="cpu", pin_memory=pin_memory, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]
        output_gpu = [
            torch.zeros((item_count, output_dim), device=device, dtype=torch.float32)
            for _ in range(cls.buffer_depth)
        ]

        cur_buffer = (input_cpu, expert_ids_cpu, weights_cpu, output_cpu, output_gpu)
        if key in cls.capture_keys:
            cls.capture_buffers[key] = cur_buffer
        cls.temp_key = key
        cls.temp_buffer = cur_buffer
        return cur_buffer


class CPURoutedExpertsBackend:
    def __init__(
        self,
        layer_idx: int,
        experts: Sequence,
        experts_start_idx: int,
        experts_end_idx: int,
        num_experts_per_tok: int,
        output_dim: int,
    ):
        self.layer_idx = layer_idx
        self.experts = experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_end_idx
        self.num_experts_per_tok = num_experts_per_tok
        self.output_dim = output_dim
        self._pending: _PendingTask | None = None
        self._native_mod = _load_native_mod()
        self._cuda_ext = None
        self._native_ready = False
        self._native_enabled = False
        self._inter_dim = 0
        self._swiglu_limit = 0.0
        self._cpu_expert_int8_enabled = _env_enabled("DEEPSEEK_CPU_EXPERT_INT8") or any(
            expert is not None and expert.w1.weight.dtype == torch.int8 for expert in experts
        )
        self._native_int8_enabled = False
        self._native_fp4_raw_enabled = False
        self._native_w1_ptrs = None
        self._native_w2_ptrs = None
        self._native_w3_ptrs = None
        self._native_s1_ptrs = None
        self._native_s2_ptrs = None
        self._native_s3_ptrs = None
        self._native_int8_w1_ptrs = None
        self._native_int8_w2_ptrs = None
        self._native_int8_w3_ptrs = None
        self._native_int8_s1_ptrs = None
        self._native_int8_s2_ptrs = None
        self._native_int8_s3_ptrs = None
        self._expert_int8_cache = {}
        self._int8_weights_prepared = False
        # Pinned arena: one contiguous pinned host tensor per (w1q, w1s, w2q, w2s, w3q, w3s),
        # shaped [n_local_experts, ...]. _expert_int8_cache slices point into the arenas so that
        # the GPU prefill MoE backend can launch a single non-blocking H2D copy per layer instead of
        # rebuilding a pinned workspace from non-pinned per-expert tensors on every request.
        # Default ON (validated to be the only physical store of int8 expert weights, releasing
        # the duplicate nn.Parameter storage). Set DEEPSEEK_GPU_PREFILL_MOE_ARENA=0 to disable.
        self._int8_arena_enabled = _env_enabled_default_on("DEEPSEEK_GPU_PREFILL_MOE_ARENA")
        self._int8_arena = None  # tuple of 6 pinned tensors or None
        self._int8_arena_filled = False
        # FP4 arena: single CPU-side store for official FP4 routed experts. It holds
        # raw [n, O, K/2] weight bytes plus raw ue8m0 [n, O, K/32] scale bytes, and
        # both CPU prefill and GPU active-expert decode read from this arena.
        self._fp4_arena_enabled = _env_enabled_default_on("DEEPSEEK_GPU_PREFILL_MOE_FP4_ARENA")
        self._fp4_arena = None
        self._expert_fp4_cache = {}
        self._fp4_weights_prepared = False
        self._executor = _get_worker_executor()
        self._persistent_worker_enabled = _env_enabled("DEEPSEEK_CPU_PERSISTENT_WORKER")
        self._native_persistent_team_enabled = _env_enabled("DEEPSEEK_CPU_NATIVE_PERSISTENT_TEAM")
        self._native_topk_persistent_enabled = _env_enabled("DEEPSEEK_CPU_TOPK_PERSISTENT")
        self._native_topk_parallel_enabled = _env_enabled("DEEPSEEK_CPU_TOPK_PARALLEL")
        self._copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._decode_inline_threshold = int(os.getenv("DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD", "0"))
        self._topk_limit = int(os.getenv("DEEPSEEK_CPU_TOPK_LIMIT", "0"))
        self._profile_enabled = _env_enabled("DEEPSEEK_CPU_MOE_PROFILE")
        self._route_profile_enabled = _env_enabled("DEEPSEEK_CPU_ROUTE_PROFILE")
        self._gguf_routes_native_enabled = _env_enabled_default_on("DEEPSEEK_GGUF_ROUTES_NATIVE")
        serving_prefill_chunk = int(os.getenv("DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS", os.getenv("DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS", "256")) or "256")
        self._gguf_routes_native_max_batch = max(1, int(os.getenv("DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH", str(serving_prefill_chunk))))
        self._gguf_grouped_expert_enabled = _env_enabled_default_on("DEEPSEEK_GGUF_GROUPED_EXPERT_PREFILL")
        self._gguf_raw_block_cache = {}
        self._has_gguf_raw_experts = None
        self._gguf_grid_cpu = None
        self._gguf_route_capacity = 0
        self._gguf_route_tokens_buf = None
        self._gguf_route_weights_buf = None
        self._gguf_w1_ptrs_buf = None
        self._gguf_w3_ptrs_buf = None
        self._gguf_w2_ptrs_buf = None
        self._host_reduce_name = f"/deepseek_cpu_moe_reduce_{os.getppid()}_{self.layer_idx}"
        self._host_reduce_slot = 0

    def __del__(self):
        native_mod = getattr(self, "_native_mod", None)
        host_reduce_name = getattr(self, "_host_reduce_name", None)
        if native_mod is not None and host_reduce_name and hasattr(native_mod, "host_float_allreduce_unlink"):
            try:
                native_mod.host_float_allreduce_unlink(host_reduce_name)
            except Exception:
                pass

    def has_gguf_raw_experts(self) -> bool:
        if self._has_gguf_raw_experts is not None:
            return self._has_gguf_raw_experts
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[expert_id]
            if expert is not None and getattr(expert, "_gguf_raw_backing", None) is not None:
                self._has_gguf_raw_experts = True
                return True
        self._has_gguf_raw_experts = False
        return False

    def _apply_topk_limit(self, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._topk_limit <= 0 or self._topk_limit >= topk_ids.shape[-1]:
            return topk_ids, topk_weights
        keep = self._topk_limit
        limited_ids = topk_ids.clone()
        limited_weights = topk_weights.clone()
        limited_ids[..., keep:] = -1
        limited_weights[..., keep:] = 0
        return limited_ids, limited_weights

    def _copy_inputs_to_cpu(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        (
            input_cpu,
            expert_ids_cpu,
            weights_cpu,
            output_cpu,
            output_gpu,
        ) = CPURoutedExpertsBuffer.get_buffer(hidden_states, self.num_experts_per_tok, self.output_dim)
        current_slot = self.layer_idx % CPURoutedExpertsBuffer.buffer_depth
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        batch_size = flat_hidden_states.shape[0]
        if hidden_states.device.type == "cuda" and self._copy_stream is not None:
            current_stream = torch.cuda.current_stream(hidden_states.device)
            self._copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._copy_stream):
                input_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
                expert_ids_cpu[current_slot].copy_(topk_ids.to(torch.long), non_blocking=True)
                weights_cpu[current_slot].copy_(topk_weights.to(torch.float32), non_blocking=True)
            ready_event = torch.cuda.Event()
            ready_event.record(self._copy_stream)
        else:
            input_cpu[current_slot].copy_(flat_hidden_states, non_blocking=True)
            expert_ids_cpu[current_slot].copy_(topk_ids.to(torch.long), non_blocking=True)
            weights_cpu[current_slot].copy_(topk_weights.to(torch.float32), non_blocking=True)
            ready_event = None
            if hidden_states.device.type == "cuda":
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(hidden_states.device))
        return current_slot, batch_size, hidden_states.device, ready_event, input_cpu, expert_ids_cpu, weights_cpu, output_cpu, output_gpu

    def _copy_output_to_device(
        self,
        current_slot: int,
        batch_size: int,
        device: torch.device,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        (
            _input_cpu,
            _expert_ids_cpu,
            _weights_cpu,
            output_cpu,
            output_gpu,
        ) = CPURoutedExpertsBuffer.get_buffer(hidden_states.view(-1, hidden_states.shape[-1]), self.num_experts_per_tok, self.output_dim)
        if device.type == "cuda":
            output_gpu[current_slot][:batch_size].copy_(output_cpu[current_slot][:batch_size], non_blocking=True)
            return output_gpu[current_slot][:batch_size]
        return output_cpu[current_slot][:batch_size]

    def _should_run_decode_inline(self, batch_size: int) -> bool:
        return batch_size <= self._decode_inline_threshold

    def _should_run_int8_native(self, batch_size: int) -> bool:
        return self._int8_weights_prepared and self._native_int8_enabled

    def _materialize_int8_expert(self, expert_id: int):
        cached = self._expert_int8_cache.get(expert_id)
        if cached is not None:
            return cached
        expert = self.experts[expert_id]
        if expert is None or getattr(expert, "_gguf_raw_backing", None) is not None:
            return None
        # Fast path: int8 checkpoint. Materialize directly into the per-layer pinned arena so
        # there is exactly one physical copy of each expert's int8 weights/scales.
        is_int8 = expert.w1.weight.dtype == torch.int8
        arena_slot = self._reserve_arena_slot_int8(expert, expert_id) if is_int8 else None
        if is_int8 and arena_slot is not None:
            arena_w1q, arena_w1s, arena_w2q, arena_w2s, arena_w3q, arena_w3s = arena_slot
            try:
                arena_w1q.copy_(expert.w1.weight.detach())
                arena_w1s.copy_(expert.w1.weight.scale.detach().to(torch.float32))
                arena_w2q.copy_(expert.w2.weight.detach())
                arena_w2s.copy_(expert.w2.weight.scale.detach().to(torch.float32))
                arena_w3q.copy_(expert.w3.weight.detach())
                arena_w3s.copy_(expert.w3.weight.scale.detach().to(torch.float32))
            except RuntimeError:
                arena_slot = None
            if arena_slot is not None:
                # Repoint expert's CPU int8 buffers to arena slices and release the original
                # nn.Parameter storages so the only remaining backing store is the pinned arena.
                expert._cpu_w1 = arena_w1q
                expert._cpu_w2 = arena_w2q
                expert._cpu_w3 = arena_w3q
                expert._cpu_w1_scale = arena_w1s
                expert._cpu_w2_scale = arena_w2s
                expert._cpu_w3_scale = arena_w3s
                expert._cpu_materialized = True
                self._release_expert_parameter_storage(expert)
                cached = (arena_w1q, arena_w1s, arena_w2q, arena_w2s, arena_w3q, arena_w3s)
                self._expert_int8_cache[expert_id] = cached
                return cached
        # Legacy / fp4 / arena-disabled path: materialize into private CPU tensors first.
        expert._materialize_cpu_weights()
        if is_int8:
            w1 = expert._cpu_w1 if expert._cpu_w1.is_contiguous() else expert._cpu_w1.contiguous()
            w2 = expert._cpu_w2 if expert._cpu_w2.is_contiguous() else expert._cpu_w2.contiguous()
            w3 = expert._cpu_w3 if expert._cpu_w3.is_contiguous() else expert._cpu_w3.contiguous()
            w1s = expert._cpu_w1_scale if expert._cpu_w1_scale.is_contiguous() else expert._cpu_w1_scale.contiguous()
            w2s = expert._cpu_w2_scale if expert._cpu_w2_scale.is_contiguous() else expert._cpu_w2_scale.contiguous()
            w3s = expert._cpu_w3_scale if expert._cpu_w3_scale.is_contiguous() else expert._cpu_w3_scale.contiguous()
            cached = self._store_int8_cached(expert_id, (w1, w1s, w2, w2s, w3, w3s))
            return cached
        if isinstance(expert._cpu_w1, torch.Tensor) and expert._cpu_w1.dtype == torch.uint8:
            gpu_cached = self._gpu_convert_fp4_expert_to_int8(expert_id)
            if gpu_cached is not None:
                return gpu_cached
            w1_packed = Packed4BitWeightAlongK.convert_from(expert._cpu_w1.view(torch.int8).view(torch.float4_e2m1fn_x2))
            w2_packed = Packed4BitWeightAlongK.convert_from(expert._cpu_w2.view(torch.int8).view(torch.float4_e2m1fn_x2))
            w3_packed = Packed4BitWeightAlongK.convert_from(expert._cpu_w3.view(torch.int8).view(torch.float4_e2m1fn_x2))
            w1_scale = expert._cpu_w1_scale.view(torch.float8_e8m0fnu).to(torch.float32).contiguous()
            w2_scale = expert._cpu_w2_scale.view(torch.float8_e8m0fnu).to(torch.float32).contiguous()
            w3_scale = expert._cpu_w3_scale.view(torch.float8_e8m0fnu).to(torch.float32).contiguous()
            w1 = _dequant_fp4_weight_torch(w1_packed, w1_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
            w2 = _dequant_fp4_weight_torch(w2_packed, w2_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
            w3 = _dequant_fp4_weight_torch(w3_packed, w3_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
            w1_q, w1_s = _quantize_int8_weight_torch(w1)
            w2_q, w2_s = _quantize_int8_weight_torch(w2)
            w3_q, w3_s = _quantize_int8_weight_torch(w3)
            cached = self._store_int8_cached(expert_id, (w1_q, w1_s, w2_q, w2_s, w3_q, w3_s))
            return cached
        if not hasattr(expert._cpu_w1, "layout_tensor"):
            return None
        w1 = _dequant_fp4_weight_torch(expert._cpu_w1, expert._cpu_w1_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
        w2 = _dequant_fp4_weight_torch(expert._cpu_w2, expert._cpu_w2_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
        w3 = _dequant_fp4_weight_torch(expert._cpu_w3, expert._cpu_w3_scale, block_size=_FP4_BLOCK_SIZE).contiguous()
        w1_q, w1_s = _quantize_int8_weight_torch(w1)
        w2_q, w2_s = _quantize_int8_weight_torch(w2)
        w3_q, w3_s = _quantize_int8_weight_torch(w3)
        cached = self._store_int8_cached(expert_id, (w1_q, w1_s, w2_q, w2_s, w3_q, w3_s))
        return cached

    def _reserve_arena_slot_int8(self, expert, expert_id: int):
        """Allocate the per-layer pinned arena (using this expert's int8 shapes as the template)
        and return the per-expert slot tuple. Returns None if arena is disabled or unavailable."""
        if not self._int8_arena_enabled:
            return None
        if self._int8_arena is None:
            n = self.experts_end_idx - self.experts_start_idx
            if n <= 0:
                return None
            try:
                w1_shape = tuple(expert.w1.weight.shape)
                w2_shape = tuple(expert.w2.weight.shape)
                w3_shape = tuple(expert.w3.weight.shape)
                w1_scale_shape = tuple(expert.w1.weight.scale.shape)
                w2_scale_shape = tuple(expert.w2.weight.scale.shape)
                w3_scale_shape = tuple(expert.w3.weight.scale.shape)
            except AttributeError:
                return None
            try:
                arena = (
                    torch.empty((n, *w1_shape), device="cpu", dtype=torch.int8, pin_memory=True),
                    torch.empty((n, *w1_scale_shape), device="cpu", dtype=torch.float32, pin_memory=True),
                    torch.empty((n, *w2_shape), device="cpu", dtype=torch.int8, pin_memory=True),
                    torch.empty((n, *w2_scale_shape), device="cpu", dtype=torch.float32, pin_memory=True),
                    torch.empty((n, *w3_shape), device="cpu", dtype=torch.int8, pin_memory=True),
                    torch.empty((n, *w3_scale_shape), device="cpu", dtype=torch.float32, pin_memory=True),
                )
            except RuntimeError:
                self._int8_arena_enabled = False
                self._int8_arena = None
                return None
            self._int8_arena = arena
        local_id = expert_id - self.experts_start_idx
        if local_id < 0 or local_id >= self._int8_arena[0].shape[0]:
            return None
        arena_w1q, arena_w1s, arena_w2q, arena_w2s, arena_w3q, arena_w3s = self._int8_arena
        return (
            arena_w1q[local_id],
            arena_w1s[local_id],
            arena_w2q[local_id],
            arena_w2s[local_id],
            arena_w3q[local_id],
            arena_w3s[local_id],
        )

    @staticmethod
    def _release_expert_parameter_storage(expert) -> None:
        for sub in (expert.w1, expert.w2, expert.w3):
            scale = getattr(sub.weight, "scale", None)
            empty_like_dtype = torch.empty(0, dtype=sub.weight.dtype)
            empty_scale_like_dtype = torch.empty(0, dtype=scale.dtype) if scale is not None else None
            try:
                sub.weight.requires_grad_(False)
            except Exception:
                pass
            sub.weight.data = empty_like_dtype
            if scale is not None:
                try:
                    scale.requires_grad_(False)
                except Exception:
                    pass
                try:
                    scale.data = empty_scale_like_dtype
                except Exception:
                    pass
                try:
                    sub.weight.scale = empty_scale_like_dtype
                except AttributeError:
                    pass

    def _ensure_int8_arena(self, packed) -> None:
        """Lazily allocate one pinned arena per layer using the shape/dtype of the first
        materialized int8 expert as a template. Subsequent experts must match that template."""
        if not self._int8_arena_enabled:
            return
        if self._int8_arena is not None:
            return
        n = self.experts_end_idx - self.experts_start_idx
        if n <= 0:
            return
        e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
        try:
            arena = (
                torch.empty((n, *e_w1q.shape), device="cpu", dtype=e_w1q.dtype, pin_memory=True),
                torch.empty((n, *e_w1s.shape), device="cpu", dtype=e_w1s.dtype, pin_memory=True),
                torch.empty((n, *e_w2q.shape), device="cpu", dtype=e_w2q.dtype, pin_memory=True),
                torch.empty((n, *e_w2s.shape), device="cpu", dtype=e_w2s.dtype, pin_memory=True),
                torch.empty((n, *e_w3q.shape), device="cpu", dtype=e_w3q.dtype, pin_memory=True),
                torch.empty((n, *e_w3s.shape), device="cpu", dtype=e_w3s.dtype, pin_memory=True),
            )
        except RuntimeError:
            # Fall back to non-arena path on pin failure.
            self._int8_arena_enabled = False
            self._int8_arena = None
            return
        self._int8_arena = arena

    def _store_int8_cached(self, expert_id: int, packed):
        """Either return packed as-is (legacy path) or copy it into the per-layer pinned arena
        and replace the cached entry with arena slices."""
        if self._int8_arena_enabled:
            self._ensure_int8_arena(packed)
        if self._int8_arena is None:
            self._expert_int8_cache[expert_id] = packed
            return packed
        local_id = expert_id - self.experts_start_idx
        if local_id < 0 or local_id >= self._int8_arena[0].shape[0]:
            self._expert_int8_cache[expert_id] = packed
            return packed
        arena_w1q, arena_w1s, arena_w2q, arena_w2s, arena_w3q, arena_w3s = self._int8_arena
        e_w1q, e_w1s, e_w2q, e_w2s, e_w3q, e_w3s = packed
        try:
            arena_w1q[local_id].copy_(e_w1q)
            arena_w1s[local_id].copy_(e_w1s)
            arena_w2q[local_id].copy_(e_w2q)
            arena_w2s[local_id].copy_(e_w2s)
            arena_w3q[local_id].copy_(e_w3q)
            arena_w3s[local_id].copy_(e_w3s)
        except RuntimeError:
            self._expert_int8_cache[expert_id] = packed
            return packed
        sliced = (
            arena_w1q[local_id],
            arena_w1s[local_id],
            arena_w2q[local_id],
            arena_w2s[local_id],
            arena_w3q[local_id],
            arena_w3s[local_id],
        )
        self._expert_int8_cache[expert_id] = sliced
        return sliced

    def release_int8_prepare_cache(self) -> None:
        self._expert_int8_cache.clear()
        self._int8_arena = None
        self._int8_weights_prepared = False
        self._native_int8_w1_ptrs = None
        self._native_int8_w2_ptrs = None
        self._native_int8_w3_ptrs = None
        self._native_int8_s1_ptrs = None
        self._native_int8_s2_ptrs = None
        self._native_int8_s3_ptrs = None

    def get_int8_arena(self):
        """Return the per-layer pinned arena tuple if every local expert has been materialized
        into it, otherwise None. The GPU prefill MoE backend uses this to skip its own pinned
        workspace memcpy and launch a single H2D copy per layer."""
        if self._int8_arena is None or not self._int8_arena_enabled:
            return None
        n = self._int8_arena[0].shape[0]
        # All local experts must have arena-backed cache entries.
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            cached = self._expert_int8_cache.get(expert_id)
            if cached is None:
                return None
            # Cheap sanity: arena-backed slice shares storage with arena_w1q.
            if cached[0].data_ptr() < self._int8_arena[0].data_ptr():
                return None
        if n != (self.experts_end_idx - self.experts_start_idx):
            return None
        return self._int8_arena

    # ------------------------------------------------------------------
    # FP4 arena: parallel to int8 arena. Stores packed FP4 (e2m1fn_x2) weight
    # bytes + ue8m0 block-scale bytes for routed experts. Used by the GPU
    # active-expert decode path when the official FP4 ckpt is loaded — avoids
    # the dequant->requant->int8 round-trip and halves H2D bytes per expert.
    # ------------------------------------------------------------------

    def _materialize_fp4_expert(self, expert_id: int):
        """Copy expert's packed FP4 weights and e8m0 block scales (uint8 byte
        views) into the per-layer pinned FP4 arena slot. Returns the slot tuple
        or None on failure."""
        cached = self._expert_fp4_cache.get(expert_id)
        if cached is not None:
            return cached
        expert = self.experts[expert_id]
        if expert is None or getattr(expert, "_gguf_raw_backing", None) is not None:
            return None
        if expert.w1.weight.dtype != torch.float4_e2m1fn_x2:
            return None
        slot = self._reserve_arena_slot_fp4(expert, expert_id)
        if slot is None:
            return None
        arena_w1q, arena_w1s, arena_w2q, arena_w2s, arena_w3q, arena_w3s = slot
        try:
            arena_w1q.copy_(expert.w1.weight.detach().view(torch.uint8))
            arena_w1s.copy_(expert.w1.weight.scale.detach().view(torch.uint8))
            arena_w2q.copy_(expert.w2.weight.detach().view(torch.uint8))
            arena_w2s.copy_(expert.w2.weight.scale.detach().view(torch.uint8))
            arena_w3q.copy_(expert.w3.weight.detach().view(torch.uint8))
            arena_w3s.copy_(expert.w3.weight.scale.detach().view(torch.uint8))
        except RuntimeError:
            return None
        expert._cpu_w1 = arena_w1q
        expert._cpu_w2 = arena_w2q
        expert._cpu_w3 = arena_w3q
        expert._cpu_w1_scale = arena_w1s
        expert._cpu_w2_scale = arena_w2s
        expert._cpu_w3_scale = arena_w3s
        expert._cpu_w2_tiled = None
        expert._cpu_w2_scale_tiled = None
        expert._cpu_materialized = True
        self._release_expert_parameter_storage(expert)
        cached = (arena_w1q, arena_w1s, arena_w2q, arena_w2s, arena_w3q, arena_w3s)
        self._expert_fp4_cache[expert_id] = cached
        return cached

    def _reserve_arena_slot_fp4(self, expert, expert_id: int):
        if not self._fp4_arena_enabled:
            return None
        if self._fp4_arena is None:
            n = self.experts_end_idx - self.experts_start_idx
            if n <= 0:
                return None
            try:
                w1_shape = tuple(expert.w1.weight.shape)              # [O, K/2]
                w2_shape = tuple(expert.w2.weight.shape)
                w3_shape = tuple(expert.w3.weight.shape)
                w1_scale_shape = tuple(expert.w1.weight.scale.shape)  # [O, K/32]
                w2_scale_shape = tuple(expert.w2.weight.scale.shape)
                w3_scale_shape = tuple(expert.w3.weight.scale.shape)
            except AttributeError:
                return None
            try:
                arena = (
                    torch.empty((n, *w1_shape), device="cpu", dtype=torch.uint8, pin_memory=True),
                    torch.empty((n, *w1_scale_shape), device="cpu", dtype=torch.uint8, pin_memory=True),
                    torch.empty((n, *w2_shape), device="cpu", dtype=torch.uint8, pin_memory=True),
                    torch.empty((n, *w2_scale_shape), device="cpu", dtype=torch.uint8, pin_memory=True),
                    torch.empty((n, *w3_shape), device="cpu", dtype=torch.uint8, pin_memory=True),
                    torch.empty((n, *w3_scale_shape), device="cpu", dtype=torch.uint8, pin_memory=True),
                )
            except RuntimeError:
                self._fp4_arena_enabled = False
                self._fp4_arena = None
                return None
            self._fp4_arena = arena
        local_id = expert_id - self.experts_start_idx
        if local_id < 0 or local_id >= self._fp4_arena[0].shape[0]:
            return None
        a_w1q, a_w1s, a_w2q, a_w2s, a_w3q, a_w3s = self._fp4_arena
        return (
            a_w1q[local_id], a_w1s[local_id],
            a_w2q[local_id], a_w2s[local_id],
            a_w3q[local_id], a_w3s[local_id],
        )

    def _gpu_convert_fp4_expert_to_int8(self, expert_id: int):
        cached = self._expert_int8_cache.get(expert_id)
        if cached is not None:
            return cached
        fp4_cached = self._expert_fp4_cache.get(expert_id)
        if fp4_cached is None:
            fp4_cached = self._materialize_fp4_expert(expert_id)
        if fp4_cached is None:
            return None
        if not torch.cuda.is_available():
            return None
        if self._cuda_ext is None:
            self._cuda_ext = load_cuda_kernel()
        if self._cuda_ext is None or not hasattr(self._cuda_ext, "fp4_weight_to_int8_forward"):
            return None
        device = torch.device("cuda", torch.cuda.current_device())
        w1q, w1s, w2q, w2s, w3q, w3s = fp4_cached
        with torch.cuda.device(device):
            w1q_i8, w1s_i8 = self._cuda_ext.fp4_weight_to_int8_forward(
                w1q.unsqueeze(0).to(device, non_blocking=True),
                w1s.unsqueeze(0).to(device, non_blocking=True),
            )
            w2q_i8, w2s_i8 = self._cuda_ext.fp4_weight_to_int8_forward(
                w2q.unsqueeze(0).to(device, non_blocking=True),
                w2s.unsqueeze(0).to(device, non_blocking=True),
            )
            w3q_i8, w3s_i8 = self._cuda_ext.fp4_weight_to_int8_forward(
                w3q.unsqueeze(0).to(device, non_blocking=True),
                w3s.unsqueeze(0).to(device, non_blocking=True),
            )
            torch.cuda.synchronize(device)
        packed = (
            w1q_i8.squeeze(0).cpu().contiguous(),
            w1s_i8.squeeze(0).cpu().contiguous(),
            w2q_i8.squeeze(0).cpu().contiguous(),
            w2s_i8.squeeze(0).cpu().contiguous(),
            w3q_i8.squeeze(0).cpu().contiguous(),
            w3s_i8.squeeze(0).cpu().contiguous(),
        )
        return self._store_int8_cached(expert_id, packed)

    def prepare_fp4_weights(self, expert_ids: Sequence[int] | None = None) -> bool:
        """Materialize routed experts into the pinned FP4 arena. Returns True
        when every local expert has an arena slot ready."""
        if self.has_gguf_raw_experts():
            self._fp4_weights_prepared = False
            return False
        if not self._fp4_arena_enabled:
            return False
        if expert_ids is None:
            expert_ids = range(self.experts_start_idx, self.experts_end_idx)
        ok = True
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                continue
            ok = (self._materialize_fp4_expert(expert_id) is not None) and ok
        self._fp4_weights_prepared = bool(self._expert_fp4_cache) and ok
        return self._fp4_weights_prepared

    def get_fp4_arena(self):
        """Return the per-layer pinned FP4 arena tuple
        (w1q, w1s, w2q, w2s, w3q, w3s) when every local expert has been
        materialized into it. Same fast-path semantics as get_int8_arena()."""
        if self._fp4_arena is None or not self._fp4_arena_enabled:
            return None
        n = self._fp4_arena[0].shape[0]
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            cached = self._expert_fp4_cache.get(expert_id)
            if cached is None:
                return None
            if cached[0].data_ptr() < self._fp4_arena[0].data_ptr():
                return None
        if n != (self.experts_end_idx - self.experts_start_idx):
            return None
        return self._fp4_arena

    def _gpu_convert_fp4_arena_to_int8(self) -> bool:
        t0 = time.perf_counter() if self._profile_enabled else 0.0
        fp4_arena = self.get_fp4_arena()
        if fp4_arena is None or not torch.cuda.is_available():
            return False
        if self._cuda_ext is None:
            self._cuda_ext = load_cuda_kernel()
        if self._cuda_ext is None or not hasattr(self._cuda_ext, "fp4_weight_to_int8_forward"):
            return False
        int8_arena = self._allocate_int8_arena_from_fp4_arena(fp4_arena)
        if int8_arena is None:
            return False
        device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(device):
            for src_q, src_s, dst_q, dst_s in (
                (fp4_arena[0], fp4_arena[1], int8_arena[0], int8_arena[1]),
                (fp4_arena[2], fp4_arena[3], int8_arena[2], int8_arena[3]),
                (fp4_arena[4], fp4_arena[5], int8_arena[4], int8_arena[5]),
            ):
                src_q_gpu = src_q.to(device, non_blocking=True)
                src_s_gpu = src_s.to(device, non_blocking=True)
                q_gpu, s_gpu = self._cuda_ext.fp4_weight_to_int8_forward(src_q_gpu, src_s_gpu)
                dst_q.copy_(q_gpu, non_blocking=False)
                dst_s.copy_(s_gpu, non_blocking=False)
                del src_q_gpu, src_s_gpu, q_gpu, s_gpu
        self._int8_arena = int8_arena
        self._int8_arena_enabled = True
        self._expert_int8_cache.clear()
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            local_id = expert_id - self.experts_start_idx
            self._expert_int8_cache[expert_id] = (
                int8_arena[0][local_id], int8_arena[1][local_id],
                int8_arena[2][local_id], int8_arena[3][local_id],
                int8_arena[4][local_id], int8_arena[5][local_id],
            )
        if self._profile_enabled:
            print(
                f"cpu_moe layer={self.layer_idx} fp4_gpu_prepare_int8 experts={self.experts_end_idx - self.experts_start_idx} time={time.perf_counter() - t0:.3f}s",
                flush=True,
            )
        return True

    def _allocate_int8_arena_from_fp4_arena(self, fp4_arena):
        w1q, w1s, w2q, w2s, w3q, w3s = fp4_arena
        shapes = (
            (tuple(w1q.shape[:-1]) + (w1q.shape[-1] * 2,), torch.int8),
            (tuple(w1s.shape[:2]), torch.float32),
            (tuple(w2q.shape[:-1]) + (w2q.shape[-1] * 2,), torch.int8),
            (tuple(w2s.shape[:2]), torch.float32),
            (tuple(w3q.shape[:-1]) + (w3q.shape[-1] * 2,), torch.int8),
            (tuple(w3s.shape[:2]), torch.float32),
        )
        try:
            return tuple(torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True) for shape, dtype in shapes)
        except RuntimeError:
            return tuple(torch.empty(shape, device="cpu", dtype=dtype) for shape, dtype in shapes)

    def prepare_int8_weights(self, expert_ids: Sequence[int] | None = None) -> bool:
        if self._inter_dim is None:
            for expert in self.experts:
                if expert is not None:
                    self._inter_dim = expert.w1.out_features
                    self._swiglu_limit = float(expert.swiglu_limit)
                    break
        if self.has_gguf_raw_experts():
            self.release_int8_prepare_cache()
            self._int8_weights_prepared = False
            return False
        full_prepare = expert_ids is None
        if expert_ids is None:
            expert_ids = range(self.experts_start_idx, self.experts_end_idx)
        if full_prepare and self.get_int8_arena() is not None:
            self._refresh_int8_ptrs()
            self._int8_weights_prepared = True
            return True
        if full_prepare and self.get_fp4_arena() is not None and self._gpu_convert_fp4_arena_to_int8():
            self._refresh_int8_ptrs()
            self._int8_weights_prepared = True
            return True
        ok = True
        prepared_any = False
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                continue
            expert = self.experts[expert_id]
            if expert is None:
                continue
            if expert.w1.weight.dtype == torch.float4_e2m1fn_x2:
                if not self._fp4_weights_prepared:
                    self.prepare_fp4_weights([expert_id])
                cached = self._gpu_convert_fp4_expert_to_int8(expert_id)
            else:
                cached = self._materialize_int8_expert(expert_id)
            ok = cached is not None and ok
            prepared_any = True
        if prepared_any:
            self._refresh_int8_ptrs()
        self._int8_weights_prepared = bool(self._expert_int8_cache) and ok
        return self._int8_weights_prepared

    def _refresh_int8_ptrs(self) -> None:
        if self._native_int8_w1_ptrs is None:
            num_experts = len(self.experts)
            self._native_int8_w1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_w2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_w3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_s1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_s2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
            self._native_int8_s3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        for expert_id, packed in self._expert_int8_cache.items():
            w1_q, w1_s, w2_q, w2_s, w3_q, w3_s = packed
            self._native_int8_w1_ptrs[expert_id] = w1_q.data_ptr()
            self._native_int8_w2_ptrs[expert_id] = w2_q.data_ptr()
            self._native_int8_w3_ptrs[expert_id] = w3_q.data_ptr()
            self._native_int8_s1_ptrs[expert_id] = w1_s.data_ptr()
            self._native_int8_s2_ptrs[expert_id] = w2_s.data_ptr()
            self._native_int8_s3_ptrs[expert_id] = w3_s.data_ptr()

    def _run_reference_items_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        output_cpu.zero_()
        if input_cpu.numel() == 0:
            return
        expert_ids_flat = expert_ids_cpu.reshape(-1)
        sort_order = torch.argsort(expert_ids_flat)
        rows = torch.arange(expert_ids_flat.numel(), device=expert_ids_flat.device)[sort_order]
        expert_ids = expert_ids_flat[sort_order]
        unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        offset = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                offset += count
                continue
            group_rows = rows[offset:offset + count]
            expert = self.experts[expert_id]
            output_cpu[group_rows] = expert.forward_cpu(
                input_cpu[group_rows],
                weights_cpu[group_rows],
                x_cpu=input_cpu[group_rows],
            )
            offset += count

    def dispatch_forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        item_count = hidden_states.shape[0]
        if item_count == 0:
            return torch.zeros((0, self.output_dim), device=hidden_states.device, dtype=torch.float32)
        (
            input_cpu,
            expert_ids_cpu,
            weights_cpu,
            output_cpu,
            output_gpu,
        ) = CPURoutedItemsBuffer.get_buffer(item_count, hidden_states.shape[-1], self.output_dim, hidden_states.device)
        current_slot = self.layer_idx % CPURoutedItemsBuffer.buffer_depth
        topk_ids, topk_weights = self._apply_topk_limit(expert_ids.reshape(item_count, 1), route_weights.reshape(item_count, 1))
        input_cpu[current_slot].copy_(hidden_states.reshape(item_count, -1), non_blocking=False)
        expert_ids_cpu[current_slot].copy_(topk_ids.to(torch.long).reshape(item_count, 1), non_blocking=False)
        weights_cpu[current_slot].copy_(topk_weights.to(torch.float32).reshape(item_count, 1), non_blocking=False)
        native_ready = self._ensure_native_ready()
        if native_ready and self._should_run_int8_native(item_count):
            runner = self._run_native_int8_decode_forward
        elif self._native_enabled:
            runner = self._run_native_forward
        else:
            runner = self._run_reference_items_forward
        if self._profile_enabled:
            t0 = time.perf_counter()
            runner(
                input_cpu[current_slot],
                expert_ids_cpu[current_slot],
                weights_cpu[current_slot],
                output_cpu[current_slot],
            )
            print(
                f"cpu_moe layer={self.layer_idx} items={item_count} runner={getattr(runner, '__name__', runner)} time={time.perf_counter() - t0:.3f}s",
                flush=True,
            )
        else:
            runner(
                input_cpu[current_slot],
                expert_ids_cpu[current_slot],
                weights_cpu[current_slot],
                output_cpu[current_slot],
            )
        if hidden_states.device.type == "cuda":
            output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=False)
            return output_gpu[current_slot]
        return output_cpu[current_slot]

    def dispatch_forward_reference(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.dispatch_forward(hidden_states, expert_ids, route_weights)

    def _ensure_native_ready(self) -> bool:
        if _layer_imatrix_capture_enabled(self.layer_idx):
            self._native_ready = True
            self._native_enabled = False
            self._native_int8_enabled = False
            self._native_fp4_raw_enabled = False
            return False
        if self.has_gguf_raw_experts():
            self._native_ready = True
            self._native_enabled = False
            self._native_int8_enabled = False
            self._native_fp4_raw_enabled = False
            return False
        if self._native_ready:
            return self._native_enabled
        self._native_ready = True
        if self._native_mod is None:
            return False

        _apply_native_runtime_config(self._native_mod)
        template_expert = None
        num_experts = len(self.experts)
        w1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        w2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        w3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        s1_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        s2_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        s3_ptrs = torch.zeros(num_experts, device="cpu", dtype=torch.long)
        native_int8_enabled = self._cpu_expert_int8_enabled and hasattr(self._native_mod, "routed_int8_moe_forward")
        fp4_raw_enabled = False
        fp4_legacy_enabled = False
        for expert_id in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[expert_id]
            if expert is None:
                continue
            expert._materialize_cpu_weights()
            if template_expert is None:
                template_expert = expert
            if expert.w1.weight.dtype == torch.int8:
                continue
            if expert.w1.weight.dtype != torch.float4_e2m1fn_x2:
                self._native_enabled = False
                self._native_int8_enabled = False
                self._native_fp4_raw_enabled = False
                return False
            if isinstance(expert._cpu_w1, torch.Tensor):
                if fp4_legacy_enabled or not hasattr(self._native_mod, "routed_fp4_moe_forward_raw"):
                    self._native_enabled = False
                    self._native_int8_enabled = False
                    self._native_fp4_raw_enabled = False
                    return False
                fp4_raw_enabled = True
                w1_ptrs[expert_id] = expert._cpu_w1.data_ptr()
                w2_ptrs[expert_id] = expert._cpu_w2.data_ptr()
                w3_ptrs[expert_id] = expert._cpu_w3.data_ptr()
                s1_ptrs[expert_id] = expert._cpu_w1_scale.data_ptr()
                s2_ptrs[expert_id] = expert._cpu_w2_scale.data_ptr()
                s3_ptrs[expert_id] = expert._cpu_w3_scale.data_ptr()
            else:
                if fp4_raw_enabled:
                    self._native_enabled = False
                    self._native_int8_enabled = False
                    self._native_fp4_raw_enabled = False
                    return False
                fp4_legacy_enabled = True
                w1_ptrs[expert_id] = expert._cpu_w1.layout_tensor.data_ptr()
                w2_ptrs[expert_id] = expert._cpu_w2_tiled.data_ptr() if expert._cpu_w2_tiled is not None else expert._cpu_w2.layout_tensor.data_ptr()
                w3_ptrs[expert_id] = expert._cpu_w3.layout_tensor.data_ptr()
                s1_ptrs[expert_id] = expert._cpu_w1_scale.data_ptr()
                s2_ptrs[expert_id] = expert._cpu_w2_scale_tiled.data_ptr() if expert._cpu_w2_scale_tiled is not None else expert._cpu_w2_scale.data_ptr()
                s3_ptrs[expert_id] = expert._cpu_w3_scale.data_ptr()

        if template_expert is None:
            self._native_enabled = False
            self._native_int8_enabled = False
            self._native_fp4_raw_enabled = False
            return False

        self._inter_dim = template_expert.w1.out_features
        self._swiglu_limit = float(template_expert.swiglu_limit)
        self._native_w1_ptrs = w1_ptrs
        self._native_w2_ptrs = w2_ptrs
        self._native_w3_ptrs = w3_ptrs
        self._native_s1_ptrs = s1_ptrs
        self._native_s2_ptrs = s2_ptrs
        self._native_s3_ptrs = s3_ptrs
        self._native_enabled = True
        self._native_int8_enabled = native_int8_enabled
        self._native_fp4_raw_enabled = fp4_raw_enabled
        return True

    def _run_native_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        forward_name = "routed_fp4_moe_forward_raw" if self._native_fp4_raw_enabled else "routed_fp4_moe_forward"
        getattr(self._native_mod, forward_name)(
            input_cpu.data_ptr(),
            expert_ids_cpu.data_ptr(),
            weights_cpu.data_ptr(),
            output_cpu.data_ptr(),
            input_cpu.shape[0],
            input_cpu.shape[1],
            expert_ids_cpu.shape[1],
            self._inter_dim,
            len(self.experts),
            self.experts_start_idx,
            self.experts_end_idx,
            self._native_w1_ptrs.data_ptr(),
            self._native_w2_ptrs.data_ptr(),
            self._native_w3_ptrs.data_ptr(),
            self._native_s1_ptrs.data_ptr(),
            self._native_s2_ptrs.data_ptr(),
            self._native_s3_ptrs.data_ptr(),
            self._swiglu_limit,
        )

    def _run_native_int8_decode_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        if not self._native_int8_enabled:
            self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
            return
        self._refresh_int8_ptrs()
        forward_name = "routed_int8_moe_forward"
        if self._native_topk_persistent_enabled and input_cpu.shape[0] == 1 and hasattr(self._native_mod, "routed_int8_moe_forward_topk_persistent"):
            forward_name = "routed_int8_moe_forward_topk_persistent"
        elif self._native_topk_parallel_enabled and input_cpu.shape[0] == 1 and hasattr(self._native_mod, "routed_int8_moe_forward_topk_parallel"):
            forward_name = "routed_int8_moe_forward_topk_parallel"
        elif self._native_persistent_team_enabled and input_cpu.shape[0] == 1 and hasattr(self._native_mod, "routed_int8_moe_forward_persistent"):
            forward_name = "routed_int8_moe_forward_persistent"
        getattr(self._native_mod, forward_name)(
            input_cpu.data_ptr(),
            expert_ids_cpu.data_ptr(),
            weights_cpu.data_ptr(),
            output_cpu.data_ptr(),
            input_cpu.shape[0],
            input_cpu.shape[1],
            expert_ids_cpu.shape[1],
            self._inter_dim,
            len(self.experts),
            self.experts_start_idx,
            self.experts_end_idx,
            self._native_int8_w1_ptrs.data_ptr(),
            self._native_int8_w2_ptrs.data_ptr(),
            self._native_int8_w3_ptrs.data_ptr(),
            self._native_int8_s1_ptrs.data_ptr(),
            self._native_int8_s2_ptrs.data_ptr(),
            self._native_int8_s3_ptrs.data_ptr(),
            self._swiglu_limit,
        )

    def _should_run_gguf_grouped_expert_prefill(self, batch_size: int) -> bool:
        return (
            not _layer_imatrix_capture_enabled(self.layer_idx)
            and self._gguf_grouped_expert_enabled
            and batch_size > self._gguf_routes_native_max_batch
            and self.has_gguf_raw_experts()
        )

    def _run_gguf_grouped_expert_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> tuple[int, int] | None:
        output_cpu.zero_()
        flat_ids = expert_ids_cpu.reshape(-1)
        token_ids = torch.arange(expert_ids_cpu.size(0), dtype=torch.long, device=expert_ids_cpu.device).repeat_interleave(expert_ids_cpu.size(1))
        local_mask = (flat_ids >= self.experts_start_idx) & (flat_ids < self.experts_end_idx)
        if not local_mask.any():
            return (0, 0) if self._profile_enabled else None
        route_experts = flat_ids[local_mask]
        route_tokens = token_ids[local_mask]
        route_weights = weights_cpu.reshape(-1, 1)[local_mask]
        sort_order = torch.argsort(route_experts)
        route_experts = route_experts[sort_order]
        route_tokens = route_tokens[sort_order]
        route_weights = route_weights[sort_order]
        unique_ids, counts = torch.unique_consecutive(route_experts, return_counts=True)
        call_count = 0
        routed_rows = 0
        offset = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            group_slice = slice(offset, offset + count)
            group_rows = route_tokens[group_slice]
            expert = self.experts[expert_id]
            contribution = expert.forward_cpu(
                input_cpu[group_rows],
                route_weights[group_slice],
                x_cpu=input_cpu[group_rows],
            )
            output_cpu.index_add_(0, group_rows, contribution)
            call_count += 1
            routed_rows += count
            offset += count
        if self._profile_enabled:
            return call_count, routed_rows
        return None

    def _should_run_gguf_routes_native(self, batch_size: int) -> bool:
        return (
            not _layer_imatrix_capture_enabled(self.layer_idx)
            and self._gguf_routes_native_enabled
            and batch_size <= self._gguf_routes_native_max_batch
            and self.has_gguf_raw_experts()
        )

    def _choose_runner(self, batch_size: int):
        native_ready = self._ensure_native_ready()
        if native_ready and self._should_run_int8_native(batch_size):
            return self._run_native_int8_decode_forward
        if self._native_enabled:
            return self._run_native_forward
        if self._should_run_gguf_routes_native(batch_size):
            return self._run_gguf_routes_native_forward
        if self._should_run_gguf_grouped_expert_prefill(batch_size):
            return self._run_gguf_grouped_expert_forward
        return self._run_reference_forward

    def _gguf_expert_raw_blocks(self, expert, backing):
        cached = self._gguf_raw_block_cache.get(backing.expert_id)
        if cached is not None:
            return cached
        try:
            from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader
        except Exception:
            return None
        reader = get_cached_gguf_tensor_reader(backing.gguf_path)
        w1, w1_type, w1_in_dim = reader.read_routed_expert_blocks(backing.w1_name, backing.expert_id, 0, expert.w1.out_features)
        w3, w3_type, w3_in_dim = reader.read_routed_expert_blocks(backing.w3_name, backing.expert_id, 0, expert.w3.out_features)
        w2, w2_type, w2_in_dim = reader.read_routed_expert_blocks(backing.w2_name, backing.expert_id, 0, expert.w2.out_features)
        cached = (w1, w1_type, w1_in_dim, w3, w3_type, w3_in_dim, w2, w2_type, w2_in_dim)
        self._gguf_raw_block_cache[backing.expert_id] = cached
        return cached

    def _run_gguf_routes_native_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> tuple[int, int] | None:
        if self._native_mod is None or not hasattr(self._native_mod, "gguf_routes_forward"):
            return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
        try:
            from src.loader.gguf.tensor_reader import get_iq2xxs_signed_grid_tensor
        except Exception:
            return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)

        max_routes = expert_ids_cpu.size(0) * expert_ids_cpu.size(1)
        if self._gguf_route_capacity < max_routes:
            self._gguf_route_tokens_buf = torch.empty(max_routes, dtype=torch.long, device="cpu")
            self._gguf_route_weights_buf = torch.empty(max_routes, dtype=torch.float32, device="cpu")
            self._gguf_w1_ptrs_buf = torch.empty(max_routes, dtype=torch.long, device="cpu")
            self._gguf_w3_ptrs_buf = torch.empty(max_routes, dtype=torch.long, device="cpu")
            self._gguf_w2_ptrs_buf = torch.empty(max_routes, dtype=torch.long, device="cpu")
            self._gguf_route_capacity = max_routes
        if self._gguf_grid_cpu is None:
            self._gguf_grid_cpu = get_iq2xxs_signed_grid_tensor().to(device="cpu")

        route_tokens_t = self._gguf_route_tokens_buf
        route_weights_t = self._gguf_route_weights_buf
        w1_ptrs_t = self._gguf_w1_ptrs_buf
        w3_ptrs_t = self._gguf_w3_ptrs_buf
        w2_ptrs_t = self._gguf_w2_ptrs_buf
        route_count = 0
        type_meta = None
        for token in range(expert_ids_cpu.size(0)):
            for top_idx in range(expert_ids_cpu.size(1)):
                expert_id = int(expert_ids_cpu[token, top_idx])
                if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                    continue
                expert = self.experts[expert_id]
                backing = getattr(expert, "_gguf_raw_backing", None) if expert is not None else None
                if backing is None:
                    return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
                blocks = self._gguf_expert_raw_blocks(expert, backing)
                if blocks is None:
                    return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
                w1, w1_type, w1_in_dim, w3, w3_type, w3_in_dim, w2, w2_type, w2_in_dim = blocks
                if w1_in_dim != expert.w1.in_features or w3_in_dim != expert.w3.in_features or w2_in_dim != expert.w2.in_features:
                    return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
                w1_type_id = _native_gguf_type_id(w1_type)
                w3_type_id = _native_gguf_type_id(w3_type)
                w2_type_id = _native_gguf_type_id(w2_type)
                if w1_type_id is None or w3_type_id is None or w2_type_id is None:
                    # The native kernel only implements iq2_xxs/q2_k; iq1_m (and any
                    # other type) must fall back to the reference decode rather than
                    # being silently treated as q2_k.
                    return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
                current_meta = (
                    w1.shape[1], w1.shape[2], w1_type_id,
                    w3.shape[1], w3.shape[2], w3_type_id,
                    w2.shape[1], w2.shape[2], w2_type_id,
                    expert.w1.in_features, expert.w1.out_features,
                )
                if type_meta is None:
                    type_meta = current_meta
                elif type_meta != current_meta:
                    return self._run_reference_forward(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
                route_tokens_t[route_count] = token
                route_weights_t[route_count] = float(weights_cpu[token, top_idx])
                w1_ptrs_t[route_count] = w1.data_ptr()
                w3_ptrs_t[route_count] = w3.data_ptr()
                w2_ptrs_t[route_count] = w2.data_ptr()
                route_count += 1
        if route_count == 0:
            return (0, 0) if self._profile_enabled else None
        assert type_meta is not None
        input_contig = input_cpu if input_cpu.device.type == "cpu" and input_cpu.dtype == torch.float32 and input_cpu.is_contiguous() else input_cpu.contiguous().to(device="cpu", dtype=torch.float32)
        self._native_mod.gguf_routes_forward(
            input_contig.data_ptr(),
            route_tokens_t[:route_count].data_ptr(),
            route_weights_t[:route_count].data_ptr(),
            output_cpu.data_ptr(),
            route_count,
            input_contig.shape[0],
            type_meta[9],
            type_meta[10],
            w1_ptrs_t[:route_count].data_ptr(),
            w3_ptrs_t[:route_count].data_ptr(),
            w2_ptrs_t[:route_count].data_ptr(),
            type_meta[0],
            type_meta[1],
            type_meta[2],
            type_meta[3],
            type_meta[4],
            type_meta[5],
            type_meta[6],
            type_meta[7],
            type_meta[8],
            float(getattr(self.experts[self.experts_start_idx], "swiglu_limit", 0.0)) if self.experts_end_idx > self.experts_start_idx else 0.0,
            self._gguf_grid_cpu.data_ptr(),
        )
        if self._profile_enabled:
            return route_count, route_count
        return None

    def _run_reference_forward(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> tuple[int, int] | None:
        output_cpu.zero_()
        call_count = 0
        routed_rows = 0
        for top_idx in range(expert_ids_cpu.size(1)):
            ids_col = expert_ids_cpu[:, top_idx]
            local_mask = (ids_col >= self.experts_start_idx) & (ids_col < self.experts_end_idx)
            if not local_mask.any():
                continue
            rows = torch.nonzero(local_mask, as_tuple=False).flatten()
            expert_ids = ids_col[rows]
            sort_order = torch.argsort(expert_ids)
            rows = rows[sort_order]
            expert_ids = expert_ids[sort_order]
            unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
            offset = 0
            for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
                group_rows = rows[offset:offset + count]
                expert = self.experts[expert_id]
                output_cpu[group_rows] += expert.forward_cpu(
                    input_cpu[group_rows],
                    weights_cpu[group_rows, top_idx:top_idx + 1],
                    x_cpu=input_cpu[group_rows],
                )
                call_count += 1
                routed_rows += count
                offset += count
        if self._profile_enabled:
            return call_count, routed_rows
        return None

    def run_forward_cpu_into(
        self,
        input_cpu: torch.Tensor,
        expert_ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
    ) -> None:
        expert_ids_cpu, weights_cpu = self._apply_topk_limit(expert_ids_cpu, weights_cpu)
        runner = self._choose_runner(input_cpu.shape[0])
        if self._profile_enabled:
            t0 = time.perf_counter()
            stats = runner(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)
            stats_text = ""
            if isinstance(stats, tuple) and len(stats) == 2:
                stats_text = f" calls={stats[0]} routed_rows={stats[1]}"
            print(
                f"cpu_moe layer={self.layer_idx} batch={input_cpu.shape[0]} runner={getattr(runner, '__name__', runner)} time={time.perf_counter() - t0:.3f}s{stats_text}",
                flush=True,
            )
        else:
            runner(input_cpu, expert_ids_cpu, weights_cpu, output_cpu)

    def submit_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("CPU routed expert task already pending")
        topk_ids, topk_weights = self._apply_topk_limit(topk_ids, topk_weights)
        current_slot, batch_size, device, ready_event, input_cpu, expert_ids_cpu, weights_cpu, output_cpu, _output_gpu = self._copy_inputs_to_cpu(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        if self._route_profile_enabled and batch_size > 16:
            ids = topk_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
            local = ids[(ids >= self.experts_start_idx) & (ids < self.experts_end_idx)]
            if local.numel() > 0:
                counts = torch.bincount(local - self.experts_start_idx, minlength=self.experts_end_idx - self.experts_start_idx)
                active = counts[counts > 0]
                print(
                    f"route_profile layer={self.layer_idx} batch={batch_size} active={active.numel()} total={int(active.sum())} max={int(active.max())} mean={float(active.float().mean()):.1f}",
                    flush=True,
                )
        native_ready = self._ensure_native_ready()
        if native_ready and self._should_run_int8_native(batch_size):
            runner = self._run_native_int8_decode_forward
        elif self._native_enabled:
            runner = self._run_native_forward
        elif self._should_run_gguf_routes_native(batch_size):
            runner = self._run_gguf_routes_native_forward
        elif self._should_run_gguf_grouped_expert_prefill(batch_size):
            runner = self._run_gguf_grouped_expert_forward
        else:
            runner = self._run_reference_forward
        def run_profiled(*_ignored):
            t0 = time.perf_counter()
            stats = runner(
                input_cpu[current_slot][:batch_size],
                expert_ids_cpu[current_slot][:batch_size],
                weights_cpu[current_slot][:batch_size],
                output_cpu[current_slot][:batch_size],
            )
            stats_text = ""
            if isinstance(stats, tuple) and len(stats) == 2:
                stats_text = f" calls={stats[0]} routed_rows={stats[1]}"
            print(
                f"cpu_moe layer={self.layer_idx} batch={batch_size} runner={getattr(runner, '__name__', runner)} time={time.perf_counter() - t0:.3f}s{stats_text}",
                flush=True,
            )

        if self._should_run_decode_inline(batch_size):
            if ready_event is not None:
                ready_event.synchronize()
            if self._profile_enabled:
                run_profiled()
            else:
                runner(
                    input_cpu[current_slot][:batch_size],
                    expert_ids_cpu[current_slot][:batch_size],
                    weights_cpu[current_slot][:batch_size],
                    output_cpu[current_slot][:batch_size],
                )
            self._pending = _PendingTask(current_slot, batch_size, device, future=None)
            return
        if self._persistent_worker_enabled:
            if self._profile_enabled:
                worker_task = _get_persistent_worker().submit(
                    ready_event,
                    run_profiled,
                    input_cpu[current_slot][:0],
                    expert_ids_cpu[current_slot][:0],
                    weights_cpu[current_slot][:0],
                    output_cpu[current_slot][:0],
                )
            else:
                worker_task = _get_persistent_worker().submit(
                    ready_event,
                    runner,
                    input_cpu[current_slot][:batch_size],
                    expert_ids_cpu[current_slot][:batch_size],
                    weights_cpu[current_slot][:batch_size],
                    output_cpu[current_slot][:batch_size],
                )
            self._pending = _PendingTask(current_slot, batch_size, device, future=None, worker_task=worker_task)
            return
        if self._profile_enabled:
            future = self._executor.submit(
                _run_cpu_task,
                ready_event,
                run_profiled,
                input_cpu[current_slot][:0],
                expert_ids_cpu[current_slot][:0],
                weights_cpu[current_slot][:0],
                output_cpu[current_slot][:0],
            )
        else:
            future = self._executor.submit(
                _run_cpu_task,
                ready_event,
                runner,
                input_cpu[current_slot][:batch_size],
                expert_ids_cpu[current_slot][:batch_size],
                weights_cpu[current_slot][:batch_size],
                output_cpu[current_slot][:batch_size],
            )
        self._pending = _PendingTask(current_slot, batch_size, device, future)

    def _finish_pending(self) -> tuple[int, int, torch.device]:
        if self._pending is None:
            raise RuntimeError("No pending CPU routed expert task")
        current_slot = self._pending.slot
        batch_size = self._pending.batch_size
        device = self._pending.device
        future = self._pending.future
        worker_task = self._pending.worker_task
        self._pending = None
        if future is not None:
            future.result()
        if worker_task is not None:
            _get_persistent_worker().wait(worker_task)
        return current_slot, batch_size, device

    def sync_forward_cpu(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, int, int, torch.device]:
        current_slot, batch_size, device = self._finish_pending()
        (
            _input_cpu,
            _expert_ids_cpu,
            _weights_cpu,
            output_cpu,
            _output_gpu,
        ) = CPURoutedExpertsBuffer.get_buffer(hidden_states.view(-1, hidden_states.shape[-1]), self.num_experts_per_tok, self.output_dim)
        return output_cpu[current_slot][:batch_size], current_slot, batch_size, device

    def copy_cpu_output_to_device(
        self,
        output: torch.Tensor,
        current_slot: int,
        batch_size: int,
        device: torch.device,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if device.type == "cuda":
            (
                _input_cpu,
                _expert_ids_cpu,
                _weights_cpu,
                _output_cpu,
                output_gpu,
            ) = CPURoutedExpertsBuffer.get_buffer(hidden_states.view(-1, hidden_states.shape[-1]), self.num_experts_per_tok, self.output_dim)
            output_gpu[current_slot][:batch_size].copy_(output, non_blocking=True)
            return output_gpu[current_slot][:batch_size]
        return output

    def host_float_allreduce(self, output_cpu: torch.Tensor, rank: int, world_size: int) -> None:
        if self._native_mod is None or not hasattr(self._native_mod, "host_float_allreduce"):
            raise RuntimeError("native host_float_allreduce is unavailable")
        if not output_cpu.is_contiguous() or output_cpu.dtype != torch.float32 or output_cpu.device.type != "cpu":
            raise RuntimeError("host_float_allreduce expects a contiguous CPU float32 tensor")
        self._native_mod.host_float_allreduce(
            output_cpu.data_ptr(),
            output_cpu.numel(),
            int(rank),
            int(world_size),
            int(self._host_reduce_slot),
            self._host_reduce_name,
        )
        self._host_reduce_slot += 1

    def sync_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output, current_slot, batch_size, device = self.sync_forward_cpu(hidden_states)
        return self.copy_cpu_output_to_device(output, current_slot, batch_size, device, hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.submit_forward(hidden_states, topk_ids, topk_weights)
        return self.sync_forward(hidden_states)


def _layer_pointer_tensor_from_backends(backends: Sequence[CPURoutedExpertsBackend], attr: str) -> torch.Tensor:
    ptrs = torch.empty(len(backends), dtype=torch.long)
    for idx, backend in enumerate(backends):
        value = getattr(backend, attr, None)
        if value is None:
            raise RuntimeError(f"backend layer={backend.layer_idx} missing {attr}")
        ptrs[idx] = value.data_ptr()
    return ptrs


def get_in_process_cpu_moe_server_name() -> str | None:
    global _INPROC_SERVER
    with _INPROC_SERVER_LOCK:
        return None if _INPROC_SERVER is None else _INPROC_SERVER.shm_name


def stop_in_process_cpu_moe_server() -> None:
    global _INPROC_SERVER
    with _INPROC_SERVER_LOCK:
        server = _INPROC_SERVER
        _INPROC_SERVER = None
    if server is not None:
        server.close()


def start_in_process_cpu_moe_server(
    backends: Sequence[CPURoutedExpertsBackend],
    dim: int,
    topk: int,
    inter_dim: int,
    n_routed_experts: int,
    swiglu_limit: float,
    shm_name: str | None = None,
    use_v2: bool = True,
) -> str:
    global _INPROC_SERVER
    with _INPROC_SERVER_LOCK:
        if _INPROC_SERVER is not None:
            return _INPROC_SERVER.shm_name

        from src.components.moe.ipc import CPUMoESharedMemory

        if shm_name is None:
            shm_name = os.getenv("DEEPSEEK_CPU_MOE_SERVER_SHM", f"pocketllm_cpu_moe_inproc_{os.getpid()}")

        shm = CPUMoESharedMemory(shm_name, dim, topk, create=True)
        w1_layers = _layer_pointer_tensor_from_backends(backends, "_native_int8_w1_ptrs")
        w2_layers = _layer_pointer_tensor_from_backends(backends, "_native_int8_w2_ptrs")
        w3_layers = _layer_pointer_tensor_from_backends(backends, "_native_int8_w3_ptrs")
        s1_layers = _layer_pointer_tensor_from_backends(backends, "_native_int8_s1_ptrs")
        s2_layers = _layer_pointer_tensor_from_backends(backends, "_native_int8_s2_ptrs")
        s3_layers = _layer_pointer_tensor_from_backends(backends, "_native_int8_s3_ptrs")

        def _run_loop() -> None:
            try:
                run_native_int8_loop(
                    shm_name,
                    w1_layers,
                    w2_layers,
                    w3_layers,
                    s1_layers,
                    s2_layers,
                    s3_layers,
                    len(backends),
                    dim,
                    topk,
                    inter_dim,
                    n_routed_experts,
                    shm.output_slots,
                    float(swiglu_limit),
                    use_v2=use_v2,
                )
            finally:
                try:
                    shm.close(unlink=True)
                except Exception:
                    pass

        thread = threading.Thread(target=_run_loop, name="deepseek-cpu-moe-inproc", daemon=True)
        thread.start()
        _INPROC_SERVER = _InProcessCPUMoEServer(shm_name=shm_name, shm=shm, thread=thread)
        return shm_name
