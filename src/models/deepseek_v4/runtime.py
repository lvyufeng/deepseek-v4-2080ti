import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.profiler import record_function

from src.components.moe.cpu_backend import CPURoutedExpertsBackend, start_in_process_cpu_moe_server
from src.components.moe.gpu_prefill_backend import GPUPrefillMoEBackend
from src.kernels.ops import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn, Packed4BitWeightAlongK, _quantize_int8_weight_torch, soft_bf16_weight_gemm_int8, soft_bf16_weight_gemm_int8_pair_cuda_ext, _SHARED_EXPERT_PAIR_INT8_CUDA, _dequant_fp4_weight_torch, soft_fp8_blockfp8_weight_dequant, q8_0_weight_gemm
from src.kernels.cuda_loader import load_cuda_kernel


world_size = 1
rank = 0
tp_world_size = 1
tp_rank = 0
_cpu_moe_server_ipc = None
_cpu_moe_server_ipc_name = None
_cpu_moe_server_last_seq = 0
_cpu_moe_server_predict_seq = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32
_GGUF_GPU_PREFILL_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}


def _get_gguf_gpu_prefill_copy_stream(device: torch.device) -> torch.cuda.Stream:
    dev_index = int(device.index if device.index is not None else torch.cuda.current_device())
    stream = _GGUF_GPU_PREFILL_COPY_STREAMS.get(dev_index)
    if stream is None:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
        _GGUF_GPU_PREFILL_COPY_STREAMS[dev_index] = stream
    return stream


class _GGUFGpuExpertLRU:
    def __init__(self):
        cap_env = os.getenv("DEEPSEEK_GGUF_GPU_DECODE_CACHE_BYTES")
        if cap_env:
            self.capacity_bytes = int(cap_env)
        else:
            self.capacity_bytes = int(8.0 * 1024 ** 3)
        self.used_bytes = 0
        self.entries: "OrderedDict[tuple, tuple[torch.Tensor, str, int]]" = OrderedDict()
        self.profile = os.getenv("DEEPSEEK_GGUF_GPU_DECODE_CACHE_PROFILE", "0").lower() in {"1", "true", "yes"}
        self.profile_every = max(1, int(os.getenv("DEEPSEEK_GGUF_GPU_DECODE_CACHE_PROFILE_EVERY", "256")))
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._last_profile_total = 0

    @staticmethod
    def _nbytes(gpu_blocks: torch.Tensor) -> int:
        return int(gpu_blocks.numel() * gpu_blocks.element_size())

    def _maybe_profile(self) -> None:
        total_lookups = self.hits + self.misses
        if self.profile and total_lookups >= self._last_profile_total + self.profile_every:
            self._last_profile_total = total_lookups
            print(
                f"gguf_gpu_expert_lru entries={len(self.entries)} used={self.used_bytes / 1024 ** 2:.1f}MiB "
                f"hits={self.hits} misses={self.misses} evictions={self.evictions}",
                flush=True,
            )

    def lookup(self, key):
        entry = self.entries.pop(key, None)
        if entry is None:
            self.misses += 1
            self._maybe_profile()
            return None
        self.entries[key] = entry
        self.hits += 1
        self._maybe_profile()
        return entry

    def insert(self, key, value: "tuple[torch.Tensor, str, int]"):
        gpu_blocks, _type_name, _in_dim = value
        existing = self.entries.pop(key, None)
        if existing is not None:
            self.used_bytes -= self._nbytes(existing[0])
        nbytes = self._nbytes(gpu_blocks)
        if nbytes > self.capacity_bytes:
            return False
        while self.entries and self.used_bytes + nbytes > self.capacity_bytes:
            _old_key, old_entry = self.entries.popitem(last=False)
            self.used_bytes -= self._nbytes(old_entry[0])
            self.evictions += 1
        self.entries[key] = value
        self.used_bytes += nbytes
        return True

    def contains(self, key) -> bool:
        return key in self.entries

    def remove(self, key) -> None:
        existing = self.entries.pop(key, None)
        if existing is not None:
            self.used_bytes -= self._nbytes(existing[0])

    def clear(self) -> None:
        self.entries.clear()
        self.used_bytes = 0


_GGUF_GPU_EXPERT_LRU = _GGUFGpuExpertLRU()


def _get_cpu_moe_server_shm_name() -> str:
    global _cpu_moe_server_ipc_name
    if _cpu_moe_server_ipc_name is not None:
        return _cpu_moe_server_ipc_name
    default_name = os.getenv("DEEPSEEK_CPU_MOE_SERVER_SHM", "pocketllm_cpu_moe_server")
    if dist.is_initialized() and _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER") and world_size > 1:
        objects = [default_name if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0)
        _cpu_moe_server_ipc_name = str(objects[0])
    else:
        _cpu_moe_server_ipc_name = default_name
    return _cpu_moe_server_ipc_name


def _get_cpu_moe_server_ipc(dim: int, topk: int):
    global _cpu_moe_server_ipc, _cpu_moe_server_ipc_name
    from src.components.moe.ipc import CPUMoESharedMemory
    name = _get_cpu_moe_server_shm_name()
    if _cpu_moe_server_ipc is None or _cpu_moe_server_ipc_name != name:
        _cpu_moe_server_ipc = CPUMoESharedMemory(name, dim, topk, create=False)
        _cpu_moe_server_ipc_name = name
    return _cpu_moe_server_ipc


def _env_enabled(name: str) -> bool:
    active_name = name.replace("DEEPSEEK_", "DEEPSEEK_ACTIVE_", 1)
    if active_name in os.environ:
        return os.getenv(active_name, "0").lower() in {"1", "true", "yes"}
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _env_enabled_default_on(name: str) -> bool:
    active_name = name.replace("DEEPSEEK_", "DEEPSEEK_ACTIVE_", 1)
    if active_name in os.environ:
        return os.getenv(active_name, "1").lower() not in {"0", "false", "no", "off"}
    val = os.getenv(name)
    if val is None:
        return True
    return val.lower() not in {"0", "false", "no", "off"}


def _any_phase_env_enabled(suffix: str) -> bool:
    return any(
        os.getenv(f"DEEPSEEK_PD_{phase}_{suffix}", "0").lower() in {"1", "true", "yes"}
        for phase in ("PREFILL", "DECODE")
    )

def _phase_env_enabled(suffix: str, default: bool = False) -> bool:
    phase = os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "")
    active_key = f"DEEPSEEK_ACTIVE_{suffix}"
    if active_key in os.environ:
        return _env_enabled(active_key)
    if phase in {"prefill", "decode"}:
        phase_key = f"DEEPSEEK_PD_{phase.upper()}_{suffix}"
        if phase_key in os.environ:
            return _env_enabled(phase_key)
    global_key = f"DEEPSEEK_{suffix}"
    if global_key in os.environ:
        return _env_enabled(global_key)
    return default


def _parse_int_list_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name, "")
    if not raw:
        return default
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return tuple(sorted(set(values))) or default


_CUSTOM_ALLREDUCE_CACHE: dict[tuple[int, torch.dtype, int], "_DecodeCustomAllreduce"] = {}
_CUSTOM_ALLREDUCE_DISABLED = False
_CUSTOM_ALLREDUCE_WARNED = False


class _DecodeCustomAllreduce:
    def __init__(self, dim: int, dtype: torch.dtype, device: torch.device):
        self.dim = int(dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.seq = 0
        self.timeout_us = int(os.getenv("DEEPSEEK_MOE_CUSTOM_ALLREDUCE_TIMEOUT_US", "5000"))
        for src_dev in range(world_size):
            for dst_dev in range(world_size):
                if src_dev != dst_dev and not torch.cuda.can_device_access_peer(src_dev, dst_dev):
                    raise RuntimeError(f"CUDA peer access is not supported between device {src_dev} and {dst_dev}")
        self.ext = load_cuda_kernel()
        if self.ext is None:
            raise RuntimeError("cuda_kernel extension is not available")
        for name in (
            "custom_allreduce_ipc_handle",
            "custom_allreduce_open",
            "custom_allreduce_close",
            "custom_allreduce_inplace",
        ):
            if not hasattr(self.ext, name):
                raise RuntimeError(f"cuda_kernel missing {name}")
        self.buffer = torch.empty((self.dim,), device=self.device, dtype=self.dtype)
        self.flags = torch.zeros((1,), device=self.device, dtype=torch.int32)
        local_handles = self.ext.custom_allreduce_ipc_handle(self.buffer, self.flags)
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, {
            "ok": True,
            "buffer": bytes(local_handles[0]),
            "flag": bytes(local_handles[1]),
        })
        if len(gathered) != 4 or not all(item and item.get("ok") for item in gathered):
            raise RuntimeError("custom allreduce IPC handle exchange failed")
        dtype_code = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}[self.dtype]
        handle = None
        open_error = ""
        try:
            handle = self.ext.custom_allreduce_open(
                self.buffer,
                self.flags,
                [item["buffer"] for item in gathered],
                [item["flag"] for item in gathered],
                int(rank),
                int(world_size),
                self.dim,
                dtype_code,
            )
        except Exception as exc:
            open_error = str(exc)
        open_status = [None for _ in range(world_size)]
        dist.all_gather_object(open_status, {"ok": handle is not None, "error": open_error})
        if not all(item and item.get("ok") for item in open_status):
            if handle is not None:
                self.ext.custom_allreduce_close(handle)
            reasons = [item.get("error", "unknown") for item in open_status if item and not item.get("ok")]
            raise RuntimeError("custom allreduce IPC open failed: " + (reasons[0] if reasons else "unknown"))
        self.handle = handle
        if os.getenv("DEEPSEEK_MOE_CUSTOM_ALLREDUCE_DEBUG", "0").lower() in {"1", "true", "yes"}:
            print(f"custom_moe_allreduce enabled rank={rank} dim={self.dim} dtype={self.dtype}", flush=True)

    def reduce_(self, tensor: torch.Tensor) -> bool:
        self.seq += 1
        return bool(self.ext.custom_allreduce_inplace(self.handle, tensor, self.seq, self.timeout_us))


def _get_decode_custom_allreduce(dim: int, dtype: torch.dtype, device: torch.device) -> Optional[_DecodeCustomAllreduce]:
    global _CUSTOM_ALLREDUCE_DISABLED, _CUSTOM_ALLREDUCE_WARNED
    if _CUSTOM_ALLREDUCE_DISABLED:
        return None
    dev_index = int(device.index if device.index is not None else torch.cuda.current_device())
    key = (dev_index, dtype, int(dim))
    cached = _CUSTOM_ALLREDUCE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        reducer = _DecodeCustomAllreduce(dim, dtype, torch.device("cuda", dev_index))
    except Exception as exc:
        _CUSTOM_ALLREDUCE_DISABLED = True
        if not _CUSTOM_ALLREDUCE_WARNED and os.getenv("DEEPSEEK_MOE_CUSTOM_ALLREDUCE_DEBUG", "0").lower() in {"1", "true", "yes"}:
            print(f"custom_moe_allreduce disabled rank={rank}: {exc}", flush=True)
            _CUSTOM_ALLREDUCE_WARNED = True
        return None
    _CUSTOM_ALLREDUCE_CACHE[key] = reducer
    return reducer


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype is torch.bfloat16:
        return 1
    if dtype is torch.float16:
        return 2
    if dtype is torch.float32:
        return 3
    return 0


def _maybe_fused_moe_finalize(y_reduce: torch.Tensor, shared: torch.Tensor, out_dtype: torch.dtype) -> Optional[torch.Tensor]:
    if not _env_enabled("DEEPSEEK_MOE_FUSED_FINALIZE"):
        return None
    out_code = _dtype_code(out_dtype)
    if out_code == 0 or not y_reduce.is_cuda or not shared.is_cuda or y_reduce.shape != shared.shape:
        return None
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "moe_finalize_reduce_forward"):
        return None
    return ext.moe_finalize_reduce_forward(y_reduce.contiguous(), shared.contiguous(), out_code)


class _CrossLayerGateProfiler:
    def __init__(self):
        self.enabled = _env_enabled("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE")
        self.ks = _parse_int_list_env("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE_KS", (6, 8, 12, 16, 24, 32))
        self.max_k = max(self.ks) if self.ks else 0
        self.percentile = float(os.getenv("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE_PERCENTILE", "75"))
        self.print_every = max(1, int(os.getenv("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE_EVERY", "256")))
        self.max_tokens = max(1, int(os.getenv("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE_MAX_TOKENS", "4")))
        self.events = 0
        self.tokens = 0
        self.routes = 0
        self.local_true = 0
        self.hits = {k: 0 for k in self.ks}
        self.local_hits = {k: 0 for k in self.ks}
        self.local_pred = {k: 0 for k in self.ks}
        self.local_waste = {k: 0 for k in self.ks}
        self.percentile_hits = 0
        self.percentile_local_hits = 0
        self.percentile_local_pred = 0
        self.percentile_local_waste = 0
        self.percentile_pred = 0
        self._announced = False

    def should_profile(self, token_count: int) -> bool:
        return self.enabled and os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "") == "decode" and token_count <= self.max_tokens

    def observe(
        self,
        layer_id: int,
        source_layer_id: int,
        pred_topk: torch.Tensor,
        pred_percentile: torch.Tensor | None,
        true_indices: torch.Tensor,
        experts_start_idx: int,
        experts_end_idx: int,
    ) -> None:
        if not self.enabled:
            return
        if pred_topk.numel() == 0 or true_indices.numel() == 0:
            return
        pred_cpu = pred_topk.detach().to(device="cpu", dtype=torch.long)
        true_cpu = true_indices.detach().to(device="cpu", dtype=torch.long)
        pct_cpu = pred_percentile.detach().to(device="cpu", dtype=torch.long) if pred_percentile is not None else None
        if pred_cpu.dim() != 2 or true_cpu.dim() != 2 or pred_cpu.size(0) != true_cpu.size(0):
            return
        self.events += 1
        self.tokens += int(true_cpu.size(0))
        self.routes += int(true_cpu.numel())
        for row in range(true_cpu.size(0)):
            true_values = [int(v) for v in true_cpu[row].tolist()]
            true_set = set(true_values)
            local_true_set = {v for v in true_set if experts_start_idx <= v < experts_end_idx}
            self.local_true += len(local_true_set)
            pred_values = [int(v) for v in pred_cpu[row].tolist() if int(v) >= 0]
            for k in self.ks:
                pred_set = set(pred_values[:k])
                self.hits[k] += sum(1 for v in true_values if v in pred_set)
                local_pred_set = {v for v in pred_set if experts_start_idx <= v < experts_end_idx}
                self.local_hits[k] += len(local_true_set & local_pred_set)
                self.local_pred[k] += len(local_pred_set)
                self.local_waste[k] += len(local_pred_set - local_true_set)
            if pct_cpu is not None:
                pct_set = {int(v) for v in pct_cpu[row].tolist() if int(v) >= 0}
                self.percentile_hits += sum(1 for v in true_values if v in pct_set)
                pct_local_set = {v for v in pct_set if experts_start_idx <= v < experts_end_idx}
                self.percentile_local_hits += len(local_true_set & pct_local_set)
                self.percentile_local_pred += len(pct_local_set)
                self.percentile_local_waste += len(pct_local_set - local_true_set)
                self.percentile_pred += len(pct_set)
        if not self._announced:
            print(
                f"cross_layer_gate_profile_config rank={rank} ks={','.join(str(k) for k in self.ks)} "
                f"percentile={self.percentile:g} max_tokens={self.max_tokens}",
                flush=True,
            )
            self._announced = True
        if self.events % self.print_every == 0:
            self.print(layer_id, source_layer_id)

    def print(self, layer_id: int, source_layer_id: int) -> None:
        routes = max(1, self.routes)
        local_true = max(1, self.local_true)
        events = max(1, self.events)
        parts = []
        for k in self.ks:
            parts.append(
                f"k{k}_route_hit={self.hits[k] / routes:.4f} "
                f"k{k}_local_hit={self.local_hits[k] / local_true:.4f} "
                f"k{k}_local_pred_per_event={self.local_pred[k] / events:.2f} "
                f"k{k}_local_waste_per_event={self.local_waste[k] / events:.2f}"
            )
        pct = ""
        if self.percentile_pred > 0:
            pct = (
                f" pct_route_hit={self.percentile_hits / routes:.4f}"
                f" pct_local_hit={self.percentile_local_hits / local_true:.4f}"
                f" pct_pred_per_token={self.percentile_pred / max(1, self.tokens):.2f}"
                f" pct_local_pred_per_event={self.percentile_local_pred / events:.2f}"
                f" pct_local_waste_per_event={self.percentile_local_waste / events:.2f}"
            )
        print(
            f"cross_layer_gate_profile rank={rank} events={self.events} tokens={self.tokens} "
            f"routes={self.routes} local_true_per_event={self.local_true / events:.2f} "
            f"last={source_layer_id}->{layer_id} " + " ".join(parts) + pct,
            flush=True,
        )


_cross_layer_gate_profiler_instance = None
_cross_layer_gate_profile_env_enabled = _env_enabled("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE")
_cross_layer_gate_prefetch_env_enabled = _env_enabled("DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH")
_cross_layer_gate_prefetch_k = max(0, int(os.getenv("DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K", "10")))
_cross_layer_gate_prefetch_local_limit = max(0, int(os.getenv("DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT", "2")))
_cross_layer_gate_prediction_max_tokens = max(1, int(os.getenv("DEEPSEEK_GPU_MOE_CROSS_LAYER_PROFILE_MAX_TOKENS", "4")))


def _cross_layer_gate_profile_enabled() -> bool:
    return _cross_layer_gate_profile_env_enabled


def _cross_layer_gate_prefetch_enabled() -> bool:
    return _cross_layer_gate_prefetch_env_enabled and _cross_layer_gate_prefetch_k > 0


def _cross_layer_gate_prediction_enabled() -> bool:
    return _cross_layer_gate_profile_enabled() or _cross_layer_gate_prefetch_enabled()


def _cross_layer_gate_prediction_should_run(token_count: int) -> bool:
    return (
        _cross_layer_gate_prediction_enabled()
        and os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "") == "decode"
        and token_count <= _cross_layer_gate_prediction_max_tokens
    )


def _cross_layer_gate_prediction_max_k() -> int:
    max_k = _cross_layer_gate_prefetch_k if _cross_layer_gate_prefetch_enabled() else 0
    if _cross_layer_gate_profile_enabled():
        max_k = max(max_k, _cross_layer_gate_profiler().max_k)
    return max_k


def _cross_layer_gate_profiler() -> _CrossLayerGateProfiler:
    global _cross_layer_gate_profiler_instance
    if _cross_layer_gate_profiler_instance is None:
        _cross_layer_gate_profiler_instance = _CrossLayerGateProfiler()
    return _cross_layer_gate_profiler_instance


def _pack_fp4_weight_rows_for_tile_decode(
    weight: Packed4BitWeightAlongK,
    scale: torch.Tensor,
    tile_rows: int = 64,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = weight.layout_tensor.contiguous()
    out_features, packed_k = raw.shape
    bytes_per_block = block_size // 2
    out_blocks = packed_k // bytes_per_block
    num_tiles = (out_features + tile_rows - 1) // tile_rows
    packed = torch.zeros((num_tiles, out_blocks, bytes_per_block, tile_rows), dtype=torch.uint8, device="cpu")
    packed_scale = torch.zeros((num_tiles, out_blocks, tile_rows), dtype=torch.float32, device="cpu")
    raw_blocks = raw.view(out_features, out_blocks, bytes_per_block)
    scale = scale.contiguous().view(out_features, out_blocks)
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_rows
        end = min(start + tile_rows, out_features)
        rows = end - start
        packed[tile_idx, :, :, :rows] = raw_blocks[start:end].permute(1, 2, 0)
        packed_scale[tile_idx, :, :rows] = scale[start:end].transpose(0, 1)
    return packed.view(-1), packed_scale.view(-1)


FP4_CPU_TILE_ROWS = 64


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4", "int8"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    routed_experts_device: Literal["gpu", "cpu"] = "gpu"
    attn_int8: bool = False
    shared_expert_int8: bool = False
    mtp_int8: bool = False
    preloaded_attn_int8: bool = False
    preloaded_shared_expert_int8: bool = False
    preload_wq_a_int8: bool = False
    preload_wq_b_int8: bool = False
    preload_wkv_int8: bool = False
    preload_wo_a_int8: bool = False
    preload_wo_b_int8: bool = False
    preload_indexer_wq_b_int8: bool = False
    preload_shared_w1_int8: bool = False
    preload_shared_w2_int8: bool = False
    preload_shared_w3_int8: bool = False
    preload_mtp_e_proj_int8: bool = False
    preload_mtp_h_proj_int8: bool = False
    preload_routed_fp4_dequant: bool = False
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # dspark: multi-token draft stages under the mtp.* namespace. All default to
    # 0/() so a checkpoint without them builds exactly as before; the loader
    # fills them from the checkpoint's config.json when present.
    n_dspark_stages: int = 0
    dspark_block_size: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 0
    dspark_noise_token_id: int = 0
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # Explicit rank layout policy.
    partition_policy: Literal["legacy", "baseline_4gpu", "layer_pp_4gpu"] = "legacy"


@dataclass(frozen=True)
class GGUFExpertRawBacking:
    gguf_path: str
    w1_name: str
    w2_name: str
    w3_name: str
    expert_id: int


def _routed_imatrix_capture_enabled(layer_id: int | None, role: str) -> bool:
    if layer_id is None or os.getenv("DEEPSEEK_IMATRIX_CAPTURE", "0").lower() not in {"1", "true", "yes"}:
        return False
    try:
        from src.loader.gguf.imatrix import collector
        return collector().should_capture(int(layer_id), role)
    except Exception:
        return False


def _maybe_capture_routed_imatrix(layer_id: int | None, expert_id: int | None, role: str, x: torch.Tensor) -> None:
    if not _routed_imatrix_capture_enabled(layer_id, role):
        return
    try:
        from src.loader.gguf.imatrix import capture_routed_expert
        capture_routed_expert(layer_id, expert_id, role, x)
    except Exception as exc:
        if os.getenv("DEEPSEEK_IMATRIX_VERBOSE", "0").lower() in {"1", "true", "yes"}:
            print(f"imatrix_capture_skip layer={layer_id} expert={expert_id} role={role} error={exc}", flush=True)


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // tp_world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % tp_world_size == 0, f"Vocabulary size must be divisible by TP world size (tp_world_size={tp_world_size})"
        self.part_vocab_size = (vocab_size // tp_world_size)
        self.vocab_start_idx = tp_rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tp_world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if tp_world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.int8:
        return soft_bf16_weight_gemm_int8(x, weight, weight.scale)
    else:
        return F.linear(x, weight)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.online_int8_enabled = False
        self._online_int8_ready = False
        self.phase_env_suffix: str | None = None
        self.online_int8_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_ONLINE_INT8_CHUNK_TOKENS", "4096")))
        self._raw_q8_0_ready = False
        self._raw_q8_0_chunk_rows = max(1, int(os.getenv("DEEPSEEK_Q8_0_CHUNK_ROWS", "256")))
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.int8:
            self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scale", torch.empty(out_features, dtype=torch.float32))
            self.weight.scale = self.scale
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def set_int8_storage(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        if weight.dtype != torch.int8:
            raise TypeError(f"expected int8 weight, got {weight.dtype}")
        if scale.dtype != torch.float32:
            raise TypeError(f"expected float32 scale, got {scale.dtype}")
        if weight.shape != self.weight.shape:
            raise ValueError(f"weight shape mismatch: got {tuple(weight.shape)}, expected {tuple(self.weight.shape)}")
        if scale.shape != self.scale.shape:
            raise ValueError(f"scale shape mismatch: got {tuple(scale.shape)}, expected {tuple(self.scale.shape)}")
        self._buffers["weight"] = weight
        self._buffers["scale"] = scale
        self.weight = weight
        self.scale = scale
        self.weight.scale = self.scale
        self._online_int8_ready = False
        self._raw_q8_0_ready = False

    def set_q8_0_storage(self, blocks: torch.Tensor, row_elems: int | None = None) -> None:
        if blocks.dtype != torch.uint8 or blocks.dim() != 3 or blocks.size(-1) != 34:
            raise TypeError(f"expected q8_0 block tensor [rows, blocks, 34] uint8, got {blocks.dtype} {tuple(blocks.shape)}")
        rows, blocks_per_row, _ = blocks.shape
        expected_rows = self.out_features
        expected_blocks = (self.in_features + 31) // 32
        if rows != expected_rows or blocks_per_row != expected_blocks:
            raise ValueError(
                f"q8_0 block shape mismatch: got {tuple(blocks.shape)}, expected ({expected_rows}, {expected_blocks}, 34)"
            )
        row_elems = self.in_features if row_elems is None else int(row_elems)
        if row_elems != self.in_features:
            raise ValueError(f"q8_0 row elems mismatch: got {row_elems}, expected {self.in_features}")
        self.register_buffer("raw_q8_0_weight", blocks.detach().to(device=self.weight.device, dtype=torch.uint8), persistent=False)
        weight = getattr(self, "weight", None)
        if isinstance(weight, torch.Tensor):
            empty_weight = torch.empty(0, dtype=weight.dtype, device=weight.device)
            try:
                weight.requires_grad_(False)
            except Exception:
                pass
            try:
                weight.data = empty_weight
            except Exception:
                if "weight" in self._buffers:
                    self._buffers["weight"] = empty_weight
                    self.weight = empty_weight
        scale = getattr(self, "scale", None)
        if isinstance(scale, torch.Tensor):
            empty_scale = torch.empty(0, dtype=scale.dtype, device=scale.device)
            try:
                scale.requires_grad_(False)
            except Exception:
                pass
            try:
                scale.data = empty_scale
            except Exception:
                if "scale" in self._buffers:
                    self._buffers["scale"] = empty_scale
                else:
                    self.scale = empty_scale
            try:
                self.weight.scale = empty_scale
            except Exception:
                pass
        self._raw_q8_0_ready = True
        self._online_int8_ready = False

    def _raw_q8_0_active(self) -> bool:
        return bool(getattr(self, "_raw_q8_0_ready", False)) and hasattr(self, "raw_q8_0_weight")

    def _raw_q8_0_forward(self, x: torch.Tensor) -> torch.Tensor:
        return q8_0_weight_gemm(
            x,
            self.raw_q8_0_weight,
            row_elems=self.in_features,
            chunk_rows=self._raw_q8_0_chunk_rows,
            out_dtype=torch.get_default_dtype(),
        )

    def enable_online_int8(self) -> None:
        self.online_int8_enabled = True

    def set_preloaded_int8(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.online_int8_enabled = True
        self.register_buffer("online_int8_weight", weight.detach().to(device=self.weight.device, dtype=torch.int8), persistent=False)
        self.register_buffer("online_int8_scale", scale.detach().to(device=self.weight.device, dtype=torch.float32), persistent=False)
        self._online_int8_ready = True

    def _online_int8_active(self) -> bool:
        if self.phase_env_suffix is not None:
            return _phase_env_enabled(self.phase_env_suffix, self.online_int8_enabled)
        return self.online_int8_enabled

    def _ensure_online_int8(self) -> None:
        if not self._online_int8_active() or self._online_int8_ready:
            return
        if self._raw_q8_0_active():
            return
        if self.weight.dtype == torch.int8:
            self.register_buffer("online_int8_weight", self.weight.detach(), persistent=False)
            self.register_buffer("online_int8_scale", self.weight.scale.detach(), persistent=False)
            self._online_int8_ready = True
            return
        if self.weight.dtype == torch.float8_e4m3fn:
            weight = soft_fp8_blockfp8_weight_dequant(self.weight.detach(), self.weight.scale.detach()).float()
        else:
            weight = self.weight.detach().float()
        weight_q, weight_s = _quantize_int8_weight_torch(weight)
        self.register_buffer("online_int8_weight", weight_q, persistent=False)
        self.register_buffer("online_int8_scale", weight_s, persistent=False)
        self._online_int8_ready = True

    def _online_int8_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_online_int8()
        if self._raw_q8_0_active():
            return self._raw_q8_0_forward(x)
        chunk = self.online_int8_chunk_tokens
        if chunk > 0 and x.numel() // x.shape[-1] > chunk:
            flat = x.reshape(-1, x.shape[-1])
            y = torch.empty((flat.size(0), self.out_features), device=x.device, dtype=torch.get_default_dtype())
            for start in range(0, flat.size(0), chunk):
                end = min(start + chunk, flat.size(0))
                y[start:end].copy_(soft_bf16_weight_gemm_int8(
                    flat[start:end].contiguous(),
                    self.online_int8_weight,
                    self.online_int8_scale,
                ))
            return y.view(*x.shape[:-1], self.out_features)
        return soft_bf16_weight_gemm_int8(x, self.online_int8_weight, self.online_int8_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._raw_q8_0_active():
            return self._raw_q8_0_forward(x)
        if self._online_int8_active():
            return self._online_int8_forward(x)
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % tp_world_size == 0, f"Output features must be divisible by world size (tp_world_size={tp_world_size})"
        self.part_out_features = out_features // tp_world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def _online_int8_forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_online_int8()
        if self._raw_q8_0_active():
            return self._raw_q8_0_forward(x)
        chunk = self.online_int8_chunk_tokens
        if chunk > 0 and x.numel() // x.shape[-1] > chunk:
            flat = x.reshape(-1, x.shape[-1])
            y = torch.empty((flat.size(0), self.out_features), device=x.device, dtype=torch.get_default_dtype())
            for start in range(0, flat.size(0), chunk):
                end = min(start + chunk, flat.size(0))
                y[start:end].copy_(soft_bf16_weight_gemm_int8(
                    flat[start:end].contiguous(),
                    self.online_int8_weight,
                    self.online_int8_scale,
                ))
            return y.view(*x.shape[:-1], self.out_features)
        return soft_bf16_weight_gemm_int8(x, self.online_int8_weight, self.online_int8_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._raw_q8_0_active():
            return self._raw_q8_0_forward(x)
        if self._online_int8_active():
            return self._online_int8_forward(x)
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % tp_world_size == 0, f"Input features must be divisible by world size (tp_world_size={tp_world_size})"
        self.part_in_features = in_features // tp_world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._raw_q8_0_active():
            y = self._raw_q8_0_forward(x)
        elif self._online_int8_active():
            y = self._online_int8_forward(x)
        else:
            y = linear(x, self.weight, None)
        if tp_world_size > 1:
            reduce_dtype = x.dtype
            y_reduce = y.to(reduce_dtype)
            dist.all_reduce(y_reduce)
            y = y_reduce.to(torch.float32)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    ext = load_cuda_kernel() if x.is_cuda and x.size(-1) == 128 else None
    if ext is not None and hasattr(ext, "hadamard128_forward"):
        return ext.hadamard128_forward(x)
    return x


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos == 0:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
        return matrix.unsqueeze(0).expand(bsz, -1, -1)
    # start_pos > 0: produce one window-row per query position. Matrix shape
    # is [seqlen, window]. For seqlen=1 this collapses to the original
    # single-row implementation. For seqlen>1 (speculative-verify decode)
    # we additionally mask out slots that the kv_cache write for a *later*
    # position in the same chunk has already overwritten — for query at
    # local index i, those are slots holding positions
    # (start_pos + i+1, ..., start_pos + seqlen - 1).
    rows = []
    for i in range(seqlen):
        cur_pos = start_pos + i
        if cur_pos >= window_size - 1:
            cur = cur_pos % window_size
            row = torch.cat([torch.arange(cur + 1, window_size), torch.arange(0, cur + 1)], dim=0)
        else:
            row = F.pad(torch.arange(cur_pos + 1), (0, window_size - cur_pos - 1), value=-1)
        if seqlen > 1:
            for j in range(i + 1, seqlen):
                future_slot = (start_pos + j) % window_size
                row = torch.where(row == future_slot, torch.full_like(row, -1), row)
        rows.append(row)
    matrix = torch.stack(rows, dim=0)  # [seqlen, window]
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos == 0:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
        return matrix.unsqueeze(0).expand(bsz, -1, -1)
    # start_pos > 0
    if seqlen == 1:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
        return matrix.unsqueeze(0).expand(bsz, -1, -1)
    rows = []
    for i in range(seqlen):
        cur_pos = start_pos + i
        rows.append(torch.arange(0, (cur_pos + 1) // ratio) + offset)
    # Pad rows to the same length so we can stack.
    max_len = max(r.numel() for r in rows)
    padded = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.full((max_len - r.numel(),), -1, dtype=r.dtype)
            r = torch.cat([r, pad], dim=0)
        padded.append(r)
    matrix = torch.stack(padded, dim=0)  # [seqlen, max_len]
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # When overlap, the first half of dims is for overlapping compression, second half for normal.
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        # State buffers for decode-phase incremental compression.
        # With overlap: state[:, :ratio] = overlapping window, state[:, ratio:] = current window.
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # compression need fp32
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.replicated_c4_indexer = _env_enabled("DEEPSEEK_REPLICATED_C4_INDEXER")
        self.full_indexer = args.partition_policy == "layer_pp_4gpu"
        if self.full_indexer:
            self.n_local_heads = self.n_heads
        else:
            self.n_local_heads = self.n_heads if self.replicated_c4_indexer else self.n_heads // tp_world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        indexer_linear = Linear if self.replicated_c4_indexer else ColumnParallelLinear
        self.wq_b = indexer_linear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wq_b.phase_env_suffix = "INDEXER_WQ_B_INT8"
        self.wq_b.replicated_c4_indexer = self.replicated_c4_indexer
        preload_indexer_wq_b = args.preloaded_attn_int8 or args.preload_indexer_wq_b_int8
        preload_indexer_wq_b = preload_indexer_wq_b or _any_phase_env_enabled("INDEXER_WQ_B_INT8")
        if preload_indexer_wq_b:
            self.wq_b.enable_online_int8()
        if _env_enabled("DEEPSEEK_INDEXER_WQ_B_INT8"):
            self.wq_b.enable_online_int8()
        self.weights_proj = indexer_linear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.weights_proj.replicated_c4_indexer = self.replicated_c4_indexer
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None
        self.fused_c4_indexer_enabled = _env_enabled("DEEPSEEK_FUSED_C4_INDEXER_CUDA")
        self._fused_c4_indexer_ext = load_cuda_kernel() if self.fused_c4_indexer_enabled else None
        self._prefill_chunk_tokens = max(
            1, int(os.getenv("DEEPSEEK_INDEXER_PREFILL_CHUNK_TOKENS", "512"))
        )

    def _fused_decode_forward(self, x: torch.Tensor, q: torch.Tensor, start_pos: int, offset: int):
        if self._fused_c4_indexer_ext is None or start_pos == 0 or x.size(1) != 1 or self.compress_ratio != 4:
            return None
        freqs_cis = self.freqs_cis[start_pos:start_pos + 1]
        rd = self.rope_head_dim
        q = q.contiguous()
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.compressor.wkv(x.float()).contiguous()
        score = self.compressor.wgate(x.float()).contiguous()
        weights = (self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)).contiguous()
        freqs = torch.view_as_real(self.freqs_cis[start_pos + 1 - self.compress_ratio]).flatten().contiguous()
        result = self._fused_c4_indexer_ext.fused_c4_indexer_decode_forward(
            q,
            kv,
            score,
            weights,
            self.compressor.ape,
            self.compressor.norm.weight,
            freqs,
            self.compressor.kv_state,
            self.compressor.score_state,
            self.kv_cache,
            int(start_pos),
            int(offset),
            int(self.index_topk),
            float(self.compressor.norm.eps),
            tp_world_size > 1 and not self.replicated_c4_indexer and not self.full_indexer
        )
        if tp_world_size > 1 and not self.replicated_c4_indexer and not self.full_indexer:
            dist.all_reduce(result)
            return self._fused_c4_indexer_ext.c4_topk_from_scores(
                result.contiguous(),
                int(offset),
                int(min(self.index_topk, (start_pos + 1) // self.compress_ratio)),
            )
        return result

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        if self.fused_c4_indexer_enabled:
            fused = self._fused_decode_forward(x, q, start_pos, offset)
            if fused is not None:
                return fused
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # use fp4 simulation for q and kv in indexer
        fp4_act_quant(q, fp4_block_size, True)
        if start_pos > 0 and seqlen > 1:
            # Decode-time speculative verify: advance compressor state position
            # by position. The compressor's stateful kv_state/score_state
            # depend on `start_pos % ratio`, so feeding a length-2 chunk
            # directly would confuse it.
            for i in range(seqlen):
                self.compressor(x[:, i:i+1], start_pos + i)
        else:
            self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        kv_window = self.kv_cache[:bsz, :end_pos // ratio]
        if seqlen > self._prefill_chunk_tokens:
            t_dim = kv_window.size(1)
            index_score = q.new_empty((bsz, seqlen, t_dim))
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                chunk_score = torch.einsum("bshd,btd->bsht", q[:, qs:qe], kv_window)
                chunk_score = (chunk_score.relu_() * weights[:, qs:qe].unsqueeze(-1)).sum(dim=2)
                index_score[:, qs:qe] = chunk_score
                del chunk_score
        else:
            index_score = torch.einsum("bshd,btd->bsht", q, kv_window)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if tp_world_size > 1 and not self.replicated_c4_indexer and not self.full_indexer:
            dist.all_reduce(index_score)
        if start_pos == 0:
            t_dim = index_score.size(-1)
            t_idx = torch.arange(t_dim, device=index_score.device).unsqueeze(0)
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                valid_per_row = ((torch.arange(qs + 1, qe + 1, device=index_score.device)) // ratio).unsqueeze(-1)
                mask = t_idx >= valid_per_row
                index_score[:, qs:qe].masked_fill_(mask.unsqueeze(0), float("-inf"))
        elif seqlen > 1:
            # Speculative-verify decode: each query position `i` may only attend
            # to compressor cells `[0 .. (start_pos+i+1)//ratio)`. Mask the
            # later ones (which exist in the slice because end_pos//ratio
            # rounds up over the seqlen=2 window).
            t_dim = index_score.size(-1)
            t_idx = torch.arange(t_dim, device=index_score.device).unsqueeze(0)  # [1, t]
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                valid_per_row = torch.tensor(
                    [min(t_dim, (start_pos + i + 1) // ratio) for i in range(qs, qe)],
                    device=index_score.device,
                )
                mask = t_idx >= valid_per_row.unsqueeze(-1)  # [chunk, t]
                index_score[:, qs:qe].masked_fill_(mask.unsqueeze(0), float("-inf"))
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                part = topk_idxs[:, qs:qe]
                valid_per_row = ((torch.arange(qs + 1, qe + 1, device=part.device)) // ratio).view(1, qe - qs, 1)
                invalid = part >= valid_per_row
                part.add_(offset)
                part.masked_fill_(invalid, -1)
        elif seqlen > 1:
            # Per-row masking: position i can only validly index cells up to
            # (start_pos+i+1)//ratio. Beyond that, mark as -1.
            chunk = self._prefill_chunk_tokens
            for qs in range(0, seqlen, chunk):
                qe = min(qs + chunk, seqlen)
                part = topk_idxs[:, qs:qe]
                valid_per_row = torch.tensor(
                    [(start_pos + i + 1) // ratio for i in range(qs, qe)],
                    device=part.device,
                ).view(1, qe - qs, 1)
                invalid = part >= valid_per_row
                part.add_(offset)
                part.masked_fill_(invalid, -1)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        if args.partition_policy == "layer_pp_4gpu":
            self.n_local_heads = args.n_heads
            self.n_local_groups = self.n_groups
        else:
            self.n_local_heads = args.n_heads // tp_world_size
            self.n_local_groups = self.n_groups // tp_world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        preload_wq_a = args.preloaded_attn_int8 or args.preload_wq_a_int8
        preload_wq_b = args.preloaded_attn_int8 or args.preload_wq_b_int8
        preload_wkv = args.preloaded_attn_int8 or args.preload_wkv_int8
        preload_wo_a = args.preloaded_attn_int8 or args.preload_wo_a_int8
        preload_wo_b = args.preloaded_attn_int8 or args.preload_wo_b_int8
        preload_wq_a = preload_wq_a or _any_phase_env_enabled("WQ_A_INT8")
        preload_wq_b = preload_wq_b or _any_phase_env_enabled("WQ_B_INT8")
        preload_wkv = preload_wkv or _any_phase_env_enabled("WKV_INT8")
        preload_wo_a = preload_wo_a or _any_phase_env_enabled("WO_A_INT8")
        preload_wo_b = preload_wo_b or _any_phase_env_enabled("WO_B_INT8")

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank, dtype=torch.int8 if args.attn_int8 else None)
        self.wq_a.phase_env_suffix = "WQ_A_INT8"
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wq_b.phase_env_suffix = "WQ_B_INT8"
        self.wkv = Linear(self.dim, self.head_dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wkv.phase_env_suffix = "WKV_INT8"
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.int8 if args.attn_int8 else torch.bfloat16)
        self.wo_a_int8_enabled = args.attn_int8 or _env_enabled("DEEPSEEK_WO_A_INT8") or preload_wo_a
        self.wo_a_fp16_enabled = _env_enabled("DEEPSEEK_WO_A_FP16")
        self.wo_a_bmm_enabled = _env_enabled("DEEPSEEK_WO_A_BMM")
        self._wo_a_int8_ready = False
        self._wo_a_phase_env_enabled = preload_wo_a
        self._wo_a_cuda_ext = load_cuda_kernel() if self.wo_a_int8_enabled else None
        self._wo_a_cuda_enabled = self._wo_a_cuda_ext is not None
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim, dtype=torch.int8 if args.attn_int8 else None)
        self.wo_b.phase_env_suffix = "WO_B_INT8"
        if preload_wq_a:
            self.wq_a.enable_online_int8()
        if preload_wq_b:
            self.wq_b.enable_online_int8()
        if preload_wkv:
            self.wkv.enable_online_int8()
        if preload_wo_b:
            self.wo_b.enable_online_int8()
        if _env_enabled("DEEPSEEK_WQ_A_INT8"):
            self.wq_a.enable_online_int8()
        if _env_enabled("DEEPSEEK_WQ_B_INT8"):
            self.wq_b.enable_online_int8()
        if _env_enabled("DEEPSEEK_WKV_INT8"):
            self.wkv.enable_online_int8()
        if _env_enabled("DEEPSEEK_WO_B_INT8"):
            self.wo_b.enable_online_int8()
        self.softmax_scale = self.head_dim ** -0.5
        self.attn_profile_enabled = _env_enabled("DEEPSEEK_ATTN_PROFILE")
        # Plan B-小-v2: fused attention decode prefuse kernels (q rmsnorm+rope; kv norm+rope+actquant).
        self._fused_attn_prefuse_enabled = _env_enabled("DEEPSEEK_FUSED_ATTN_PREFUSE")
        self._fused_attn_prefuse_ext = load_cuda_kernel() if self._fused_attn_prefuse_enabled else None
        if self._fused_attn_prefuse_ext is not None and (
            not hasattr(self._fused_attn_prefuse_ext, "fused_q_rmsnorm_rope_inplace")
            or not hasattr(self._fused_attn_prefuse_ext, "fused_kv_rope_actquant_inplace")
        ):
            # Built extension does not contain the new ops; disable rather than crash.
            self._fused_attn_prefuse_enabled = False
            self._fused_attn_prefuse_ext = None
        # Plan B-小-v3 gate: fold the per-step kv_cache write into the fused_kv
        # kernel. The kernel writes the produced row directly into
        # kv_cache[:, start_pos % win, :] in addition to the inplace kv buffer,
        # so the Python-side `self.kv_cache[:bsz, slot] = kv.squeeze(1)` and its
        # 80+ select/copy_/as_strided dispatcher ops per layer per step are
        # skipped. Default OFF: A/B over 7 long_long runs (governor pinned)
        # gave baseline {2.049, 2.054, 2.014} vs fold {2.036, 2.103, 1.797,
        # 2.061} -- mean delta -2.0% (1.999 vs 2.039), inside +/-10% jitter.
        # Same dispatcher-bound failure pattern as IMMA / inv-rope: GPU is
        # ~92% idle, so per-op fusion does not move wallclock. Re-test once
        # CUDA Graph capture or async double-buffered MoE lands.
        self._fused_kv_cache_fold_enabled = (
            self._fused_attn_prefuse_enabled
            and os.environ.get("DEEPSEEK_FUSED_KV_CACHE_FOLD", "0") == "1"
        )

        # Separate gate for the inverse-rope back-end fuse so we can A/B it
        # without disabling the prefuse front-end. Default OFF: kernel is
        # bit-exact but A/B over 6 long_long runs (governor pinned) showed mean
        # delta +1.7% (baseline {1.761, 2.074, 2.002} vs inv-rope {2.089, 2.007,
        # 1.841}), well inside the ±10% per-run jitter. Same pattern as IMMA:
        # GPU is ~92% idle so saving 5 dispatches/layer does not move wallclock.
        # Re-test once CUDA Graph capture lands.
        self._fused_attn_inv_rope_enabled = (
            self._fused_attn_prefuse_enabled
            and os.environ.get("DEEPSEEK_FUSED_ATTN_INV_ROPE", "0") == "1"
        )

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False)
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                         rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def set_preloaded_wo_a_int8(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        self.wo_a_int8_enabled = True
        self.register_buffer(
            "wo_a_int8_weight",
            weight.detach().to(device=self.wo_a.weight.device, dtype=torch.int8).view(self.n_local_groups, self.o_lora_rank, -1),
            persistent=False,
        )
        self.register_buffer(
            "wo_a_int8_scale",
            scale.detach().to(device=self.wo_a.weight.device, dtype=torch.float32).view(self.n_local_groups, self.o_lora_rank),
            persistent=False,
        )
        self._wo_a_int8_ready = True

    def _wo_a_int8_active(self) -> bool:
        return _phase_env_enabled("WO_A_INT8", self.wo_a_int8_enabled or self._wo_a_phase_env_enabled)

    def _get_prefuse_freqs(self, start_pos: int, seqlen: int):
        # Returns (real, imag) fp32 contiguous tensors of shape [S, rd/2] on the freqs_cis device.
        # Builds two persistent fp32 buffers once (full sequence length), then narrows per call so
        # decode steps only pay one .narrow() each (no copy, no dtype cast).
        full_r = getattr(self, "_freqs_real_full", None)
        full_i = getattr(self, "_freqs_imag_full", None)
        if full_r is None or full_r.size(0) != self.freqs_cis.size(0) or full_r.device != self.freqs_cis.device:
            full_r = self.freqs_cis.real.to(torch.float32).contiguous()
            full_i = self.freqs_cis.imag.to(torch.float32).contiguous()
            self._freqs_real_full = full_r
            self._freqs_imag_full = full_i
        return (
            full_r.narrow(0, start_pos, seqlen),
            full_i.narrow(0, start_pos, seqlen),
        )

    def _wo_a_fp16_weight(self) -> torch.Tensor:
        weight = getattr(self, "wo_a_fp16_weight", None)
        if weight is None or weight.device != self.wo_a.weight.device:
            weight = self.wo_a.weight.detach().to(dtype=torch.float16).view(self.n_local_groups, self.o_lora_rank, -1)
            self.register_buffer("wo_a_fp16_weight", weight, persistent=False)
        return weight

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        profile = self.attn_profile_enabled
        def mark():
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter() if profile else 0.0
        t0 = mark()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # Decide once per call whether the prefuse kernels apply (decode bf16 path only).
        use_prefuse = (
            self._fused_attn_prefuse_enabled
            and self._fused_attn_prefuse_ext is not None
            and seqlen == 1
            and x.is_cuda
            and x.dtype == torch.bfloat16
        )
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        if use_prefuse:
            fr, fi = self._get_prefuse_freqs(start_pos, seqlen)
            self._fused_attn_prefuse_ext.fused_q_rmsnorm_rope_inplace(q, fr, fi, float(self.eps))
        else:
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            apply_rotary_emb(q[..., -rd:], freqs_cis)
        t_q = mark()

        # win kv & topk_idxs
        # Decide if the fused-kv kernel can also fold in the kv_cache slot
        # write (decode-only fast path: seqlen == 1 and start_pos != 0).
        kv_cache_fold = (
            use_prefuse
            and self._fused_kv_cache_fold_enabled
            and seqlen == 1
            and start_pos != 0
            and bsz <= self.kv_cache.size(0)
        )
        kv = self.wkv(x)
        if use_prefuse:
            fr, fi = self._get_prefuse_freqs(start_pos, seqlen)
            if kv_cache_fold:
                self._fused_attn_prefuse_ext.fused_kv_rope_actquant_inplace(
                    kv, self.kv_norm.weight, fr, fi, 64,
                    float(self.kv_norm.eps),
                    self.kv_cache, int(start_pos % win),
                )
            else:
                self._fused_attn_prefuse_ext.fused_kv_rope_actquant_inplace(
                    kv, self.kv_norm.weight, fr, fi, 64, float(self.kv_norm.eps)
                )
        else:
            kv = self.kv_norm(kv)
            apply_rotary_emb(kv[..., -rd:], freqs_cis)
            # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        t_kv = mark()
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).cuda()
        t_window_idx = mark()
        t_compress_idx = t_window_idx
        t_compressor = t_window_idx
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).cuda()
            t_compress_idx = mark()
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
                t_compressor = mark()
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            # Decode path. seqlen==1 is the standard fast path; seqlen>1 is
            # the speculative-verify path used when speculative decoding is
            # active. For seqlen>1 we write `seqlen` consecutive slots into
            # the windowed kv_cache (with wrap-around), call the per-position
            # compressor seqlen times, and let sparse_attn handle a multi-q
            # batch.
            if not kv_cache_fold:
                if seqlen == 1:
                    self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
                else:
                    for i in range(seqlen):
                        self.kv_cache[:bsz, (start_pos + i) % win] = kv[:, i]
            if self.compress_ratio:
                if seqlen == 1:
                    self.compressor(x, start_pos)
                else:
                    for i in range(seqlen):
                        self.compressor(x[:, i:i+1], start_pos + i)
                t_compressor = mark()
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        t_sparse = mark()
        if use_prefuse and self._fused_attn_inv_rope_enabled and o.is_cuda and o.dtype == torch.bfloat16 and hasattr(self._fused_attn_prefuse_ext, "fused_o_inverse_rope_inplace"):
            fr, fi = self._get_prefuse_freqs(start_pos, seqlen)
            self._fused_attn_prefuse_ext.fused_o_inverse_rope_inplace(o, fr, fi)
        else:
            apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        if getattr(self.wo_a, "_raw_q8_0_active", lambda: False)():
            blocks = self.wo_a.raw_q8_0_weight.view(self.n_local_groups, self.o_lora_rank, self.wo_a.raw_q8_0_weight.size(1), 34)
            proj_chunks = []
            for g in range(self.n_local_groups):
                proj_g = q8_0_weight_gemm(
                    o[:, :, g, :],
                    blocks[g],
                    row_elems=self.wo_a.in_features,
                    chunk_rows=self.wo_a._raw_q8_0_chunk_rows,
                    out_dtype=torch.get_default_dtype(),
                )
                proj_chunks.append(proj_g.unsqueeze(2))
            o = torch.cat(proj_chunks, dim=2)
        elif self._wo_a_int8_active():
            if not self._wo_a_int8_ready:
                if getattr(self.wo_a, "_online_int8_ready", False) and hasattr(self.wo_a, "online_int8_weight"):
                    self.register_buffer(
                        "wo_a_int8_weight",
                        self.wo_a.online_int8_weight.detach().view(self.n_local_groups, self.o_lora_rank, -1),
                        persistent=False,
                    )
                    self.register_buffer(
                        "wo_a_int8_scale",
                        self.wo_a.online_int8_scale.detach().view(self.n_local_groups, self.o_lora_rank),
                        persistent=False,
                    )
                else:
                    weight = self.wo_a.weight.detach().view(self.n_local_groups, self.o_lora_rank, -1)
                    if self.wo_a.weight.dtype == torch.int8:
                        scale = self.wo_a.weight.scale.detach()
                        self.register_buffer("wo_a_int8_weight", weight, persistent=False)
                        self.register_buffer("wo_a_int8_scale", scale.view(self.n_local_groups, -1), persistent=False)
                    else:
                        q_chunks = []
                        s_chunks = []
                        for g in range(self.n_local_groups):
                            weight_q, weight_s = _quantize_int8_weight_torch(weight[g])
                            q_chunks.append(weight_q)
                            s_chunks.append(weight_s)
                        self.register_buffer("wo_a_int8_weight", torch.stack(q_chunks, dim=0), persistent=False)
                        self.register_buffer("wo_a_int8_scale", torch.stack(s_chunks, dim=0), persistent=False)
                self._wo_a_int8_ready = True
            if self._wo_a_cuda_enabled and o.is_cuda:
                o = self._wo_a_cuda_ext.wo_a_int8_forward(
                    o.contiguous(),
                    self.wo_a_int8_weight.contiguous(),
                    self.wo_a_int8_scale.contiguous(),
                ).to(torch.get_default_dtype())
            else:
                proj_chunks = []
                for g in range(self.n_local_groups):
                    proj_g = soft_bf16_weight_gemm_int8(o[:, :, g, :], self.wo_a_int8_weight[g], self.wo_a_int8_scale[g])
                    proj_chunks.append(proj_g.unsqueeze(2))
                o = torch.cat(proj_chunks, dim=2)
        else:
            if self.wo_a_fp16_enabled and o.is_cuda and self.wo_a.weight.is_cuda:
                wo_a = self._wo_a_fp16_weight()
                o = torch.einsum("bsgd,grd->bsgr", o.to(torch.float16), wo_a).to(torch.get_default_dtype())
            else:
                wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
                if self.wo_a_bmm_enabled and o.is_cuda:
                    o_shape = o.shape
                    o = torch.bmm(
                        o.permute(2, 0, 1, 3).reshape(self.n_local_groups, -1, o_shape[-1]),
                        wo_a.transpose(1, 2),
                    ).view(self.n_local_groups, o_shape[0], o_shape[1], self.o_lora_rank).permute(1, 2, 0, 3)
                else:
                    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        t_wo_a = mark()
        x = self.wo_b(o.flatten(2))
        t_wo_b = mark()
        if profile:
            print(
                f"attn_detail layer={self.layer_id} pos={start_pos} batch={bsz} seqlen={seqlen} ratio={ratio} topk={topk_idxs.size(-1)} "
                f"q={t_q - t0:.6f}s kv={t_kv - t_q:.6f}s win_idx={t_window_idx - t_kv:.6f}s "
                f"cmp_idx={t_compress_idx - t_window_idx:.6f}s compressor={t_compressor - t_compress_idx:.6f}s "
                f"sparse={t_sparse - t_compressor:.6f}s wo_a={t_wo_a - t_sparse:.6f}s wo_b={t_wo_b - t_wo_a:.6f}s",
                flush=True,
            )
        return x


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts.
    Supports hash-based routing (first n_hash_layers) where expert indices are
    predetermined per token ID, and score-based routing (remaining layers)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))

    def predict_indices_from_proxy(
        self,
        x: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        max_k: int,
        percentile: float,
        compute_percentile: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.hash:
            if input_ids is None:
                return torch.empty((0, 0), device=x.device, dtype=torch.long), None
            indices = self.tid2eid[input_ids].to(torch.long)
            return indices, indices if compute_percentile else None
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        if self.bias is not None:
            scores = scores + self.bias
        topk_count = min(max_k, scores.size(-1))
        pred_topk = scores.topk(topk_count, dim=-1)[1]
        if not compute_percentile:
            return pred_topk, None
        pct_fraction = max(0.0, min(1.0, (100.0 - percentile) / 100.0))
        pct_count = min(scores.size(-1), max(self.topk, int(math.ceil(scores.size(-1) * pct_fraction))))
        pred_percentile = scores.topk(pct_count, dim=-1)[1] if pct_count > 0 else None
        return pred_topk, pred_percentile

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0, shared_int8_enabled: bool = False):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit
        self.preload_fp4_dequant = False
        self._cpu_predequantized = False
        self._cpu_materialized = False
        self._cpu_w1 = None
        self._cpu_w2 = None
        self._cpu_w3 = None
        self._cpu_w1_scale = None
        self._cpu_w2_scale = None
        self._cpu_w3_scale = None
        self._cpu_w2_tiled = None
        self._cpu_w2_scale_tiled = None
        self._cpu_tile_rows = FP4_CPU_TILE_ROWS
        self._gguf_raw_backing: GGUFExpertRawBacking | None = None
        self._gguf_chunk_rows = max(1, int(os.getenv("DEEPSEEK_GGUF_CPU_CHUNK_ROWS", "128")))
        self._gguf_raw_chunk_rows = max(1, int(os.getenv("DEEPSEEK_GGUF_NATIVE_RAW_CHUNK_ROWS", "2048")))
        self._gguf_profile_enabled = _env_enabled("DEEPSEEK_GGUF_EXPERT_PROFILE")
        self._gguf_profile_limit = int(os.getenv("DEEPSEEK_GGUF_EXPERT_PROFILE_LIMIT", "64"))
        self._gguf_raw_matmul_enabled = os.getenv("DEEPSEEK_GGUF_NATIVE_RAW_MATMUL", "1").lower() in {"1", "true", "yes"}
        self._gguf_fused_expert_enabled = os.getenv("DEEPSEEK_GGUF_FUSED_EXPERT", "0").lower() in {"1", "true", "yes"}
        self._gguf_cuda_fused_expert_enabled = _env_enabled("DEEPSEEK_GGUF_GPU_FUSED_EXPERT")
        self._gguf_native_mod = None
        self._gguf_cuda_mod = None
        self._gguf_iq2_grid = None
        self._gguf_cuda_iq2_grid = None
        self._gguf_cuda_iq1_grid = None
        self._gguf_cuda_prefetch_enabled = _env_enabled("DEEPSEEK_GGUF_GPU_PREFETCH")
        self._gguf_cuda_profile_enabled = _env_enabled("DEEPSEEK_GGUF_GPU_PROFILE")
        self._gguf_cuda_profile_limit = max(0, int(os.getenv("DEEPSEEK_GGUF_GPU_PROFILE_LIMIT", "32")))
        self._gguf_cuda_prefill_gemm_enabled = _env_enabled("DEEPSEEK_GGUF_GPU_PREFILL_GEMM")
        self._gguf_cuda_profile_count = 0
        self._gguf_cuda_cache: dict[tuple[str, int, int], tuple[torch.Tensor, str, int]] = {}
        self._gguf_cuda_prefetch_events: dict[tuple[str, int, int], torch.cuda.Event] = {}
        self._gguf_cuda_prefetch_host: dict[tuple[str, int, int], torch.Tensor] = {}
        self._gguf_cuda_prefetch_pending: set[tuple[str, int, int]] = set()
        self._gguf_profile_count = 0
        self._profile_layer_idx: int | None = None
        self._profile_expert_idx: int | None = None
        self.shared_expert_int8_enabled = shared_int8_enabled
        self.shared_expert_fp16_enabled = _env_enabled("DEEPSEEK_SHARED_EXPERT_FP16")
        self.shared_expert_phase_fp16_enabled = _env_enabled("DEEPSEEK_PD_SHARED_EXPERT_FP16")
        self._shared_fp16_ready = False
        self._shared_w13_fp16 = None
        self._shared_w2_fp16 = None
        self._shared_int8_ready = False
        self.shared_expert_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_SHARED_EXPERT_CHUNK_TOKENS", "8192")))
        self.shared_expert_profile_enabled = os.getenv("DEEPSEEK_SHARED_EXPERT_PROFILE", "0").lower() in {"1", "true", "yes"}

    def _predequantize_fp4_weights_on_gpu(self):
        if self._cpu_predequantized or self.w1.weight.dtype != torch.float4_e2m1fn_x2:
            return
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else self.w1.weight.device
        w1 = _dequant_fp4_weight_torch(
            Packed4BitWeightAlongK.convert_from(self.w1.weight.detach().to(device)),
            self.w1.weight.scale.detach().to(device, dtype=torch.float32),
            block_size=fp4_block_size,
        ).cpu().contiguous()
        w2 = _dequant_fp4_weight_torch(
            Packed4BitWeightAlongK.convert_from(self.w2.weight.detach().to(device)),
            self.w2.weight.scale.detach().to(device, dtype=torch.float32),
            block_size=fp4_block_size,
        ).cpu().contiguous()
        w3 = _dequant_fp4_weight_torch(
            Packed4BitWeightAlongK.convert_from(self.w3.weight.detach().to(device)),
            self.w3.weight.scale.detach().to(device, dtype=torch.float32),
            block_size=fp4_block_size,
        ).cpu().contiguous()
        self._cpu_w1 = w1
        self._cpu_w2 = w2
        self._cpu_w3 = w3
        self._cpu_predequantized = True
        self._cpu_materialized = True

    def set_gguf_raw_storage(self, gguf_path: str, w1_name: str, w2_name: str, w3_name: str, expert_id: int) -> None:
        self._gguf_raw_backing = GGUFExpertRawBacking(gguf_path, w1_name, w2_name, w3_name, int(expert_id))
        self._cpu_predequantized = False
        self._cpu_materialized = False
        self._cpu_w1 = None
        self._cpu_w2 = None
        self._cpu_w3 = None
        self._cpu_w1_scale = None
        self._cpu_w2_scale = None
        self._cpu_w3_scale = None
        self._cpu_w2_tiled = None
        self._cpu_w2_scale_tiled = None
        for sub in (self.w1, self.w2, self.w3):
            weight = getattr(sub, "weight", None)
            if weight is None:
                continue
            empty_weight = torch.empty(0, dtype=weight.dtype, device=weight.device)
            try:
                weight.requires_grad_(False)
            except Exception:
                pass
            try:
                weight.data = empty_weight
            except Exception:
                if "weight" in sub._buffers:
                    sub._buffers["weight"] = empty_weight
                    sub.weight = empty_weight
            scale = getattr(weight, "scale", None)
            if scale is not None:
                empty_scale = torch.empty(0, dtype=scale.dtype, device=scale.device)
                try:
                    scale.requires_grad_(False)
                except Exception:
                    pass
                try:
                    scale.data = empty_scale
                except Exception:
                    if "scale" in sub._buffers:
                        sub._buffers["scale"] = empty_scale
                    else:
                        sub.scale = empty_scale
                try:
                    sub.weight.scale = empty_scale
                except Exception:
                    pass

    def _gguf_cuda_key(self, name: str, device: torch.device) -> tuple[str, str, int, int]:
        dev_index = int(device.index if device.index is not None else torch.cuda.current_device())
        expert_id = int(self._gguf_raw_backing.expert_id) if self._gguf_raw_backing is not None else -1
        gguf_path = self._gguf_raw_backing.gguf_path if self._gguf_raw_backing is not None else ""
        return gguf_path, name, expert_id, dev_index

    def _gguf_type_id(self, type_name: str) -> int | None:
        if type_name == "iq2_xxs":
            return 0
        if type_name == "q2_k":
            return 1
        if type_name == "iq1_m":
            return 2
        return None

    def _gguf_cuda_grid(self, device: torch.device) -> torch.Tensor:
        from src.loader.gguf.tensor_reader import get_iq2xxs_signed_grid_tensor

        if self._gguf_cuda_iq2_grid is None or self._gguf_cuda_iq2_grid.device != device:
            self._gguf_cuda_iq2_grid = get_iq2xxs_signed_grid_tensor().to(device=device, non_blocking=False).contiguous()
        return self._gguf_cuda_iq2_grid

    def _gguf_cuda_iq1_grid_tensor(self, device: torch.device) -> torch.Tensor:
        from src.loader.gguf.tensor_reader import get_iq1_grid_tensor

        if self._gguf_cuda_iq1_grid is None or self._gguf_cuda_iq1_grid.device != device:
            self._gguf_cuda_iq1_grid = get_iq1_grid_tensor().to(device=device, non_blocking=False).contiguous()
        return self._gguf_cuda_iq1_grid

    def _gguf_cuda_quant_grid(self, type_name: str, device: torch.device) -> torch.Tensor:
        """Return the codebook tensor a kernel needs for ``type_name``.

        iq2_xxs and iq1_m use different grids; q2_k uses none.  Passing the
        wrong grid would silently corrupt output, so this is the single source
        of truth for grid selection.
        """
        if type_name == "iq2_xxs":
            return self._gguf_cuda_grid(device)
        if type_name == "iq1_m":
            return self._gguf_cuda_iq1_grid_tensor(device)
        return torch.empty(0, dtype=torch.int8, device=device)

    def _gguf_cuda_group_grid(self, type_names: tuple[str, ...], device: torch.device) -> torch.Tensor:
        """Pick the single grid a paired/grouped kernel call needs.

        The CUDA kernels accept one grid argument shared across w1/w3/w2, so the
        weight types may not mix iq2_xxs with iq1_m.  q2_k contributes no grid.
        """
        has_iq2 = "iq2_xxs" in type_names
        has_iq1 = "iq1_m" in type_names
        if has_iq2 and has_iq1:
            raise RuntimeError("cannot mix iq2_xxs and iq1_m in one GGUF CUDA grouped/pair call")
        if has_iq2:
            return self._gguf_cuda_grid(device)
        if has_iq1:
            return self._gguf_cuda_iq1_grid_tensor(device)
        return torch.empty(0, dtype=torch.int8, device=device)

    def _gguf_cuda_read_blocks(self, reader, name: str) -> tuple[torch.Tensor, str, int] | None:
        if self._gguf_raw_backing is None:
            return None
        try:
            if name == self._gguf_raw_backing.w1_name:
                rows = self.w1.out_features
                expected_in = self.w1.in_features
            elif name == self._gguf_raw_backing.w3_name:
                rows = self.w3.out_features
                expected_in = self.w3.in_features
            elif name == self._gguf_raw_backing.w2_name:
                rows = self.w2.out_features
                expected_in = self.w2.in_features
            else:
                return None
            blocks, type_name, in_dim = reader.read_routed_expert_blocks(name, self._gguf_raw_backing.expert_id, 0, rows)
            if in_dim != expected_in:
                return None
            return blocks, type_name, in_dim
        except Exception:
            return None

    def prefetch_cuda_gguf(self, reader, device: torch.device, force: bool = False) -> bool:
        if ((not force and not self._gguf_cuda_prefetch_enabled) or not self._gguf_cuda_fused_expert_enabled or self._gguf_raw_backing is None):
            return False
        if device.type != "cuda":
            return False
        copy_stream = _get_gguf_gpu_prefill_copy_stream(device)
        prepared = False
        for name in (self._gguf_raw_backing.w1_name, self._gguf_raw_backing.w3_name, self._gguf_raw_backing.w2_name):
            key = self._gguf_cuda_key(name, device)
            if _GGUF_GPU_EXPERT_LRU.contains(key) or key in self._gguf_cuda_prefetch_pending:
                prepared = True
                continue
            packed = self._gguf_cuda_read_blocks(reader, name)
            if packed is None:
                continue
            blocks, type_name, in_dim = packed
            with torch.cuda.stream(copy_stream):
                host_blocks = torch.empty_like(blocks, device="cpu", pin_memory=True)
                host_blocks.copy_(blocks)
                gpu_blocks = torch.empty_like(host_blocks, device=device)
                gpu_blocks.copy_(host_blocks, non_blocking=True)
                gpu_blocks.record_stream(copy_stream)
                event = torch.cuda.Event()
                event.record(copy_stream)
            if not _GGUF_GPU_EXPERT_LRU.insert(key, (gpu_blocks, type_name, in_dim)):
                self._gguf_cuda_cache[key] = (gpu_blocks, type_name, in_dim)
            self._gguf_cuda_prefetch_host[key] = host_blocks
            self._gguf_cuda_prefetch_events[key] = event
            self._gguf_cuda_prefetch_pending.add(key)
            prepared = True
        return prepared

    def _gguf_cuda_get_blocks(self, reader, name: str, device: torch.device) -> tuple[torch.Tensor, str, int] | None:
        key = self._gguf_cuda_key(name, device)
        cached = _GGUF_GPU_EXPERT_LRU.lookup(key)
        if cached is None:
            cached = self._gguf_cuda_cache.get(key)
        if cached is not None:
            event = self._gguf_cuda_prefetch_events.pop(key, None)
            if event is not None:
                torch.cuda.current_stream(device).wait_event(event)
            self._gguf_cuda_prefetch_host.pop(key, None)
            self._gguf_cuda_prefetch_pending.discard(key)
            return cached
        packed = self._gguf_cuda_read_blocks(reader, name)
        if packed is None:
            return None
        blocks, type_name, in_dim = packed
        gpu_blocks = blocks.to(device=device, non_blocking=True).contiguous()
        value = (gpu_blocks, type_name, in_dim)
        if not _GGUF_GPU_EXPERT_LRU.insert(key, value):
            self._gguf_cuda_cache[key] = value
        return value

    def clear_cuda_gguf_cache(self) -> None:
        for event in self._gguf_cuda_prefetch_events.values():
            event.synchronize()
        if self._gguf_raw_backing is not None:
            for name in (
                self._gguf_raw_backing.w1_name,
                self._gguf_raw_backing.w3_name,
                self._gguf_raw_backing.w2_name,
            ):
                for dev_index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
                    key = (
                        self._gguf_raw_backing.gguf_path,
                        name,
                        int(self._gguf_raw_backing.expert_id),
                        dev_index,
                    )
                    _GGUF_GPU_EXPERT_LRU.remove(key)
        self._gguf_cuda_cache.clear()
        self._gguf_cuda_prefetch_events.clear()
        self._gguf_cuda_prefetch_host.clear()
        self._gguf_cuda_prefetch_pending.clear()

    def _gguf_cuda_fused_forward(self, reader, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor | None:
        if not self._gguf_cuda_fused_expert_enabled or self._gguf_raw_backing is None or not x.is_cuda:
            return None
        try:
            cuda_mod = self._gguf_cuda_mod
            if cuda_mod is None:
                cuda_mod = load_cuda_kernel()
                if cuda_mod is None or not hasattr(cuda_mod, "gguf_quant_gemm_forward"):
                    return None
                self._gguf_cuda_mod = cuda_mod
            profile = self._gguf_cuda_profile_enabled and self._gguf_cuda_profile_count < self._gguf_cuda_profile_limit
            t0 = time.perf_counter() if profile else 0.0
            w1_packed = self._gguf_cuda_get_blocks(reader, self._gguf_raw_backing.w1_name, x.device)
            w3_packed = self._gguf_cuda_get_blocks(reader, self._gguf_raw_backing.w3_name, x.device)
            w2_packed = self._gguf_cuda_get_blocks(reader, self._gguf_raw_backing.w2_name, x.device)
            if w1_packed is None or w3_packed is None or w2_packed is None:
                return None
            w1_gpu, w1_type, w1_in_dim = w1_packed
            w3_gpu, w3_type, w3_in_dim = w3_packed
            w2_gpu, w2_type, w2_in_dim = w2_packed
            if w1_in_dim != self.w1.in_features or w3_in_dim != self.w3.in_features or w2_in_dim != self.w2.in_features:
                return None
            current_stream = torch.cuda.current_stream(x.device)
            w1_gpu.record_stream(current_stream)
            w3_gpu.record_stream(current_stream)
            w2_gpu.record_stream(current_stream)
            w1_type_id = self._gguf_type_id(w1_type)
            w3_type_id = self._gguf_type_id(w3_type)
            w2_type_id = self._gguf_type_id(w2_type)
            if w1_type_id is None or w3_type_id is None or w2_type_id is None:
                return None
            if profile:
                torch.cuda.synchronize(x.device)
                t_copy = time.perf_counter()
            x_contig = x.contiguous()
            if hasattr(cuda_mod, "gguf_quant_gemm_pair_forward") and w1_in_dim == w3_in_dim and w1_type == w3_type:
                pair_grid = self._gguf_cuda_quant_grid(w1_type, x.device)
                pair = cuda_mod.gguf_quant_gemm_pair_forward(
                    x_contig,
                    w1_gpu,
                    int(self.w1.in_features),
                    w1_type_id,
                    w3_gpu,
                    int(self.w3.in_features),
                    w3_type_id,
                    pair_grid,
                )
                gate = pair[0].float()
                up = pair[1].float()
            else:
                gate = cuda_mod.gguf_quant_gemm_forward(x_contig, w1_gpu, int(self.w1.in_features), w1_type_id, self._gguf_cuda_quant_grid(w1_type, x.device)).float()
                up = cuda_mod.gguf_quant_gemm_forward(x_contig, w3_gpu, int(self.w3.in_features), w3_type_id, self._gguf_cuda_quant_grid(w3_type, x.device)).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            if weights is not None:
                hidden = hidden * weights.to(device=x.device, dtype=torch.float32)
            out = cuda_mod.gguf_quant_gemm_forward(hidden, w2_gpu, int(self.w2.in_features), w2_type_id, self._gguf_cuda_quant_grid(w2_type, x.device)).float()
            if profile:
                torch.cuda.synchronize(x.device)
                t_done = time.perf_counter()
                self._gguf_cuda_profile_count += 1
                print(
                    f"gguf_gpu_expert_profile rank={rank} layer={self._profile_layer_idx} expert={self._gguf_raw_backing.expert_id} "
                    f"batch={x.shape[0]} cache={t_copy - t0:.6f}s compute={t_done - t_copy:.6f}s total={t_done - t0:.6f}s",
                    flush=True,
                )
            return out
        except Exception as exc:
            if self._gguf_cuda_profile_enabled and self._gguf_cuda_profile_count < self._gguf_cuda_profile_limit:
                self._gguf_cuda_profile_count += 1
                print(f"gguf_gpu_expert_fallback rank={rank} layer={self._profile_layer_idx} expert={self._gguf_raw_backing.expert_id} error={exc}", flush=True)
            return None

    def _gguf_native_fused_forward(self, reader, x_cpu: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor | None:
        if not self._gguf_fused_expert_enabled or self._gguf_raw_backing is None:
            return None
        try:
            from src.loader.gguf.tensor_reader import get_iq2xxs_signed_grid_tensor
            from src.components.moe.cpu_backend import _load_native_mod, _apply_native_runtime_config

            native_mod = self._gguf_native_mod
            if native_mod is None:
                native_mod = _load_native_mod()
                if native_mod is None or not hasattr(native_mod, "gguf_expert_forward"):
                    return None
                _apply_native_runtime_config(native_mod)
                self._gguf_native_mod = native_mod
            elif not hasattr(native_mod, "gguf_expert_forward"):
                return None
            w1_blocks, w1_type, w1_in_dim = reader.read_routed_expert_blocks(
                self._gguf_raw_backing.w1_name,
                self._gguf_raw_backing.expert_id,
                0,
                self.w1.out_features,
            )
            w3_blocks, w3_type, w3_in_dim = reader.read_routed_expert_blocks(
                self._gguf_raw_backing.w3_name,
                self._gguf_raw_backing.expert_id,
                0,
                self.w3.out_features,
            )
            w2_blocks, w2_type, w2_in_dim = reader.read_routed_expert_blocks(
                self._gguf_raw_backing.w2_name,
                self._gguf_raw_backing.expert_id,
                0,
                self.w2.out_features,
            )
            if w1_in_dim != self.w1.in_features or w3_in_dim != self.w3.in_features or w2_in_dim != self.w2.in_features:
                return None
            w1_type_id = self._gguf_type_id(w1_type)
            w3_type_id = self._gguf_type_id(w3_type)
            w2_type_id = self._gguf_type_id(w2_type)
            if w1_type_id is None or w3_type_id is None or w2_type_id is None:
                return None
            x_contig = x_cpu.contiguous().to(device="cpu", dtype=torch.float32)
            weights_contig = weights.detach().cpu().to(torch.float32).contiguous() if weights is not None else torch.empty(0, dtype=torch.float32, device="cpu")
            out = torch.empty((x_contig.shape[0], self.w2.out_features), dtype=torch.float32, device="cpu")
            if self._gguf_iq2_grid is None:
                self._gguf_iq2_grid = get_iq2xxs_signed_grid_tensor()
            native_mod.gguf_expert_forward(
                x_contig.data_ptr(),
                weights_contig.data_ptr() if weights is not None else 0,
                out.data_ptr(),
                w1_blocks.data_ptr(),
                w3_blocks.data_ptr(),
                w2_blocks.data_ptr(),
                x_contig.shape[0],
                self.w1.in_features,
                self.w1.out_features,
                w1_blocks.shape[1],
                w1_blocks.shape[2],
                w1_type_id,
                w3_blocks.shape[1],
                w3_blocks.shape[2],
                w3_type_id,
                w2_blocks.shape[1],
                w2_blocks.shape[2],
                w2_type_id,
                float(self.swiglu_limit),
                self._gguf_iq2_grid.data_ptr(),
            )
            return out
        except Exception:
            return None

    def _gguf_native_raw_matmul(self, reader, name: str, x_cpu: torch.Tensor, row_start: int, rows: int) -> torch.Tensor | None:
        if not self._gguf_raw_matmul_enabled or self._gguf_raw_backing is None:
            return None
        try:
            from src.loader.gguf.tensor_reader import get_iq2xxs_signed_grid_tensor
            from src.components.moe.cpu_backend import _load_native_mod, _apply_native_runtime_config

            native_mod = self._gguf_native_mod
            if native_mod is None:
                native_mod = _load_native_mod()
                if native_mod is None or not hasattr(native_mod, "gguf_quantized_matmul"):
                    return None
                _apply_native_runtime_config(native_mod)
                self._gguf_native_mod = native_mod
            blocks, type_name, in_dim = reader.read_routed_expert_blocks(
                name,
                self._gguf_raw_backing.expert_id,
                row_start,
                rows,
            )
            x_contig = x_cpu.contiguous().to(device="cpu", dtype=torch.float32)
            out = torch.empty((x_contig.shape[0], rows), dtype=torch.float32, device="cpu")
            type_id = self._gguf_type_id(type_name)
            if type_id is None:
                return None
            if type_name == "iq2_xxs":
                if self._gguf_iq2_grid is None:
                    self._gguf_iq2_grid = get_iq2xxs_signed_grid_tensor()
                grid = self._gguf_iq2_grid
            else:
                grid = torch.empty(0, dtype=torch.int8, device="cpu")
            native_mod.gguf_quantized_matmul(
                x_contig.data_ptr(),
                blocks.data_ptr(),
                out.data_ptr(),
                x_contig.shape[0],
                in_dim,
                rows,
                blocks.shape[1],
                blocks.shape[2],
                type_id,
                grid.data_ptr(),
            )
            return out
        except Exception:
            return None

    def _forward_cpu_gguf(self, x_cpu: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._gguf_raw_backing is None:
            raise RuntimeError("GGUF raw backing is not attached")
        from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

        profile = self._gguf_profile_enabled and self._gguf_profile_count < self._gguf_profile_limit
        # imatrix capture only *records* the w1/w3 input (x_cpu) and the w2 input
        # (post-SwiGLU hidden y); both are available regardless of which GEMM path
        # runs, so we keep the fast native/fused matmul and just observe alongside it.
        capture_w13 = _routed_imatrix_capture_enabled(self._profile_layer_idx, "ffn_gate_exps") or _routed_imatrix_capture_enabled(self._profile_layer_idx, "ffn_up_exps")
        capture_w2 = _routed_imatrix_capture_enabled(self._profile_layer_idx, "ffn_down_exps")
        t0 = time.perf_counter() if profile else 0.0
        read_w13_s = 0.0
        linear_w13_s = 0.0
        read_w2_s = 0.0
        linear_w2_s = 0.0
        reader = get_cached_gguf_tensor_reader(self._gguf_raw_backing.gguf_path)
        if capture_w13:
            _maybe_capture_routed_imatrix(self._profile_layer_idx, self._gguf_raw_backing.expert_id, "ffn_gate_exps", x_cpu)
            _maybe_capture_routed_imatrix(self._profile_layer_idx, self._gguf_raw_backing.expert_id, "ffn_up_exps", x_cpu)
        # The fused kernel returns the final expert output directly and never
        # exposes the post-SwiGLU hidden, so only take it when we do not need the
        # w2 input for capture.
        if not capture_w2:
            fused = self._gguf_native_fused_forward(reader, x_cpu, weights)
            if fused is not None:
                return fused
        chunk = self._gguf_raw_chunk_rows if self._gguf_raw_matmul_enabled else self._gguf_chunk_rows
        gate_parts = []
        up_parts = []
        w13_chunks = 0
        for start in range(0, self.w1.out_features, chunk):
            rows = min(chunk, self.w1.out_features - start)
            if profile:
                t_read = time.perf_counter()
            gate_part = self._gguf_native_raw_matmul(reader, self._gguf_raw_backing.w1_name, x_cpu, start, rows)
            up_part = self._gguf_native_raw_matmul(reader, self._gguf_raw_backing.w3_name, x_cpu, start, rows)
            if gate_part is None or up_part is None:
                w1 = reader.read_routed_expert(self._gguf_raw_backing.w1_name, self._gguf_raw_backing.expert_id, start, rows)
                w3 = reader.read_routed_expert(self._gguf_raw_backing.w3_name, self._gguf_raw_backing.expert_id, start, rows)
                if profile:
                    t_linear = time.perf_counter()
                    read_w13_s += t_linear - t_read
                gate_part = torch.nn.functional.linear(x_cpu, w1, None)
                up_part = torch.nn.functional.linear(x_cpu, w3, None)
            elif profile:
                t_linear = time.perf_counter()
                read_w13_s += t_linear - t_read
            if profile:
                linear_w13_s += time.perf_counter() - t_linear
            gate_parts.append(gate_part)
            up_parts.append(up_part)
            w13_chunks += 1
        if profile:
            t_cat = time.perf_counter()
        gate = torch.cat(gate_parts, dim=-1)
        up = torch.cat(up_parts, dim=-1)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = torch.nn.functional.silu(gate) * up
        if weights is not None:
            y = weights.detach().cpu().to(torch.float32) * y
        _maybe_capture_routed_imatrix(self._profile_layer_idx, self._gguf_raw_backing.expert_id, "ffn_down_exps", y)
        if profile:
            act_s = time.perf_counter() - t_cat
        else:
            act_s = 0.0
        out_parts = []
        w2_chunks = 0
        for start in range(0, self.w2.out_features, chunk):
            rows = min(chunk, self.w2.out_features - start)
            if profile:
                t_read = time.perf_counter()
            out_part = self._gguf_native_raw_matmul(reader, self._gguf_raw_backing.w2_name, y, start, rows)
            if out_part is None:
                w2 = reader.read_routed_expert(self._gguf_raw_backing.w2_name, self._gguf_raw_backing.expert_id, start, rows)
                if profile:
                    t_linear = time.perf_counter()
                    read_w2_s += t_linear - t_read
                out_part = torch.nn.functional.linear(y, w2, None)
            elif profile:
                t_linear = time.perf_counter()
                read_w2_s += t_linear - t_read
            out_parts.append(out_part)
            if profile:
                linear_w2_s += time.perf_counter() - t_linear
            w2_chunks += 1
        if profile:
            t_out_cat = time.perf_counter()
        out = torch.cat(out_parts, dim=-1)
        if profile:
            out_cat_s = time.perf_counter() - t_out_cat
            total_s = time.perf_counter() - t0
            self._gguf_profile_count += 1
            print(
                f"gguf_expert_profile rank={rank} layer={self._profile_layer_idx} expert={self._gguf_raw_backing.expert_id} batch={x_cpu.shape[0]} "
                f"chunk={chunk} w13_chunks={w13_chunks} w2_chunks={w2_chunks} "
                f"native_load_w13={read_w13_s:.6f}s linear_w13={linear_w13_s:.6f}s "
                f"act={act_s:.6f}s native_load_w2={read_w2_s:.6f}s linear_w2={linear_w2_s:.6f}s "
                f"out_cat={out_cat_s:.6f}s total={total_s:.6f}s",
                flush=True,
            )
        return out

    def _materialize_cpu_weights(self):
        if self._cpu_materialized:
            return
        if self.preload_fp4_dequant:
            self._predequantize_fp4_weights_on_gpu()
            if self._cpu_materialized:
                return
        if self.w1.weight.dtype == torch.float4_e2m1fn_x2:
            self._cpu_w1 = Packed4BitWeightAlongK.convert_from(self.w1.weight.detach().cpu())
            self._cpu_w2 = Packed4BitWeightAlongK.convert_from(self.w2.weight.detach().cpu())
            self._cpu_w3 = Packed4BitWeightAlongK.convert_from(self.w3.weight.detach().cpu())
            self._cpu_w1_scale = self.w1.weight.scale.detach().to(device="cpu", dtype=torch.float32).clone()
            self._cpu_w2_scale = self.w2.weight.scale.detach().to(device="cpu", dtype=torch.float32).clone()
            self._cpu_w3_scale = self.w3.weight.scale.detach().to(device="cpu", dtype=torch.float32).clone()
            self._cpu_w2_tiled, self._cpu_w2_scale_tiled = _pack_fp4_weight_rows_for_tile_decode(
                self._cpu_w2,
                self._cpu_w2_scale,
                tile_rows=self._cpu_tile_rows,
                block_size=fp4_block_size,
            )
        elif self.w1.weight.dtype == torch.int8:
            self._cpu_w1 = self.w1.weight.detach().cpu().contiguous()
            self._cpu_w2 = self.w2.weight.detach().cpu().contiguous()
            self._cpu_w3 = self.w3.weight.detach().cpu().contiguous()
            self._cpu_w1_scale = self.w1.weight.scale.detach().cpu().float().contiguous()
            self._cpu_w2_scale = self.w2.weight.scale.detach().cpu().float().contiguous()
            self._cpu_w3_scale = self.w3.weight.scale.detach().cpu().float().contiguous()
        else:
            self._cpu_w1 = self.w1.weight.detach().cpu().to(torch.float32)
            self._cpu_w2 = self.w2.weight.detach().cpu().to(torch.float32)
            self._cpu_w3 = self.w3.weight.detach().cpu().to(torch.float32)
        self._cpu_materialized = True

    def _ensure_shared_fp16(self):
        if not (self.shared_expert_fp16_enabled or self.shared_expert_phase_fp16_enabled) or self._shared_fp16_ready:
            return
        if self.w1.weight.dtype == torch.float8_e4m3fn:
            w1 = soft_fp8_blockfp8_weight_dequant(self.w1.weight.detach(), self.w1.weight.scale.detach()).float()
            w2 = soft_fp8_blockfp8_weight_dequant(self.w2.weight.detach(), self.w2.weight.scale.detach()).float()
            w3 = soft_fp8_blockfp8_weight_dequant(self.w3.weight.detach(), self.w3.weight.scale.detach()).float()
        elif self.w1.weight.dtype == torch.int8:
            w1 = self.w1.weight.detach().float() * self.w1.weight.scale.detach().float().unsqueeze(1)
            w2 = self.w2.weight.detach().float() * self.w2.weight.scale.detach().float().unsqueeze(1)
            w3 = self.w3.weight.detach().float() * self.w3.weight.scale.detach().float().unsqueeze(1)
        else:
            w1 = self.w1.weight.detach().float()
            w2 = self.w2.weight.detach().float()
            w3 = self.w3.weight.detach().float()
        self.register_buffer("shared_w13_fp16", torch.cat([w1, w3], dim=0).to(device=self.w1.weight.device, dtype=torch.float16).contiguous(), persistent=False)
        self.register_buffer("shared_w2_fp16", w2.to(device=self.w1.weight.device, dtype=torch.float16).contiguous(), persistent=False)
        self._shared_fp16_ready = True

    def _ensure_shared_int8(self):
        if not self.shared_expert_int8_enabled or self._shared_int8_ready:
            return
        if self.w1.weight.dtype == torch.int8:
            self.register_buffer("shared_int8_w1", self.w1.weight.detach(), persistent=False)
            self.register_buffer("shared_int8_s1", self.w1.weight.scale.detach(), persistent=False)
            self.register_buffer("shared_int8_w2", self.w2.weight.detach(), persistent=False)
            self.register_buffer("shared_int8_s2", self.w2.weight.scale.detach(), persistent=False)
            self.register_buffer("shared_int8_w3", self.w3.weight.detach(), persistent=False)
            self.register_buffer("shared_int8_s3", self.w3.weight.scale.detach(), persistent=False)
            self._shared_int8_ready = True
            return
        if self.w1.weight.dtype == torch.float8_e4m3fn:
            w1 = soft_fp8_blockfp8_weight_dequant(self.w1.weight.detach(), self.w1.weight.scale.detach()).float()
            w2 = soft_fp8_blockfp8_weight_dequant(self.w2.weight.detach(), self.w2.weight.scale.detach()).float()
            w3 = soft_fp8_blockfp8_weight_dequant(self.w3.weight.detach(), self.w3.weight.scale.detach()).float()
        else:
            w1 = self.w1.weight.detach().float()
            w2 = self.w2.weight.detach().float()
            w3 = self.w3.weight.detach().float()
        w1_q, w1_s = _quantize_int8_weight_torch(w1)
        w2_q, w2_s = _quantize_int8_weight_torch(w2)
        w3_q, w3_s = _quantize_int8_weight_torch(w3)
        self.register_buffer("shared_int8_w1", w1_q, persistent=False)
        self.register_buffer("shared_int8_s1", w1_s, persistent=False)
        self.register_buffer("shared_int8_w2", w2_q, persistent=False)
        self.register_buffer("shared_int8_s2", w2_s, persistent=False)
        self.register_buffer("shared_int8_w3", w3_q, persistent=False)
        self.register_buffer("shared_int8_s3", w3_s, persistent=False)
        self._shared_int8_ready = True

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        profile = self.shared_expert_profile_enabled and x.is_cuda and weights is None
        if profile:
            torch.cuda.synchronize(x.device)
            t0 = time.perf_counter()
        use_raw_q8_0 = (
            self.w1._raw_q8_0_active()
            or self.w2._raw_q8_0_active()
            or self.w3._raw_q8_0_active()
        )
        chunk = self.shared_expert_chunk_tokens if (
            not use_raw_q8_0
            and (self.shared_expert_int8_enabled or self.shared_expert_fp16_enabled or self.shared_expert_phase_fp16_enabled)
        ) else 0
        if chunk > 0 and x.dim() == 2 and x.size(0) > chunk:
            parts = []
            for start in range(0, x.size(0), chunk):
                end = min(start + chunk, x.size(0))
                weight_part = weights[start:end] if weights is not None else None
                parts.append(self.forward(x[start:end].contiguous(), weight_part))
            return torch.cat(parts, dim=0).to(dtype)
        use_shared_fp16 = not use_raw_q8_0 and (
            self.shared_expert_fp16_enabled or (
                self.shared_expert_phase_fp16_enabled
                and os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "") == "prefill"
            )
        )
        if use_shared_fp16 and x.is_cuda:
            self._ensure_shared_fp16()
            gate_up = F.linear(x.to(torch.float16), self.shared_w13_fp16).float()
            gate, up = gate_up.chunk(2, dim=-1)
        elif self.shared_expert_int8_enabled:
            self._ensure_shared_int8()
            if _SHARED_EXPERT_PAIR_INT8_CUDA and x.is_cuda:
                try:
                    gate, up = soft_bf16_weight_gemm_int8_pair_cuda_ext(
                        x,
                        self.shared_int8_w1,
                        self.shared_int8_s1,
                        self.shared_int8_w3,
                        self.shared_int8_s3,
                    )
                    gate = gate.float()
                    up = up.float()
                except Exception:
                    gate = soft_bf16_weight_gemm_int8(x, self.shared_int8_w1, self.shared_int8_s1).float()
                    up = soft_bf16_weight_gemm_int8(x, self.shared_int8_w3, self.shared_int8_s3).float()
            else:
                gate = soft_bf16_weight_gemm_int8(x, self.shared_int8_w1, self.shared_int8_s1).float()
                up = soft_bf16_weight_gemm_int8(x, self.shared_int8_w3, self.shared_int8_s3).float()
        else:
            gate = self.w1(x).float()
            up = self.w3(x).float()
        if profile:
            torch.cuda.synchronize(x.device)
            t_w13 = time.perf_counter()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        if profile:
            torch.cuda.synchronize(x.device)
            t_act = time.perf_counter()
        if use_shared_fp16 and x.is_cuda:
            out = F.linear(x.to(torch.float16), self.shared_w2_fp16).to(dtype)
        elif self.shared_expert_int8_enabled:
            out = soft_bf16_weight_gemm_int8(x.to(dtype), self.shared_int8_w2, self.shared_int8_s2).to(dtype)
        else:
            out = self.w2(x.to(dtype))
        if profile:
            torch.cuda.synchronize(out.device)
            t_w2 = time.perf_counter()
            print(
                f"shared_expert_profile w13={t_w13 - t0:.6f}s act={t_act - t_w13:.6f}s w2={t_w2 - t_act:.6f}s total={t_w2 - t0:.6f}s",
                flush=True,
            )
        return out

    def forward_cuda_gguf_staged(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor],
        w1_packed: tuple[torch.Tensor, str, int],
        w3_packed: tuple[torch.Tensor, str, int],
        w2_packed: tuple[torch.Tensor, str, int],
    ) -> torch.Tensor | None:
        if not self._gguf_cuda_fused_expert_enabled or not x.is_cuda:
            return None
        try:
            cuda_mod = self._gguf_cuda_mod
            if cuda_mod is None:
                cuda_mod = load_cuda_kernel()
                if cuda_mod is None or not hasattr(cuda_mod, "gguf_quant_gemm_forward"):
                    return None
                self._gguf_cuda_mod = cuda_mod
            w1_gpu, w1_type, w1_in_dim = w1_packed
            w3_gpu, w3_type, w3_in_dim = w3_packed
            w2_gpu, w2_type, w2_in_dim = w2_packed
            if w1_in_dim != self.w1.in_features or w3_in_dim != self.w3.in_features or w2_in_dim != self.w2.in_features:
                return None
            w1_type_id = self._gguf_type_id(w1_type)
            w3_type_id = self._gguf_type_id(w3_type)
            w2_type_id = self._gguf_type_id(w2_type)
            if w1_type_id is None or w3_type_id is None or w2_type_id is None:
                return None
            current_stream = torch.cuda.current_stream(x.device)
            w1_gpu.record_stream(current_stream)
            w3_gpu.record_stream(current_stream)
            w2_gpu.record_stream(current_stream)
            x_contig = x.contiguous()
            use_prefill_gemm = self._gguf_cuda_prefill_gemm_enabled and x_contig.shape[0] > 1 and hasattr(cuda_mod, "gguf_quant_gemm_prefill_forward")
            if hasattr(cuda_mod, "gguf_quant_gemm_pair_forward") and not use_prefill_gemm and w1_in_dim == w3_in_dim and w1_type == w3_type:
                pair_grid = self._gguf_cuda_quant_grid(w1_type, x.device)
                pair = cuda_mod.gguf_quant_gemm_pair_forward(
                    x_contig,
                    w1_gpu,
                    int(self.w1.in_features),
                    w1_type_id,
                    w3_gpu,
                    int(self.w3.in_features),
                    w3_type_id,
                    pair_grid,
                )
                gate = pair[0].float()
                up = pair[1].float()
            else:
                gemm = cuda_mod.gguf_quant_gemm_prefill_forward if use_prefill_gemm else cuda_mod.gguf_quant_gemm_forward
                gate = gemm(x_contig, w1_gpu, int(self.w1.in_features), w1_type_id, self._gguf_cuda_quant_grid(w1_type, x.device)).float()
                up = gemm(x_contig, w3_gpu, int(self.w3.in_features), w3_type_id, self._gguf_cuda_quant_grid(w3_type, x.device)).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            if weights is not None:
                hidden = hidden * weights.to(device=x.device, dtype=torch.float32)
            w2_gemm = cuda_mod.gguf_quant_gemm_prefill_forward if use_prefill_gemm else cuda_mod.gguf_quant_gemm_forward
            return w2_gemm(hidden, w2_gpu, int(self.w2.in_features), w2_type_id, self._gguf_cuda_quant_grid(w2_type, x.device)).float()
        except Exception as exc:
            if self._gguf_cuda_profile_enabled and self._gguf_cuda_profile_count < self._gguf_cuda_profile_limit:
                self._gguf_cuda_profile_count += 1
                print(f"gguf_gpu_staged_fallback rank={rank} layer={self._profile_layer_idx} expert={self._gguf_raw_backing.expert_id if self._gguf_raw_backing is not None else -1} error={exc}", flush=True)
            return None


    def forward_cuda_gguf_dense_staged(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor],
        w1_host_packed: tuple[torch.Tensor, str, int],
        w3_host_packed: tuple[torch.Tensor, str, int],
        w2_host_packed: tuple[torch.Tensor, str, int],
        expert_local_idx: int,
    ) -> torch.Tensor | None:
        if not x.is_cuda:
            return None
        try:
            w1_host, _w1_type, w1_in_dim = w1_host_packed
            w3_host, _w3_type, w3_in_dim = w3_host_packed
            w2_host, _w2_type, w2_in_dim = w2_host_packed
            if w1_in_dim != self.w1.in_features or w3_in_dim != self.w3.in_features or w2_in_dim != self.w2.in_features:
                return None
            device = x.device
            w1 = w1_host[expert_local_idx].to(dtype=torch.float32).to(device=device, non_blocking=False).to(torch.bfloat16)
            w3 = w3_host[expert_local_idx].to(dtype=torch.float32).to(device=device, non_blocking=False).to(torch.bfloat16)
            w2 = w2_host[expert_local_idx].to(dtype=torch.float32).to(device=device, non_blocking=False).to(torch.bfloat16)
            x_contig = x.contiguous().to(torch.bfloat16)
            gate = F.linear(x_contig, w1, None).float()
            up = F.linear(x_contig, w3, None).float()
            if self.swiglu_limit > 0:
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
                gate = torch.clamp(gate, max=self.swiglu_limit)
            hidden = torch.nn.functional.silu(gate) * up
            if weights is not None:
                hidden = hidden * weights.to(device=device, dtype=torch.float32)
            return F.linear(hidden.to(torch.bfloat16), w2, None).float()
        except Exception as exc:
            if self._gguf_cuda_profile_enabled and self._gguf_cuda_profile_count < self._gguf_cuda_profile_limit:
                self._gguf_cuda_profile_count += 1
                print(f"gguf_gpu_dense_staged_fallback rank={rank} layer={self._profile_layer_idx} expert={self._gguf_raw_backing.expert_id if self._gguf_raw_backing is not None else -1} error={exc}", flush=True)
            return None

    def forward_cuda_gguf(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor | None:
        if self._gguf_raw_backing is None:
            return None
        from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

        reader = get_cached_gguf_tensor_reader(self._gguf_raw_backing.gguf_path)
        return self._gguf_cuda_fused_forward(reader, x, weights)

    def forward_cpu(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None, x_cpu: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._gguf_raw_backing is not None:
            if x_cpu is None:
                x_cpu = x.detach().cpu().to(torch.float32)
            return self._forward_cpu_gguf(x_cpu, weights)
        self._materialize_cpu_weights()
        if x_cpu is None:
            x_cpu = x.detach().cpu().to(torch.float32)
        if isinstance(self._cpu_w1, Packed4BitWeightAlongK):
            w1 = _dequant_fp4_weight_torch(self._cpu_w1, self._cpu_w1_scale, block_size=fp4_block_size)
            w2 = _dequant_fp4_weight_torch(self._cpu_w2, self._cpu_w2_scale, block_size=fp4_block_size)
            w3 = _dequant_fp4_weight_torch(self._cpu_w3, self._cpu_w3_scale, block_size=fp4_block_size)
        elif isinstance(self._cpu_w1, torch.Tensor) and self._cpu_w1.dtype == torch.uint8:
            # The FP4 arena keeps the physical checkpoint representation as
            # uint8 bytes (two FP4 values per byte) and stores UE8M0 scales as
            # uint8 bytes. Reinterpret both views before dequantizing; passing
            # the packed half-width matrix to F.linear would silently produce
            # a shape mismatch (or, for other dimensions, incorrect results).
            w1_packed = Packed4BitWeightAlongK.convert_from(
                self._cpu_w1.view(torch.int8).view(torch.float4_e2m1fn_x2)
            )
            w2_packed = Packed4BitWeightAlongK.convert_from(
                self._cpu_w2.view(torch.int8).view(torch.float4_e2m1fn_x2)
            )
            w3_packed = Packed4BitWeightAlongK.convert_from(
                self._cpu_w3.view(torch.int8).view(torch.float4_e2m1fn_x2)
            )
            w1_scale = self._cpu_w1_scale.view(torch.float8_e8m0fnu).to(torch.float32).contiguous()
            w2_scale = self._cpu_w2_scale.view(torch.float8_e8m0fnu).to(torch.float32).contiguous()
            w3_scale = self._cpu_w3_scale.view(torch.float8_e8m0fnu).to(torch.float32).contiguous()
            w1 = _dequant_fp4_weight_torch(w1_packed, w1_scale, block_size=fp4_block_size)
            w2 = _dequant_fp4_weight_torch(w2_packed, w2_scale, block_size=fp4_block_size)
            w3 = _dequant_fp4_weight_torch(w3_packed, w3_scale, block_size=fp4_block_size)
        else:
            w1 = self._cpu_w1
            w2 = self._cpu_w2
            w3 = self._cpu_w3
        _maybe_capture_routed_imatrix(self._profile_layer_idx, self._profile_expert_idx, "ffn_gate_exps", x_cpu)
        _maybe_capture_routed_imatrix(self._profile_layer_idx, self._profile_expert_idx, "ffn_up_exps", x_cpu)
        gate = F.linear(x_cpu, w1, None)
        up = F.linear(x_cpu, w3, None)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights.detach().cpu().to(torch.float32) * y
        _maybe_capture_routed_imatrix(self._profile_layer_idx, self._profile_expert_idx, "ffn_down_exps", y)
        y = F.linear(y, w2, None)
        return y


class MoE(nn.Module):
    _gguf_layer_cache_lru = OrderedDict()
    _gguf_max_cached_layers = int(os.getenv("DEEPSEEK_GGUF_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", os.getenv("DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS", "3")))
    _gguf_prefill_profile_count = 0

    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert.
    Experts are sharded across TP ranks; each rank handles n_routed_experts // tp_world_size experts."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        if args.partition_policy == "layer_pp_4gpu":
            self.n_local_experts = args.n_routed_experts
        else:
            assert args.n_routed_experts % tp_world_size == 0, f"Number of experts must be divisible by TP world size (tp_world_size={tp_world_size})"
            self.n_local_experts = args.n_routed_experts // tp_world_size
        self.n_activated_experts = args.n_activated_experts
        self.routed_experts_device = args.routed_experts_device
        self.pd_phase_auto_select_enabled = _env_enabled("DEEPSEEK_PD_PHASE_AUTO_SELECT")
        self.pd_single_cpu_moe_weights_enabled = _env_enabled("DEEPSEEK_PD_SINGLE_CPU_MOE_WEIGHTS")
        self.cpu_moe_inproc_server_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER")
        self.cpu_moe_rank0_server_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_RANK0_SERVER")
        self.cpu_moe_external_server_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_EXTERNAL_SERVER")
        self.cpu_moe_external_prefill_local_enabled = self.cpu_moe_external_server_enabled and os.getenv("DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL", "1").lower() in {"1", "true", "yes"}
        self.cpu_moe_predict_seq_enabled = (self.cpu_moe_external_server_enabled or self.cpu_moe_inproc_server_enabled) and _env_enabled("DEEPSEEK_CPU_MOE_PREDICT_SEQ")
        if self.cpu_moe_external_server_enabled and not self.cpu_moe_external_prefill_local_enabled and not self.pd_single_cpu_moe_weights_enabled:
            self.experts_start_idx = 0
            self.experts_end_idx = 0
        elif self.cpu_moe_inproc_server_enabled:
            # The rank-0 in-process daemon thread runs the full-routes int8
            # CPU MoE kernel; it requires the full per-layer pointer table to
            # cover all routed experts. We therefore have rank 0 own all
            # experts (OOM caveat: increases per-rank GPU prefill MoE staging
            # cache; lower DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS or
            # disable DEEPSEEK_GPU_PREFILL_MOE for long prefill if needed).
            # Ranks 1-3 own no experts and only contribute the shared expert
            # plus the post-MoE allreduce skip.
            if rank == 0:
                self.experts_start_idx = 0
                self.experts_end_idx = self.n_routed_experts
            else:
                self.experts_start_idx = 0
                self.experts_end_idx = 0
        elif self.cpu_moe_rank0_server_enabled and rank == 0:
            self.experts_start_idx = 0
            self.experts_end_idx = self.n_routed_experts
        elif args.partition_policy == "layer_pp_4gpu":
            self.experts_start_idx = 0
            self.experts_end_idx = self.n_routed_experts
        else:
            self.experts_start_idx = tp_rank * self.n_local_experts
            self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = None
        if args.expert_dtype == "fp4":
            expert_dtype = torch.float4_e2m1fn_x2
        elif args.expert_dtype == "int8":
            expert_dtype = torch.int8
        if self.routed_experts_device == "cpu":
            if self.cpu_moe_external_server_enabled and not self.cpu_moe_external_prefill_local_enabled:
                self.experts = [None for _ in range(self.n_routed_experts)]
                self.cpu_backend = None
            else:
                with torch.device("cpu"):
                    self.experts = nn.ModuleList([
                        Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit, shared_int8_enabled=False)
                        if self.experts_start_idx <= i < self.experts_end_idx else None
                        for i in range(self.n_routed_experts)
                    ])
                    for expert_id, expert in enumerate(self.experts):
                        if expert is not None:
                            expert.preload_fp4_dequant = args.preload_routed_fp4_dequant
                            expert._profile_layer_idx = self.layer_id
                            expert._profile_expert_idx = expert_id
                self.cpu_backend = CPURoutedExpertsBackend(
                    layer_idx=self.layer_id,
                    experts=self.experts,
                    experts_start_idx=self.experts_start_idx,
                    experts_end_idx=self.experts_end_idx,
                    num_experts_per_tok=self.n_activated_experts,
                    output_dim=self.dim,
                )
        else:
            self.experts = nn.ModuleList([
                Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit)
                if self.experts_start_idx <= i < self.experts_end_idx else None
                for i in range(self.n_routed_experts)
            ])
            for expert_id, expert in enumerate(self.experts):
                if expert is not None:
                    expert._profile_layer_idx = self.layer_id
                    expert._profile_expert_idx = expert_id
            self.cpu_backend = None
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim,
            args.moe_inter_dim,
            dtype=torch.int8 if args.shared_expert_int8 else None,
            shared_int8_enabled=args.shared_expert_int8 or _env_enabled("DEEPSEEK_SHARED_EXPERT_INT8"),
        )
        if args.preloaded_shared_expert_int8 or args.preload_shared_w1_int8:
            self.shared_experts.w1.enable_online_int8()
        if args.preloaded_shared_expert_int8 or args.preload_shared_w2_int8:
            self.shared_experts.w2.enable_online_int8()
        if args.preloaded_shared_expert_int8 or args.preload_shared_w3_int8:
            self.shared_experts.w3.enable_online_int8()
        self.async_allreduce_enabled = os.getenv("DEEPSEEK_MOE_ASYNC_ALLREDUCE", "0").lower() in {"1", "true", "yes"}
        self.cpu_host_reduce_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_CPU_MOE_HOST_REDUCE")
        self.reduce_fp16_enabled = _env_enabled("DEEPSEEK_MOE_REDUCE_FP16")
        self._profile_enabled = os.getenv("DEEPSEEK_MOE_PROFILE", "0").lower() in {"1", "true", "yes"}
        self._active_profile_enabled = os.getenv("DEEPSEEK_GPU_MOE_ACTIVE_PROFILE", "0").lower() in {"1", "true", "yes"}
        self.gpu_active_overlap_reduce_enabled = _env_enabled("DEEPSEEK_GPU_MOE_ACTIVE_OVERLAP_REDUCE")
        self._rank_route_profile_enabled = _env_enabled("DEEPSEEK_RANK_ROUTE_PROFILE")
        self._allreduce_stream = torch.cuda.Stream() if self.async_allreduce_enabled and torch.cuda.is_available() else None
        self._gguf_decode_shared_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.gpu_prefill_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GPU_PREFILL_MOE")
        self.gpu_gguf_prefill_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_PREFILL_MOE")
        self.gpu_gguf_prefill_active_expert_enabled = self.gpu_gguf_prefill_moe_enabled and _env_enabled("DEEPSEEK_GGUF_GPU_PREFILL_ACTIVE_EXPERT")
        self.gpu_gguf_prefill_active_keep_cache = _env_enabled("DEEPSEEK_GGUF_GPU_PREFILL_ACTIVE_KEEP_CACHE")
        self.gpu_gguf_decode_active_expert_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_ACTIVE_EXPERT")
        self.gpu_gguf_decode_grouped_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_GROUPED")
        self.gpu_gguf_decode_single_token_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_SINGLE_TOKEN")
        self.gpu_gguf_decode_slot_cache_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE")
        self.gpu_gguf_decode_slot_cache_size = max(0, int(os.getenv("DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE_SIZE", "4")))
        self.gpu_gguf_decode_grouped_profile = _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_GROUPED_PROFILE")
        self.gpu_gguf_decode_keep_cache = _env_enabled_default_on("DEEPSEEK_GGUF_GPU_DECODE_KEEP_CACHE")
        self.gpu_gguf_decode_active_prefetch_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_ACTIVE_PREFETCH")
        self.gpu_gguf_decode_overlap_shared_route = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_OVERLAP_SHARED_ROUTE")
        self.gpu_decode_active_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GPU_MOE_DECODE_ACTIVE")
        self.gpu_spec_token2_moe_enabled = self.routed_experts_device == "cpu" and _env_enabled("DEEPSEEK_GPU_MOE_SPEC_TOKEN2")
        self.gpu_prefill_moe_min_tokens = int(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_MIN_TOKENS", "64"))
        self.gpu_prefill_moe_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS", "0")))
        self.moe_reduce_cast_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_MOE_REDUCE_CAST_CHUNK_TOKENS", "8192")))
        self.gpu_prefill_backend = None
        self._gguf_layer_stage = None
        self._gguf_layer_stage_host_cache = None
        self._gguf_layer_stage_event: torch.cuda.Event | None = None
        self._gguf_layer_stage_pending = False
        self._gguf_layer_stage_device: torch.device | None = None
        self._gguf_layer_stage_profile = _env_enabled("DEEPSEEK_GGUF_LAYER_STAGE_PROFILE")
        self._gguf_decode_slot_cache_device: torch.device | None = None
        self._gguf_decode_slot_cache: "OrderedDict[int, int]" = OrderedDict()
        self._gguf_decode_slot_w1_blocks: torch.Tensor | None = None
        self._gguf_decode_slot_w3_blocks: torch.Tensor | None = None
        self._gguf_decode_slot_w2_blocks: torch.Tensor | None = None
        self._gguf_decode_slot_meta: tuple[str, int, str, int, str, int] | None = None
        self._cross_layer_pred_source_layer_id = -1
        self._cross_layer_pred_topk = None
        self._cross_layer_pred_percentile = None
        self._cross_layer_pred_prefetch_done = False

    def _gguf_layer_cache_key(self, device: torch.device) -> tuple[int, int, int, int]:
        dev_index = int(device.index if device.index is not None else torch.cuda.current_device())
        return (int(self.layer_id), dev_index, int(self.experts_start_idx), int(self.experts_end_idx))

    def _touch_gguf_layer_cache(self, device: torch.device) -> None:
        key = self._gguf_layer_cache_key(device)
        self._gguf_layer_cache_lru.pop(key, None)
        self._gguf_layer_cache_lru[key] = self
        while self._gguf_max_cached_layers > 0 and len(self._gguf_layer_cache_lru) > self._gguf_max_cached_layers:
            _old_key, old_moe = self._gguf_layer_cache_lru.popitem(last=False)
            if old_moe is not self:
                old_moe.release_gguf_gpu_prefill_moe()

    def _has_gguf_raw_experts(self) -> bool:
        if self.cpu_backend is None:
            return False
        checker = getattr(self.cpu_backend, "has_gguf_raw_experts", None)
        return bool(checker()) if callable(checker) else False

    def set_cross_layer_gate_prediction(
        self,
        source_layer_id: int,
        pred_topk: torch.Tensor,
        pred_percentile: torch.Tensor | None,
    ) -> None:
        self._cross_layer_pred_source_layer_id = source_layer_id
        self._cross_layer_pred_topk = pred_topk
        self._cross_layer_pred_percentile = pred_percentile
        self._cross_layer_pred_prefetch_done = False

    def _stage_gguf_layer_blocks(self, device: torch.device, async_copy: bool) -> None:
        if not self.gpu_gguf_prefill_moe_enabled or not self._has_gguf_raw_experts() or device.type != "cuda":
            return
        if self._gguf_layer_stage is not None and self._gguf_layer_stage_device == device:
            self._touch_gguf_layer_cache(device)
            return
        first_expert = next((expert for expert in self.experts if expert is not None and expert._gguf_raw_backing is not None), None)
        if first_expert is None:
            return

        from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

        t0 = time.perf_counter() if self._gguf_layer_stage_profile else 0.0
        host_cache = self._gguf_layer_stage_host_cache
        if host_cache is None:
            reader = get_cached_gguf_tensor_reader(first_expert._gguf_raw_backing.gguf_path)
            backing = first_expert._gguf_raw_backing
            w1_cpu, w1_type, w1_in_dim = reader.read_routed_layer_blocks(
                backing.w1_name,
                self.experts_start_idx,
                self.experts_end_idx - self.experts_start_idx,
            )
            w3_cpu, w3_type, w3_in_dim = reader.read_routed_layer_blocks(
                backing.w3_name,
                self.experts_start_idx,
                self.experts_end_idx - self.experts_start_idx,
            )
            w2_cpu, w2_type, w2_in_dim = reader.read_routed_layer_blocks(
                backing.w2_name,
                self.experts_start_idx,
                self.experts_end_idx - self.experts_start_idx,
            )
            w1_host = torch.empty_like(w1_cpu, device="cpu", pin_memory=True)
            w3_host = torch.empty_like(w3_cpu, device="cpu", pin_memory=True)
            w2_host = torch.empty_like(w2_cpu, device="cpu", pin_memory=True)
            w1_host.copy_(w1_cpu)
            w3_host.copy_(w3_cpu)
            w2_host.copy_(w2_cpu)
            host_cache = (
                (w1_host, w1_type, w1_in_dim),
                (w3_host, w3_type, w3_in_dim),
                (w2_host, w2_type, w2_in_dim),
            )
            self._gguf_layer_stage_host_cache = host_cache
        (w1_host, w1_type, w1_in_dim), (w3_host, w3_type, w3_in_dim), (w2_host, w2_type, w2_in_dim) = host_cache
        copy_stream = _get_gguf_gpu_prefill_copy_stream(device) if async_copy else torch.cuda.current_stream(device)
        current_stream = torch.cuda.current_stream(device)
        if async_copy and os.getenv("DEEPSEEK_GGUF_PREFILL_COPY_WAIT_CURRENT", "1").lower() not in {"0", "false", "no"}:
            copy_stream.wait_stream(current_stream)
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            w1_gpu = torch.empty_like(w1_host, device=device)
            w3_gpu = torch.empty_like(w3_host, device=device)
            w2_gpu = torch.empty_like(w2_host, device=device)
            w1_gpu.copy_(w1_host, non_blocking=True)
            w3_gpu.copy_(w3_host, non_blocking=True)
            w2_gpu.copy_(w2_host, non_blocking=True)
            w1_gpu.record_stream(copy_stream)
            w3_gpu.record_stream(copy_stream)
            w2_gpu.record_stream(copy_stream)
            event.record(copy_stream)
        self._gguf_layer_stage = (
            (w1_gpu, w1_type, w1_in_dim),
            (w3_gpu, w3_type, w3_in_dim),
            (w2_gpu, w2_type, w2_in_dim),
            (w1_host, w3_host, w2_host),
        )
        self._gguf_layer_stage_event = event
        self._gguf_layer_stage_pending = True
        self._gguf_layer_stage_device = device
        self._touch_gguf_layer_cache(device)
        if self._gguf_layer_stage_profile:
            print(
                f"gguf_gpu_layer_stage layer={self.layer_id} rank={rank} async={int(async_copy)} "
                f"bytes={(w1_host.numel() + w3_host.numel() + w2_host.numel()) / 1024 ** 2:.1f}MiB "
                f"total={time.perf_counter() - t0:.4f}s",
                flush=True,
            )

    def prefetch_gguf_gpu_prefill_moe(self, device: torch.device, token_count: int) -> None:
        if self.gpu_gguf_prefill_active_expert_enabled:
            return
        if token_count < self.gpu_prefill_moe_min_tokens:
            return
        self._stage_gguf_layer_blocks(device, async_copy=True)

    def wait_gguf_gpu_prefill_moe(self, device: torch.device) -> bool:
        if self._gguf_layer_stage is None:
            self._stage_gguf_layer_blocks(device, async_copy=False)
        if self._gguf_layer_stage is None:
            return False
        if self._gguf_layer_stage_pending and self._gguf_layer_stage_event is not None:
            torch.cuda.current_stream(device).wait_event(self._gguf_layer_stage_event)
            self._gguf_layer_stage_pending = False
        self._touch_gguf_layer_cache(device)
        return True

    def release_gguf_gpu_prefill_moe(self) -> None:
        if self._gguf_layer_stage_device is not None:
            key = self._gguf_layer_cache_key(self._gguf_layer_stage_device)
            if self._gguf_layer_cache_lru.get(key) is self:
                self._gguf_layer_cache_lru.pop(key, None)
        if self._gguf_layer_stage_pending and self._gguf_layer_stage_event is not None:
            self._gguf_layer_stage_event.synchronize()
        self._gguf_layer_stage = None
        self._gguf_layer_stage_host_cache = None
        self._gguf_layer_stage_event = None
        self._gguf_layer_stage_pending = False
        self._gguf_layer_stage_device = None

    def _prefetch_gguf_decode_expert_ids(self, device: torch.device, expert_ids: list[int]) -> int:
        if not expert_ids or not self._has_gguf_raw_experts() or device.type != "cuda":
            return 0
        try:
            from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

            first_expert = next((expert for expert in self.experts if expert is not None and expert._gguf_raw_backing is not None), None)
            if first_expert is None:
                return 0
            reader = get_cached_gguf_tensor_reader(first_expert._gguf_raw_backing.gguf_path)
            prepared = 0
            for expert_id in sorted(set(int(eid) for eid in expert_ids)):
                if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
                    continue
                expert = self.experts[expert_id]
                if expert is None or expert._gguf_raw_backing is None:
                    continue
                if expert.prefetch_cuda_gguf(reader, device, force=True):
                    prepared += 1
            return prepared
        except Exception as exc:
            if _env_enabled("DEEPSEEK_GGUF_GPU_PROFILE"):
                print(f"gguf_decode_active_prefetch_fallback rank={rank} layer={self.layer_id} error={exc}", flush=True)
            return 0

    def _prefetch_gguf_decode_local_experts(self, device: torch.device, local_ids: list[int]) -> None:
        if not local_ids or not self._has_gguf_raw_experts() or device.type != "cuda":
            return
        prepared = self._prefetch_gguf_decode_expert_ids(
            device,
            [int(local_id) + int(self.experts_start_idx) for local_id in local_ids],
        )
        if _env_enabled("DEEPSEEK_GGUF_GPU_PROFILE") and prepared:
            for local_id in local_ids:
                expert_id = int(local_id) + int(self.experts_start_idx)
                if self.experts_start_idx <= expert_id < self.experts_end_idx:
                    print(
                        f"gguf_decode_cross_layer_prefetch rank={rank} layer={self.layer_id} local={int(local_id)} expert={expert_id}",
                        flush=True,
                    )

    def prefetch_cross_layer_gate_prediction(self, device: torch.device) -> None:
        if self._cross_layer_pred_prefetch_done or not _cross_layer_gate_prefetch_enabled():
            return
        pred_topk = self._cross_layer_pred_topk
        if pred_topk is None or pred_topk.numel() == 0:
            return
        if (
            self._pd_active_phase() != "decode"
            or self.cpu_backend is None
            or self.experts_end_idx <= self.experts_start_idx
            or device.type != "cuda"
        ):
            return
        K = _cross_layer_gate_prefetch_k
        limit = _cross_layer_gate_prefetch_local_limit
        if pred_topk.size(0) == 1 and limit > 0:
            row_ids = pred_topk[0, :K].detach().to("cpu", dtype=torch.long).tolist()
            start, end = self.experts_start_idx, self.experts_end_idx
            picked: list[int] = []
            for eid in row_ids:
                if start <= eid < end:
                    picked.append(eid - start)
                    if len(picked) >= limit:
                        break
            if not picked:
                self._cross_layer_pred_prefetch_done = True
                return
            if self._has_gguf_raw_experts() and self.gpu_gguf_decode_grouped_enabled:
                self._prefetch_gguf_decode_local_experts(device, picked)
                self._cross_layer_pred_prefetch_done = True
                return
            if not self.gpu_decode_active_moe_enabled:
                return
            local_tensor = torch.tensor(picked, dtype=torch.long, device="cpu")
            backend = self._ensure_gpu_prefill_backend()
            backend._stage_local_experts(device, local_tensor, async_copy=True)
            self._cross_layer_pred_prefetch_done = True
            return
        if self._has_gguf_raw_experts():
            self._cross_layer_pred_prefetch_done = True
            return
        if not self.gpu_decode_active_moe_enabled:
            return
        pred = pred_topk[:, :K].contiguous()
        if limit > 0:
            local_mask = (pred >= self.experts_start_idx) & (pred < self.experts_end_idx)
            if not bool(local_mask.any().item()):
                self._cross_layer_pred_prefetch_done = True
                return
            rows = []
            for row in range(pred.size(0)):
                local_row = pred[row][local_mask[row]][:limit]
                if local_row.numel() > 0:
                    rows.append(local_row)
            if not rows:
                self._cross_layer_pred_prefetch_done = True
                return
            max_len = max(row.numel() for row in rows)
            limited = torch.full((len(rows), max_len), self.experts_start_idx - 1, device=pred.device, dtype=pred.dtype)
            for row_idx, row in enumerate(rows):
                limited[row_idx, :row.numel()] = row
            pred = limited
        self._ensure_gpu_prefill_backend().prefetch_active_experts(device, pred)
        self._cross_layer_pred_prefetch_done = True

    def _observe_cross_layer_gate_prediction(self, indices: torch.Tensor) -> None:
        pred_topk = self._cross_layer_pred_topk
        if pred_topk is None:
            return
        pred_percentile = self._cross_layer_pred_percentile
        source_layer_id = self._cross_layer_pred_source_layer_id
        self._cross_layer_pred_source_layer_id = -1
        self._cross_layer_pred_topk = None
        self._cross_layer_pred_percentile = None
        self._cross_layer_pred_prefetch_done = False
        if not _cross_layer_gate_profile_enabled():
            return
        _cross_layer_gate_profiler().observe(
            layer_id=self.layer_id,
            source_layer_id=source_layer_id,
            pred_topk=pred_topk,
            pred_percentile=pred_percentile,
            true_indices=indices,
            experts_start_idx=self.experts_start_idx,
            experts_end_idx=self.experts_end_idx,
        )

    def _finalize_reduced_moe(self, y_reduce: torch.Tensor, shared: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        chunk = self.moe_reduce_cast_chunk_tokens
        if chunk > 0 and y_reduce.dim() == 2 and y_reduce.size(0) > chunk:
            out = torch.empty((y_reduce.size(0), y_reduce.size(1)), device=y_reduce.device, dtype=out_dtype)
            for start in range(0, y_reduce.size(0), chunk):
                end = min(start + chunk, y_reduce.size(0))
                part = y_reduce[start:end].to(torch.float32)
                part.add_(shared[start:end])
                out[start:end].copy_(part)
                del part
            return out
        y = y_reduce.to(torch.float32)
        y += shared
        return y.to(out_dtype)

    def _pd_active_phase(self) -> Optional[str]:
        if not self.pd_phase_auto_select_enabled:
            return None
        phase = os.getenv("DEEPSEEK_PD_ACTIVE_PHASE", "")
        return phase if phase in {"prefill", "decode"} else None

    def _should_use_gpu_prefill_moe(self, x: torch.Tensor) -> bool:
        phase = self._pd_active_phase()
        return (
            not self._has_gguf_raw_experts()
            and phase != "decode"
            and self.gpu_prefill_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
            and x.shape[0] >= self.gpu_prefill_moe_min_tokens
        )

    def _should_use_gpu_gguf_prefill_moe(self, x: torch.Tensor) -> bool:
        phase = self._pd_active_phase()
        return (
            self._has_gguf_raw_experts()
            and phase != "decode"
            and self.gpu_gguf_prefill_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
            and x.shape[0] >= self.gpu_prefill_moe_min_tokens
        )

    def _should_use_gpu_gguf_decode_active_expert(self, x: torch.Tensor) -> bool:
        return (
            self._has_gguf_raw_experts()
            and self._pd_active_phase() == "decode"
            and self.gpu_gguf_decode_active_expert_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
        )

    def _should_use_gpu_gguf_decode_grouped(self, x: torch.Tensor) -> bool:
        return (
            self._has_gguf_raw_experts()
            and self._pd_active_phase() == "decode"
            and self.gpu_gguf_decode_grouped_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
        )

    def _gguf_decode_slot_cache_get(
        self,
        reader,
        local_ids_cpu: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, str, int, str, int] | None:
        if not self.gpu_gguf_decode_slot_cache_enabled or self.gpu_gguf_decode_slot_cache_size <= 0:
            return None
        if len(local_ids_cpu) > self.gpu_gguf_decode_slot_cache_size:
            return None

        def read_values(local_id: int):
            expert_id = int(local_id) + int(self.experts_start_idx)
            expert = self.experts[expert_id]
            if expert is None or expert._gguf_raw_backing is None:
                return None
            w1 = expert._gguf_cuda_read_blocks(reader, expert._gguf_raw_backing.w1_name)
            w3 = expert._gguf_cuda_read_blocks(reader, expert._gguf_raw_backing.w3_name)
            w2 = expert._gguf_cuda_read_blocks(reader, expert._gguf_raw_backing.w2_name)
            if w1 is None or w3 is None or w2 is None:
                return None
            return w1, w3, w2

        first_values = None
        meta = self._gguf_decode_slot_meta
        if (
            self._gguf_decode_slot_cache_device != device
            or meta is None
            or self._gguf_decode_slot_w1_blocks is None
            or self._gguf_decode_slot_w3_blocks is None
            or self._gguf_decode_slot_w2_blocks is None
            or self._gguf_decode_slot_w1_blocks.size(0) != self.gpu_gguf_decode_slot_cache_size
        ):
            first_values = read_values(local_ids_cpu[0])
            if first_values is None:
                return None
            w1_first, w3_first, w2_first = first_values
            w1_type, w1_in_dim = w1_first[1], int(w1_first[2])
            w3_type, w3_in_dim = w3_first[1], int(w3_first[2])
            w2_type, w2_in_dim = w2_first[1], int(w2_first[2])
            if w1_type != "iq2_xxs" or w3_type != "iq2_xxs" or w2_type != "q2_k":
                return None
            meta = (w1_type, w1_in_dim, w3_type, w3_in_dim, w2_type, w2_in_dim)
            capacity = int(self.gpu_gguf_decode_slot_cache_size)
            self._gguf_decode_slot_cache_device = device
            self._gguf_decode_slot_cache.clear()
            self._gguf_decode_slot_meta = meta
            self._gguf_decode_slot_w1_blocks = torch.empty(
                (capacity,) + tuple(w1_first[0].shape), device=device, dtype=w1_first[0].dtype
            )
            self._gguf_decode_slot_w3_blocks = torch.empty(
                (capacity,) + tuple(w3_first[0].shape), device=device, dtype=w3_first[0].dtype
            )
            self._gguf_decode_slot_w2_blocks = torch.empty(
                (capacity,) + tuple(w2_first[0].shape), device=device, dtype=w2_first[0].dtype
            )
        assert meta is not None
        assert self._gguf_decode_slot_w1_blocks is not None
        assert self._gguf_decode_slot_w3_blocks is not None
        assert self._gguf_decode_slot_w2_blocks is not None
        w1_type, w1_in_dim, w3_type, w3_in_dim, w2_type, w2_in_dim = meta
        capacity = int(self.gpu_gguf_decode_slot_cache_size)
        route_slot_values = []
        for local_id in local_ids_cpu:
            local_id = int(local_id)
            slot = self._gguf_decode_slot_cache.pop(local_id, None)
            if slot is None:
                if len(self._gguf_decode_slot_cache) >= capacity:
                    _evicted_local, slot = self._gguf_decode_slot_cache.popitem(last=False)
                else:
                    used_slots = set(self._gguf_decode_slot_cache.values())
                    slot = next(idx for idx in range(capacity) if idx not in used_slots)
                values = first_values if first_values is not None and local_id == int(local_ids_cpu[0]) else read_values(local_id)
                if values is None:
                    return None
                w1_value, w3_value, w2_value = values
                if (
                    w1_value[1] != w1_type
                    or int(w1_value[2]) != w1_in_dim
                    or w3_value[1] != w3_type
                    or int(w3_value[2]) != w3_in_dim
                    or w2_value[1] != w2_type
                    or int(w2_value[2]) != w2_in_dim
                ):
                    return None
                self._gguf_decode_slot_w1_blocks[slot].copy_(w1_value[0])
                self._gguf_decode_slot_w3_blocks[slot].copy_(w3_value[0])
                self._gguf_decode_slot_w2_blocks[slot].copy_(w2_value[0])
            self._gguf_decode_slot_cache[local_id] = int(slot)
            route_slot_values.append(int(slot))
        route_slots = torch.tensor(route_slot_values, device=device, dtype=torch.long)
        return (
            self._gguf_decode_slot_w1_blocks,
            self._gguf_decode_slot_w3_blocks,
            self._gguf_decode_slot_w2_blocks,
            route_slots,
            w1_type,
            w1_in_dim,
            w3_type,
            w3_in_dim,
            w2_type,
            w2_in_dim,
        )

    def _forward_gguf_decode_grouped(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor | None:
        cuda_mod = load_cuda_kernel()
        if cuda_mod is None or not hasattr(cuda_mod, "gguf_moe_prefill_grouped_forward"):
            return None
        profile = self.gpu_gguf_decode_grouped_profile
        t0 = time.perf_counter() if profile else 0.0
        route_slots = None
        compact_starts = None
        use_cpu_single_route = x.shape[0] == 1 and not _env_enabled("DEEPSEEK_GGUF_GPU_DECODE_DEVICE_ROUTE")
        local_ids = None
        seg_starts = None
        if use_cpu_single_route:
            used_cpp_single_route = False
            if hasattr(cuda_mod, "gguf_single_token_route_slots"):
                try:
                    grouped = cuda_mod.gguf_single_token_route_slots(
                        indices.contiguous(),
                        weights.contiguous().to(torch.float32),
                        int(self.experts_start_idx),
                        int(self.n_local_experts),
                    )
                    local_ids_cpu_tensor, route_tokens, route_weights, route_slots, compact_starts = grouped
                    if local_ids_cpu_tensor.numel() == 0:
                        return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
                    local_ids_cpu = local_ids_cpu_tensor.to(torch.long).tolist()
                    used_cpp_single_route = True
                except Exception:
                    used_cpp_single_route = False
            if not used_cpp_single_route:
                idx_cpu = indices[0].detach().cpu().to(torch.long).tolist()
                routes_by_local: dict[int, list[int]] = {}
                for route_pos, expert_id in enumerate(idx_cpu):
                    if self.experts_start_idx <= int(expert_id) < self.experts_end_idx:
                        routes_by_local.setdefault(int(expert_id) - int(self.experts_start_idx), []).append(route_pos)
                local_ids_cpu = sorted(routes_by_local.keys())
                if not local_ids_cpu:
                    return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
                route_positions = []
                route_slot_values = []
                seg_values = [0]
                for slot, local_id in enumerate(local_ids_cpu):
                    local_positions = routes_by_local[local_id]
                    route_positions.extend(local_positions)
                    route_slot_values.extend([slot] * len(local_positions))
                    seg_values.append(len(route_positions))
                route_tokens = torch.zeros((len(route_positions),), device=x.device, dtype=torch.long)
                route_pos_tensor = torch.tensor(route_positions, device=x.device, dtype=torch.long)
                route_weights = weights[0].to(torch.float32).index_select(0, route_pos_tensor)
                route_slots = torch.tensor(route_slot_values, device=x.device, dtype=torch.long)
                compact_starts = torch.tensor(seg_values, device=x.device, dtype=torch.int32)
            if profile:
                torch.cuda.synchronize(x.device)
                t_group = t_ids = time.perf_counter()
        else:
            if not hasattr(cuda_mod, "moe_group_routes"):
                return None
            grouped = cuda_mod.moe_group_routes(
                indices.contiguous(),
                weights.contiguous().to(torch.float32),
                int(self.experts_start_idx),
                int(self.n_local_experts),
            )
            if profile:
                torch.cuda.synchronize(x.device)
                t_group = time.perf_counter()
            if grouped is None:
                return None
            local_ids, route_tokens, route_weights, seg_starts = grouped
            if local_ids.numel() == 0:
                return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
            local_ids_cpu = local_ids.detach().cpu().to(torch.long).tolist()
            if route_slots is None and x.shape[0] == 1 and self.gpu_gguf_decode_single_token_enabled:
                seg_i32 = seg_starts if seg_starts.dtype == torch.int32 else seg_starts.to(torch.int32)
                route_slots = torch.repeat_interleave(torch.arange(local_ids.numel(), device=x.device, dtype=torch.long), (seg_i32[local_ids + 1] - seg_i32[local_ids]).to(torch.long))
            if profile:
                torch.cuda.synchronize(x.device)
                t_ids = time.perf_counter()

        try:
            from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader

            first_expert = next((expert for expert in self.experts if expert is not None and expert._gguf_raw_backing is not None), None)
            if first_expert is None:
                return None
            reader = get_cached_gguf_tensor_reader(first_expert._gguf_raw_backing.gguf_path)
            slot_cached = None
            if route_slots is not None and self.gpu_gguf_decode_single_token_enabled:
                slot_cached = self._gguf_decode_slot_cache_get(reader, local_ids_cpu, x.device)
            if slot_cached is not None:
                (
                    w1_blocks,
                    w3_blocks,
                    w2_blocks,
                    route_slots,
                    w1_type,
                    w1_in_dim,
                    w3_type,
                    w3_in_dim,
                    w2_type,
                    w2_in_dim,
                ) = slot_cached
            else:
                w1_values = []
                w3_values = []
                w2_values = []
                for local_id in local_ids_cpu:
                    expert_id = int(local_id) + int(self.experts_start_idx)
                    expert = self.experts[expert_id]
                    if expert is None or expert._gguf_raw_backing is None:
                        return None
                    w1_packed = expert._gguf_cuda_get_blocks(reader, expert._gguf_raw_backing.w1_name, x.device)
                    w3_packed = expert._gguf_cuda_get_blocks(reader, expert._gguf_raw_backing.w3_name, x.device)
                    w2_packed = expert._gguf_cuda_get_blocks(reader, expert._gguf_raw_backing.w2_name, x.device)
                    if w1_packed is None or w3_packed is None or w2_packed is None:
                        return None
                    w1_values.append(w1_packed)
                    w3_values.append(w3_packed)
                    w2_values.append(w2_packed)
                w1_type = w1_values[0][1]
                w3_type = w3_values[0][1]
                w2_type = w2_values[0][1]
                w1_in_dim = int(w1_values[0][2])
                w3_in_dim = int(w3_values[0][2])
                w2_in_dim = int(w2_values[0][2])
                if any(v[1] != w1_type or int(v[2]) != w1_in_dim for v in w1_values):
                    return None
                if any(v[1] != w3_type or int(v[2]) != w3_in_dim for v in w3_values):
                    return None
                if any(v[1] != w2_type or int(v[2]) != w2_in_dim for v in w2_values):
                    return None
                w1_blocks = torch.stack([v[0] for v in w1_values], dim=0).contiguous()
                w3_blocks = torch.stack([v[0] for v in w3_values], dim=0).contiguous()
                w2_blocks = torch.stack([v[0] for v in w2_values], dim=0).contiguous()
            if profile:
                torch.cuda.synchronize(x.device)
                t_fetch = time.perf_counter()
                t_stack = t_fetch
            if compact_starts is None:
                seg_i32 = seg_starts if seg_starts.dtype == torch.int32 else seg_starts.to(torch.int32)
                compact_starts = torch.cat([seg_i32[local_ids.to(seg_i32.device)], seg_i32[-1:]], dim=0).contiguous()
            iq2_grid = first_expert._gguf_cuda_grid(x.device)
            if (
                route_slots is not None
                and self.gpu_gguf_decode_single_token_enabled
                and hasattr(cuda_mod, "gguf_moe_single_token_iq2_q2k_forward")
                and w1_type == "iq2_xxs"
                and w3_type == "iq2_xxs"
                and w2_type == "q2_k"
            ):
                y = cuda_mod.gguf_moe_single_token_iq2_q2k_forward(
                    x.contiguous(),
                    route_slots,
                    route_weights.contiguous(),
                    w1_blocks,
                    w3_blocks,
                    w2_blocks,
                    iq2_grid,
                    float(self.cpu_backend._swiglu_limit),
                )
            elif (
                route_slots is not None
                and self.gpu_gguf_decode_single_token_enabled
                and hasattr(cuda_mod, "gguf_moe_single_token_iq1m_forward")
                and w1_type == "iq1_m"
                and w3_type == "iq1_m"
                and w2_type == "iq1_m"
            ):
                y = cuda_mod.gguf_moe_single_token_iq1m_forward(
                    x.contiguous(),
                    route_slots,
                    route_weights.contiguous(),
                    w1_blocks,
                    w3_blocks,
                    w2_blocks,
                    first_expert._gguf_cuda_iq1_grid_tensor(x.device),
                    float(self.cpu_backend._swiglu_limit),
                )
            else:
                w1_type_id = first_expert._gguf_type_id(w1_type)
                w3_type_id = first_expert._gguf_type_id(w3_type)
                w2_type_id = first_expert._gguf_type_id(w2_type)
                if w1_type_id is None or w3_type_id is None or w2_type_id is None:
                    return None
                quant_grid = first_expert._gguf_cuda_group_grid((w1_type, w3_type, w2_type), x.device)
                y = cuda_mod.gguf_moe_prefill_grouped_forward(
                    x.contiguous(),
                    route_tokens,
                    route_weights.contiguous(),
                    compact_starts,
                    w1_blocks,
                    w3_blocks,
                    w2_blocks,
                    w1_in_dim,
                    w1_type_id,
                    w3_in_dim,
                    w3_type_id,
                    w2_in_dim,
                    w2_type_id,
                    quant_grid,
                    float(self.cpu_backend._swiglu_limit),
                )
            if profile:
                torch.cuda.synchronize(x.device)
                t_kernel = time.perf_counter()
                print(
                    f"gguf_decode_grouped_profile layer={self.layer_id} rank={rank} "
                    f"active={len(local_ids_cpu)} routes={int(route_tokens.numel())} "
                    f"group={t_group - t0:.6f}s ids={t_ids - t_group:.6f}s fetch={t_fetch - t_ids:.6f}s "
                    f"stack={t_stack - t_fetch:.6f}s kernel={t_kernel - t_stack:.6f}s total={t_kernel - t0:.6f}s",
                    flush=True,
                )
            return y
        except Exception as exc:
            if _env_enabled("DEEPSEEK_GGUF_GPU_PROFILE"):
                print(f"gguf_decode_grouped_fallback rank={rank} layer={self.layer_id} error={exc}", flush=True)
            return None

    def _forward_gguf_decode_active_expert(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        y = torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
        routes_by_expert: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for top_idx in range(indices.size(1)):
            ids_col = indices[:, top_idx]
            local_mask = (ids_col >= self.experts_start_idx) & (ids_col < self.experts_end_idx)
            if not bool(local_mask.any().item()):
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
                group_weights = weights[group_rows, top_idx:top_idx + 1].contiguous()
                routes_by_expert.setdefault(int(expert_id), []).append((group_rows, group_weights))
                offset += count
        reader = None
        try:
            from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader
            first_expert = next((expert for expert in self.experts if expert is not None and expert._gguf_raw_backing is not None), None)
            if first_expert is not None:
                reader = get_cached_gguf_tensor_reader(first_expert._gguf_raw_backing.gguf_path)
        except Exception:
            reader = None
        for expert_id in sorted(routes_by_expert.keys()):
            expert = self.experts[expert_id]
            if expert is None:
                continue
            route_groups = routes_by_expert[expert_id]
            group_rows = torch.cat([rows for rows, _weights in route_groups], dim=0)
            group_x = x[group_rows].contiguous()
            group_weights = torch.cat([weights_part for _rows, weights_part in route_groups], dim=0)
            out = None
            if reader is not None:
                try:
                    expert.prefetch_cuda_gguf(reader, x.device)
                    out = expert.forward_cuda_gguf(group_x, group_weights)
                except Exception:
                    out = None
                finally:
                    if not self.gpu_gguf_decode_keep_cache:
                        expert.clear_cuda_gguf_cache()
            if out is None:
                out = self.cpu_backend.dispatch_forward(
                    group_x,
                    group_rows.new_full((group_rows.numel(),), int(expert_id)),
                    group_weights.view(-1).contiguous(),
                ).to(device=x.device, dtype=torch.float32)
            y.index_add_(0, group_rows, out.to(torch.float32))
        return y

    def _should_use_gpu_decode_active_moe(self, x: torch.Tensor) -> bool:
        return (
            not self._has_gguf_raw_experts()
            and self._pd_active_phase() == "decode"
            and self.gpu_decode_active_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
        )

    def _should_use_gpu_spec_token2_moe(self, x: torch.Tensor) -> bool:
        return (
            not self._has_gguf_raw_experts()
            and self._pd_active_phase() == "decode"
            and self.gpu_spec_token2_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and x.is_cuda
            and x.shape[0] == 2
        )

    def _ensure_gpu_prefill_backend(self) -> GPUPrefillMoEBackend:
        if self.gpu_prefill_backend is None:
            self.gpu_prefill_backend = GPUPrefillMoEBackend(
                self.cpu_backend,
                dim=self.dim,
                num_experts=self.n_routed_experts,
                experts_start_idx=self.experts_start_idx,
                experts_end_idx=self.experts_end_idx,
            )
        return self.gpu_prefill_backend


    def prefetch_gpu_prefill_moe(self, device: torch.device, token_count: int) -> None:
        if self._has_gguf_raw_experts():
            self.prefetch_gguf_gpu_prefill_moe(device, token_count)
            return
        if self.gpu_decode_active_moe_enabled and self._pd_active_phase() == "decode":
            return
        if (
            self.gpu_prefill_moe_enabled
            and self.cpu_backend is not None
            and self.experts_end_idx > self.experts_start_idx
            and device.type == "cuda"
            and token_count >= self.gpu_prefill_moe_min_tokens
        ):
            prefer_fp4 = _env_enabled("DEEPSEEK_GPU_PREFILL_MOE_FP4_DIRECT_GROUPED")
            self._ensure_gpu_prefill_backend().prefetch(device, prefer_fp4=prefer_fp4)

    def _should_use_in_process_cpu_server(self, x: torch.Tensor) -> bool:
        if self._has_gguf_raw_experts():
            return False
        if not self.cpu_moe_inproc_server_enabled or world_size <= 1 or x.shape[0] != 1:
            return False
        return self._pd_active_phase() != "prefill"

    def _should_use_rank0_cpu_server(self, x: torch.Tensor) -> bool:
        return not self._has_gguf_raw_experts() and self.cpu_moe_rank0_server_enabled and world_size > 1 and x.shape[0] == 1


    def _should_use_external_cpu_server(self, x: torch.Tensor) -> bool:
        if self._has_gguf_raw_experts():
            return False
        if not self.cpu_moe_external_server_enabled or world_size <= 1 or x.shape[0] != 1:
            return False
        return self._pd_active_phase() != "prefill"

    def _external_cpu_server_submit_decode(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> int:
        global _cpu_moe_server_last_seq, _cpu_moe_server_predict_seq
        if rank != 0:
            if self.cpu_moe_predict_seq_enabled:
                _cpu_moe_server_predict_seq += 1
                return _cpu_moe_server_predict_seq
            return -1
        ipc = _get_cpu_moe_server_ipc(self.dim, self.n_activated_experts)
        req, _resp, _layer, _stop = ipc.read_header()
        seq_to_reuse = req + 1 - ipc.output_slots
        if seq_to_reuse > 0:
            ipc.wait_slot_acks(seq_to_reuse, world_size)
        seq = ipc.submit(self.layer_id, x, indices, weights)
        _cpu_moe_server_last_seq = seq
        _cpu_moe_server_predict_seq = seq
        return seq

    def _external_cpu_server_wait_decode(self, seq: int, x: torch.Tensor) -> torch.Tensor:
        ipc = _get_cpu_moe_server_ipc(self.dim, self.n_activated_experts)
        if self.cpu_moe_predict_seq_enabled:
            if rank == 0:
                global _cpu_moe_server_predict_seq
                _cpu_moe_server_predict_seq = seq
        else:
            seq_tensor = torch.empty(1, device=x.device, dtype=torch.long)
            if rank == 0:
                seq_tensor.fill_(seq)
            dist.broadcast(seq_tensor, src=0)
            seq = int(seq_tensor.item())
        ipc.wait_response(seq)
        y = ipc.output_tensor(seq).to(device=x.device, dtype=torch.float32, non_blocking=False)
        if rank != 0:
            ipc.ack(rank, seq)
        return y


    def _external_cpu_server_sync(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        global _cpu_moe_server_last_seq, _cpu_moe_server_predict_seq
        profile = _env_enabled("DEEPSEEK_CPU_MOE_SERVER_PROFILE")
        ipc = _get_cpu_moe_server_ipc(self.dim, self.n_activated_experts)
        outputs = []
        seq_tensor = torch.empty(1, device=x.device, dtype=torch.long)
        t_total = time.perf_counter() if profile else 0.0
        for row in range(x.shape[0]):
            if rank == 0:
                req, _resp, _layer, _stop = ipc.read_header()
                seq_to_reuse = req + 1 - ipc.output_slots
                if seq_to_reuse > 0:
                    ipc.wait_slot_acks(seq_to_reuse, world_size)
                seq = ipc.submit(self.layer_id, x[row:row + 1], indices[row:row + 1], weights[row:row + 1])
                _cpu_moe_server_last_seq = seq
                _cpu_moe_server_predict_seq = seq
                seq_tensor.fill_(seq)
            dist.broadcast(seq_tensor, src=0)
            seq = int(seq_tensor.item())
            if self.cpu_moe_predict_seq_enabled and rank != 0:
                _cpu_moe_server_predict_seq = seq
            ipc.wait_response(seq)
            y_row = ipc.output_tensor(seq).to(device=x.device, dtype=torch.float32, non_blocking=False)
            if rank != 0:
                ipc.ack(rank, seq)
            outputs.append(y_row)
        y = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        if profile:
            print(
                f"cpu_moe_external_profile layer={self.layer_id} rank={rank} batch={x.shape[0]} total={time.perf_counter() - t_total:.6f}s",
                flush=True,
            )
        return y

    def _rank0_cpu_server_sync(self, x: torch.Tensor) -> torch.Tensor:
        if rank == 0:
            y = self.cpu_backend.sync_forward(x).to(torch.float32)
        else:
            y = torch.empty_like(x, dtype=torch.float32)
        dist.broadcast(y, src=0)
        return y

    def _should_use_cpu_host_reduce(self, x: torch.Tensor) -> bool:
        phase = self._pd_active_phase()
        return phase != "prefill" and not self._has_gguf_raw_experts() and self.cpu_host_reduce_enabled and tp_world_size > 1 and x.shape[0] == 1

    def _cpu_host_reduce(self, x: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
        t0 = time.perf_counter() if self._profile_enabled else 0.0
        y_cpu, current_slot, batch_size, device = self.cpu_backend.sync_forward_cpu(x)
        t_sync = time.perf_counter() if self._profile_enabled else 0.0
        if not hasattr(self.cpu_backend, "host_float_allreduce"):
            raise RuntimeError("DEEPSEEK_CPU_MOE_HOST_REDUCE requires native host_float_allreduce")
        self.cpu_backend.host_float_allreduce(y_cpu, rank, tp_world_size)
        t_reduce = time.perf_counter() if self._profile_enabled else 0.0
        y = self.cpu_backend.copy_cpu_output_to_device(y_cpu, current_slot, batch_size, device, x).to(torch.float32)
        t_h2d = time.perf_counter() if self._profile_enabled else 0.0
        return y, t_sync - t0, t_reduce - t_sync, t_h2d - t_reduce

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor, prefetch_next: Optional[nn.Module] = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        self._observe_cross_layer_gate_prediction(indices)
        if self._rank_route_profile_enabled and x.shape[0] == 1:
            local = ((indices >= self.experts_start_idx) & (indices < self.experts_end_idx)).sum().item()
            print(f"rank_route layer={self.layer_id} rank={rank} local={int(local)} ids={indices.flatten().tolist()}", flush=True)
        used_gpu_prefill_moe = False
        reduced_moe_ready = False
        reduced_moe = False
        used_remote_only = False
        if self.routed_experts_device == "cpu" and self._should_use_gpu_prefill_moe(x):
            shared = self.shared_experts(x)
            backend = self._ensure_gpu_prefill_backend()
            chunk = self.gpu_prefill_moe_chunk_tokens
            if chunk > 0 and x.shape[0] > chunk:
                y = torch.empty((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
                for start in range(0, x.shape[0], chunk):
                    end = min(start + chunk, x.shape[0])
                    y_part = backend.forward(
                        x[start:end].contiguous(),
                        indices[start:end].contiguous(),
                        weights[start:end].contiguous(),
                        self.cpu_backend._swiglu_limit,
                    )
                    y[start:end].copy_(y_part)
                    del y_part
            else:
                y = backend.forward(
                    x,
                    indices,
                    weights,
                    self.cpu_backend._swiglu_limit,
                )
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_gguf_prefill_moe(x):
            gguf_prefill_profile = False
            if _env_enabled("DEEPSEEK_GGUF_PREFILL_PROFILE"):
                profile_limit = max(0, int(os.getenv("DEEPSEEK_GGUF_PREFILL_PROFILE_LIMIT", "64")))
                profile_every = max(1, int(os.getenv("DEEPSEEK_GGUF_PREFILL_PROFILE_EVERY", "1")))
                profile_index = MoE._gguf_prefill_profile_count
                MoE._gguf_prefill_profile_count += 1
                gguf_prefill_profile = profile_index < profile_limit and profile_index % profile_every == 0
            if gguf_prefill_profile:
                torch.cuda.synchronize(x.device)
                t_gguf_start = time.perf_counter()
            shared = self.shared_experts(x)
            if gguf_prefill_profile:
                torch.cuda.synchronize(x.device)
                t_gguf_shared = time.perf_counter()
            y = torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)
            grouped = None
            grouped_routes = 0
            expert_ids_order = []
            cuda_mod = load_cuda_kernel()
            if cuda_mod is not None and hasattr(cuda_mod, "moe_group_routes"):
                grouped = cuda_mod.moe_group_routes(
                    indices.contiguous(),
                    weights.contiguous().to(torch.float32),
                    int(self.experts_start_idx),
                    int(self.n_local_experts),
                )
            if gguf_prefill_profile:
                torch.cuda.synchronize(x.device)
                t_gguf_group = time.perf_counter()
            if grouped is None:
                routes_by_expert: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
                for top_idx in range(indices.size(1)):
                    ids_col = indices[:, top_idx]
                    local_mask = (ids_col >= self.experts_start_idx) & (ids_col < self.experts_end_idx)
                    if not bool(local_mask.any().item()):
                        continue
                    rows = torch.nonzero(local_mask, as_tuple=False).flatten()
                    if rows.numel() == 0:
                        continue
                    expert_ids = ids_col[rows]
                    sort_order = torch.argsort(expert_ids)
                    rows = rows[sort_order]
                    expert_ids = expert_ids[sort_order]
                    unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
                    offset = 0
                    for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
                        group_rows = rows[offset:offset + count]
                        group_weights = weights[group_rows, top_idx:top_idx + 1].contiguous()
                        routes_by_expert.setdefault(int(expert_id), []).append((group_rows, group_weights))
                        offset += count
                expert_ids_order = sorted(routes_by_expert.keys())
                grouped_routes = sum(
                    int(rows.numel())
                    for route_groups in routes_by_expert.values()
                    for rows, _weights in route_groups
                )
            else:
                _local_ids, route_tokens, route_weights, seg_starts = grouped
                grouped_routes = int(route_tokens.numel())
                if _local_ids.numel() == 0:
                    routes_by_expert = {}
                    expert_ids_order = []
                else:
                    routes_by_expert = {}
                    expert_ids_order = []
            staged_ready = False if self.gpu_gguf_prefill_active_expert_enabled else self.wait_gguf_gpu_prefill_moe(x.device)
            if gguf_prefill_profile:
                torch.cuda.synchronize(x.device)
                t_gguf_stage = time.perf_counter()
            used_grouped_gguf = False
            if (
                _env_enabled("DEEPSEEK_GGUF_GPU_GROUPED_MOE")
                and grouped is not None
                and staged_ready
                and self._gguf_layer_stage is not None
                and cuda_mod is not None
                and hasattr(cuda_mod, "gguf_moe_prefill_grouped_forward")
            ):
                w1_layer, w3_layer, w2_layer, _host_layer = self._gguf_layer_stage
                stream = torch.cuda.current_stream(x.device)
                for tensor in (w1_layer[0], w3_layer[0], w2_layer[0]):
                    tensor.record_stream(stream)
                try:
                    first_expert = next((expert for expert in self.experts if expert is not None and expert._gguf_raw_backing is not None), None)
                    if first_expert is not None:
                        w1_type_id = first_expert._gguf_type_id(w1_layer[1])
                        w3_type_id = first_expert._gguf_type_id(w3_layer[1])
                        w2_type_id = first_expert._gguf_type_id(w2_layer[1])
                        quant_grid = first_expert._gguf_cuda_group_grid((w1_layer[1], w3_layer[1], w2_layer[1]), x.device)
                    else:
                        w1_type_id = w3_type_id = w2_type_id = None
                        quant_grid = torch.empty(0, dtype=torch.int8, device=x.device)
                    if w1_type_id is None or w3_type_id is None or w2_type_id is None:
                        raise RuntimeError("unsupported GGUF staged quant type")
                    w1_blocks = w1_layer[0]
                    w3_blocks = w3_layer[0]
                    w2_blocks = w2_layer[0]
                    y = cuda_mod.gguf_moe_prefill_grouped_forward(
                        x.contiguous(),
                        route_tokens,
                        route_weights.contiguous(),
                        seg_starts,
                        w1_blocks,
                        w3_blocks,
                        w2_blocks,
                        int(w1_layer[2]),
                        w1_type_id,
                        int(w3_layer[2]),
                        w3_type_id,
                        int(w2_layer[2]),
                        w2_type_id,
                        quant_grid,
                        float(self.cpu_backend._swiglu_limit),
                    )
                    used_grouped_gguf = True
                except Exception as exc:
                    if _env_enabled("DEEPSEEK_GGUF_GPU_PROFILE"):
                        print(f"gguf_gpu_grouped_fallback rank={rank} layer={self.layer_id} error={exc}", flush=True)
            if self.gpu_gguf_prefill_active_expert_enabled:
                reader = None
                if staged_ready:
                    self.release_gguf_gpu_prefill_moe()
                try:
                    from src.loader.gguf.tensor_reader import get_cached_gguf_tensor_reader
                    first_expert = next((expert for expert in self.experts if expert is not None and expert._gguf_raw_backing is not None), None)
                    if first_expert is not None:
                        reader = get_cached_gguf_tensor_reader(first_expert._gguf_raw_backing.gguf_path)
                except Exception:
                    reader = None
                if grouped is not None:
                    routes_by_expert = {}
                    seg_cpu = seg_starts.detach().cpu()
                    for local_id in _local_ids.detach().cpu().to(torch.long).tolist():
                        start_i = int(seg_cpu[local_id].item())
                        end_i = int(seg_cpu[local_id + 1].item())
                        expert_id = int(local_id) + int(self.experts_start_idx)
                        if end_i <= start_i:
                            continue
                        rows = route_tokens[start_i:end_i]
                        group_weights = route_weights[start_i:end_i].view(-1, 1).contiguous()
                        routes_by_expert[expert_id] = [(rows, group_weights)]
                    expert_ids_order = sorted(routes_by_expert.keys())
                for expert_id in expert_ids_order:
                    expert = self.experts[expert_id]
                    if expert is None:
                        continue
                    route_groups = routes_by_expert[expert_id]
                    group_rows = torch.cat([rows for rows, _weights in route_groups], dim=0)
                    group_x = x[group_rows].contiguous()
                    group_weights = torch.cat([weights_part for _rows, weights_part in route_groups], dim=0)
                    if reader is not None:
                        expert.prefetch_cuda_gguf(reader, x.device)
                    out = expert.forward_cuda_gguf(group_x, group_weights)
                    if not self.gpu_gguf_prefill_active_keep_cache:
                        expert.clear_cuda_gguf_cache()
                    if out is None:
                        out = self.cpu_backend.dispatch_forward(
                            group_x,
                            group_rows.new_full((group_rows.numel(),), int(expert_id)),
                            group_weights.view(-1).contiguous(),
                        ).to(device=x.device, dtype=torch.float32)
                    y.index_add_(0, group_rows, out.to(torch.float32))
                used_grouped_gguf = True
            if not used_grouped_gguf:
                if grouped is not None:
                    routes_by_expert = {}
                    seg_cpu = seg_starts.detach().cpu()
                    for local_id in _local_ids.detach().cpu().to(torch.long).tolist():
                        start_i = int(seg_cpu[local_id].item())
                        end_i = int(seg_cpu[local_id + 1].item())
                        expert_id = int(local_id) + int(self.experts_start_idx)
                        if end_i <= start_i:
                            continue
                        rows = route_tokens[start_i:end_i]
                        group_weights = route_weights[start_i:end_i].view(-1, 1).contiguous()
                        routes_by_expert[expert_id] = [(rows, group_weights)]
                    expert_ids_order = sorted(routes_by_expert.keys())
                if staged_ready and self._gguf_layer_stage is not None:
                    w1_layer, w3_layer, w2_layer, _host_layer = self._gguf_layer_stage
                    for expert_id in expert_ids_order:
                        expert = self.experts[expert_id]
                        if expert is None:
                            continue
                        route_groups = routes_by_expert[expert_id]
                        group_rows = torch.cat([rows for rows, _weights in route_groups], dim=0)
                        group_x = x[group_rows].contiguous()
                        group_weights = torch.cat([weights_part for _rows, weights_part in route_groups], dim=0)
                        out = expert.forward_cuda_gguf_staged(
                            group_x,
                            group_weights,
                            (w1_layer[0][expert_id - self.experts_start_idx], w1_layer[1], w1_layer[2]),
                            (w3_layer[0][expert_id - self.experts_start_idx], w3_layer[1], w3_layer[2]),
                            (w2_layer[0][expert_id - self.experts_start_idx], w2_layer[1], w2_layer[2]),
                        )
                        if out is None:
                            out = self.cpu_backend.dispatch_forward(
                                group_x,
                                group_rows.new_full((group_rows.numel(),), int(expert_id)),
                                group_weights.view(-1).contiguous(),
                            ).to(device=x.device, dtype=torch.float32)
                        y.index_add_(0, group_rows, out.to(torch.float32))
                else:
                    for expert_id in expert_ids_order:
                        route_groups = routes_by_expert[expert_id]
                        group_rows = torch.cat([rows for rows, _weights in route_groups], dim=0)
                        group_x = x[group_rows].contiguous()
                        group_weights = torch.cat([weights_part for _rows, weights_part in route_groups], dim=0)
                        out = self.cpu_backend.dispatch_forward(
                            group_x,
                            group_rows.new_full((group_rows.numel(),), int(expert_id)),
                            group_weights.view(-1).contiguous(),
                        ).to(device=x.device, dtype=torch.float32)
                        y.index_add_(0, group_rows, out.to(torch.float32))
            if gguf_prefill_profile:
                torch.cuda.synchronize(x.device)
                t_gguf_done = time.perf_counter()
                print(
                    f"gguf_prefill_profile layer={self.layer_id} rank={rank} tokens={x.shape[0]} "
                    f"routes={grouped_routes} experts={len(expert_ids_order)} staged={int(staged_ready)} grouped={int(used_grouped_gguf)} "
                    f"shared={t_gguf_shared - t_gguf_start:.4f}s group={t_gguf_group - t_gguf_shared:.4f}s "
                    f"stage={t_gguf_stage - t_gguf_group:.4f}s compute={t_gguf_done - t_gguf_stage:.4f}s "
                    f"total={t_gguf_done - t_gguf_start:.4f}s",
                    flush=True,
                )
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_gguf_decode_grouped(x):
            if self.gpu_gguf_decode_active_prefetch_enabled:
                self._prefetch_gguf_decode_expert_ids(x.device, indices.flatten().detach().to("cpu", dtype=torch.long).tolist())
            if prefetch_next is not None and hasattr(prefetch_next, "ffn"):
                prefetch_next.ffn.prefetch_cross_layer_gate_prediction(x.device)
            if self.gpu_gguf_decode_overlap_shared_route and self._gguf_decode_shared_stream is not None:
                shared_stream = self._gguf_decode_shared_stream
                shared_stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(shared_stream):
                    shared = self.shared_experts(x)
                y = self._forward_gguf_decode_grouped(x, indices, weights)
                torch.cuda.current_stream(x.device).wait_stream(shared_stream)
            else:
                shared = self.shared_experts(x)
                y = self._forward_gguf_decode_grouped(x, indices, weights)
            if y is None:
                y = self._forward_gguf_decode_active_expert(x, indices, weights)
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_gguf_decode_active_expert(x):
            shared = self.shared_experts(x)
            y = self._forward_gguf_decode_active_expert(x, indices, weights)
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_spec_token2_moe(x):
            # Speculative verify mixed path: keep the first token on the normal
            # CPU MoE path, and run only the draft/second token's routed experts
            # on GPU with route-aware active-expert staging. This is the only
            # form that can help on the heterogeneous CPU-MoE architecture: it
            # avoids doubling CPU MoE work while using otherwise-idle GPU time.
            self.cpu_backend.submit_forward(x[:1], indices[:1], weights[:1])
            backend = self._ensure_gpu_prefill_backend()
            backend.prefetch_active_experts(x.device, indices[1:2])
            shared = self.shared_experts(x)
            y2 = backend.forward(
                x[1:2],
                indices[1:2],
                weights[1:2],
                self.cpu_backend._swiglu_limit,
            )
            y1 = self.cpu_backend.sync_forward(x[:1])
            y = torch.empty((2, self.dim), device=x.device, dtype=torch.float32)
            y[:1] = y1
            y[1:2] = y2
            used_gpu_prefill_moe = True
        elif self.routed_experts_device == "cpu" and self._should_use_gpu_decode_active_moe(x):
            active_profile = self._active_profile_enabled and x.is_cuda
            t_active0 = time.perf_counter() if active_profile else 0.0
            backend = self._ensure_gpu_prefill_backend()
            backend.prefetch_active_experts(x.device, indices)
            if hasattr(backend, "prefetch_active_experts_to_cache"):
                backend.prefetch_active_experts_to_cache(x.device, indices)
            t_active_prefetch = time.perf_counter() if active_profile else 0.0
            if prefetch_next is not None and hasattr(prefetch_next, "ffn"):
                prefetch_next.ffn.prefetch_cross_layer_gate_prediction(x.device)
            t_active_cross = time.perf_counter() if active_profile else 0.0
            overlap_shared_reduce = (
                _env_enabled("DEEPSEEK_GPU_MOE_ACTIVE_OVERLAP_SHARED_REDUCE")
                and self.async_allreduce_enabled
                and tp_world_size > 1
                and self._allreduce_stream is not None
            )
            overlap_reduce = (
                not overlap_shared_reduce
                and self.gpu_active_overlap_reduce_enabled
                and self.async_allreduce_enabled
                and tp_world_size > 1
                and self._allreduce_stream is not None
            )
            if overlap_shared_reduce or overlap_reduce:
                shared = None
                t_active_shared = t_active_cross
            else:
                shared = self.shared_experts(x)
                if active_profile:
                    torch.cuda.synchronize(x.device)
                t_active_shared = time.perf_counter() if active_profile else 0.0
            y = backend.forward(
                x,
                indices,
                weights,
                self.cpu_backend._swiglu_limit,
            )
            if active_profile:
                torch.cuda.synchronize(x.device)
            t_active_routed = time.perf_counter() if active_profile else 0.0
            if overlap_shared_reduce:
                reduce_dtype = torch.float16 if self.reduce_fp16_enabled else x.dtype
                y_reduce = y.to(reduce_dtype)
                if active_profile:
                    torch.cuda.synchronize(x.device)
                    t_active_reduce_cast = time.perf_counter()
                self._allreduce_stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(self._allreduce_stream):
                    dist.all_reduce(y_reduce, async_op=False)
                shared = self.shared_experts(x)
                if active_profile:
                    torch.cuda.synchronize(x.device)
                    t_active_shared = time.perf_counter()
                torch.cuda.current_stream(x.device).wait_stream(self._allreduce_stream)
                if active_profile:
                    torch.cuda.synchronize(x.device)
                    t_active_allreduce = time.perf_counter()
                fused = _maybe_fused_moe_finalize(y_reduce, shared, x.dtype)
                if fused is not None:
                    y = fused
                    if active_profile:
                        torch.cuda.synchronize(x.device)
                        t_active_reduce_to_fp32 = time.perf_counter()
                        t_active_add_shared = t_active_reduce_to_fp32
                        t_active_type_cast = t_active_reduce_to_fp32
                else:
                    y = y_reduce.to(torch.float32)
                    if active_profile:
                        torch.cuda.synchronize(x.device)
                        t_active_reduce_to_fp32 = time.perf_counter()
                reduced_moe_ready = True
                reduced_moe = True
            elif overlap_reduce:
                reduce_dtype = torch.float16 if self.reduce_fp16_enabled else x.dtype
                self._allreduce_stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(self._allreduce_stream):
                    y_reduce = y.to(reduce_dtype)
                    dist.all_reduce(y_reduce, async_op=False)
                shared = self.shared_experts(x)
                if active_profile:
                    torch.cuda.synchronize(x.device)
                    t_active_shared = time.perf_counter()
                torch.cuda.current_stream(x.device).wait_stream(self._allreduce_stream)
                y = y_reduce.to(torch.float32)
                reduced_moe_ready = True
            used_gpu_prefill_moe = True
            if active_profile and not overlap_shared_reduce:
                t_active_reduce_cast = t_active_allreduce = t_active_reduce_to_fp32 = t_active_add_shared = t_active_type_cast = 0.0
        elif self.routed_experts_device == "cpu" and self._should_use_in_process_cpu_server(x):
            profile = self._profile_enabled
            t0 = time.perf_counter() if profile else 0.0
            seq = self._external_cpu_server_submit_decode(x, indices, weights)
            t_submit = time.perf_counter() if profile else 0.0
            shared = self.shared_experts(x)
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_shared = time.perf_counter() if profile else 0.0
            y = self._external_cpu_server_wait_decode(seq, x)
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_wait = time.perf_counter() if profile else 0.0
            if profile:
                print(
                    f"moe_inproc_overlap_profile layer={self.layer_id} rank={rank} "
                    f"submit={t_submit - t0:.6f}s shared={t_shared - t_submit:.6f}s wait_h2d={t_wait - t_shared:.6f}s total={t_wait - t0:.6f}s",
                    flush=True,
                )
        elif self.routed_experts_device == "cpu" and self._should_use_external_cpu_server(x):
            if x.shape[0] == 1:
                profile = self._profile_enabled
                t0 = time.perf_counter() if profile else 0.0
                seq = self._external_cpu_server_submit_decode(x, indices, weights)
                t_submit = time.perf_counter() if profile else 0.0
                shared = self.shared_experts(x)
                if profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_shared = time.perf_counter() if profile else 0.0
                y = self._external_cpu_server_wait_decode(seq, x)
                if profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_wait = time.perf_counter() if profile else 0.0
                if profile:
                    print(
                        f"moe_external_overlap_profile layer={self.layer_id} rank={rank} "
                        f"submit={t_submit - t0:.6f}s shared={t_shared - t_submit:.6f}s wait_h2d={t_wait - t_shared:.6f}s total={t_wait - t0:.6f}s",
                        flush=True,
                    )
            else:
                shared = self.shared_experts(x)
                y = self._external_cpu_server_sync(x, indices, weights)
        elif self.routed_experts_device == "cpu" and self._should_use_rank0_cpu_server(x):
            if rank == 0:
                self.cpu_backend.submit_forward(x, indices, weights)
            shared = self.shared_experts(x)
            y = self._rank0_cpu_server_sync(x)
        elif self.routed_experts_device == "cpu" and self.cpu_moe_inproc_server_enabled and rank != 0:
            y = torch.zeros_like(x, dtype=torch.float32)
            shared = self.shared_experts(x)
            used_remote_only = True
        elif self.routed_experts_device == "cpu" and self.async_allreduce_enabled and tp_world_size > 1 and self._allreduce_stream is not None:
            profile = self._profile_enabled
            t0 = time.perf_counter() if profile else 0.0
            self.cpu_backend.submit_forward(x, indices, weights)
            t_submit = time.perf_counter() if profile else 0.0
            shared = self.shared_experts(x)
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_shared = time.perf_counter() if profile else 0.0
            with torch.cuda.stream(self._allreduce_stream):
                if self._should_use_cpu_host_reduce(x):
                    y, host_sync_time, host_reduce_time, host_h2d_time = self._cpu_host_reduce(x)
                else:
                    y = self.cpu_backend.sync_forward(x)
                    if profile and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_sync = time.perf_counter() if profile else 0.0
                    y_reduce = y.to(torch.float16 if self.reduce_fp16_enabled else x.dtype)
                    chunked_finalize = (
                        self.moe_reduce_cast_chunk_tokens > 0
                        and y_reduce.dim() == 2
                        and y_reduce.size(0) > self.moe_reduce_cast_chunk_tokens
                    )
                    if chunked_finalize:
                        del y
                    dist.all_reduce(y_reduce, async_op=False)
                    if profile and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_reduce = time.perf_counter() if profile else 0.0
                    if chunked_finalize:
                        y = y_reduce
                        reduced_moe_ready = True
                    else:
                        y = y_reduce.to(torch.float32)
            torch.cuda.current_stream(y.device).wait_stream(self._allreduce_stream)
            if 'host_sync_time' in locals():
                if profile and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_sync = time.perf_counter() if profile else 0.0
                t_reduce = t_sync
                if profile:
                    print(
                        f"moe_host_reduce_profile layer={self.layer_id} batch={x.shape[0]} cpu_sync={host_sync_time:.4f}s host_reduce={host_reduce_time:.4f}s h2d={host_h2d_time:.4f}s",
                        flush=True,
                    )
            if profile:
                print(
                    f"moe_profile layer={self.layer_id} batch={x.shape[0]} submit={t_submit - t0:.4f}s shared={t_shared - t_submit:.4f}s sync={t_sync - t_shared:.4f}s reduce={t_reduce - t_sync:.4f}s",
                    flush=True,
                )
        elif self.routed_experts_device == "cpu":
            self.cpu_backend.submit_forward(x, indices, weights)
            shared = self.shared_experts(x)
            if self._should_use_cpu_host_reduce(x):
                y, _host_sync_time, _host_reduce_time, _host_h2d_time = self._cpu_host_reduce(x)
            else:
                y = self.cpu_backend.sync_forward(x)
        else:
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
            for i in range(self.experts_start_idx, self.experts_end_idx):
                if counts[i] == 0:
                    continue
                expert = self.experts[i]
                idx, top = torch.where(indices == i)
                y[idx] += expert(x[idx], weights[idx, top, None])
            shared = self.shared_experts(x)
        if not reduced_moe_ready and tp_world_size > 1 and not (
            self.routed_experts_device == "cpu"
            and (
                (self.async_allreduce_enabled and not used_gpu_prefill_moe and not used_remote_only)
                or self._should_use_cpu_host_reduce(x)
                or self._should_use_in_process_cpu_server(x)
                or self._should_use_external_cpu_server(x)
                or self._should_use_rank0_cpu_server(x)
            )
        ):
            reduce_dtype = torch.float16 if self.reduce_fp16_enabled else x.dtype
            y_reduce = y.to(reduce_dtype)
            if 'active_profile' in locals() and active_profile:
                torch.cuda.synchronize(x.device)
                t_active_reduce_cast = time.perf_counter()
            chunked_finalize = (
                self.moe_reduce_cast_chunk_tokens > 0
                and y_reduce.dim() == 2
                and y_reduce.size(0) > self.moe_reduce_cast_chunk_tokens
            )
            if chunked_finalize:
                del y
            custom_reducer = None
            if (
                _env_enabled("DEEPSEEK_MOE_CUSTOM_ALLREDUCE")
                and dist.is_initialized()
                and world_size == 4
                and self._pd_active_phase() == "decode"
                and y_reduce.is_cuda
                and y_reduce.is_contiguous()
                and y_reduce.dim() == 2
                and y_reduce.size(0) == 1
                and y_reduce.size(1) == self.dim
                and y_reduce.dtype in (torch.bfloat16, torch.float16, torch.float32)
            ):
                custom_reducer = _get_decode_custom_allreduce(self.dim, y_reduce.dtype, y_reduce.device)
            if custom_reducer is not None:
                ok = custom_reducer.reduce_(y_reduce)
                if not ok:
                    dist.all_reduce(y_reduce)
            else:
                dist.all_reduce(y_reduce)
            if 'active_profile' in locals() and active_profile:
                torch.cuda.synchronize(x.device)
                t_active_allreduce = time.perf_counter()
            if chunked_finalize:
                y = self._finalize_reduced_moe(y_reduce, shared, x.dtype)
                if 'active_profile' in locals() and active_profile:
                    torch.cuda.synchronize(x.device)
                    t_active_type_cast = time.perf_counter()
                reduced_moe = True
            else:
                fused = _maybe_fused_moe_finalize(y_reduce, shared, x.dtype)
                if fused is not None:
                    y = fused
                    if 'active_profile' in locals() and active_profile:
                        torch.cuda.synchronize(x.device)
                        t_active_reduce_to_fp32 = time.perf_counter()
                        t_active_add_shared = t_active_reduce_to_fp32
                        t_active_type_cast = t_active_reduce_to_fp32
                    reduced_moe = True
                else:
                    y = y_reduce.to(torch.float32)
                    if 'active_profile' in locals() and active_profile:
                        torch.cuda.synchronize(x.device)
                        t_active_reduce_to_fp32 = time.perf_counter()
        if reduced_moe_ready and not reduced_moe:
            y = self._finalize_reduced_moe(y, shared, x.dtype)
            if 'active_profile' in locals() and active_profile:
                torch.cuda.synchronize(x.device)
                t_active_type_cast = time.perf_counter()
            reduced_moe = True
        if not reduced_moe:
            y += shared
            if 'active_profile' in locals() and active_profile:
                torch.cuda.synchronize(x.device)
                t_active_add_shared = time.perf_counter()
            y = y.type_as(x)
            if 'active_profile' in locals() and active_profile:
                torch.cuda.synchronize(x.device)
                t_active_type_cast = time.perf_counter()
        if 'active_profile' in locals() and active_profile:
            torch.cuda.synchronize(x.device)
            t_active_done = time.perf_counter()
            reduce_cast = t_active_reduce_cast - t_active_routed if t_active_reduce_cast else 0.0
            allreduce = t_active_allreduce - (t_active_reduce_cast or t_active_routed) if t_active_allreduce else 0.0
            reduce_to_fp32 = t_active_reduce_to_fp32 - (t_active_allreduce or t_active_routed) if t_active_reduce_to_fp32 else 0.0
            add_shared = t_active_add_shared - (t_active_reduce_to_fp32 or t_active_allreduce or t_active_routed) if t_active_add_shared else 0.0
            type_cast = t_active_type_cast - (t_active_add_shared or t_active_reduce_to_fp32 or t_active_allreduce or t_active_routed) if t_active_type_cast else 0.0
            print(
                f"gpu_moe_active_profile layer={self.layer_id} rank={rank} "
                f"prefetch={t_active_prefetch - t_active0:.6f}s cross={t_active_cross - t_active_prefetch:.6f}s "
                f"shared={t_active_shared - t_active_cross:.6f}s routed={t_active_routed - t_active_shared:.6f}s "
                f"finalize={t_active_done - t_active_routed:.6f}s reduce_cast={reduce_cast:.6f}s allreduce={allreduce:.6f}s "
                f"reduce_to_fp32={reduce_to_fp32:.6f}s add_shared={add_shared:.6f}s type_cast={type_cast:.6f}s "
                f"total={t_active_done - t_active0:.6f}s",
                flush=True,
            )
        return y.view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        self.hc_int8_enabled = _env_enabled("DEEPSEEK_HC_INT8")
        self.hc_pre_cuda_enabled = _env_enabled_default_on("DEEPSEEK_HC_PRE_CUDA")
        self.hc_post_cuda_enabled = _env_enabled_default_on("DEEPSEEK_HC_POST_CUDA")
        self.hc_fp16_mode = os.getenv("DEEPSEEK_HC_FP16", "0").lower()
        self._hc_int8_ready = False
        self._hc_cuda_ext = load_cuda_kernel() if self.hc_int8_enabled or self.hc_pre_cuda_enabled or self.hc_post_cuda_enabled else None
        self._hc_cuda_enabled = self._hc_cuda_ext is not None
        self._hc_pre_cuda_available = self._hc_cuda_enabled and hasattr(self._hc_cuda_ext, "hc_split_pre_forward")
        self._hc_post_cuda_available = self._hc_cuda_enabled and hasattr(self._hc_cuda_ext, "hc_post_forward")
        self.layer_profile_enabled = _env_enabled("DEEPSEEK_LAYER_PROFILE")
        self.prefetch_moe_before_ffn = _env_enabled("DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN")
        self.hc_pre_chunk_tokens = max(0, int(os.getenv("DEEPSEEK_HC_PRE_CHUNK_TOKENS", "16384")))

    def prefetch_gpu_prefill_moe(self, device: torch.device, token_count: int) -> None:
        self.ffn.prefetch_gpu_prefill_moe(device, token_count)

    def release_gpu_prefill_moe_cache(self) -> None:
        backend = getattr(self.ffn, "gpu_prefill_backend", None)
        if backend is not None:
            backend.release_cache()
        if hasattr(self.ffn, "release_gguf_gpu_prefill_moe"):
            self.ffn.release_gguf_gpu_prefill_moe()
        for expert in getattr(self.ffn, "experts", []):
            if expert is not None and hasattr(expert, "clear_cuda_gguf_cache"):
                expert.clear_cuda_gguf_cache()

    def _ensure_hc_int8(self):
        if self._hc_int8_ready:
            return
        attn_q, attn_s = _quantize_int8_weight_torch(self.hc_attn_fn.detach())
        ffn_q, ffn_s = _quantize_int8_weight_torch(self.hc_ffn_fn.detach())
        self.register_buffer("hc_attn_int8_weight", attn_q.unsqueeze(0), persistent=False)
        self.register_buffer("hc_attn_int8_scale", attn_s.unsqueeze(0), persistent=False)
        self.register_buffer("hc_ffn_int8_weight", ffn_q.unsqueeze(0), persistent=False)
        self.register_buffer("hc_ffn_int8_scale", ffn_s.unsqueeze(0), persistent=False)
        self._hc_int8_ready = True

    def _hc_linear_int8(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        self._ensure_hc_int8()
        if kind == "attn":
            weight_q = self.hc_attn_int8_weight
            weight_s = self.hc_attn_int8_scale
        else:
            weight_q = self.hc_ffn_int8_weight
            weight_s = self.hc_ffn_int8_scale
        if self._hc_cuda_enabled and x.is_cuda:
            y = self._hc_cuda_ext.wo_a_int8_forward(
                x.unsqueeze(2).contiguous(),
                weight_q.contiguous(),
                weight_s.contiguous(),
            ).squeeze(2)
            return y.to(torch.float32)
        return soft_bf16_weight_gemm_int8(x, weight_q[0], weight_s[0]).to(torch.float32)

    def _hc_linear_fp16(self, x: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
        weight_name = "hc_attn_fp16_weight" if hc_fn is self.hc_attn_fn else "hc_ffn_fp16_weight"
        weight = getattr(self, weight_name, None)
        if weight is None or weight.device != hc_fn.device:
            weight = hc_fn.detach().to(dtype=torch.float16)
            self.register_buffer(weight_name, weight, persistent=False)
        return F.linear(x.to(torch.float16), weight).to(torch.float32)

    def _hc_should_use_fp16(self, kind: str, x: torch.Tensor, hc_fn: torch.Tensor) -> bool:
        if not x.is_cuda or not hc_fn.is_cuda:
            return False
        return self.hc_fp16_mode in {"1", "true", "yes", "all", kind}

    def _hc_pre_impl(self, x: torch.Tensor, shape: torch.Size, dtype: torch.dtype, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        kind = "attn" if hc_fn is self.hc_attn_fn else "ffn"
        if self.hc_int8_enabled:
            mixes = self._hc_linear_int8(x, kind) * rsqrt
        elif self._hc_should_use_fp16(kind, x, hc_fn):
            mixes = self._hc_linear_fp16(x, hc_fn) * rsqrt
        else:
            mixes = F.linear(x, hc_fn) * rsqrt
        if self.hc_pre_cuda_enabled and self._hc_pre_cuda_available and x.is_cuda and self.hc_mult == 4:
            y, _pre, post, comb = self._hc_cuda_ext.hc_split_pre_forward(
                mixes.contiguous(),
                x.view(shape).contiguous(),
                hc_scale.contiguous(),
                hc_base.contiguous(),
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
            )
            return y.to(dtype), post, comb
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        chunk = self.hc_pre_chunk_tokens
        if chunk > 0 and x.size(1) > chunk:
            y = torch.empty((shape[0], shape[1], shape[3]), device=x.device, dtype=dtype)
            post = torch.empty((shape[0], shape[1], self.hc_mult), device=x.device, dtype=torch.float32)
            comb = torch.empty((shape[0], shape[1], self.hc_mult, self.hc_mult), device=x.device, dtype=torch.float32)
            for start in range(0, x.size(1), chunk):
                end = min(start + chunk, x.size(1))
                part = x[:, start:end].flatten(2).float().contiguous()
                part_shape = torch.Size((shape[0], end - start, shape[2], shape[3]))
                y_part, post_part, comb_part = self._hc_pre_impl(
                    part,
                    part_shape,
                    dtype,
                    hc_fn,
                    hc_scale,
                    hc_base,
                )
                y[:, start:end].copy_(y_part)
                post[:, start:end].copy_(post_part)
                comb[:, start:end].copy_(comb_part)
                del part, y_part, post_part, comb_part
            return y, post, comb
        return self._hc_pre_impl(x.flatten(2).float(), shape, dtype, hc_fn, hc_scale, hc_base)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        if (
            self.hc_post_cuda_enabled
            and self._hc_post_cuda_available
            and x.is_cuda
            and residual.dim() == 4
            and residual.size(2) == 4
            and residual.size(0) == x.size(0)
            and residual.size(1) == x.size(1)
            and residual.size(3) == x.size(2)
        ):
            return self._hc_cuda_ext.hc_post_forward(
                x.contiguous(),
                residual.contiguous(),
                post.contiguous(),
                comb.contiguous(),
            )
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def _profile_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor], prefetch_next: Optional[nn.Module] = None) -> torch.Tensor:
        profile = self.layer_profile_enabled
        t0 = self._profile_sync() if profile else 0.0
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        t_hc_attn_pre = self._profile_sync() if profile else 0.0
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        t_attn = self._profile_sync() if profile else 0.0
        x = self.hc_post(x, residual, post, comb)
        t_hc_attn_post = self._profile_sync() if profile else 0.0

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        t_hc_ffn_pre = self._profile_sync() if profile else 0.0
        x = self.ffn_norm(x)
        if input_ids is not None and prefetch_next is not None and _cross_layer_gate_prediction_should_run(input_ids.numel()):
            max_k = _cross_layer_gate_prediction_max_k()
            if max_k > 0 and hasattr(prefetch_next, "ffn") and hasattr(prefetch_next.ffn, "gate"):
                pred_topk, pred_percentile = prefetch_next.ffn.gate.predict_indices_from_proxy(
                    x.view(-1, x.size(-1)),
                    input_ids.flatten(),
                    max_k,
                    _cross_layer_gate_profiler().percentile if _cross_layer_gate_profile_enabled() else 75.0,
                    compute_percentile=_cross_layer_gate_profile_enabled(),
                )
                prefetch_next.ffn.set_cross_layer_gate_prediction(self.layer_id, pred_topk, pred_percentile)
        gguf_next_managed = (
            prefetch_next is not None
            and hasattr(prefetch_next, "ffn")
            and getattr(prefetch_next.ffn, "gpu_gguf_prefill_moe_enabled", False)
            and not getattr(prefetch_next.ffn, "gpu_gguf_prefill_active_expert_enabled", False)
        )
        prefetch_before_ffn = self.prefetch_moe_before_ffn and not gguf_next_managed
        if prefetch_before_ffn and prefetch_next is not None and input_ids is not None:
            prefetch_next.prefetch_gpu_prefill_moe(x.device, input_ids.numel())
        x = self.ffn(x, input_ids, prefetch_next=prefetch_next)
        if not prefetch_before_ffn and not gguf_next_managed and prefetch_next is not None and input_ids is not None:
            prefetch_next.prefetch_gpu_prefill_moe(x.device, input_ids.numel())
        t_moe = self._profile_sync() if profile else 0.0
        x = self.hc_post(x, residual, post, comb)
        t_hc_ffn_post = self._profile_sync() if profile else 0.0
        if profile:
            print(
                f"layer_profile layer={self.layer_id} pos={start_pos} batch={x.shape[0]} "
                f"hc_attn_pre={t_hc_attn_pre - t0:.4f}s "
                f"attn={t_attn - t_hc_attn_pre:.4f}s "
                f"hc_attn_post={t_hc_attn_post - t_attn:.4f}s "
                f"hc_ffn_pre={t_hc_ffn_pre - t_hc_attn_post:.4f}s "
                f"moe={t_moe - t_hc_ffn_pre:.4f}s "
                f"hc_ffn_post={t_hc_ffn_post - t_moe:.4f}s",
                flush=True,
            )
        return x


class ParallelHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = (vocab_size // tp_world_size)
        self.hc_fp16_enabled = _env_enabled("DEEPSEEK_HEAD_HC_FP16")
        self.lm_head_fp16_enabled = _env_enabled("DEEPSEEK_LM_HEAD_FP16")
        self._raw_q8_0_ready = False
        self._raw_q8_0_chunk_rows = max(1, int(os.getenv("DEEPSEEK_Q8_0_HEAD_CHUNK_ROWS", "256")))
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))

    def set_q8_0_storage(self, blocks: torch.Tensor, row_elems: int | None = None) -> None:
        if blocks.dtype != torch.uint8 or blocks.dim() != 3 or blocks.size(-1) != 34:
            raise TypeError(f"expected q8_0 block tensor [rows, blocks, 34] uint8, got {blocks.dtype} {tuple(blocks.shape)}")
        expected_blocks = (self.dim + 31) // 32
        if blocks.size(0) != self.part_vocab_size or blocks.size(1) != expected_blocks:
            raise ValueError(
                f"q8_0 block shape mismatch: got {tuple(blocks.shape)}, expected ({self.part_vocab_size}, {expected_blocks}, 34)"
            )
        row_elems = self.dim if row_elems is None else int(row_elems)
        if row_elems != self.dim:
            raise ValueError(f"q8_0 row elems mismatch: got {row_elems}, expected {self.dim}")
        self.register_buffer("raw_q8_0_weight", blocks.detach().to(device=self.weight.device, dtype=torch.uint8), persistent=False)
        empty_weight = torch.empty(0, dtype=self.weight.dtype, device=self.weight.device)
        try:
            self.weight.requires_grad_(False)
        except Exception:
            pass
        try:
            self.weight.data = empty_weight
        except Exception:
            self.weight = empty_weight
        self._raw_q8_0_ready = True

    def _lm_head_fp16_weight(self) -> torch.Tensor:
        weight = getattr(self, "lm_head_fp16_weight", None)
        if weight is None or weight.device != self.weight.device:
            weight = self.weight.detach().to(dtype=torch.float16)
            self.register_buffer("lm_head_fp16_weight", weight, persistent=False)
        return weight

    def get_logits(self, x, keep_all_positions: bool = False):
        if not keep_all_positions:
            x = x[:, -1]
        if self._raw_q8_0_ready and hasattr(self, "raw_q8_0_weight"):
            return q8_0_weight_gemm(
                x.float(),
                self.raw_q8_0_weight,
                row_elems=self.dim,
                chunk_rows=self._raw_q8_0_chunk_rows,
                out_dtype=torch.float32,
            )
        if self.lm_head_fp16_enabled and x.is_cuda and self.weight.is_cuda:
            return F.linear(x.to(torch.float16), self._lm_head_fp16_weight()).to(torch.float32)
        return F.linear(x.float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm, keep_all_positions: bool = False):
        if not keep_all_positions:
            x = x[:, -1:]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x), keep_all_positions=keep_all_positions)
        if tp_world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(tp_world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def next_token(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm, keep_all_positions: bool = False) -> torch.Tensor:
        if not keep_all_positions:
            x = x[:, -1:]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x), keep_all_positions=keep_all_positions)
        values, indices = logits.max(dim=-1)
        if tp_world_size > 1:
            all_values = [torch.empty_like(values) for _ in range(tp_world_size)]
            all_indices = [torch.empty_like(indices) for _ in range(tp_world_size)]
            dist.all_gather(all_values, values)
            dist.all_gather(all_indices, indices)
            values = torch.stack(all_values, dim=0)
            indices = torch.stack(all_indices, dim=0)
            best_rank = values.argmax(dim=0)
            next_token = indices.gather(0, best_rank.unsqueeze(0)).squeeze(0)
            return next_token + best_rank.to(next_token.dtype) * self.part_vocab_size
        return indices

    def _hc_linear_fp16(self, x: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "hc_fp16_weight", None)
        if weight is None or weight.device != hc_fn.device:
            weight = hc_fn.detach().to(dtype=torch.float16)
            self.register_buffer("hc_fp16_weight", weight, persistent=False)
        return F.linear(x.to(torch.float16), weight).to(torch.float32)

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if self.hc_fp16_enabled and x.is_cuda and hc_fn.is_cuda:
            mixes = self._hc_linear_fp16(x, hc_fn) * rsqrt
        else:
            mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPBlock(Block):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim, dtype=torch.int8 if args.mtp_int8 else None)
        self.h_proj = Linear(args.dim, args.dim, dtype=torch.int8 if args.mtp_int8 else None)
        if args.preload_mtp_e_proj_int8:
            self.e_proj.enable_online_int8()
        if args.preload_mtp_h_proj_int8:
            self.h_proj.enable_online_int8()
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [b,s,hc,d]
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


class Transformer(nn.Module):
    """Full DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits.
    Sets global state (tp_world_size, rank, default_dtype, scale_fmt, scale_dtype) in __init__."""
    def __init__(self, args: ModelArgs):
        global world_size, rank, tp_world_size, tp_rank, default_dtype, scale_fmt, scale_dtype
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        from src.models.deepseek_v4.partition import (
            assert_baseline_compatible_env,
            is_layer_pp_policy,
            log_partition_layout,
            normalize_policy,
        )
        partition_policy = normalize_policy(args.partition_policy)
        args.partition_policy = partition_policy
        assert_baseline_compatible_env(partition_policy, world_size)
        if is_layer_pp_policy(partition_policy):
            tp_world_size = 1
            tp_rank = 0
        else:
            tp_world_size = world_size
            tp_rank = rank
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        super().__init__()
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.partition_policy = partition_policy
        self.is_layer_pp = partition_policy == "layer_pp_4gpu"
        self.pp_size = world_size if self.is_layer_pp else 1
        self.pp_rank = rank if self.is_layer_pp else 0
        self.layer_start = (self.pp_rank * args.n_layers) // self.pp_size
        self.layer_end = ((self.pp_rank + 1) * args.n_layers) // self.pp_size
        self.owns_embedding = not self.is_layer_pp or self.pp_rank == 0
        self.owns_head = not self.is_layer_pp or self.pp_rank == self.pp_size - 1
        self.embed = ParallelEmbedding(args.vocab_size, args.dim) if self.owns_embedding else None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args) if self.layer_start <= layer_id < self.layer_end else None)
        self.norm = RMSNorm(args.dim, self.norm_eps) if self.owns_head else None
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps) if self.owns_head else None
        self.mtp = torch.nn.ModuleList()
        if not self.is_layer_pp:
            for layer_id in range(args.n_mtp_layers):
                self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
                self.mtp[-1].embed = self.embed
                self.mtp[-1].head = self.head
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            if self.owns_head:
                self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
                self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
                self.hc_head_scale = nn.Parameter(torch.empty(1))
            else:
                self.register_parameter("hc_head_fn", None)
                self.register_parameter("hc_head_base", None)
                self.register_parameter("hc_head_scale", None)
        if partition_policy in {"baseline_4gpu", "layer_pp_4gpu"}:
            log_partition_layout(self, world_size, rank, partition_policy)

    def prepare_cpu_expert_int8(self) -> None:
        timing_enabled = os.getenv("DEEPSEEK_LOAD_TIMING", "0").lower() in {"1", "true", "yes"}
        any_gguf_raw = False
        for layer in self.layers:
            if layer is None:
                continue
            backend = getattr(layer.ffn, "cpu_backend", None)
            if backend is not None:
                has_gguf_raw = bool(getattr(layer.ffn, "_has_gguf_raw_experts", lambda: False)())
                any_gguf_raw = any_gguf_raw or has_gguf_raw
                if has_gguf_raw:
                    if getattr(backend, "_inter_dim", 0) in {0, None}:
                        for expert in backend.experts:
                            if expert is not None:
                                backend._inter_dim = expert.w1.out_features
                                backend._swiglu_limit = float(expert.swiglu_limit)
                                break
                    continue
                t0 = time.perf_counter() if timing_enabled else 0.0
                prepare_fp4 = getattr(backend, "prepare_fp4_weights", None)
                if callable(prepare_fp4):
                    prepare_fp4()
                t1 = time.perf_counter() if timing_enabled else 0.0
                backend.prepare_int8_weights()
                if timing_enabled:
                    print(
                        f"load_timing rank={rank} layer={layer.layer_id} stage=cpu_expert_prepare "
                        f"fp4={t1 - t0:.3f}s int8={time.perf_counter() - t1:.3f}s total={time.perf_counter() - t0:.3f}s",
                        flush=True,
                    )
        for layer in self.mtp:
            backend = getattr(layer.ffn, "cpu_backend", None)
            if backend is not None:
                has_gguf_raw = bool(getattr(layer.ffn, "_has_gguf_raw_experts", lambda: False)())
                any_gguf_raw = any_gguf_raw or has_gguf_raw
                if has_gguf_raw:
                    if getattr(backend, "_inter_dim", 0) in {0, None}:
                        for expert in backend.experts:
                            if expert is not None:
                                backend._inter_dim = expert.w1.out_features
                                backend._swiglu_limit = float(expert.swiglu_limit)
                                break
                    continue
                t0 = time.perf_counter() if timing_enabled else 0.0
                prepare_fp4 = getattr(backend, "prepare_fp4_weights", None)
                if callable(prepare_fp4):
                    prepare_fp4()
                t1 = time.perf_counter() if timing_enabled else 0.0
                backend.prepare_int8_weights()
                if timing_enabled:
                    print(
                        f"load_timing rank={rank} layer={layer.layer_id} stage=mtp_cpu_expert_prepare "
                        f"fp4={t1 - t0:.3f}s int8={time.perf_counter() - t1:.3f}s total={time.perf_counter() - t0:.3f}s",
                        flush=True,
                    )
        if any_gguf_raw:
            return
        if (
            _env_enabled("DEEPSEEK_CPU_MOE_INPROC_SERVER")
            and rank == 0
            and world_size > 1
            and any(layer is not None for layer in self.layers)
        ):
            backends = []
            template_ffn = None
            for layer in self.layers:
                if layer is None:
                    continue
                ffn = getattr(layer, "ffn", None)
                backend = getattr(ffn, "cpu_backend", None) if ffn is not None else None
                if backend is None:
                    raise RuntimeError(f"layer {layer.layer_id} missing cpu_backend; in-process CPU MoE server requires routed-experts-device=cpu")
                backends.append(backend)
                if template_ffn is None:
                    template_ffn = ffn
            if template_ffn is None:
                return
            shm_name = start_in_process_cpu_moe_server(
                backends=backends,
                dim=template_ffn.dim,
                topk=template_ffn.n_activated_experts,
                inter_dim=backends[0]._inter_dim,
                n_routed_experts=template_ffn.n_routed_experts,
                swiglu_limit=backends[0]._swiglu_limit,
                shm_name=os.getenv("DEEPSEEK_CPU_MOE_SERVER_SHM"),
                use_v2=True,
            )
            os.environ["DEEPSEEK_CPU_MOE_SERVER_SHM"] = shm_name
            print(f"deepseek inproc cpu moe server started shm={shm_name}", flush=True)

    def release_cpu_expert_int8_prepare_cache(self) -> None:
        for layer in self.layers:
            if layer is None:
                continue
            backend = getattr(layer.ffn, "cpu_backend", None)
            if backend is not None and hasattr(backend, "release_int8_prepare_cache"):
                backend.release_int8_prepare_cache()
        for layer in self.mtp:
            backend = getattr(layer.ffn, "cpu_backend", None)
            if backend is not None and hasattr(backend, "release_int8_prepare_cache"):
                backend.release_int8_prepare_cache()

    def release_gpu_prefill_moe_cache(self) -> None:
        for layer in self.layers:
            if layer is None:
                continue
            layer.release_gpu_prefill_moe_cache()
        for layer in self.mtp:
            layer.release_gpu_prefill_moe_cache()

    def _local_layers(self):
        return [layer for layer in self.layers if layer is not None]

    def _run_local_layers(self, h: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        local_layers = self._local_layers()
        token_count = input_ids.numel()
        prefetch_window = max(1, int(os.getenv("DEEPSEEK_GGUF_GPU_PREFILL_MOE_PREFETCH_LAYERS", str(MoE._gguf_max_cached_layers))))
        gguf_prefetch_window = any(
            getattr(layer.ffn, "gpu_gguf_prefill_moe_enabled", False)
            for layer in local_layers[:prefetch_window]
        )
        keep_staged_after_forward = os.getenv("DEEPSEEK_GGUF_GPU_PREFILL_MOE_KEEP_STAGED", "1").lower() not in {"0", "false", "no"}
        release_staged_after_forward = (
            gguf_prefetch_window
            and not keep_staged_after_forward
            and any(token_count >= getattr(layer.ffn, "gpu_prefill_moe_min_tokens", 64) for layer in local_layers)
        )
        if gguf_prefetch_window:
            for layer in local_layers[:prefetch_window]:
                layer.prefetch_gpu_prefill_moe(h.device, token_count)
        elif local_layers:
            local_layers[0].prefetch_gpu_prefill_moe(h.device, token_count)
        for layer_idx, layer in enumerate(local_layers):
            prefetch_next = local_layers[layer_idx + 1] if layer_idx + 1 < len(local_layers) else None
            h = layer(h, start_pos, input_ids, prefetch_next=prefetch_next)
            if gguf_prefetch_window:
                if release_staged_after_forward:
                    layer.release_gpu_prefill_moe_cache()
                ahead_idx = layer_idx + prefetch_window
                if ahead_idx < len(local_layers):
                    local_layers[ahead_idx].prefetch_gpu_prefill_moe(h.device, token_count)
        return h

    def _send_activation(self, h: torch.Tensor, dst: int) -> None:
        dist.send(h.contiguous(), dst=dst)

    def _recv_activation(self, input_ids: torch.Tensor, src: int) -> torch.Tensor:
        shape = (input_ids.size(0), input_ids.size(1), self.hc_mult, self.dim)
        h = torch.empty(shape, device=input_ids.device, dtype=torch.get_default_dtype())
        dist.recv(h, src=src)
        return h

    def _forward_legacy(self, input_ids: torch.Tensor, start_pos: int = 0, return_next_token: bool = False, return_hidden: bool = False, keep_all_positions: bool = False):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        h = self._run_local_layers(h, start_pos, input_ids)
        if return_next_token:
            next_token = self.head.next_token(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm, keep_all_positions=keep_all_positions)
            if return_hidden:
                return next_token, h
            return next_token
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm, keep_all_positions=keep_all_positions)
        if return_hidden:
            return logits, h
        return logits

    def _forward_layer_pp(self, input_ids: torch.Tensor, start_pos: int = 0, return_next_token: bool = False, return_hidden: bool = False, keep_all_positions: bool = False):
        if return_hidden:
            raise RuntimeError("layer_pp_4gpu does not support return_hidden/MTP yet")
        if self.pp_rank == 0:
            h = self.embed(input_ids)
            h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        else:
            h = self._recv_activation(input_ids, self.pp_rank - 1)
        h = self._run_local_layers(h, start_pos, input_ids)
        if self.pp_rank + 1 < self.pp_size:
            self._send_activation(h, self.pp_rank + 1)
            if not return_next_token:
                return None
            next_token = torch.empty(input_ids.size(0), device=input_ids.device, dtype=torch.long)
        else:
            if not return_next_token:
                return None
            next_token = self.head.next_token(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm, keep_all_positions=keep_all_positions)
        dist.broadcast(next_token, src=self.pp_size - 1)
        return next_token

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, return_next_token: bool = False, return_hidden: bool = False, keep_all_positions: bool = False):
        if self.is_layer_pp:
            return self._forward_layer_pp(input_ids, start_pos, return_next_token, return_hidden, keep_all_positions)
        return self._forward_legacy(input_ids, start_pos, return_next_token, return_hidden, keep_all_positions)

    @torch.inference_mode()
    def draft_with_mtp(self, h: torch.Tensor, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Run a single MTP block over (h, input_ids) and return the greedy draft token.

        h:         [b, s, hc_mult, dim] — last main backbone hidden state at positions
                   [start_pos-s+1 .. start_pos]. Caller typically slices h[:, -1:] when
                   only the most recent position matters.
        input_ids: [b, s] — the tokens at the same positions as h.
        start_pos: position of the FIRST entry in input_ids (matches the convention used
                   by Transformer.forward / Block.forward).

        Returns a token tensor of shape [b] — argmax of the MTP logits at the last
        position, i.e. the draft token for position (start_pos + s).
        """
        if not self.mtp:
            raise RuntimeError("draft_with_mtp called but Transformer has no MTP layers")
        logits = self.mtp[0](h, start_pos, input_ids)  # MTPBlock already calls head; returns [b, vocab]
        # head.get_logits already slices x[:, -1] internally and head.forward all-gathers
        # across TP tp_world_size, so logits is [b, vocab] for the LAST position only.
        return logits.argmax(dim=-1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(n_hash_layers=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    print(model(x).size())
    for i in range(128, 150):
        print(i, model(x[:, 0:1], i).size())

    h = torch.randn(2, 128, args.hc_mult, args.dim)
    mtp = model.mtp[0]
    print(mtp(h, 0, x).size())
    print(mtp(h[:, 0:1], 1, x[:, 0:1]).size())
