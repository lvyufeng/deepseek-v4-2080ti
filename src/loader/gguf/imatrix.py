from __future__ import annotations

import atexit
import math
import os
import struct
import threading
from dataclasses import dataclass
from typing import BinaryIO, Sequence

import numpy as np
import torch

from src.loader.gguf.reader import GGUF_MAGIC, align_up


GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGML_TYPE_F32 = 0
DEFAULT_EXPERT_COUNT = 256


@dataclass
class _Stats:
    sum2: np.ndarray
    counts: np.ndarray


def _rank_output_path(path: str) -> str:
    if not path:
        return path
    rank = os.getenv("RANK")
    local_rank = os.getenv("LOCAL_RANK")
    if "{rank}" in path or "{local_rank}" in path:
        return path.format(rank=rank or "0", local_rank=local_rank or "0")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1:
        return path
    rank_id = int(rank or "0")
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank_id}{ext or '.gguf'}"


class RoutedIMatrixCollector:
    """Collect llama.cpp-compatible routed-MoE importance matrices.

    llama.cpp stores MoE imatrix data per merged GGUF expert tensor.  For a
    tensor with shape [in_dim, out_dim, n_experts], the imatrix payload is a
    [in_dim, n_experts] F32 tensor named ``<tensor>.in_sum2`` plus a
    [1, n_experts] F32 counts tensor named ``<tensor>.counts``.  Quantize then
    divides sum2 by counts per expert.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("DEEPSEEK_IMATRIX_CAPTURE", "0").lower() in {"1", "true", "yes"}
        self.output_path = _rank_output_path(os.getenv("DEEPSEEK_IMATRIX_CAPTURE_PATH", ""))
        self.dataset = os.getenv("DEEPSEEK_IMATRIX_DATASET", "pocketllm_runtime_capture")
        self.chunk_size = max(1, int(os.getenv("DEEPSEEK_IMATRIX_CHUNK_SIZE", "512")))
        self.n_experts = max(1, int(os.getenv("DEEPSEEK_IMATRIX_EXPERTS", str(DEFAULT_EXPERT_COUNT))))
        self.roles = _parse_str_set(os.getenv("DEEPSEEK_IMATRIX_ROLES", ""))
        self.layers = _parse_int_set(os.getenv("DEEPSEEK_IMATRIX_LAYERS", ""))
        self.verbose = os.getenv("DEEPSEEK_IMATRIX_VERBOSE", "0").lower() in {"1", "true", "yes"}
        self._stats: dict[str, _Stats] = {}
        self._lock = threading.Lock()
        self._registered = False
        if self.enabled and self.output_path:
            atexit.register(self.save)
            self._registered = True

    def should_capture(self, layer: int, role: str) -> bool:
        if not self.enabled:
            return False
        if self.layers is not None and int(layer) not in self.layers:
            return False
        if self.roles is not None and role not in self.roles:
            return False
        return True

    def observe(self, layer: int, expert: int, role: str, x: torch.Tensor) -> None:
        if not self.should_capture(layer, role):
            return
        if expert < 0 or expert >= self.n_experts:
            return
        if x.numel() == 0:
            return
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32).reshape(-1, x.shape[-1]).contiguous()
        if x_cpu.size(0) == 0:
            return
        # NumPy does the feature-wise reduction without keeping the full square
        # tensor alive longer than this call.
        arr = x_cpu.numpy()
        sum2 = np.einsum("bi,bi->i", arr, arr, dtype=np.float64).astype(np.float32)
        count = float(arr.shape[0])
        name = f"blk.{int(layer)}.{role}.weight"
        with self._lock:
            stat = self._stats.get(name)
            if stat is None:
                stat = _Stats(
                    sum2=np.zeros((self.n_experts, arr.shape[1]), dtype=np.float32),
                    counts=np.zeros((self.n_experts,), dtype=np.float32),
                )
                self._stats[name] = stat
            elif stat.sum2.shape[1] != arr.shape[1]:
                raise ValueError(f"imatrix width mismatch for {name}: {stat.sum2.shape[1]} vs {arr.shape[1]}")
            stat.sum2[int(expert)] += sum2
            stat.counts[int(expert)] += count
        if self.verbose:
            print(f"imatrix_capture {name} expert={int(expert)} rows={int(count)}", flush=True)

    def save(self, path: str | None = None) -> None:
        path = path or self.output_path
        if not path or not self._stats:
            return
        with self._lock:
            entries = [(name, stat.sum2.copy(), stat.counts.copy()) for name, stat in self._stats.items()]
        entries.sort(key=lambda item: item[0])
        max_count = max(float(counts.max()) for _name, _sum2, counts in entries) if entries else 0.0
        chunk_count = int(math.ceil(max_count / self.chunk_size)) if max_count > 0 else 0
        tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, tmp_path + ".prev")
        _write_imatrix_gguf(tmp_path, entries, dataset=self.dataset, chunk_size=self.chunk_size, chunk_count=chunk_count)
        if os.path.exists(path):
            os.replace(path, path + ".prev")
        os.replace(tmp_path, path)
        total_counts = sum(float(counts.sum()) for _name, _sum2, counts in entries)
        print(
            f"imatrix_capture_saved path={path} entries={len(entries)} total_counts={total_counts:.0f} chunk_count={chunk_count}",
            flush=True,
        )


_COLLECTOR: RoutedIMatrixCollector | None = None


def _parse_str_set(raw: str) -> set[str] | None:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _parse_int_set(raw: str) -> set[int] | None:
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    return values or None


def collector() -> RoutedIMatrixCollector:
    global _COLLECTOR
    if _COLLECTOR is None:
        _COLLECTOR = RoutedIMatrixCollector()
    return _COLLECTOR


def capture_routed_expert(layer: int | None, expert: int | None, role: str, x: torch.Tensor) -> None:
    if layer is None or expert is None:
        return
    c = collector()
    if c.enabled:
        c.observe(int(layer), int(expert), role, x)


def save_imatrix(path: str | None = None) -> None:
    collector().save(path)


def merge_imatrix_files(
    inputs: Sequence[str],
    output_path: str,
    *,
    dataset: str = "pocketllm_runtime_capture_merged",
    chunk_size: int | None = None,
) -> None:
    if not inputs:
        raise ValueError("at least one imatrix input is required")

    merged: dict[str, _Stats] = {}
    inferred_chunk_size = chunk_size
    for path in inputs:
        from src.loader.gguf.tensor_reader import GGUFTensorDataReader

        with GGUFTensorDataReader(str(path)) as reader:
            if reader.gguf.metadata.get("general.type") != "imatrix":
                raise ValueError(f"{path} is not an imatrix GGUF")
            if inferred_chunk_size is None:
                raw_chunk_size = reader.gguf.metadata.get("imatrix.chunk_size")
                if raw_chunk_size is not None:
                    inferred_chunk_size = int(raw_chunk_size)
            names = sorted(t.name for t in reader.gguf.tensors if t.name.endswith(".in_sum2"))
            for sum_name in names:
                base_name = sum_name[: -len(".in_sum2")]
                counts_name = f"{base_name}.counts"
                if counts_name not in reader.gguf.tensors_by_name:
                    raise ValueError(f"{path} is missing {counts_name}")
                sum2 = reader.read_tensor(sum_name).numpy().astype(np.float32, copy=False)
                counts = reader.read_tensor(counts_name).numpy().reshape(-1).astype(np.float32, copy=False)
                if sum2.ndim != 2:
                    raise ValueError(f"bad in_sum2 shape for {sum_name}: {sum2.shape}")
                if sum2.shape[0] != counts.shape[0]:
                    raise ValueError(f"expert count mismatch for {base_name}: {sum2.shape} vs {counts.shape}")
                stat = merged.get(base_name)
                if stat is None:
                    stat = _Stats(sum2=np.zeros_like(sum2), counts=np.zeros_like(counts))
                    merged[base_name] = stat
                elif stat.sum2.shape != sum2.shape or stat.counts.shape != counts.shape:
                    raise ValueError(f"shape mismatch while merging {base_name}")
                stat.sum2 += sum2
                stat.counts += counts

    entries = [(name, stat.sum2, stat.counts) for name, stat in sorted(merged.items())]
    if not entries:
        raise ValueError("no imatrix entries found")
    effective_chunk_size = int(inferred_chunk_size or 512)
    max_count = max(float(counts.max()) for _name, _sum2, counts in entries)
    chunk_count = int(math.ceil(max_count / effective_chunk_size)) if max_count > 0 else 0
    tmp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if os.path.exists(tmp_path):
        os.replace(tmp_path, tmp_path + ".prev")
    _write_imatrix_gguf(tmp_path, entries, dataset=dataset, chunk_size=effective_chunk_size, chunk_count=chunk_count)
    if os.path.exists(output_path):
        os.replace(output_path, output_path + ".prev")
    os.replace(tmp_path, output_path)
    total_counts = sum(float(counts.sum()) for _name, _sum2, counts in entries)
    print(
        f"imatrix_merge_saved path={output_path} inputs={len(inputs)} entries={len(entries)} total_counts={total_counts:.0f} chunk_count={chunk_count}",
        flush=True,
    )


def _pack_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_metadata(f: BinaryIO, dataset: str, chunk_size: int, chunk_count: int) -> None:
    _write_kv_string(f, "general.type", "imatrix")
    _write_kv_string_array(f, "imatrix.datasets", [dataset or "pocketllm_runtime_capture"])
    _write_kv_u32(f, "imatrix.chunk_count", int(chunk_count))
    _write_kv_u32(f, "imatrix.chunk_size", int(chunk_size))


def _write_kv_u32(f: BinaryIO, key: str, value: int) -> None:
    f.write(_pack_string(key))
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", int(value)))


def _write_kv_string(f: BinaryIO, key: str, value: str) -> None:
    f.write(_pack_string(key))
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    f.write(_pack_string(value))


def _write_kv_string_array(f: BinaryIO, key: str, values: list[str]) -> None:
    f.write(_pack_string(key))
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    f.write(struct.pack("<Q", len(values)))
    for value in values:
        f.write(_pack_string(value))


def _pad_file(f: BinaryIO, alignment: int = 32) -> None:
    pos = f.tell()
    padded = align_up(pos, alignment)
    if padded > pos:
        f.write(b"\0" * (padded - pos))


def _write_tensor_record(f: BinaryIO, name: str, dims: tuple[int, ...], type_id: int, offset: int) -> None:
    f.write(_pack_string(name))
    f.write(struct.pack("<I", len(dims)))
    for dim in dims:
        f.write(struct.pack("<Q", int(dim)))
    f.write(struct.pack("<I", int(type_id)))
    f.write(struct.pack("<Q", int(offset)))


def _write_imatrix_gguf(
    path: str,
    entries: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    dataset: str,
    chunk_size: int,
    chunk_count: int,
) -> None:
    tensors: list[tuple[str, tuple[int, ...], np.ndarray]] = []
    for name, sum2, counts in entries:
        if sum2.ndim != 2 or counts.ndim != 1:
            raise ValueError(f"bad imatrix stats for {name}")
        n_experts, width = sum2.shape
        if counts.shape[0] != n_experts:
            raise ValueError(f"bad imatrix counts for {name}")
        tensors.append((f"{name}.in_sum2", (width, n_experts), np.ascontiguousarray(sum2.astype("<f4", copy=False))))
        tensors.append((f"{name}.counts", (1, n_experts), np.ascontiguousarray(counts.reshape(n_experts, 1).astype("<f4", copy=False))))

    offsets: list[int] = []
    off = 0
    for _name, _dims, data in tensors:
        offsets.append(off)
        off += align_up(data.nbytes, 32)

    with open(path, "xb") as f:
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", 4))
        _write_metadata(f, dataset, chunk_size, chunk_count)
        for (name, dims, _data), offset in zip(tensors, offsets):
            _write_tensor_record(f, name, dims, GGML_TYPE_F32, offset)
        _pad_file(f, 32)
        for _name, _dims, data in tensors:
            data.tofile(f)
            _pad_file(f, 32)
