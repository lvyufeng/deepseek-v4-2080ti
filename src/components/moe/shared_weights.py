import ctypes
import json
import os
import re
import struct
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from typing import Iterable

import torch

_HEADER_MAGIC = b"PKTLMOE1"
_HEADER_VERSION = 1
_HEADER_STRUCT = struct.Struct("<8sQQQQQQ")
_ALIGN = 64


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total

    @property
    def nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class SharedTensorView:
    spec: TensorSpec
    offset: int


def _align_up(value: int, align: int = _ALIGN) -> int:
    return ((int(value) + align - 1) // align) * align


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


class SharedCPUMoEWeightArena:
    """POSIX shared-memory int8 routed expert storage for one TP rank shard."""

    def __init__(
        self,
        root_dir: str,
        rank: int,
        world_size: int,
        n_layers: int,
        n_routed_experts: int,
        dim: int,
        moe_inter_dim: int,
        create: bool,
        shm_name: str | None = None,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.n_layers = int(n_layers)
        self.n_routed_experts = int(n_routed_experts)
        self.dim = int(dim)
        self.moe_inter_dim = int(moe_inter_dim)
        if self.n_routed_experts % self.world_size != 0:
            raise ValueError("n_routed_experts must be divisible by world_size")
        self.n_local_experts = self.n_routed_experts // self.world_size
        self.expert_start = self.rank * self.n_local_experts
        self.expert_end = self.expert_start + self.n_local_experts
        self.meta_path = os.path.join(self.root_dir, "metadata.json")
        self.rank_meta_path = os.path.join(self.root_dir, f"rank{self.rank}.json")
        self.ready_path = os.path.join(self.root_dir, f"rank{self.rank}.ready")
        self.shm_name = shm_name or self._default_shm_name()
        self._shm: shared_memory.SharedMemory | None = None
        self._buffer = None
        self._views: dict[tuple[int, int, str], SharedTensorView] = {}
        self._tensors: dict[tuple[int, int, str], torch.Tensor] = {}
        self._size = 0
        self._create = bool(create)
        self._open(create=create)

    @staticmethod
    def enabled() -> bool:
        return os.getenv("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS", "0").lower() in {"1", "true", "yes"}

    @staticmethod
    def root_dir_from_env() -> str | None:
        value = os.getenv("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR")
        return None if not value else os.path.abspath(value)

    @staticmethod
    def build_specs(dim: int, moe_inter_dim: int) -> list[TensorSpec]:
        return [
            TensorSpec("w1.weight", (moe_inter_dim, dim), torch.int8),
            TensorSpec("w1.scale", (moe_inter_dim,), torch.float32),
            TensorSpec("w2.weight", (dim, moe_inter_dim), torch.int8),
            TensorSpec("w2.scale", (dim,), torch.float32),
            TensorSpec("w3.weight", (moe_inter_dim, dim), torch.int8),
            TensorSpec("w3.scale", (moe_inter_dim,), torch.float32),
        ]

    def _default_shm_name(self) -> str:
        run_name = _safe_name(os.path.basename(self.root_dir.rstrip(os.sep)) or "run")
        return f"pocketllm_moe_{run_name}_rank{self.rank}"

    def _layout_entries(self) -> Iterable[tuple[int, int, TensorSpec]]:
        specs = self.build_specs(self.dim, self.moe_inter_dim)
        for layer_id in range(self.n_layers):
            for expert_id in range(self.expert_start, self.expert_end):
                for spec in specs:
                    yield layer_id, expert_id, spec

    def _compute_layout(self) -> tuple[int, dict[tuple[int, int, str], SharedTensorView]]:
        views: dict[tuple[int, int, str], SharedTensorView] = {}
        offset = _align_up(_HEADER_STRUCT.size)
        for layer_id, expert_id, spec in self._layout_entries():
            offset = _align_up(offset)
            views[(layer_id, expert_id, spec.name)] = SharedTensorView(spec=spec, offset=offset)
            offset += spec.nbytes
        size = _align_up(offset)
        return size, views

    def _metadata(self) -> dict:
        return {
            "version": _HEADER_VERSION,
            "world_size": self.world_size,
            "n_layers": self.n_layers,
            "n_routed_experts": self.n_routed_experts,
            "n_local_experts": self.n_local_experts,
            "dim": self.dim,
            "moe_inter_dim": self.moe_inter_dim,
            "size": self._size,
        }

    def _rank_metadata(self) -> dict:
        return {
            **self._metadata(),
            "rank": self.rank,
            "shm_name": self.shm_name,
            "expert_start": self.expert_start,
            "expert_end": self.expert_end,
        }

    def _write_json_atomic(self, path: str, payload: dict) -> None:
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    def _write_metadata(self) -> None:
        self._write_json_atomic(self.rank_meta_path, self._rank_metadata())
        if self.rank == 0:
            self._write_json_atomic(self.meta_path, self._metadata())

    def _validate_header(self) -> None:
        magic, version, world_size, n_layers, n_routed_experts, dim, moe_inter_dim = _HEADER_STRUCT.unpack_from(self._buffer, 0)
        if magic != _HEADER_MAGIC:
            raise RuntimeError(f"shared arena {self.shm_name} has invalid magic")
        if version != _HEADER_VERSION:
            raise RuntimeError(f"shared arena {self.shm_name} has unsupported version {version}")
        expected = (self.world_size, self.n_layers, self.n_routed_experts, self.dim, self.moe_inter_dim)
        actual = (world_size, n_layers, n_routed_experts, dim, moe_inter_dim)
        if actual != expected:
            raise RuntimeError(f"shared arena header mismatch: got {actual}, expected {expected}")

    def _write_header(self) -> None:
        _HEADER_STRUCT.pack_into(
            self._buffer,
            0,
            _HEADER_MAGIC,
            _HEADER_VERSION,
            self.world_size,
            self.n_layers,
            self.n_routed_experts,
            self.dim,
            self.moe_inter_dim,
        )

    def _unlink_stale(self) -> None:
        try:
            stale = shared_memory.SharedMemory(name=self.shm_name, create=False)
        except FileNotFoundError:
            return
        stale.close()
        stale.unlink()

    def _open(self, create: bool) -> None:
        if create:
            os.makedirs(self.root_dir, exist_ok=True)
            self._unlink_stale()
        self._size, self._views = self._compute_layout()
        self._shm = shared_memory.SharedMemory(name=self.shm_name, create=create, size=self._size if create else 0)
        if not create:
            try:
                resource_tracker.unregister(self._shm._name, "shared_memory")
            except Exception:
                pass
            if self._shm.size != self._size:
                raise RuntimeError(f"shared arena {self.shm_name} size mismatch: got {self._shm.size}, expected {self._size}")
        self._buffer = self._shm.buf
        if create:
            self._write_header()
            self._write_metadata()
        else:
            self._validate_header()

    def close(self, unlink: bool = False) -> None:
        self._tensors.clear()
        if self._buffer is not None:
            self._buffer.release()
            self._buffer = None
        if self._shm is not None:
            self._shm.close()
            if unlink:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
            self._shm = None

    def tensor(self, layer_id: int, expert_id: int, name: str) -> torch.Tensor:
        key = (int(layer_id), int(expert_id), name)
        cached = self._tensors.get(key)
        if cached is not None:
            return cached
        view = self._views.get(key)
        if view is None:
            raise KeyError(f"unknown shared tensor {key}")
        tensor = torch.frombuffer(self._buffer, dtype=view.spec.dtype, count=view.spec.numel, offset=view.offset).view(view.spec.shape)
        self._tensors[key] = tensor
        return tensor

    def pre_touch(self) -> None:
        libc = ctypes.CDLL(None)
        address = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
        libc.memset(ctypes.c_void_p(address), 0, ctypes.c_size_t(self._size))
        self._write_header()
        try:
            libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(self._size), ctypes.c_int(14))
        except Exception:
            pass

    def mark_ready(self) -> None:
        self._write_metadata()
        tmp_path = f"{self.ready_path}.tmp"
        with open(tmp_path, "w") as f:
            f.write("ready\n")
        os.replace(tmp_path, self.ready_path)

    @classmethod
    def wait_until_ready(cls, root_dir: str, world_size: int, timeout_s: float = 600.0) -> None:
        root_dir = os.path.abspath(root_dir)
        deadline = time.time() + timeout_s
        for rank in range(int(world_size)):
            ready_path = os.path.join(root_dir, f"rank{rank}.ready")
            while not os.path.exists(ready_path):
                if time.time() > deadline:
                    raise TimeoutError(f"timeout waiting for {ready_path}")
                time.sleep(0.05)

    @classmethod
    def from_metadata(cls, root_dir: str, rank: int, create: bool = False):
        root_dir = os.path.abspath(root_dir)
        rank_meta_path = os.path.join(root_dir, f"rank{rank}.json")
        with open(rank_meta_path) as f:
            meta = json.load(f)
        return cls(
            root_dir=root_dir,
            rank=int(rank),
            world_size=int(meta["world_size"]),
            n_layers=int(meta["n_layers"]),
            n_routed_experts=int(meta["n_routed_experts"]),
            dim=int(meta["dim"]),
            moe_inter_dim=int(meta["moe_inter_dim"]),
            create=create,
            shm_name=str(meta["shm_name"]),
        )


class SharedCPUMoEWeightSet:
    def __init__(self, arenas: list[SharedCPUMoEWeightArena]):
        self.arenas = arenas
        if not arenas:
            raise ValueError("arenas must not be empty")
        self.world_size = arenas[0].world_size
        self.n_layers = arenas[0].n_layers
        self.n_routed_experts = arenas[0].n_routed_experts
        self.dim = arenas[0].dim
        self.moe_inter_dim = arenas[0].moe_inter_dim

    @classmethod
    def attach_all(cls, root_dir: str, world_size: int):
        arenas = [SharedCPUMoEWeightArena.from_metadata(root_dir, rank, create=False) for rank in range(int(world_size))]
        return cls(arenas)

    def close(self) -> None:
        for arena in self.arenas:
            arena.close()

    def tensor(self, layer_id: int, expert_id: int, name: str) -> torch.Tensor:
        expert_id = int(expert_id)
        rank = expert_id // self.arenas[0].n_local_experts
        return self.arenas[rank].tensor(layer_id, expert_id, name)

    def build_layer_pointer_tensor(self, name: str) -> torch.Tensor:
        ptrs = torch.empty((self.n_layers, self.n_routed_experts), dtype=torch.long)
        for layer_id in range(self.n_layers):
            for expert_id in range(self.n_routed_experts):
                ptrs[layer_id, expert_id] = self.tensor(layer_id, expert_id, name).data_ptr()
        return ptrs
