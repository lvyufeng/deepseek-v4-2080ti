import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
import re
import time

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("DEEPSEEK_CPU_MOE_SERVER_CUDA_VISIBLE_DEVICES", "")

import torch
from src.components.moe.ipc import CPUMoESharedMemory
from src.loader.safetensors import iter_safetensors_shards, read_safetensors_index
from src.components.moe.shared_weights import SharedCPUMoEWeightArena, SharedCPUMoEWeightSet
from src.models.deepseek_v4 import runtime as model_module
from src.components.moe.cpu_backend import CPURoutedExpertsBackend, run_native_int8_loop
from src.models.deepseek_v4.runtime import Expert, ModelArgs


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes"}


def _enable_numa_interleave() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(0b11)
    ret = libc.syscall(238, 3, ctypes.byref(mask), ctypes.c_ulong(64))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")


class RoutedLayer:
    def __init__(self, layer_id: int, args: ModelArgs, server_shards: int = 1):
        expert_dtype = None
        if args.expert_dtype == "fp4":
            expert_dtype = torch.float4_e2m1fn_x2
        elif args.expert_dtype == "int8":
            expert_dtype = torch.int8
        experts = torch.nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit, shared_int8_enabled=False)
            for _ in range(args.n_routed_experts)
        ])
        self.experts = experts
        self.cpu_backend = CPURoutedExpertsBackend(
            layer_idx=layer_id,
            experts=experts,
            experts_start_idx=0,
            experts_end_idx=args.n_routed_experts,
            num_experts_per_tok=args.n_activated_experts,
            output_dim=args.dim,
        )
        if server_shards > 1:
            if args.n_routed_experts % server_shards != 0:
                raise ValueError(f"n_routed_experts={args.n_routed_experts} must be divisible by server_shards={server_shards}")
            shard_size = args.n_routed_experts // server_shards
            self.shard_backends = [
                CPURoutedExpertsBackend(
                    layer_idx=layer_id,
                    experts=experts,
                    experts_start_idx=shard * shard_size,
                    experts_end_idx=(shard + 1) * shard_size,
                    num_experts_per_tok=args.n_activated_experts,
                    output_dim=args.dim,
                )
                for shard in range(server_shards)
            ]
        else:
            self.shard_backends = []
        self.shard_outputs: list[torch.Tensor] = []

    def prepare_int8_weights(self) -> None:
        if self.shard_backends:
            for backend in self.shard_backends:
                backend.prepare_int8_weights()
        else:
            self.cpu_backend.prepare_int8_weights()

    def run_forward_into(
        self,
        input_cpu: torch.Tensor,
        ids_cpu: torch.Tensor,
        weights_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
        executor: ThreadPoolExecutor | None,
    ) -> None:
        if not self.shard_backends:
            self.cpu_backend.run_forward_cpu_into(input_cpu, ids_cpu, weights_cpu, output_cpu)
            return
        if len(self.shard_outputs) != len(self.shard_backends):
            self.shard_outputs = [torch.empty_like(output_cpu) for _ in self.shard_backends]
        futures = [
            executor.submit(backend.run_forward_cpu_into, input_cpu, ids_cpu, weights_cpu, self.shard_outputs[idx])
            for idx, backend in enumerate(self.shard_backends)
        ]
        for future in futures:
            future.result()
        output_cpu.copy_(self.shard_outputs[0])
        for partial in self.shard_outputs[1:]:
            output_cpu.add_(partial)


def _target_for_expert(expert: Expert, proj: str, kind: str):
    linear = getattr(expert, proj)
    return linear.weight if kind == "weight" else linear.scale


def _load_routed_experts(layers: list[RoutedLayer], ckpt_path: str) -> None:
    index = read_safetensors_index(ckpt_path)

    file_to_keys: dict[str, list[str]] = {}
    expected = set()
    for key, file_name in index.weight_map.items():
        match = _EXPERT_RE.match(key)
        if match is None:
            continue
        layer_id = int(match.group(1))
        if layer_id >= len(layers):
            continue
        file_to_keys.setdefault(file_name, []).append(key)
        expected.add(key)

    loaded = set()
    for file_idx, total_files, file_name, keys, reader in iter_safetensors_shards(index, file_to_keys, device="cpu"):
        print(f"load routed shard {file_idx}/{total_files}: {file_name}", flush=True)
        for key in keys:
            match = _EXPERT_RE.match(key)
            if match is None:
                continue
            layer_id = int(match.group(1))
            expert_id = int(match.group(2))
            proj = match.group(3)
            kind = match.group(4)
            target = _target_for_expert(layers[layer_id].experts[expert_id], proj, kind)
            tensor = reader.get_tensor(key)
            if kind == "weight" and target.dtype == torch.float4_e2m1fn_x2:
                target.view(torch.uint8).copy_(tensor.view(torch.uint8).to(device=target.device))
            else:
                if tensor.shape != target.shape:
                    raise ValueError(f"Shape mismatch for {key}: got {tuple(tensor.shape)}, expected {tuple(target.shape)}")
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(key)

    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(f"Missing {len(missing)} routed expert tensors, e.g. {missing[:10]}")



def _layer_pointer_tensor(layers: list[RoutedLayer], attr: str) -> torch.Tensor:
    ptrs = torch.empty(len(layers), dtype=torch.long)
    for idx, layer in enumerate(layers):
        ptrs[idx] = getattr(layer.cpu_backend, attr).data_ptr()
    return ptrs


def _shared_layer_pointer_tensor(shared_weights: SharedCPUMoEWeightSet, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    per_layer_tables = shared_weights.build_layer_pointer_tensor(name)
    ptrs = torch.empty(shared_weights.n_layers, dtype=torch.long)
    for layer_id in range(shared_weights.n_layers):
        ptrs[layer_id] = per_layer_tables[layer_id].data_ptr()
    return ptrs, per_layer_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shm-name", default="pocketllm_cpu_moe_server")
    parser.add_argument("--omp-threads", type=int, default=64)
    parser.add_argument("--server-shards", type=int, default=int(os.getenv("DEEPSEEK_CPU_MOE_SERVER_SHARDS", "1")))
    parser.add_argument("--shared-weight-dir", default=os.getenv("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR"))
    parser.add_argument("--shared-weight-world-size", type=int, default=int(os.getenv("WORLD_SIZE", "4")))
    args = parser.parse_args()

    if args.server_shards < 1:
        raise ValueError("--server-shards must be >= 1")
    shard_threads = max(1, args.omp_threads // args.server_shards)
    os.environ["OMP_NUM_THREADS"] = str(shard_threads)
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ["DEEPSEEK_CPU_NATIVE_PERSISTENT_TEAM"] = os.getenv("DEEPSEEK_CPU_NATIVE_PERSISTENT_TEAM", "0")
    os.environ["DEEPSEEK_CPU_TOPK_PERSISTENT"] = os.getenv("DEEPSEEK_CPU_TOPK_PERSISTENT", "0")
    os.environ.setdefault("DEEPSEEK_CPU_TOPK_PARALLEL", "0")

    import src.components.moe.cpu_backend as cpu_routed_backend
    cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=shard_threads)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    model_module.world_size = 1
    model_module.rank = 0

    with open(args.config) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = "cpu"
    model_args = ModelArgs(**config_data)

    if _env_enabled("DEEPSEEK_CPU_MOE_NUMA_INTERLEAVE"):
        _enable_numa_interleave()
        print("cpu_moe_server numa interleave enabled", flush=True)

    shm = CPUMoESharedMemory(args.shm_name, model_args.dim, model_args.n_activated_experts, create=True)
    print(f"cpu_moe_server ready shm={args.shm_name}", flush=True)

    shared_weight_mode = _env_enabled("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS") and bool(args.shared_weight_dir)
    if shared_weight_mode:
        shared_weights = None
        owner_arenas = []
        try:
            if _env_enabled("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE"):
                for shard_rank in range(args.shared_weight_world_size):
                    arena = SharedCPUMoEWeightArena(
                        root_dir=args.shared_weight_dir,
                        rank=shard_rank,
                        world_size=args.shared_weight_world_size,
                        n_layers=model_args.n_layers,
                        n_routed_experts=model_args.n_routed_experts,
                        dim=model_args.dim,
                        moe_inter_dim=model_args.moe_inter_dim,
                        create=True,
                    )
                    print(f"cpu_moe_server pre-touch shared weights rank={shard_rank} bytes={arena._size}", flush=True)
                    arena.pre_touch()
                    owner_arenas.append(arena)
                print(f"cpu_moe_server created shared weight arenas dir={args.shared_weight_dir}", flush=True)
            SharedCPUMoEWeightArena.wait_until_ready(args.shared_weight_dir, args.shared_weight_world_size)
            shared_weights = SharedCPUMoEWeightSet.attach_all(args.shared_weight_dir, args.shared_weight_world_size)
            print(f"cpu_moe_server attach shared weights dir={args.shared_weight_dir}", flush=True)
            if not _env_enabled("DEEPSEEK_CPU_MOE_CPP_LOOP"):
                raise RuntimeError("shared-weight CPU MoE server currently requires DEEPSEEK_CPU_MOE_CPP_LOOP=1")
            use_v2_loop = _env_enabled("DEEPSEEK_CPU_MOE_CPP_LOOP_V2")
            w1_layers, w1_tables = _shared_layer_pointer_tensor(shared_weights, "w1.weight")
            w2_layers, w2_tables = _shared_layer_pointer_tensor(shared_weights, "w2.weight")
            w3_layers, w3_tables = _shared_layer_pointer_tensor(shared_weights, "w3.weight")
            s1_layers, s1_tables = _shared_layer_pointer_tensor(shared_weights, "w1.scale")
            s2_layers, s2_tables = _shared_layer_pointer_tensor(shared_weights, "w2.scale")
            s3_layers, s3_tables = _shared_layer_pointer_tensor(shared_weights, "w3.scale")
            _keepalive = (w1_tables, w2_tables, w3_tables, s1_tables, s2_tables, s3_tables)
            print(f"cpu_moe_server shared native loop {'v2' if use_v2_loop else 'v1'}", flush=True)
            run_native_int8_loop(
                args.shm_name,
                w1_layers,
                w2_layers,
                w3_layers,
                s1_layers,
                s2_layers,
                s3_layers,
                model_args.n_layers,
                model_args.dim,
                model_args.n_activated_experts,
                model_args.moe_inter_dim,
                model_args.n_routed_experts,
                shm.output_slots,
                float(model_args.swiglu_limit),
                use_v2=use_v2_loop,
            )
        finally:
            if shared_weights is not None:
                shared_weights.close()
            for arena in owner_arenas:
                arena.close(unlink=True)
            shm.close(unlink=True)
        return

    print("cpu_moe_server init routed experts", flush=True)
    init_start = time.perf_counter()
    with torch.device("cpu"):
        layers = [RoutedLayer(layer_id, model_args, server_shards=args.server_shards) for layer_id in range(model_args.n_layers)]
    print(f"cpu_moe_server init time: {time.perf_counter() - init_start:.3f}s", flush=True)

    print("cpu_moe_server load routed experts", flush=True)
    load_start = time.perf_counter()
    _load_routed_experts(layers, args.ckpt_path)
    for layer in layers:
        layer.prepare_int8_weights()
    print(f"cpu_moe_server load time: {time.perf_counter() - load_start:.3f}s", flush=True)

    executor = ThreadPoolExecutor(max_workers=args.server_shards) if args.server_shards > 1 else None
    if args.server_shards > 1:
        print(f"cpu_moe_server shards={args.server_shards} shard_threads={shard_threads}", flush=True)

    if _env_enabled("DEEPSEEK_CPU_MOE_CPP_LOOP"):
        use_v2_loop = _env_enabled("DEEPSEEK_CPU_MOE_CPP_LOOP_V2")
        w1_layers = _layer_pointer_tensor(layers, "_native_int8_w1_ptrs")
        w2_layers = _layer_pointer_tensor(layers, "_native_int8_w2_ptrs")
        w3_layers = _layer_pointer_tensor(layers, "_native_int8_w3_ptrs")
        s1_layers = _layer_pointer_tensor(layers, "_native_int8_s1_ptrs")
        s2_layers = _layer_pointer_tensor(layers, "_native_int8_s2_ptrs")
        s3_layers = _layer_pointer_tensor(layers, "_native_int8_s3_ptrs")
        print(f"cpu_moe_server native loop {'v2' if use_v2_loop else 'v1'}", flush=True)
        try:
            run_native_int8_loop(
                args.shm_name,
                w1_layers,
                w2_layers,
                w3_layers,
                s1_layers,
                s2_layers,
                s3_layers,
                model_args.n_layers,
                model_args.dim,
                model_args.n_activated_experts,
                model_args.moe_inter_dim,
                model_args.n_routed_experts,
                shm.output_slots,
                float(model_args.swiglu_limit),
                use_v2=use_v2_loop,
            )
        finally:
            shm.close(unlink=True)
        return

    last_seq = 0
    profile = os.getenv("DEEPSEEK_CPU_MOE_SERVER_PROFILE", "0").lower() in {"1", "true", "yes"}
    profile_every = int(os.getenv("DEEPSEEK_CPU_MOE_SERVER_PROFILE_EVERY", "43"))
    request_count = 0
    compute_total = 0.0
    wait_total = 0.0
    try:
        while True:
            wait_start = time.perf_counter() if profile else 0.0
            req = shm.wait_request(last_seq)
            if req is None:
                break
            wait_end = time.perf_counter() if profile else 0.0
            seq, layer_id = req
            layer = layers[layer_id]
            compute_start = time.perf_counter() if profile else 0.0
            layer.run_forward_into(shm.input_tensor(), shm.ids_tensor(), shm.weights_tensor(), shm.output_tensor(seq), executor)
            compute_end = time.perf_counter() if profile else 0.0
            shm.respond(seq)
            last_seq = seq
            if profile:
                request_count += 1
                wait_total += wait_end - wait_start
                compute_total += compute_end - compute_start
                if request_count % profile_every == 0:
                    print(
                        f"cpu_moe_server_profile requests={request_count} layer={layer_id} "
                        f"avg_wait={wait_total / profile_every:.6f}s avg_compute={compute_total / profile_every:.6f}s",
                        flush=True,
                    )
                    wait_total = 0.0
                    compute_total = 0.0
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        shm.close(unlink=True)


if __name__ == "__main__":
    main()
