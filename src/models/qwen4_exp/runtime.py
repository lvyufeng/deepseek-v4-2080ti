"""TP4 heterogeneous runtime for Qwen4-Exp.

Launch layout: one process per GPU, NCCL for the per-layer all-reduce.  Dense
weights are sharded (see `builder.build_heterogeneous`); routed experts are
loaded into each rank's disjoint host-RAM shard at startup, and the 95 GiB PLE
table stays host-resident through the existing mapping.  Only active experts are
staged to GPU during a step.

Device memory per rank (BF16, 4 ranks):
  dense sharded weights   ~2.6 GiB   (attn/linear-attn projections, HC mixers)
  lm_head shard            0.30 GiB
  KV + conv/recurrent      grows with context
  staged experts        top_k * 6.6 MiB per layer, freed each step

Host memory per rank:
  resident expert shard    56.25 GiB (128 of 512 experts x 48 layers)

Entry point: `python -m src.models.qwen4_exp.runtime --model DIR --prompt ...`
under torchrun, or `run_tp4()` from another module.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from src.models.qwen4_exp.builder import build_heterogeneous
from src.models.qwen4_exp.config import Qwen4ExpConfig
from src.models.qwen4_exp.model import Qwen4ExpModel
from src.models.qwen4_exp.weights import MmapSafetensors, Qwen4ExpCheckpoint


@dataclass
class TPContext:
    rank: int
    world_size: int
    device: torch.device
    initialized: bool


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def _parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            begin, end = (int(value) for value in part.split("-", 1))
            cpus.extend(range(begin, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def _cuda_numa_node(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    pci_id = (
        f"{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:"
        f"{properties.pci_device_id:02x}.0"
    )
    try:
        with open(f"/sys/bus/pci/devices/{pci_id}/numa_node") as f:
            node = int(f.read())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def _physical_core_groups(cpus: set[int]) -> list[list[int]]:
    core_map: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(cpus):
        topology = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(os.path.join(topology, "physical_package_id")) as f:
                package_id = int(f.read())
            with open(os.path.join(topology, "core_id")) as f:
                core_id = int(f.read())
        except (OSError, ValueError):
            continue
        core_map.setdefault((package_id, core_id), []).append(cpu)
    return [sorted(siblings) for _, siblings in sorted(core_map.items())]


def _cpu_affinity_for_device(
    device: torch.device,
    local_rank: int,
    local_world_size: int,
) -> list[int] | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    node = _cuda_numa_node(device)
    if node is None:
        return None
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            node_cpus = set(_parse_cpu_list(f.read()))
    except (OSError, ValueError):
        return None

    allowed_cpus = set(os.sched_getaffinity(0))
    core_groups = _physical_core_groups(node_cpus & allowed_cpus)
    if not core_groups:
        return None

    visible_ranks = min(local_world_size, torch.cuda.device_count())
    peer_ranks = [
        rank
        for rank in range(visible_ranks)
        if _cuda_numa_node(torch.device(f"cuda:{rank}")) == node
    ]
    if local_rank not in peer_ranks:
        return None

    peer_index = peer_ranks.index(local_rank)
    base = len(core_groups) // len(peer_ranks)
    extra = len(core_groups) % len(peer_ranks)
    begin = peer_index * base + min(peer_index, extra)
    count = base + (1 if peer_index < extra else 0)
    selected = core_groups[begin : begin + count]
    return sorted(cpu for siblings in selected for cpu in siblings)


def _configure_cpu_affinity(ctx: TPContext) -> list[int] | None:
    if not _env_enabled("POCKETLLM_QWEN4_CPU_AFFINITY", "1") or ctx.device.type != "cuda":
        return None
    local_rank = ctx.device.index or 0
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(ctx.world_size)))
    cpus = _cpu_affinity_for_device(ctx.device, local_rank, local_world_size)
    if not cpus:
        if ctx.rank == 0:
            print("[qwen4exp] NUMA-local CPU affinity unavailable", flush=True)
        return None
    os.sched_setaffinity(0, cpus)
    cpu_spec = ",".join(str(cpu) for cpu in cpus)
    print(
        f"[qwen4exp] rank {ctx.rank}: GPU {local_rank} bound to CPUs "
        f"{cpu_spec} ({len(cpus)} logical CPUs)",
        flush=True,
    )
    return cpus


def init_distributed() -> TPContext:
    """Join the torchrun-provided process group, or fall back to single-rank."""
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return TPContext(rank, world_size, torch.device(f"cuda:{local_rank}"), True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return TPContext(rank, world_size, device, False)


def make_all_reduce(ctx: TPContext):
    """Sum a row-parallel block output across ranks. None when world_size == 1."""
    if ctx.world_size == 1:
        return None
    import torch.distributed as dist

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        # The tensor is a fresh block output, so an in-place reduce is safe and
        # avoids an extra allocation per layer.
        tensor = tensor.contiguous()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    return all_reduce


def gather_logits(logits: torch.Tensor, ctx: TPContext) -> torch.Tensor:
    """Concatenate the vocab-sharded lm_head outputs into full logits."""
    if ctx.world_size == 1:
        return logits
    import torch.distributed as dist

    parts = [torch.empty_like(logits) for _ in range(ctx.world_size)]
    dist.all_gather(parts, logits.contiguous())
    return torch.cat(parts, dim=-1)


def load_model(
    model_dir: str,
    ctx: TPContext,
    *,
    dtype: torch.dtype = torch.bfloat16,
    expert_cache_capacity: int = 0,
    host_expert_memory: bool = True,
    pin_experts: bool = True,
    profiler=None,
) -> tuple[Qwen4ExpModel, Qwen4ExpConfig]:
    _configure_cpu_affinity(ctx)
    config = Qwen4ExpConfig.from_pretrained(model_dir)
    checkpoint = Qwen4ExpCheckpoint(model_dir, store=MmapSafetensors(model_dir))
    if host_expert_memory:
        # Pull this rank's routed experts off the disk-backed mapping and into
        # host RAM before any step runs, so steady-state staging never seeks.
        import torch.distributed as dist

        distributed = ctx.world_size > 1 and dist.is_initialized()
        started = time.perf_counter()
        shard = checkpoint.preload_experts(
            config.text_config.num_hidden_layers,
            num_experts=config.text_config.num_experts,
            rank=ctx.rank,
            world_size=ctx.world_size,
            pin=pin_experts,
            progress=ctx.rank == 0 and os.environ.get("QWEN4EXP_PRELOAD_PROGRESS") == "1",
            barrier=dist.barrier if distributed else None,
        )
        elapsed = time.perf_counter() - started
        print(
            f"[qwen4exp] rank {ctx.rank}: {shard.num_local_experts} experts/layer "
            f"x {shard.stats()['layers']} layers resident, "
            f"{shard.resident_bytes / (1 << 30):.2f} GiB "
            f"({'pinned' if shard.pinned else 'pageable'}) in {elapsed:.1f}s",
            flush=True,
        )
        if distributed:
            dist.barrier()
    model = build_heterogeneous(
        config.text_config,
        checkpoint,
        device=ctx.device,
        dtype=dtype,
        rank=ctx.rank,
        world_size=ctx.world_size,
        all_reduce=make_all_reduce(ctx),
        expert_cache_capacity=expert_cache_capacity,
        profiler=profiler,
    )
    return model, config


def generate(
    model: Qwen4ExpModel,
    ctx: TPContext,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    eos_token_ids: tuple[int, ...] = (),
    on_token=None,
) -> tuple[list[int], dict[str, float]]:
    """Greedy decode with chunked prefill.

    Chunking keeps the QSA attention matrix bounded: a 64K single-shot prefill
    would need a 64K x 64K mask per layer, which does not fit.  The GatedDeltaNet
    and QSA caches carry state across chunks (validated by
    `test_chunked_prefill_matches_single_shot`).
    """
    prompt_len = input_ids.shape[1]
    cache = model.make_cache(batch_size=input_ids.shape[0], max_seq_len=prompt_len + max_new_tokens + 1)

    profiler = model.profiler
    prof_scope = profiler.scope if profiler else lambda _name: nullcontext()

    t0 = time.perf_counter()
    logits = None
    past = 0
    with torch.inference_mode(), prof_scope("prefill"):
        while past < prompt_len:
            piece = input_ids[:, past : past + chunk_size]
            logits = model.forward(
                piece,
                cache=cache,
                past_len=past,
                last_token_only=True,
            )
            past += piece.shape[1]
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    logits = gather_logits(logits[:, -1], ctx)
    produced: list[int] = []
    t1 = time.perf_counter()
    next_id = logits.argmax(-1, keepdim=True)
    with torch.inference_mode(), prof_scope("decode"):
        for _ in range(max_new_tokens):
            token = int(next_id.item())
            produced.append(token)
            if on_token is not None and ctx.rank == 0:
                on_token(token)
            if token in eos_token_ids:
                break
            step = model.forward(next_id.to(ctx.device), cache=cache, past_len=past)
            past += 1
            next_id = gather_logits(step[:, -1], ctx).argmax(-1, keepdim=True)
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1

    stats = {
        "prompt_tokens": float(prompt_len),
        "generated_tokens": float(len(produced)),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "prefill_tps": prompt_len / prefill_s if prefill_s > 0 else 0.0,
        "decode_tps": len(produced) / decode_s if decode_s > 0 else 0.0,
    }
    return produced, stats


def _load_tokenizer(model_dir: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def _apply_chat_template(model_dir: str, prompt: str) -> str:
    """Wrap a prompt in the checkpoint's chat template if jinja2 is available."""
    template_path = os.path.join(model_dir, "chat_template.jinja")
    if not os.path.exists(template_path):
        return prompt
    try:
        from jinja2 import Template
    except ImportError:
        return prompt
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(
        messages=[{"role": "user", "content": prompt}], add_generation_prompt=True
    )


def run_tp4(
    model_dir: str,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    expert_cache_capacity: int = 0,
    host_expert_memory: bool = True,
    pin_experts: bool = True,
    raw_prompt: bool = False,
    enable_profile: bool = False,
) -> dict:
    from src.models.qwen4_exp.profiler import Profiler

    ctx = init_distributed()
    profiler = Profiler(enabled=enable_profile, device=ctx.device) if enable_profile else None

    if ctx.rank == 0:
        print(f"[rank{ctx.rank}] loading {model_dir} world_size={ctx.world_size}", flush=True)

    t0 = time.perf_counter()
    model, config = load_model(
        model_dir,
        ctx,
        dtype=dtype,
        expert_cache_capacity=expert_cache_capacity,
        host_expert_memory=host_expert_memory,
        pin_experts=pin_experts,
        profiler=profiler,
    )
    load_s = time.perf_counter() - t0
    if ctx.rank == 0:
        mem = torch.cuda.memory_allocated(ctx.device) / 2**30 if ctx.device.type == "cuda" else 0.0
        print(f"[rank{ctx.rank}] loaded in {load_s:.1f}s, {mem:.2f} GiB on device", flush=True)

    # All layers share one MoE backend instance; reach it through layer 0.
    moe_backend = getattr(model.layers[0], "moe", None)
    moe_backend = getattr(moe_backend, "inner", moe_backend)  # unwrap ShardedMoE
    moe_phase_profile = enable_profile and os.environ.get("POCKETLLM_QWEN4_MOE_PHASE_PROFILE") == "1"
    if moe_phase_profile and hasattr(moe_backend, "phase_stats"):
        moe_backend.phase_stats = {}

    tokenizer = _load_tokenizer(model_dir)
    text = prompt if raw_prompt else _apply_chat_template(model_dir, prompt)
    input_ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.long)

    eos = tuple(config.text_config.eos_token_id)
    produced, stats = generate(
        model,
        ctx,
        input_ids,
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
        eos_token_ids=eos,
    )
    stats["load_s"] = load_s

    result = {"tokens": produced, "text": tokenizer.decode(produced), "stats": stats}
    if ctx.rank == 0:
        print(f"[rank0] output: {result['text']!r}", flush=True)
        print(f"[rank0] stats: {json.dumps(stats, indent=2)}", flush=True)

        if moe_phase_profile and getattr(moe_backend, "phase_stats", None):
            ps = moe_backend.phase_stats
            staged = moe_backend.stage_calls
            hits = moe_backend.cache_hits
            gib = moe_backend.stage_bytes / 2**30
            print(
                "\n[rank0] moe phases: "
                f"stage {ps.get('stage_s', 0.0):.3f}s  "
                f"compute {ps.get('compute_s', 0.0):.3f}s  "
                f"expert-calls {ps.get('experts', 0)} over {ps.get('calls', 0)} moe calls\n"
                f"[rank0] staging: {staged} misses / {hits} hits "
                f"({100.0 * hits / max(1, staged + hits):.1f}% hit), {gib:.2f} GiB H2D",
                flush=True,
            )

        if profiler and profiler.enabled:
            print("\n" + profiler.aggregate_report())
            if _env_enabled("POCKETLLM_QWEN4_PROFILE_TREE"):
                print("\n" + profiler.report())

    if ctx.initialized:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen4-Exp TP heterogeneous inference")
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--prompt", default="Hello, who are you?")
    parser.add_argument("--prompt-file", help="read the prompt from a file instead of --prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--expert-cache",
        type=int,
        default=0,
        help="number of staged experts to keep on device (0 = restage every step)",
    )
    parser.add_argument(
        "--mmap-experts",
        action="store_true",
        help="diagnostic fallback: read routed experts from the mmap on demand",
    )
    parser.add_argument(
        "--pageable-experts",
        action="store_true",
        help="keep the resident host shard pageable instead of requesting pinned memory",
    )
    parser.add_argument("--raw-prompt", action="store_true", help="skip the chat template")
    parser.add_argument("--profile", action="store_true", help="enable hierarchical profiler")
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()

    run_tp4(
        args.model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        chunk_size=args.chunk_size,
        dtype=getattr(torch, args.dtype),
        expert_cache_capacity=args.expert_cache,
        host_expert_memory=not args.mmap_experts,
        pin_experts=not args.pageable_experts,
        raw_prompt=args.raw_prompt,
        enable_profile=args.profile,
    )


if __name__ == "__main__":
    main()
