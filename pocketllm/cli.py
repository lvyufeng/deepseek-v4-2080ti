"""Command-line entry points for the unified PocketLLM API."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .api import ConfigurationError, EngineArgs, UnsupportedFeatureError
from .backends.factory import create_backend, select_backend
from .server.openai import serve
from .supervisor import TensorParallelSupervisor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pocketllm", description="PocketLLM unified inference interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="start the OpenAI-compatible server")
    serve_parser.add_argument("--model", required=True, help="checkpoint directory or model path")
    serve_parser.add_argument("--backend", choices=["auto", "torch", "cpp"], default="auto")
    serve_parser.add_argument("--tokenizer-path")
    serve_parser.add_argument("--config-path")
    serve_parser.add_argument("--model-format", choices=["auto", "safetensors", "gguf"], default="auto")
    serve_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    serve_parser.add_argument("--tensor-parallel-rank", type=int, default=0)
    serve_parser.add_argument(
        "--tensor-parallel-supervisor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="supervise local TP ranks automatically (default: enabled)",
    )
    serve_parser.add_argument(
        "--tensor-parallel-startup-timeout",
        type=float,
        default=300.0,
        metavar="SECONDS",
        help="rank startup timeout in seconds (default: 300)",
    )
    serve_parser.add_argument(
        "--tensor-parallel-shutdown-timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="rank shutdown timeout in seconds (default: 30)",
    )
    serve_parser.add_argument("--tensor-parallel-master-addr")
    serve_parser.add_argument("--tensor-parallel-master-port", type=int)
    serve_parser.add_argument("--tensor-parallel-rendezvous-dir")
    serve_parser.add_argument("--device")
    serve_parser.add_argument("--max-model-len", type=int)
    serve_parser.add_argument("--dtype")
    serve_parser.add_argument("--kv-cache-dtype", default="auto")
    serve_parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    serve_parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    serve_parser.add_argument("--max-batch-size", type=int, default=1)
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    # "auto" asks the native model registry which engine the checkpoint wants.
    # The explicit values stay for checkpoints that declare nothing, and for
    # forcing one runtime onto a checkpoint to compare them.
    serve_parser.add_argument("--engine-kind", default="auto",
                              choices=["auto", "qwen", "persistent"])
    # Matches the legacy server default: routed experts live in host memory so
    # a 4x2080Ti box can hold the model. "gpu" needs far more device memory.
    serve_parser.add_argument("--routed-experts-device", choices=["gpu", "cpu"], default="cpu")
    serve_parser.add_argument("--pd-mode", choices=["off", "scheduler"], default="scheduler")
    serve_parser.add_argument("--attention-window", type=int, default=0)
    serve_parser.add_argument("--attention-sink-tokens", type=int, default=0)
    serve_parser.add_argument("--speculative-method", choices=["mtp", "dspark", "dflash2"])
    serve_parser.add_argument("--speculative-tokens", type=int, default=1)
    serve_parser.add_argument("--served-model-name", help="model id reported by /v1/models")
    serve_parser.add_argument(
        "--backend-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="backend-specific option; repeatable and parsed as JSON when possible",
    )
    serve_parser.add_argument(
        "--supervised-child",
        action="store_true",
        help=argparse.SUPPRESS,  # internal flag for supervisor-launched children
    )
    return parser


def _backend_options(namespace: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {
        "engine_kind": namespace.engine_kind,
        "routed_experts_device": namespace.routed_experts_device,
        "pd_mode": namespace.pd_mode,
    }
    nccl_id_path = os.getenv("POCKETLLM_NCCL_ID_PATH") or os.getenv("NCCL_ID_PATH")
    if nccl_id_path:
        options["nccl_id_path"] = nccl_id_path
    for item in namespace.backend_option or []:
        key, separator, raw = str(item).partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"--backend-option expects KEY=VALUE, got {item!r}")
        try:
            # JSON keeps numbers, booleans, and nested values typed; a bare
            # string stays a string so paths do not need quoting.
            value = json.loads(raw)
        except ValueError:
            value = raw
        options[key.strip()] = value
    return options


def _args(namespace: argparse.Namespace) -> EngineArgs:
    return EngineArgs(
        model=namespace.model,
        backend=namespace.backend,
        tokenizer_path=namespace.tokenizer_path,
        config_path=namespace.config_path,
        model_format=namespace.model_format,
        tensor_parallel_size=namespace.tensor_parallel_size,
        tensor_parallel_rank=namespace.tensor_parallel_rank,
        device=namespace.device,
        max_model_len=namespace.max_model_len,
        dtype=namespace.dtype,
        kv_cache_dtype=namespace.kv_cache_dtype,
        prefill_chunk_tokens=namespace.prefill_chunk_tokens,
        enable_prefix_caching=namespace.enable_prefix_caching,
        max_batch_size=namespace.max_batch_size,
        attention_window=namespace.attention_window,
        attention_sink_tokens=namespace.attention_sink_tokens,
        speculative_method=namespace.speculative_method,
        speculative_tokens=namespace.speculative_tokens,
        backend_options=_backend_options(namespace),
    )


def _emit_readiness(rank: int) -> None:
    """Emit the exact marker the supervisor waits for."""
    print(f"POCKETLLM_RANK_READY rank={rank}", flush=True)


def _supervised_command(original_argv: list[str]) -> list[str]:
    """Build child argv from parent argv, preserving all serve arguments."""
    result: list[str] = []
    excluded = {
        "--tensor-parallel-supervisor",
        "--no-tensor-parallel-supervisor",
        "--tensor-parallel-rank",
        "--supervised-child",
    }
    skip_next = False
    for index, item in enumerate(original_argv):
        if skip_next:
            skip_next = False
            continue
        if item in excluded:
            if index + 1 < len(original_argv) and not original_argv[index + 1].startswith("-"):
                skip_next = True
            continue
        if any(item.startswith(prefix + "=") for prefix in excluded):
            continue
        result.append(item)
    result.extend(("--no-tensor-parallel-supervisor", "--supervised-child"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "serve":
        return 2

    engine_args = _args(args)
    world = engine_args.tensor_parallel_size
    rank = engine_args.tensor_parallel_rank
    supervised = (
        world > 1
        and rank == 0
        and getattr(args, "tensor_parallel_supervisor", True)
        and not getattr(args, "supervised_child", False)
    )

    # Supervised parent: spawn ranks and monitor.
    if world > 1 and rank == 0 and supervised:
        if args.device is not None:
            raise ConfigurationError(
                "--device cannot be used with automatic TP supervision; "
                "use --no-tensor-parallel-supervisor for manual rank control"
            )
        supervisor = TensorParallelSupervisor(
            command=[sys.executable, "-m", "pocketllm", *_supervised_command(argv or sys.argv[1:])],
            world_size=world,
            startup_timeout=args.tensor_parallel_startup_timeout,
            shutdown_timeout=args.tensor_parallel_shutdown_timeout,
            master_addr=args.tensor_parallel_master_addr,
            master_port=args.tensor_parallel_master_port,
            rendezvous_dir=args.tensor_parallel_rendezvous_dir,
        )
        try:
            return supervisor.run()
        except KeyboardInterrupt:
            supervisor.cleanup()
            return 0

    # Rank child or manual launch: construct one backend and dispatch by role.
    backend = create_backend(engine_args)
    try:
        if rank == 0:
            # A supervised rank 0 must join collectives and finish model loading
            # before any worker can become ready. Single-rank and manual launches
            # retain their existing lazy-loading behavior.
            if getattr(args, "supervised_child", False):
                backend.prepare()
            model_name = args.served_model_name or args.model
            on_ready = (lambda: _emit_readiness(rank)) if getattr(args, "supervised_child", False) else None
            serve(backend, host=args.host, port=args.port, model=model_name, on_ready=on_ready)
        else:
            # Nonzero rank enters backend worker loop.
            on_ready = (lambda: _emit_readiness(rank)) if getattr(args, "supervised_child", False) else None
            try:
                backend.run_worker(on_ready=on_ready)
            except UnsupportedFeatureError as exc:
                if supervised:
                    raise ConfigurationError(
                        f"{engine_args.backend} backend does not support automatic TP supervision; "
                        f"use an external launcher or the legacy native executable: {exc}"
                    ) from exc
                raise
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
