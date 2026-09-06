#!/usr/bin/env python3
"""Run serial real-checkpoint Qwen tensor-parallel context benchmarks.

Each context length launches the requested TP ranks, records rank-local logs,
parses the rank-0 timing line, and verifies that every rank generated the same
greedy token sequence. The token fixtures are deterministic natural-language
tokenizer IDs, not synthetic random tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TEXT = (
    "Decode context parallelism partitions the key-value history across devices "
    "while tensor parallelism partitions attention heads and matrix weights. "
    "A correct implementation must merge the distributed softmax maximum, "
    "denominator, and weighted value sum without changing greedy generation. "
    "Long-context inference on PCIe-connected GPUs is useful only when the "
    "reduced attention scan costs more than the added collectives. This benchmark "
    "uses deterministic tokenizer output from a natural-language systems paragraph. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Qwen checkpoint directory")
    parser.add_argument(
        "--binary",
        default="build/cpp_engine/pocketllm_engine",
        help="built C++ engine executable",
    )
    parser.add_argument(
        "--lengths",
        default="512,4096,8192,32768,65536",
        help="comma-separated prompt lengths",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    # Matches QwenEngineOptions::prefill_chunk_tokens. Keep these in step, or the
    # benchmark measures a chunk size production does not use.
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8192)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8", "turboquant_k8v4", "int8_per_token_head"), default="fp16")
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-top-p", type=float, default=1.0)
    parser.add_argument("--qwen-top-k", type=int, default=0)
    parser.add_argument("--qwen-seed", type=int, default=0)
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="serial fresh-process repetitions per prompt length",
    )
    parser.add_argument(
        "--topology-label",
        default="",
        help="human-readable topology label stored in the result artifact",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="environment override passed to every rank and recorded verbatim",
    )
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated physical GPU IDs; defaults to 0..tp_world-1",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="Qwen layer limit; keep 0 for the complete 64-layer model",
    )
    parser.add_argument(
        "--work-dir",
        default=".tmp/qwen_long_context",
        help="directory for token fixtures, NCCL IDs, logs, and results",
    )
    parser.add_argument(
        "--tokenizer-python",
        default=sys.executable,
        help="Python interpreter used to create tokenizer fixtures",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise SystemExit(f"invalid --env value, expected NAME=VALUE: {item}")
        overrides[name] = value
    return overrides


def checkpoint_manifest(ckpt: Path) -> dict[str, Any]:
    revision = None
    metadata_dir = ckpt / ".cache" / "huggingface" / "download"
    files: dict[str, Any] = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "model.safetensors",
        "model_mtp.safetensors",
    ):
        path = ckpt / name
        if not path.is_file():
            continue
        entry: dict[str, Any] = {"size": path.stat().st_size}
        if path.stat().st_size < 64 * 1024 * 1024:
            entry["sha256"] = sha256_file(path)
        metadata = metadata_dir / f"{name}.metadata"
        if metadata.is_file():
            lines = metadata.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if lines:
                revision = revision or lines[0]
            if len(lines) >= 2:
                hub_oid = lines[1]
                entry["hub_sha256" if len(hub_oid) == 64 else "hub_etag"] = hub_oid
        files[name] = entry
    return {"path": str(ckpt), "revision": revision, "files": files}


def make_token_fixture(
    ckpt: Path, path: Path, length: int, tokenizer_python: str
) -> list[int]:
    """Encode a deterministic real-text fixture using the checkpoint tokenizer."""
    if path.exists():
        tokens = [int(item) for item in path.read_text(encoding="ascii").split(",") if item]
        if len(tokens) >= length:
            return tokens[:length]

    # Keep tokenizer loading in the requested interpreter so the benchmark can
    # run from the project's deepseek environment without importing it here.
    code = """
from transformers import AutoTokenizer
import json
import sys
ckpt, target = sys.argv[1], int(sys.argv[2])
text = (
    "Decode context parallelism partitions the key-value history across devices "
    "while tensor parallelism partitions attention heads and matrix weights. "
    "A correct implementation must merge the distributed softmax maximum, "
    "denominator, and weighted value sum without changing greedy generation. "
    "Long-context inference on PCIe-connected GPUs is useful only when the "
    "reduced attention scan costs more than the added collectives. This benchmark "
    "uses deterministic tokenizer output from a natural-language systems paragraph. "
) * 4000
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
ids = tok.encode(text, add_special_tokens=False)
if len(ids) < target:
    raise RuntimeError(f"fixture text produced {len(ids)} tokens, need {target}")
print(json.dumps(ids[:target]))
"""
    completed = subprocess.run(
        [tokenizer_python, "-c", code, str(ckpt), str(length)],
        check=True,
        capture_output=True,
        text=True,
    )
    # transformers may print informational warnings to stderr; stdout is JSON.
    tokens = json.loads(completed.stdout)
    if not isinstance(tokens, list) or len(tokens) != length:
        raise RuntimeError(f"invalid tokenizer fixture length for {length}: {len(tokens)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(int(token)) for token in tokens), encoding="ascii")
    return [int(token) for token in tokens]


def parse_runtime_line(line: str) -> dict[str, Any] | None:
    if not line.startswith("qwen_runtime=1 "):
        return None
    result: dict[str, Any] = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", line):
        try:
            result[key] = int(value)
        except ValueError:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def binary_metadata(binary: Path) -> dict[str, Any]:
    stat = binary.stat()
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "binary_inode": stat.st_ino,
        "binary_size": stat.st_size,
        "binary_mtime_ns": stat.st_mtime_ns,
        "binary_sha256": digest.hexdigest(),
    }


def parse_log(path: Path) -> tuple[dict[str, Any], list[int], list[dict[str, Any]]]:
    phases: list[dict[str, Any]] = []
    phase_pattern = re.compile(
        r"qwen_phase tag=(\S+) rank=(\d+) phase=(\S+) "
        r"seconds=([^\s]+)(?: calls=(\d+) share=([^\s]+))?"
    )
    runtime: dict[str, Any] | None = None
    tokens: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_runtime_line(line)
        if parsed is not None:
            runtime = parsed
        phase_match = phase_pattern.match(line)
        if phase_match:
            tag, rank, phase, seconds, calls, share = phase_match.groups()
            entry: dict[str, Any] = {
                "tag": tag,
                "rank": int(rank),
                "phase": phase,
                "seconds": float(seconds),
            }
            if calls is not None:
                entry["calls"] = int(calls)
            if share is not None:
                entry["share"] = float(share)
            phases.append(entry)
        match = re.match(r"generate_step=(\d+) token=(-?\d+)", line)
        if match:
            tokens.append(int(match.group(2)))
    if runtime is None:
        raise RuntimeError(f"rank log has no qwen_runtime line: {path}")
    if not tokens:
        raise RuntimeError(f"rank log has no generated tokens: {path}")
    return runtime, tokens, phases


def phase_leaf_summary(phases: list[dict[str, Any]]) -> dict[str, float]:
    """Return primitive phase times, excluding nested aliases and containers."""
    # `projection()` and `all_reduce_half()` each have an outer scope, while
    # full_attention adds site-specific scopes around the same calls. Keep the
    # primitive projection/collective scopes once instead of summing aliases.
    primitive_prefixes = ("pr.", "pd.")
    primitive_names = {
        "full.attn_kernel",
        "tp_all_reduce",
        "gated_delta",
        "norm",
        "add",
        "resid_copy",
        "attn_resid_norm",
        "swiglu.d",
        "swiglu.r",
        "lmhead.d",
        "lmhead.r",
        "argmax",
        "top1_allreduce",
    }
    summary: dict[str, float] = {}
    for item in phases:
        name = str(item["phase"])
        if not (name.startswith(primitive_prefixes) or name in primitive_names):
            continue
        summary[name] = summary.get(name, 0.0) + float(item["seconds"])
    return summary


def phase_share_summary(
    phases: list[dict[str, Any]], total_seconds: float
) -> dict[str, float]:
    """Convert primitive phase totals to shares of the runtime prefill time."""
    if total_seconds <= 0.0:
        return {}
    return {
        name: seconds / total_seconds
        for name, seconds in phase_leaf_summary(phases).items()
    }


def phase_summary(phases: list[dict[str, Any]]) -> dict[str, float]:
    return phase_leaf_summary(phases)



def environment_metadata() -> dict[str, str]:
    names = (
        "CUDA_VISIBLE_DEVICES",
        "QWEN_PHASE_PROFILE",
        "POCKETLLM_QWEN_GQA_OPTIMIZED",
        "POCKETLLM_QWEN_GQA_LONG_TILE",
        "POCKETLLM_QWEN_GQA_FLASH_TILE",
        "POCKETLLM_QWEN_GQA_MMA_TILE",
        "POCKETLLM_QWEN_DECODE_MMA",
        "POCKETLLM_QWEN_DECODE_MMA_TARGET_SPLITS",
        "POCKETLLM_QWEN_DECODE_TARGET_SPLITS",
        "POCKETLLM_QWEN_DECODE_SPLIT_POSITIONS",
        "POCKETLLM_QWEN_DECODE_GROUP_HEADS",
        "QWEN_NCCL_COMM_STREAM",
        "QWEN_COMM_OVERLAP_SLICES",
        "POCKETLLM_QWEN_GQA_QUERY_ROWS",
        "QWEN_FP8_F16_PREFILL_CUBLAS",
        "QWEN_FP16_LOGITS_MATVEC",
        "QWEN_FUSE_ATTN_RESID_NORM",
        "QWEN_GATED_DELTA_FLASHQLA_SM75",
        "QWEN_RMSNORM_HYBRID_REDUCTION",
        "QWEN_RMSNORM_VECTOR",
        "QWEN_FUSE_QKVZ_DECODE",
        "QWEN_FUSE_FULL_QKV_DECODE",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"



def run_case(
    binary: Path,
    ckpt: Path,
    work_dir: Path,
    length: int,
    repetition: int,
    max_new_tokens: int,
    layers: int,
    tp_world: int,
    devices: list[int],
    tokenizer_python: str,
    prefill_chunk_tokens: int,
    kv_cache_dtype: str,
    qwen_temperature: float,
    qwen_top_p: float,
    qwen_top_k: int,
    qwen_seed: int,
    env_overrides: dict[str, str],
) -> dict[str, Any]:
    token_path = work_dir / f"tokens_{length}.txt"
    tokens = make_token_fixture(ckpt, token_path, length, tokenizer_python)
    case_dir = work_dir / f"run_{length}_rep{repetition}"
    case_dir.mkdir(parents=True, exist_ok=True)
    id_path = case_dir / "nccl.id"
    if id_path.exists():
        id_path.unlink()
    for old_log in case_dir.glob("rank*.log"):
        old_log.unlink()

    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    started = time.monotonic()
    statuses: dict[int, int] = {}
    try:
        for rank in range(tp_world):
            log = (case_dir / f"rank{rank}.log").open("wb")
            command = [
                str(binary),
                "--ckpt",
                str(ckpt),
                "--tp-world",
                str(tp_world),
                "--tp-rank",
                str(rank),
                "--device",
                "0",
                "--nccl-id-path",
                str(id_path),
                "--token-ids-file",
                str(token_path),
                "--generate-token",
                "123",
                "--max-new-tokens",
                str(max_new_tokens),
                "--max-context",
                str(length + max_new_tokens),
                "--prefill-chunk-tokens",
                str(prefill_chunk_tokens),
                "--kv-cache-dtype",
                kv_cache_dtype,
                "--qwen-temperature",
                str(qwen_temperature),
                "--qwen-top-p",
                str(qwen_top_p),
                "--qwen-top-k",
                str(qwen_top_k),
                "--qwen-seed",
                str(qwen_seed),
                "--smoke-layers",
                str(layers),
                "--resident-bench",
            ]
            env = os.environ.copy()
            env.update(env_overrides)
            env["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
            processes.append((rank, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)))
            log.close()
        while len(statuses) < len(processes):
            for rank, process in processes:
                if rank in statuses:
                    continue
                status = process.poll()
                if status is None:
                    continue
                statuses[rank] = status
                if status != 0:
                    for other_rank, other in processes:
                        if other_rank not in statuses and other.poll() is None:
                            other.terminate()
                    break
            if any(status != 0 for status in statuses.values()):
                break
            time.sleep(0.1)
    finally:
        for rank, process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                statuses[rank] = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                statuses[rank] = process.wait(timeout=30)

    ordered_statuses = [(rank, statuses[rank]) for rank in range(tp_world)]
    if any(status != 0 for _, status in ordered_statuses):
        details = ", ".join(f"rank{rank}={status}" for rank, status in ordered_statuses)
        raise RuntimeError(f"Qwen TP{tp_world} case {length} failed: {details}")

    parsed = [parse_log(case_dir / f"rank{rank}.log") for rank in range(tp_world)]
    rank_runtime = [item[0] for item in parsed]
    rank_tokens = [item[1] for item in parsed]
    rank_phases = [item[2] for item in parsed]
    if any(rank_tokens[0] != other for other in rank_tokens[1:]):
        raise RuntimeError(f"token parity failure at prompt length {length}")
    runtime = rank_runtime[0]
    if runtime.get("layers") != 64 and layers == 0:
        raise RuntimeError(f"complete-model run did not load 64 layers: {runtime}")
    elapsed = time.monotonic() - started
    token_bytes = token_path.read_bytes()
    command_template = [
        str(binary),
        "--ckpt", str(ckpt),
        "--tp-world", str(tp_world),
        "--tp-rank", "<rank>",
        "--device", "0",
        "--nccl-id-path", str(id_path),
        "--token-ids-file", str(token_path),
        "--generate-token", "123",
        "--max-new-tokens", str(max_new_tokens),
        "--max-context", str(length + max_new_tokens),
        "--prefill-chunk-tokens", str(prefill_chunk_tokens),
        "--kv-cache-dtype", kv_cache_dtype,
        "--qwen-temperature", str(qwen_temperature),
        "--qwen-top-p", str(qwen_top_p),
        "--qwen-top-k", str(qwen_top_k),
        "--qwen-seed", str(qwen_seed),
        "--smoke-layers", str(layers),
        "--resident-bench",
    ]
    result = {
        "prompt_tokens": length,
        "repetition": repetition,
        "generated_tokens": len(rank_tokens[0]),
        "tokens": rank_tokens[0],
        "elapsed_wall_seconds": elapsed,
        "runtime": runtime,
        "rank_runtime": rank_runtime,
        "rank_token_parity": True,
        "rank_wall_seconds": [item.get("wall") for item in rank_runtime],
        "rank_model_load_seconds": [
            item.get("model_load_seconds") for item in rank_runtime
        ],
        "rank_process_elapsed_seconds": [
            item.get("process_elapsed_final_seconds") for item in rank_runtime
        ],
        "rank_gpu_memory_used_bytes": [
            item.get("gpu_memory_used_bytes") for item in rank_runtime
        ],
        "rank_gpu_memory_total_bytes": [
            item.get("gpu_memory_total_bytes") for item in rank_runtime
        ],
        "rank_phase_summary": [phase_summary(items) for items in rank_phases],
        "phase_leaf_summary": phase_leaf_summary(rank_phases[0]),
        "phase_leaf_share_summary": phase_share_summary(
            rank_phases[0], float(runtime.get("prefill_seconds", 0.0))
        ),
        "phase_total_seconds": next(
            (float(item["seconds"]) for item in rank_phases[0]
             if item["phase"] == "TOTAL"),
            None,
        ),
        "token_fixture": str(token_path),
        "token_fixture_sha256": sha256_bytes(token_bytes),
        "command_template": command_template,
        "environment": environment_metadata(),
        "git_revision": git_revision(),
        "binary_metadata": binary_metadata(binary),
        "log_dir": str(case_dir),
    }
    result["runtime"]["actual_prompt_tokens"] = int(
        result["runtime"].get("prompt_tokens", length)
    )
    result["runtime"]["prefill_seconds_reported"] = result["runtime"].get(
        "prefill_seconds"
    )
    result["runtime"]["decode_seconds_reported"] = result["runtime"].get(
        "decode_seconds"
    )
    result["runtime"]["full_request_wall_seconds"] = result["runtime"].get("wall")
    result["runtime"]["phase_profile_present"] = bool(rank_phases[0])
    result["runtime"]["phase_profile_warning"] = (
        "QWEN_PHASE_PROFILE synchronizes around every scope; use only for attribution"
        if rank_phases[0]
        else None
    )
    result["runtime"]["cold_start_excluded"] = False
    result["runtime"]["failure_excluded"] = False
    result["runtime"]["note"] = (
        "phase_leaf_summary excludes full_attention and STACK.r containers; "
        "phase instrumentation synchronizes CUDA and is diagnostic only"
    )
    result["environment"] = {
        **result["environment"],
        "benchmark_cuda_visible_devices": ",".join(str(device) for device in devices),
        "benchmark_binary": str(binary),
        "benchmark_checkpoint": str(ckpt),
        "benchmark_prefill_chunk_tokens": str(prefill_chunk_tokens),
        "benchmark_kv_cache_dtype": kv_cache_dtype,
    }
    print(
        f"length={length} layers={runtime.get('layers')} "
        f"prefill_tps={runtime.get('prefill_tokens_per_s')} "
        f"decode_tps={runtime.get('decode_tokens_per_s')} "
        f"gpu_used_bytes={max(result['rank_gpu_memory_used_bytes'])} "
        f"rank_token_parity=PASS"
    )
    return result


def main() -> int:
    args = parse_args()
    binary = Path(args.binary).resolve()
    ckpt = Path(args.ckpt).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not binary.is_file():
        raise SystemExit(f"binary not found: {binary}")
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint directory not found: {ckpt}")
    if args.max_new_tokens < 2:
        raise SystemExit("--max-new-tokens must be at least 2 to measure decode")
    if args.prefill_chunk_tokens <= 0:
        raise SystemExit("--prefill-chunk-tokens must be positive")
    if args.qwen_temperature < 0.0:
        raise SystemExit("--qwen-temperature must be non-negative")
    if not 0.0 < args.qwen_top_p <= 1.0:
        raise SystemExit("--qwen-top-p must be in (0, 1]")
    if args.qwen_top_k < 0:
        raise SystemExit("--qwen-top-k must be non-negative")
    if args.qwen_seed < 0:
        raise SystemExit("--qwen-seed must be non-negative")
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain positive integers")
    if args.layers < 0:
        raise SystemExit("--layers must be non-negative")
    if args.tp_world <= 0:
        raise SystemExit("--tp-world must be positive")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    env_overrides = parse_env_overrides(args.env)
    devices = (
        [int(item) for item in args.devices.split(",") if item.strip()]
        if args.devices
        else list(range(args.tp_world))
    )
    if len(devices) != args.tp_world:
        raise SystemExit("--devices count must match --tp-world")
    if len(set(devices)) != len(devices):
        raise SystemExit("--devices must not contain duplicates")

    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    output = {
        "mode": "qwen_tp_long_context_baseline",
        "checkpoint": checkpoint_manifest(ckpt),
        "binary": {
            "path": str(binary),
            "size": binary.stat().st_size,
            "sha256": sha256_file(binary),
        },
        "tp_world": args.tp_world,
        "devices": devices,
        "topology_label": args.topology_label,
        "layers": args.layers,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "kv_cache_dtype": args.kv_cache_dtype,
        "git_revision": git_revision(),
        "binary_metadata": binary_metadata(binary),
        # Both the ambient CUDA/NCCL environment and the explicit --env overrides
        # are recorded. They answer different questions: the first says what the
        # machine was, the second says what the run asked for.
        "environment": environment_metadata(),
        "env_overrides": env_overrides,
        "qwen_sampling": {
            "temperature": args.qwen_temperature,
            "top_p": args.qwen_top_p,
            "top_k": args.qwen_top_k,
            "seed": args.qwen_seed,
        },
        "repetitions": args.repetitions,
        "serial_cases": True,
        "phase_profile_diagnostic_only": "QWEN_PHASE_PROFILE" in os.environ,
        "results": results,
    }
    result_path = work_dir / "results.json"
    for length in lengths:
        for repetition in range(args.repetitions):
            results.append(
                run_case(
                    binary,
                    ckpt,
                    work_dir,
                    length,
                    repetition,
                    args.max_new_tokens,
                    args.layers,
                    args.tp_world,
                    devices,
                    args.tokenizer_python,
                    args.prefill_chunk_tokens,
                    args.kv_cache_dtype,
                    args.qwen_temperature,
                    args.qwen_top_p,
                    args.qwen_top_k,
                    args.qwen_seed,
                    env_overrides,
                )
            )
            result_path.write_text(
                json.dumps(output, indent=2) + "\n", encoding="utf-8"
            )
    medians: dict[str, Any] = {}
    for length in lengths:
        cases = [item for item in results if item["prompt_tokens"] == length]
        fields = (
            "wall",
            "prefill_seconds",
            "prefill_tokens_per_s",
            "decode_seconds",
            "decode_tokens_per_s",
            "model_load_seconds",
            "process_elapsed_final_seconds",
            "gpu_memory_used_bytes",
        )
        summary: dict[str, Any] = {}
        for field in fields:
            values = sorted(
                float(case["runtime"][field])
                for case in cases
                if field in case["runtime"]
            )
            if values:
                middle = len(values) // 2
                summary[field] = (
                    values[middle]
                    if len(values) % 2
                    else (values[middle - 1] + values[middle]) / 2.0
                )
        medians[str(length)] = summary
    output["medians"] = medians
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"results={result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        raise
