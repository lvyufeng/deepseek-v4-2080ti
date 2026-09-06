#!/usr/bin/env python3
"""Validate external Qwen DSpark prefix reuse against cold TP4 engines.

The persistent sequence covers an exact repeat, monotonic append, shorter prefix,
interior branch, and a compressed-context branch. Every cached result is compared
with an independently launched cold DSpark run, and every TP rank must agree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, TextIO


FIXTURE_TEXT = (
    "Speculative decoding proposes future language-model tokens and verifies them "
    "with one target forward. Exact greedy parity requires rollback of rejected "
    "state and commit of only the accepted prefix. This natural-language fixture "
    "exercises target auxiliary features, DSpark attention, and Markov correction. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dspark", required=True)
    parser.add_argument("--binary", default="build/cpp_engine/pocketllm_engine")
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--base-tokens", type=int, default=512)
    parser.add_argument("--append-tokens", type=int, default=64)
    parser.add_argument("--shorter-tokens", type=int, default=384)
    parser.add_argument("--branch-prefix-tokens", type=int, default=320)
    parser.add_argument("--branch-suffix-tokens", type=int, default=96)
    parser.add_argument("--compression-prefix-tokens", type=int, default=256)
    parser.add_argument("--compression-suffix-tokens", type=int, default=192)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--work-dir", default=".tmp/qwen_dspark_prefix_cache")
    parser.add_argument(
        "--tokenizer-python",
        default="/home/lvyufeng/miniconda3/envs/deepseek/bin/python",
    )
    return parser.parse_args()


def make_token_fixture(
    ckpt: Path, path: Path, length: int, tokenizer_python: str
) -> list[int]:
    if path.exists():
        tokens = [int(item) for item in path.read_text(encoding="ascii").split(",") if item]
        if len(tokens) >= length:
            return tokens[:length]
    code = f"""
from transformers import AutoTokenizer
import json, sys
ckpt, target = sys.argv[1], int(sys.argv[2])
text = {FIXTURE_TEXT!r} * max(4000, target // 20 + 1)
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
ids = tok.encode(text, add_special_tokens=False)
if len(ids) < target:
    raise RuntimeError(f"fixture produced {{len(ids)}} tokens, need {{target}}")
print(json.dumps(ids[:target]))
"""
    completed = subprocess.run(
        [tokenizer_python, "-c", code, str(ckpt), str(length)],
        check=True,
        capture_output=True,
        text=True,
    )
    tokens = json.loads(completed.stdout)
    if not isinstance(tokens, list) or len(tokens) != length:
        raise RuntimeError(f"invalid fixture for length {length}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(int(token)) for token in tokens), encoding="ascii")
    return [int(token) for token in tokens]


def parse_result(line: str) -> dict[str, Any] | None:
    if not line.startswith("qwen_persistent_result=1 "):
        return None
    result: dict[str, Any] = {}
    for key, value in re.findall(r"(\w+)=([^\s]+)", line):
        if key == "tokens":
            result[key] = [int(item) for item in value.split(",") if item]
            continue
        try:
            result[key] = int(value)
        except ValueError:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def parse_worker_tokens(path: Path) -> list[list[int]]:
    results: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("qwen_persistent_worker_result=1 "):
            continue
        match = re.search(r"\btokens=([^\s]*)", line)
        if match is None:
            raise RuntimeError(f"worker result is missing tokens: {path}")
        results.append([int(item) for item in match.group(1).split(",") if item])
    return results


def require_worker_parity(
    log_dir: Path, tp_world: int, expected: list[list[int]], label: str
) -> None:
    for rank in range(1, tp_world):
        actual = parse_worker_tokens(log_dir / f"rank{rank}.log")
        if actual != expected:
            raise RuntimeError(
                f"{label} TP-rank token parity failed for rank {rank}"
            )


def wait_result(root: subprocess.Popen[str]) -> dict[str, Any]:
    while True:
        line = root.stdout.readline() if root.stdout is not None else ""
        if not line:
            raise RuntimeError("persistent Qwen root exited before returning a result")
        parsed = parse_result(line.strip())
        if parsed is not None:
            return parsed


def command_for_rank(
    *,
    binary: Path,
    ckpt: Path,
    dspark: Path,
    rank: int,
    tp_world: int,
    id_path: Path,
    max_context: int,
    prefill_chunk_tokens: int,
    kv_cache_dtype: str,
    prefix_cache: bool,
) -> list[str]:
    command = [
        str(binary),
        "--ckpt", str(ckpt),
        "--tp-world", str(tp_world),
        "--tp-rank", str(rank),
        "--device", "0",
        "--nccl-id-path", str(id_path),
        "--max-context", str(max_context),
        "--prefill-chunk-tokens", str(prefill_chunk_tokens),
        "--kv-cache-dtype", kv_cache_dtype,
        "--smoke-layers", "0",
        "--qwen-dspark", str(dspark),
        "--qwen-persistent-stdin",
    ]
    if not prefix_cache:
        command.append("--qwen-no-prefix-cache")
    return command


def cleanup_nccl_paths(id_path: Path) -> None:
    id_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (id_path, Path(str(id_path) + ".qwen")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def launch_group(
    *,
    binary: Path,
    ckpt: Path,
    dspark: Path,
    tp_world: int,
    devices: list[int],
    id_path: Path,
    log_dir: Path,
    max_context: int,
    prefill_chunk_tokens: int,
    kv_cache_dtype: str,
    prefix_cache: bool,
) -> tuple[subprocess.Popen[str], list[subprocess.Popen[bytes]], list[BinaryIO]]:
    cleanup_nccl_paths(id_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        command_for_rank(
            binary=binary,
            ckpt=ckpt,
            dspark=dspark,
            rank=rank,
            tp_world=tp_world,
            id_path=id_path,
            max_context=max_context,
            prefill_chunk_tokens=prefill_chunk_tokens,
            kv_cache_dtype=kv_cache_dtype,
            prefix_cache=prefix_cache,
        )
        for rank in range(tp_world)
    ]
    logs: list[BinaryIO] = []
    workers: list[subprocess.Popen[bytes]] = []
    for rank in range(1, tp_world):
        log = (log_dir / f"rank{rank}.log").open("wb")
        logs.append(log)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
        workers.append(
            subprocess.Popen(commands[rank], stdout=log, stderr=subprocess.STDOUT, env=env)
        )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(devices[0])
    root = subprocess.Popen(
        commands[0],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return root, workers, logs


def stop_group(
    root: subprocess.Popen[str] | None,
    workers: list[subprocess.Popen[bytes]],
    logs: list[BinaryIO],
) -> None:
    if root is not None:
        if root.stdin is not None and not root.stdin.closed:
            root.stdin.close()
        if root.poll() is None:
            root.terminate()
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    if root is not None:
        try:
            root.wait(timeout=30)
        except subprocess.TimeoutExpired:
            root.kill()
            root.wait(timeout=30)
    for worker in workers:
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=30)
    for log in logs:
        log.close()


def request(root: subprocess.Popen[str], prompt: list[int], max_new_tokens: int) -> dict[str, Any]:
    if root.stdin is None:
        raise RuntimeError("persistent root stdin is unavailable")
    root.stdin.write(
        f"{max_new_tokens} " + " ".join(str(token) for token in prompt) + "\n"
    )
    root.stdin.flush()
    return wait_result(root)


def run_cold_case(
    *,
    args: argparse.Namespace,
    binary: Path,
    ckpt: Path,
    dspark: Path,
    devices: list[int],
    work_dir: Path,
    case_index: int,
    prompt: list[int],
) -> dict[str, Any]:
    case_dir = work_dir / "cold" / f"case_{case_index}"
    root: subprocess.Popen[str] | None = None
    workers: list[subprocess.Popen[bytes]] = []
    logs: list[BinaryIO] = []
    try:
        root, workers, logs = launch_group(
            binary=binary,
            ckpt=ckpt,
            dspark=dspark,
            tp_world=args.tp_world,
            devices=devices,
            id_path=case_dir / "nccl.id",
            log_dir=case_dir,
            max_context=args.max_context,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            prefix_cache=False,
        )
        result = request(root, prompt, args.max_new_tokens)
        if root.stdin is not None:
            root.stdin.close()
        root.wait(timeout=120)
        for worker in workers:
            worker.wait(timeout=120)
        if root.returncode != 0 or any(worker.returncode != 0 for worker in workers):
            raise RuntimeError(
                f"cold case failed root={root.returncode} "
                f"workers={[worker.returncode for worker in workers]}"
            )
        require_worker_parity(
            case_dir, args.tp_world,
            [[int(token) for token in result.get("tokens", [])]],
            f"cold case {case_index}",
        )
        return result
    finally:
        stop_group(root, workers, logs)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, list[int]]:
    binary = Path(args.binary).resolve()
    ckpt = Path(args.ckpt).resolve()
    dspark = Path(args.dspark).resolve()
    work_dir = Path(args.work_dir).resolve()
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not binary.is_file() or not ckpt.is_dir():
        raise SystemExit("binary or target checkpoint is missing")
    if not (dspark / "config.json").is_file() or not (dspark / "model.safetensors").is_file():
        raise SystemExit("DSpark checkpoint is incomplete")
    if len(devices) != args.tp_world or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain one distinct device per TP rank")
    if args.max_new_tokens < 8:
        raise SystemExit("--max-new-tokens must be at least 8 for a DSpark block")
    positive = [
        args.base_tokens,
        args.append_tokens,
        args.shorter_tokens,
        args.branch_prefix_tokens,
        args.branch_suffix_tokens,
        args.compression_prefix_tokens,
        args.compression_suffix_tokens,
        args.max_context,
    ]
    if any(value <= 0 for value in positive):
        raise SystemExit("prefix fixture sizes and max-context must be positive")
    if args.shorter_tokens >= args.base_tokens:
        raise SystemExit("--shorter-tokens must be smaller than --base-tokens")
    if args.branch_prefix_tokens >= args.shorter_tokens:
        raise SystemExit(
            "--branch-prefix-tokens must be smaller than --shorter-tokens"
        )
    if args.compression_prefix_tokens >= args.branch_prefix_tokens:
        raise SystemExit(
            "--compression-prefix-tokens must be smaller than --branch-prefix-tokens"
        )
    return binary, ckpt, dspark, work_dir, devices


def main() -> int:
    args = parse_args()
    binary, ckpt, dspark, work_dir, devices = validate_args(args)
    fixture_length = args.base_tokens + args.append_tokens
    base = make_token_fixture(
        ckpt, work_dir / f"tokens_{fixture_length}.txt", fixture_length,
        args.tokenizer_python,
    )
    original = base[:args.base_tokens]
    prompts = [
        ("initial", original),
        ("exact_repeat", list(original)),
        ("monotonic_append", list(base)),
        ("shorter_prefix", original[:args.shorter_tokens]),
        (
            "interior_branch",
            original[:args.branch_prefix_tokens]
            + [70000 + index for index in range(args.branch_suffix_tokens)],
        ),
        (
            "compression",
            original[:args.compression_prefix_tokens]
            + [90000 + index for index in range(args.compression_suffix_tokens)],
        ),
    ]
    if any(len(prompt) + args.max_new_tokens > args.max_context for _, prompt in prompts):
        raise SystemExit("a fixture prompt plus generation exceeds --max-context")

    work_dir.mkdir(parents=True, exist_ok=True)
    root: subprocess.Popen[str] | None = None
    workers: list[subprocess.Popen[bytes]] = []
    logs: list[BinaryIO] = []
    cached_results: list[dict[str, Any]] = []
    cached_worker_tokens: list[list[int]] = []
    started = time.monotonic()
    try:
        root, workers, logs = launch_group(
            binary=binary,
            ckpt=ckpt,
            dspark=dspark,
            tp_world=args.tp_world,
            devices=devices,
            id_path=work_dir / "cached" / "nccl.id",
            log_dir=work_dir / "cached",
            max_context=args.max_context,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            prefix_cache=True,
        )
        for _, prompt in prompts:
            cached_results.append(request(root, prompt, args.max_new_tokens))
        if root.stdin is not None:
            root.stdin.close()
        root.wait(timeout=120)
        for worker in workers:
            worker.wait(timeout=120)
        if root.returncode != 0 or any(worker.returncode != 0 for worker in workers):
            raise RuntimeError(
                f"cached group failed root={root.returncode} "
                f"workers={[worker.returncode for worker in workers]}"
            )
        cached_worker_tokens = [
            [int(token) for token in result.get("tokens", [])]
            for result in cached_results
        ]
        require_worker_parity(
            work_dir / "cached", args.tp_world, cached_worker_tokens, "cached"
        )
    finally:
        stop_group(root, workers, logs)

    results: list[dict[str, Any]] = []
    for index, ((name, prompt), cached) in enumerate(zip(prompts, cached_results)):
        cold = run_cold_case(
            args=args,
            binary=binary,
            ckpt=ckpt,
            dspark=dspark,
            devices=devices,
            work_dir=work_dir,
            case_index=index,
            prompt=prompt,
        )
        if cached.get("tokens") != cold.get("tokens"):
            raise RuntimeError(f"cached/cold DSpark token parity failed for {name}")
        expected_reuse = {
            "initial": 0,
            "exact_repeat": args.base_tokens,
            "monotonic_append": args.base_tokens + args.max_new_tokens - 1,
            "shorter_prefix": args.shorter_tokens // 256 * 256,
            "interior_branch": args.branch_prefix_tokens // 256 * 256,
            "compression": args.compression_prefix_tokens // 256 * 256,
        }[name]
        if int(cached.get("prefix_reused_tokens", -1)) != expected_reuse:
            raise RuntimeError(
                f"unexpected reused-token count for {name}: "
                f"{cached.get('prefix_reused_tokens')} != {expected_reuse}"
            )
        if int(cold.get("prefix_reused_tokens", -1)) != 0:
            raise RuntimeError(f"cold case unexpectedly reused prefix for {name}")
        if int(cold.get("prefix_computed_tokens", -1)) != len(prompt):
            raise RuntimeError(f"cold case did not recompute full prompt for {name}")
        if int(cached.get("dspark_confidence_count", 0)) <= 0:
            raise RuntimeError(f"missing DSpark confidence telemetry for {name}")
        row = {
            "name": name,
            "prompt_tokens": len(prompt),
            "cached": cached,
            "cold": cold,
            "token_parity": True,
        }
        results.append(row)
        print(
            f"case={name} prompt={len(prompt)} "
            f"reused={cached.get('prefix_reused_tokens')} "
            f"computed={cached.get('prefix_computed_tokens')} "
            f"source={cached.get('prefix_resume_source')} "
            f"cold_parity=PASS confidence_mean={cached.get('dspark_confidence_mean')}"
        )

    print("cached_cold_token_parity=PASS dspark_rank_token_parity=PASS")
    output = {
        "mode": "qwen_dspark_prefix_cache",
        "checkpoint": str(ckpt),
        "dspark_checkpoint": str(dspark),
        "kv_cache_dtype": args.kv_cache_dtype,
        "tp_world": args.tp_world,
        "max_new_tokens": args.max_new_tokens,
        "results": results,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    result_path = work_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"results={result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stderr or error.stdout or str(error), file=sys.stderr)
        raise
    except subprocess.TimeoutExpired as error:
        print(f"timeout: {error}", file=sys.stderr)
        raise SystemExit(1)
