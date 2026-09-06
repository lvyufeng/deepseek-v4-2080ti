#!/usr/bin/env python3
"""Run serial plain-versus-external-DSpark Qwen TP benchmarks.

Each prompt length runs plain first and DSpark second on the real target/draft
checkpoints. Every TP rank must return the same greedy tokens and DSpark must
match plain token-for-token. Cases are strictly serial to avoid GPU contention.
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
from typing import Any


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
    parser.add_argument("--lengths", default="512,8192,32768")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--work-dir", default=".tmp/qwen_dspark")
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


def parse_runtime(line: str) -> dict[str, Any] | None:
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


def parse_log(path: Path) -> tuple[dict[str, Any], list[int]]:
    runtime: dict[str, Any] | None = None
    tokens: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_runtime(line)
        if parsed is not None:
            runtime = parsed
        match = re.match(r"generate_step=(\d+) token=(-?\d+)", line)
        if match:
            tokens.append(int(match.group(2)))
    if runtime is None or not tokens:
        raise RuntimeError(f"incomplete Qwen log: {path}")
    return runtime, tokens


def run_case(
    *,
    binary: Path,
    ckpt: Path,
    dspark: Path,
    work_dir: Path,
    token_path: Path,
    length: int,
    max_new_tokens: int,
    tp_world: int,
    devices: list[int],
    prefill_chunk_tokens: int,
    kv_cache_dtype: str,
    enabled: bool,
) -> dict[str, Any]:
    mode = "dspark" if enabled else "plain"
    case_dir = work_dir / f"length_{length}" / mode
    case_dir.mkdir(parents=True, exist_ok=True)
    id_path = case_dir / "nccl.id"
    try:
        id_path.unlink()
    except FileNotFoundError:
        pass
    for path in case_dir.glob("rank*.log"):
        path.unlink()

    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    logs = []
    statuses: dict[int, int] = {}
    started = time.monotonic()
    try:
        for rank in range(tp_world):
            command = [
                str(binary),
                "--ckpt", str(ckpt),
                "--tp-world", str(tp_world),
                "--tp-rank", str(rank),
                "--device", "0",
                "--nccl-id-path", str(id_path),
                "--token-ids-file", str(token_path),
                "--generate-token", "123",
                "--max-new-tokens", str(max_new_tokens),
                "--max-context", str(length + max_new_tokens),
                "--prefill-chunk-tokens", str(prefill_chunk_tokens),
                "--kv-cache-dtype", kv_cache_dtype,
                "--smoke-layers", "0",
                "--resident-bench",
            ]
            if enabled:
                command += ["--qwen-dspark", str(dspark)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
            log = (case_dir / f"rank{rank}.log").open("wb")
            logs.append(log)
            processes.append(
                (rank, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
            )
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
        for log in logs:
            log.close()

    if any(statuses.get(rank, -1) != 0 for rank in range(tp_world)):
        raise RuntimeError(f"{mode} TP case failed: {statuses}")
    parsed = [parse_log(case_dir / f"rank{rank}.log") for rank in range(tp_world)]
    runtimes = [item[0] for item in parsed]
    rank_tokens = [item[1] for item in parsed]
    if any(rank_tokens[0] != tokens for tokens in rank_tokens[1:]):
        raise RuntimeError(f"{mode} TP-rank parity failed at length {length}")
    if runtimes[0].get("layers") != 64:
        raise RuntimeError(f"{mode} did not execute all target layers")
    memory = [runtime.get("gpu_memory_used_bytes") for runtime in runtimes]
    return {
        "mode": mode,
        "prompt_tokens": length,
        "tokens": rank_tokens[0],
        "runtime": runtimes[0],
        "rank_runtime": runtimes,
        "rank_token_parity": True,
        "elapsed_process_wall_seconds": time.monotonic() - started,
        "max_gpu_memory_used_bytes": max(value for value in memory if value is not None),
        "log_dir": str(case_dir),
    }


def main() -> int:
    args = parse_args()
    binary = Path(args.binary).resolve()
    ckpt = Path(args.ckpt).resolve()
    dspark = Path(args.dspark).resolve()
    work_dir = Path(args.work_dir).resolve()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not binary.is_file() or not ckpt.is_dir():
        raise SystemExit("binary or target checkpoint is missing")
    if not (dspark / "config.json").is_file() or not (dspark / "model.safetensors").is_file():
        raise SystemExit("DSpark checkpoint is incomplete")
    if not lengths or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain positive integers")
    if len(devices) != args.tp_world or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain one distinct device per TP rank")
    if args.max_new_tokens < 8:
        raise SystemExit("--max-new-tokens must be at least 8 for a DSpark block")

    work_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "mode": "qwen_dspark_ab",
        "checkpoint": str(ckpt),
        "dspark_checkpoint": str(dspark),
        "binary": str(binary),
        "tp_world": args.tp_world,
        "devices": devices,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "kv_cache_dtype": args.kv_cache_dtype,
        "results": [],
    }
    result_path = work_dir / "results.json"
    for length in lengths:
        token_path = work_dir / f"tokens_{length}.txt"
        make_token_fixture(ckpt, token_path, length, args.tokenizer_python)
        cases = []
        for enabled in (False, True):
            result = run_case(
                binary=binary,
                ckpt=ckpt,
                dspark=dspark,
                work_dir=work_dir,
                token_path=token_path,
                length=length,
                max_new_tokens=args.max_new_tokens,
                tp_world=args.tp_world,
                devices=devices,
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                kv_cache_dtype=args.kv_cache_dtype,
                enabled=enabled,
            )
            cases.append(result)
            runtime = result["runtime"]
            print(
                f"length={length} mode={result['mode']} wall={runtime.get('wall')} "
                f"decode_tps={runtime.get('decode_tokens_per_s')} "
                f"draft_match_rate={runtime.get('mtp_accept_rate')} "
                f"accept_length={runtime.get('spec_accept_length')} "
                f"confidence_mean={runtime.get('dspark_confidence_mean')} "
                f"rank_parity=PASS"
            )
        plain, speculative = cases
        if speculative["tokens"] != plain["tokens"]:
            raise RuntimeError(f"DSpark differs from plain greedy tokens at length {length}")
        speculative["plain_token_parity"] = True
        speculative["speedup_vs_plain_wall"] = (
            float(plain["runtime"]["wall"]) / float(speculative["runtime"]["wall"])
        )
        speculative["speedup_vs_plain_decode"] = (
            float(speculative["runtime"]["decode_tokens_per_s"])
            / float(plain["runtime"]["decode_tokens_per_s"])
        )
        output["results"].append({"prompt_tokens": length, "cases": cases})
        result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(
            f"length={length} DSpark parity=PASS "
            f"wall_speedup={speculative['speedup_vs_plain_wall']:.4f} "
            f"decode_speedup={speculative['speedup_vs_plain_decode']:.4f}"
        )
    print(f"results={result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stderr or error.stdout or str(error), file=sys.stderr)
        raise
