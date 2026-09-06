#!/usr/bin/env python3
"""Run serial plain-versus-native-MTP Qwen TP benchmarks.

Each case launches every TP rank, checks rank-local greedy parity, then compares
MTP K=1/2/4 with the plain sequence for the same real-token prompt. Cases run
serially so resident-weight and attention timings do not contend across modes.
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
    "Multi-token prediction proposes several future tokens and verifies them with "
    "one target-model forward pass. End-to-end speedup depends on acceptance, "
    "batched verification cost, recurrent-state rollback, and context length. "
    "This benchmark uses deterministic natural-language tokens and checks exact "
    "greedy parity between plain decoding and native Qwen MTP. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--binary", default="build/cpp_engine/pocketllm_engine")
    parser.add_argument("--lengths", default="512,32768")
    parser.add_argument("--mtp-tokens", default="1,2,4")
    parser.add_argument(
        "--adaptive", action="store_true",
        help="enable acceptance-adaptive K for every MTP case",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--work-dir", default=".tmp/qwen_mtp")
    parser.add_argument(
        "--tokenizer-python",
        default="/home/lvyufeng/miniconda3/envs/deepseek/bin/python",
        help="interpreter with transformers for real-token fixtures",
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
    work_dir: Path,
    token_path: Path,
    length: int,
    max_new_tokens: int,
    layers: int,
    tp_world: int,
    devices: list[int],
    prefill_chunk_tokens: int,
    kv_cache_dtype: str,
    mtp_tokens: int | None,
    adaptive: bool,
) -> dict[str, Any]:
    mode = "plain" if mtp_tokens is None else (
        f"mtp_adaptive_k{mtp_tokens}" if adaptive else f"mtp_k{mtp_tokens}"
    )
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
    log_handles = []
    started = time.monotonic()
    statuses: dict[int, int] = {}
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
                "--smoke-layers", str(layers),
                "--resident-bench",
            ]
            if mtp_tokens is not None:
                command += ["--qwen-mtp-tokens", str(mtp_tokens)]
                if adaptive:
                    command.append("--qwen-mtp-adaptive")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
            log = (case_dir / f"rank{rank}.log").open("wb")
            log_handles.append(log)
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
        for log in log_handles:
            log.close()

    if any(statuses.get(rank, -1) != 0 for rank in range(tp_world)):
        raise RuntimeError(f"{mode} TP case failed: {statuses}")
    parsed = [parse_log(case_dir / f"rank{rank}.log") for rank in range(tp_world)]
    rank_runtime = [item[0] for item in parsed]
    rank_tokens = [item[1] for item in parsed]
    if any(rank_tokens[0] != tokens for tokens in rank_tokens[1:]):
        raise RuntimeError(f"{mode} TP-rank token parity failed at length {length}")
    runtime = rank_runtime[0]
    if layers == 0 and runtime.get("layers") != 64:
        raise RuntimeError(f"{mode} did not run all 64 layers: {runtime}")
    memory = [item.get("gpu_memory_used_bytes") for item in rank_runtime]
    return {
        "mode": mode,
        "prompt_tokens": length,
        "tokens": rank_tokens[0],
        "runtime": runtime,
        "rank_runtime": rank_runtime,
        "rank_token_parity": True,
        "elapsed_process_wall_seconds": time.monotonic() - started,
        "max_gpu_memory_used_bytes": max(value for value in memory if value is not None),
        "log_dir": str(case_dir),
    }


def main() -> int:
    args = parse_args()
    binary = Path(args.binary).resolve()
    ckpt = Path(args.ckpt).resolve()
    work_dir = Path(args.work_dir).resolve()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]
    mtp_values = [int(item) for item in args.mtp_tokens.split(",") if item.strip()]
    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not binary.is_file() or not ckpt.is_dir():
        raise SystemExit("binary or checkpoint is missing")
    if not lengths or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain positive integers")
    if not mtp_values or any(value <= 0 for value in mtp_values):
        raise SystemExit("--mtp-tokens must contain positive integers")
    if len(devices) != args.tp_world or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain one distinct device per TP rank")
    if args.layers != 0:
        raise SystemExit("native MTP validation requires --layers 0 (complete target)")
    if args.max_new_tokens < 2:
        raise SystemExit("--max-new-tokens must be at least 2")

    work_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "mode": "qwen_native_mtp_ab",
        "checkpoint": str(ckpt),
        "binary": str(binary),
        "tp_world": args.tp_world,
        "devices": devices,
        "layers": args.layers,
        "max_new_tokens": args.max_new_tokens,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "kv_cache_dtype": args.kv_cache_dtype,
        "results": [],
    }
    result_path = work_dir / "results.json"
    for length in lengths:
        token_path = work_dir / f"tokens_{length}.txt"
        make_token_fixture(ckpt, token_path, length, args.tokenizer_python)
        plain = run_case(
            binary=binary,
            ckpt=ckpt,
            work_dir=work_dir,
            token_path=token_path,
            length=length,
            max_new_tokens=args.max_new_tokens,
            layers=args.layers,
            tp_world=args.tp_world,
            devices=devices,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            mtp_tokens=None,
            adaptive=False,
        )
        length_results = [plain]
        print(
            f"length={length} mode=plain wall={plain['runtime'].get('wall')} "
            f"decode_tps={plain['runtime'].get('decode_tokens_per_s')} rank_parity=PASS"
        )
        for mtp_tokens in mtp_values:
            result = run_case(
                binary=binary,
                ckpt=ckpt,
                work_dir=work_dir,
                token_path=token_path,
                length=length,
                max_new_tokens=args.max_new_tokens,
                layers=args.layers,
                tp_world=args.tp_world,
                devices=devices,
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                kv_cache_dtype=args.kv_cache_dtype,
                mtp_tokens=mtp_tokens,
                adaptive=args.adaptive,
            )
            if result["tokens"] != plain["tokens"]:
                raise RuntimeError(
                    f"MTP K={mtp_tokens} differs from plain greedy tokens at length {length}"
                )
            plain_wall = float(plain["runtime"]["wall"])
            mtp_wall = float(result["runtime"]["wall"])
            result["plain_token_parity"] = True
            result["speedup_vs_plain_wall"] = plain_wall / mtp_wall
            plain_decode_tps = float(plain["runtime"]["decode_tokens_per_s"])
            mtp_decode_tps = float(result["runtime"]["decode_tokens_per_s"])
            result["speedup_vs_plain_decode"] = mtp_decode_tps / plain_decode_tps
            length_results.append(result)
            runtime = result["runtime"]
            print(
                f"length={length} mode={result['mode']} "
                f"wall={mtp_wall} wall_speedup={result['speedup_vs_plain_wall']:.4f} "
                f"decode_tps={mtp_decode_tps:.4f} "
                f"decode_speedup={result['speedup_vs_plain_decode']:.4f} "
                f"accept_rate={runtime.get('mtp_accept_rate')} "
                f"draft_s={runtime.get('mtp_draft_seconds')} "
                f"verify_s={runtime.get('mtp_verify_seconds')} "
                f"replay_s={runtime.get('mtp_replay_seconds')} parity=PASS"
            )
        output["results"].append({"prompt_tokens": length, "cases": length_results})
        result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"results={result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or exc.stdout or str(exc), file=sys.stderr)
        raise
