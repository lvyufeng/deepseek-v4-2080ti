#!/usr/bin/env python3
"""Run serial plain-versus-DFlash2 Qwen TP benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dflash2", required=True)
    parser.add_argument("--binary", default="cpp_engine/build/pocketllm_engine")
    parser.add_argument("--lengths", default="512,4096,8192")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--work-dir", default=".tmp/qwen_dflash2")
    parser.add_argument("--tokenizer-python", default="/home/lvyufeng/miniconda3/envs/deepseek/bin/python")
    parser.add_argument("--fused-swiglu", action="store_true")
    parser.add_argument("--grouped-attention", action="store_true")
    parser.add_argument(
        "--snapshot-interval", type=int, default=4096,
        help="recurrent snapshot interval; 0 disables snapshots for fresh A/B")
    return parser.parse_args()


def fixture(ckpt: Path, path: Path, length: int, tokenizer_python: str) -> list[int]:
    if path.exists():
        values = [int(item) for item in path.read_text(encoding="ascii").split(",") if item]
        if len(values) >= length:
            return values[:length]
    code = """
from transformers import AutoTokenizer
import json, sys
ckpt, target = sys.argv[1], int(sys.argv[2])
text = ("Speculative decoding proposes future language-model tokens and verifies them "
        "with one target forward. Exact greedy parity requires rollback of rejected "
        "state and commit of only the accepted prefix. This real-text fixture tests "
        "DFlash2 target taps, block attention, selector paths, and replay. ") * 5000
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
ids = tok.encode(text, add_special_tokens=False)
if len(ids) < target: raise RuntimeError(f"fixture produced {len(ids)} tokens, need {target}")
print(json.dumps(ids[:target]))
"""
    result = subprocess.run([tokenizer_python, "-c", code, str(ckpt), str(length)], check=True,
                            capture_output=True, text=True)
    values = json.loads(result.stdout)
    if not isinstance(values, list) or len(values) != length:
        raise RuntimeError(f"invalid fixture length {length}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(int(value)) for value in values), encoding="ascii")
    return [int(value) for value in values]


def parse_log(path: Path) -> tuple[dict[str, Any], list[int]]:
    runtime: dict[str, Any] | None = None
    tokens: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("qwen_runtime=1 "):
            runtime = {}
            for key, value in re.findall(r"(\w+)=([^\s]+)", line):
                try: runtime[key] = int(value)
                except ValueError:
                    try: runtime[key] = float(value)
                    except ValueError: runtime[key] = value
        match = re.match(r"generate_step=(\d+) token=(-?\d+)", line)
        if match: tokens.append(int(match.group(2)))
    if runtime is None or not tokens:
        raise RuntimeError(f"incomplete runtime log {path}")
    return runtime, tokens


def run_case(args: argparse.Namespace, ckpt: Path, token_path: Path, length: int,
             mode: str, work_dir: Path) -> dict[str, Any]:
    case_dir = work_dir / f"length_{length}" / mode
    case_dir.mkdir(parents=True, exist_ok=True)
    id_path = case_dir / "nccl.id"
    if id_path.exists(): id_path.unlink()
    for old in case_dir.glob("rank*.log"): old.unlink()
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    logs = []
    statuses: dict[int, int] = {}
    started = time.monotonic()
    try:
        for rank in range(args.tp_world):
            command = [str(Path(args.binary).resolve()), "--ckpt", str(ckpt),
                       "--tp-world", str(args.tp_world), "--tp-rank", str(rank),
                       "--device", "0", "--nccl-id-path", str(id_path),
                       "--token-ids-file", str(token_path), "--generate-token", "123",
                       "--max-new-tokens", str(args.max_new_tokens), "--max-context",
                       str(length + args.max_new_tokens), "--prefill-chunk-tokens",
                       str(args.prefill_chunk_tokens), "--kv-cache-dtype", args.kv_cache_dtype,
                       "--smoke-layers", "0", "--qwen-snapshot-interval",
                       str(args.snapshot_interval), "--resident-bench"]
            if mode == "dflash2": command += ["--qwen-dflash2", str(args.dflash2)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.devices.split(",")[rank]
            if mode == "dflash2" and args.fused_swiglu:
                env["POCKETLLM_DFLASH2_FUSED_SWIGLU"] = "1"
            if mode == "dflash2" and args.grouped_attention:
                env["POCKETLLM_DFLASH2_GROUPED_ATTN"] = "1"
            log = (case_dir / f"rank{rank}.log").open("wb")
            logs.append(log)
            processes.append((rank, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)))
        while len(statuses) < len(processes):
            for rank, process in processes:
                if rank in statuses: continue
                status = process.poll()
                if status is None: continue
                statuses[rank] = status
                if status != 0:
                    for other_rank, other in processes:
                        if other_rank not in statuses and other.poll() is None: other.terminate()
                    break
            if any(status != 0 for status in statuses.values()): break
            time.sleep(0.1)
    finally:
        for rank, process in processes:
            if process.poll() is None: process.terminate()
            try: statuses[rank] = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); statuses[rank] = process.wait(timeout=30)
        for log in logs: log.close()
    if any(statuses.get(rank, -1) != 0 for rank in range(args.tp_world)):
        raise RuntimeError(f"{mode} TP case failed: {statuses}")
    parsed = [parse_log(case_dir / f"rank{rank}.log") for rank in range(args.tp_world)]
    rank_tokens = [item[1] for item in parsed]
    if any(rank_tokens[0] != values for values in rank_tokens[1:]):
        raise RuntimeError(f"{mode} rank token parity failed at {length}")
    result = {"mode": mode, "prompt_tokens": length, "tokens": rank_tokens[0],
              "runtime": parsed[0][0], "rank_runtime": [item[0] for item in parsed],
              "rank_token_parity": True, "elapsed_process_wall_seconds": time.monotonic() - started,
              "log_dir": str(case_dir),
              "per_rank_gpu_memory": [
                  {"used_bytes": value.get("gpu_memory_used_bytes"),
                   "total_bytes": value.get("gpu_memory_total_bytes")}
                  for value in (item[0] for item in parsed)
              ]}
    if mode == "dflash2":
        runtime = result["runtime"]
        result["verification"] = {
            "verify_count": runtime.get("mtp_verify_count"),
            "proposed_drafts": runtime.get("mtp_proposed_drafts"),
            "correct_drafts": runtime.get("mtp_correct_drafts"),
            "accept_rate": runtime.get("mtp_accept_rate"),
            "accept_length": runtime.get("spec_accept_length"),
            "draft_seconds": runtime.get("mtp_draft_seconds"),
            "verify_seconds": runtime.get("mtp_verify_seconds"),
            "replay_seconds": runtime.get("mtp_replay_seconds"),
            "replay_tokens": runtime.get("mtp_replay_tokens"),
        }
        # The same memory vector is already attached to every mode; retain it
        # here as part of the DFlash2 verification record for report consumers.
        result["verification"]["per_rank_gpu_memory"] = result["per_rank_gpu_memory"]
    runtime = result["runtime"]
    print(f"length={length} mode={mode} wall={runtime.get('wall')} prefill={runtime.get('prefill_seconds')} "
          f"decode_tps={runtime.get('decode_tokens_per_s')} accept={runtime.get('mtp_accept_rate')} "
          f"accept_length={runtime.get('spec_accept_length')} token_parity=PASS")
    return result


def main() -> int:
    args = parse_args()
    if args.tp_world != len(args.devices.split(",")):
        raise SystemExit("--devices must contain one device per TP rank")
    ckpt, draft, work_dir = Path(args.ckpt).resolve(), Path(args.dflash2).resolve(), Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {"mode": "qwen_dflash2_ab", "results": []}
    for length in [int(item) for item in args.lengths.split(",") if item.strip()]:
        token_path = work_dir / f"tokens_{length}.txt"
        fixture(ckpt, token_path, length, args.tokenizer_python)
        plain = run_case(args, ckpt, token_path, length, "plain", work_dir)
        dflash = run_case(args, ckpt, token_path, length, "dflash2", work_dir)
        cross_mode_parity = plain["tokens"] == dflash["tokens"]
        plain["cross_mode_token_parity"] = cross_mode_parity
        dflash["cross_mode_token_parity"] = cross_mode_parity
        if not cross_mode_parity:
            first_difference = next(
                (index for index, (left, right) in enumerate(
                    zip(plain["tokens"], dflash["tokens"])) if left != right),
                min(len(plain["tokens"]), len(dflash["tokens"])),
            )
            raise RuntimeError(
                f"plain/DFlash2 token parity failed at {length}, "
                f"first_difference={first_difference}")
        plain_runtime, dflash_runtime = plain["runtime"], dflash["runtime"]
        print(f"length={length} cross_mode_token_parity=PASS")
        dflash["full_request_speedup"] = plain_runtime["wall"] / dflash_runtime["wall"]
        dflash["decode_speedup"] = plain_runtime["decode_seconds"] / dflash_runtime["decode_seconds"]
        output["results"].append({"plain": plain, "dflash2": dflash})
        print(f"length={length} full_request_speedup={dflash['full_request_speedup']:.4f} "
              f"decode_speedup={dflash['decode_speedup']:.4f}")
    (work_dir / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
