#!/usr/bin/env python3
"""Verify the PocketLLM C++ adapter against the reference pocketllm_engine.

Both sides load the same real checkpoint and drive the same native QwenEngine
with identical greedy settings, so the generated token ids must agree. The
adapter differs in exactly one documented way: it stops at the first EOS token
and excludes it, while the reference binary always runs the full
``--max-new-tokens`` budget. Parity therefore means the adapter's tokens are the
reference sequence truncated at the first EOS.

One process per TP rank on both sides, sharing an NCCL id file, matching how
scripts/bench_qwen_long_context.py launches the reference binary.

Usage:
  PYTHONPATH=. python scripts/verify_cpp_adapter_parity.py \
    --ckpt /mnt/data2/Qwen3.8-27B-FP8 \
    --tp-world 2 --devices 0,1 --max-new-tokens 24
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

REPO = Path(__file__).resolve().parent.parent
REFERENCE_STEP = re.compile(r"^generate_step=(\d+) token=(-?\d+)")

DEFAULT_PROMPTS = [
    "用一句话介绍你自己。",
    "What is the capital of France? Answer in one word.",
    "Name three prime numbers.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--binary", default="cpp_engine/build/pocketllm_engine")
    parser.add_argument("--native-path", default="cpp_engine/build-python/python")
    parser.add_argument("--tp-world", type=int, default=2)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp16")
    parser.add_argument("--work-dir", default="parity_work")
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def launch_ranks(command_for_rank, tp_world, devices, case_dir, timeout, tag):
    """Run one process per rank and wait for all of them to exit."""
    case_dir.mkdir(parents=True, exist_ok=True)
    id_path = case_dir / f"{tag}.nccl.id"
    if id_path.exists():
        id_path.unlink()
    logs: list[Path] = []
    processes: list[tuple[int, subprocess.Popen]] = []
    started = time.monotonic()
    try:
        for rank in range(tp_world):
            log_path = case_dir / f"{tag}_rank{rank}.log"
            logs.append(log_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(devices[rank])
            env["PYTHONPATH"] = os.pathsep.join(
                [str(REPO), str(REPO / "cpp_engine/build-python/python"), env.get("PYTHONPATH", "")]
            )
            with log_path.open("wb") as handle:
                processes.append((
                    rank,
                    subprocess.Popen(
                        command_for_rank(rank, id_path),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        cwd=str(REPO),
                    ),
                ))
        statuses: dict[int, int] = {}
        while len(statuses) < len(processes):
            for rank, process in processes:
                if rank not in statuses:
                    code = process.poll()
                    if code is not None:
                        statuses[rank] = code
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"{tag}: ranks exceeded {timeout}s; see {case_dir}")
            time.sleep(0.2)
        failed = {rank: code for rank, code in statuses.items() if code != 0}
        if failed:
            tail = logs[0].read_text(errors="replace").splitlines()[-15:]
            raise RuntimeError(f"{tag}: ranks failed {failed}\n" + "\n".join(tail))
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.kill()
    return logs


def reference_tokens(args, token_path: Path, case_dir: Path, devices) -> list[int]:
    def command_for_rank(rank: int, id_path: Path) -> list[str]:
        return [
            str((REPO / args.binary).resolve()),
            "--ckpt", args.ckpt,
            "--tp-world", str(args.tp_world),
            "--tp-rank", str(rank),
            "--device", "0",
            "--nccl-id-path", str(id_path),
            "--token-ids-file", str(token_path),
            "--generate-token", "1",
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-context", str(args.max_context),
            "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--qwen-temperature", "0",
            "--qwen-top-p", "1",
            "--qwen-top-k", "0",
            "--qwen-seed", "0",
            # The adapter builds QwenEngine with layer_count=0 (every layer);
            # the binary's default is 1, which would compare different models.
            "--smoke-layers", "0",
        ]

    logs = launch_ranks(command_for_rank, args.tp_world, devices, case_dir, args.timeout, "reference")
    steps: dict[int, int] = {}
    for line in logs[0].read_text(errors="replace").splitlines():
        match = REFERENCE_STEP.match(line.strip())
        if match:
            steps[int(match.group(1))] = int(match.group(2))
    if not steps:
        raise RuntimeError(f"reference produced no generate_step lines; see {logs[0]}")
    return [steps[key] for key in sorted(steps)]


def adapter_run(args, token_path: Path, case_dir: Path, devices, mode: str) -> list[dict]:
    runner = REPO / "scripts/_parity_adapter_rank.py"
    out_paths = [case_dir / f"adapter_{mode}_rank{rank}.json" for rank in range(args.tp_world)]
    for path in out_paths:
        if path.exists():
            path.unlink()

    def command_for_rank(rank: int, id_path: Path) -> list[str]:
        command = [
            sys.executable, str(runner),
            "--ckpt", args.ckpt,
            "--tp-world", str(args.tp_world),
            "--tp-rank", str(rank),
            "--nccl-id-path", str(id_path),
            "--token-ids-file", str(token_path),
            "--max-new-tokens", str(args.max_new_tokens),
            "--max-context", str(args.max_context),
            "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
            "--kv-cache-dtype", args.kv_cache_dtype,
            "--out", str(out_paths[rank]),
        ]
        if mode == "stream":
            command.append("--stream")
        return command

    launch_ranks(command_for_rank, args.tp_world, devices, case_dir, args.timeout, f"adapter_{mode}")
    payloads = []
    for path in out_paths:
        if not path.exists():
            raise RuntimeError(f"adapter rank produced no output: {path}")
        payloads.append(json.loads(path.read_text()))
    return payloads


def truncate_at_eos(tokens: list[int], eos_ids: set[int]) -> tuple[list[int], bool]:
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index], True
    return list(tokens), False


def main() -> int:
    args = parse_args()
    devices = [int(item) for item in args.devices.split(",")]
    if len(devices) != args.tp_world:
        raise SystemExit("--devices count must match --tp-world")
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    cases = []
    failures = 0
    for index, prompt in enumerate(prompts):
        case_dir = work_dir / f"case{index}"
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_ids = [int(token) for token in tokenizer.encode(prompt)]
        token_path = case_dir / "tokens.txt"
        # load_token_ids() in main.cpp splits on commas only.
        token_path.write_text(",".join(str(token) for token in prompt_ids), encoding="utf-8")

        print(f"[case {index}] prompt={prompt!r} prompt_tokens={len(prompt_ids)}", flush=True)
        reference = reference_tokens(args, token_path, case_dir, devices)
        offline = adapter_run(args, token_path, case_dir, devices, "offline")
        streamed = adapter_run(args, token_path, case_dir, devices, "stream")

        eos_ids = set(offline[0]["eos_token_ids"])
        expected, hit_eos = truncate_at_eos(reference, eos_ids)
        expected_reason = "stop" if hit_eos else "length"

        rank_offline_agree = all(item["token_ids"] == offline[0]["token_ids"] for item in offline)
        rank_stream_agree = all(item["token_ids"] == streamed[0]["token_ids"] for item in streamed)
        offline_ok = offline[0]["token_ids"] == expected
        stream_ok = streamed[0]["token_ids"] == expected
        reason_ok = (
            offline[0]["finish_reason"] == expected_reason
            and streamed[0]["finish_reason"] == expected_reason
        )
        usage_ok = offline[0]["usage"] == streamed[0]["usage"]
        passed = all([rank_offline_agree, rank_stream_agree, offline_ok, stream_ok, reason_ok, usage_ok])
        failures += 0 if passed else 1

        case = {
            "prompt": prompt,
            "prompt_tokens": len(prompt_ids),
            "eos_token_ids": sorted(eos_ids),
            "eos_source": offline[0]["eos_source"],
            "reference_tokens": reference,
            "expected_tokens": expected,
            "reference_hit_eos": hit_eos,
            "offline_tokens": offline[0]["token_ids"],
            "stream_tokens": streamed[0]["token_ids"],
            "offline_finish_reason": offline[0]["finish_reason"],
            "stream_finish_reason": streamed[0]["finish_reason"],
            "expected_finish_reason": expected_reason,
            "offline_usage": offline[0]["usage"],
            "stream_usage": streamed[0]["usage"],
            "text": offline[0].get("text"),
            "checks": {
                "offline_matches_reference": offline_ok,
                "stream_matches_reference": stream_ok,
                "offline_ranks_agree": rank_offline_agree,
                "stream_ranks_agree": rank_stream_agree,
                "finish_reason_correct": reason_ok,
                "usage_consistent": usage_ok,
            },
            "pass": passed,
        }
        cases.append(case)
        print(f"  eos={sorted(eos_ids)} source={case['eos_source']}", flush=True)
        print(f"  reference={reference}", flush=True)
        print(f"  expected ={expected} reason={expected_reason}", flush=True)
        print(f"  offline  ={offline[0]['token_ids']} reason={offline[0]['finish_reason']} usage={offline[0]['usage']}", flush=True)
        print(f"  stream   ={streamed[0]['token_ids']} reason={streamed[0]['finish_reason']} usage={streamed[0]['usage']}", flush=True)
        print(f"  {'PASS' if passed else 'FAIL'} {case['checks']}", flush=True)

    summary = {
        "checkpoint": args.ckpt,
        "tp_world": args.tp_world,
        "devices": devices,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_new_tokens": args.max_new_tokens,
        "cases": cases,
        "failures": failures,
        "pass": failures == 0,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} CASE(S) FAILED'} ({len(cases)} cases)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
