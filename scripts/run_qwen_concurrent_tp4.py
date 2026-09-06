#!/usr/bin/env python3
"""TP4 launcher for bench_concurrent_throughput.py on the real model.

Each invocation runs ONE measurement as a four-rank TensorParallel process
group: rank 0 drives the throughput loop through the batch scheduler, ranks
1-3 enter the native worker command channel. Rank 0's LLM.close() broadcasts
the shutdown that unwinds the workers, and the launcher waits on all four PIDs.

Usage:
  run_qwen_concurrent_tp4.py --mode serial|batch --num-concurrent N \
      [--max-context C] [--max-tokens T] [--prompt-length P] \
      [--kv-paged] [--kv-block-size B] [--kv-cache-mb M] \
      [--warmup-runs W] [--test-runs R] [--checkpoint DIR] \
      [--json-out PATH]

Runs under the same TCP/NCCL contract as the C++ parity launcher: every rank
shares one NCCL id path, created by rank 0 and passed on the command line.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "bench_concurrent_throughput.py"
PYTHON = os.environ.get("BENCH_PYTHON", "/home/lvyufeng/miniconda3/bin/python")


def build_rank_cmd(args, rank: int, nccl_path: str) -> list[str]:
    cmd = [PYTHON, str(SCRIPT)]
    cmd += [args.checkpoint, "--single", args.mode]
    cmd += ["--num-concurrent", str(args.num_concurrent)]
    cmd += ["--max-model-len", str(args.max_context)]
    cmd += ["--max-tokens", str(args.max_tokens)]
    cmd += ["--prompt-length", str(args.prompt_length)]
    cmd += ["--warmup-runs", str(args.warmup_runs)]
    cmd += ["--test-runs", str(args.test_runs)]
    cmd += ["--tensor-parallel-size", "4"]
    # Each rank must drive the engine with the same KV layout as rank 0.
    cmd += ["--backend-option", f"nccl_id_path={nccl_path}"]
    if args.kv_paged:
        cmd += ["--backend-option", "kv_paged=True"]
        cmd += ["--backend-option", f"kv_block_size={args.kv_block_size}"]
    if args.kv_cache_mb:
        cmd += ["--backend-option", f"kv_cache_bytes={args.kv_cache_mb * 1024 * 1024}"]
    if args.no_prefix_cache:
        cmd += ["--backend-option", "enable_prefix_caching=False"]
    if args.json_out:
        cmd += ["--json-out", str(args.json_out) if rank == 0 else f"{args.json_out}.r{rank}"]
    return cmd


def parse_backend_options(cmd: list[str]) -> dict:
    """Extract the --backend-option pairs from a rank command into a dict."""
    out: dict[str, str] = {}
    it = iter(cmd)
    for tok in it:
        if tok == "--backend-option":
            key, _, value = next(it).partition("=")
            out[key] = value
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to Qwen3.8 checkpoint directory")
    parser.add_argument("--mode", choices=["serial", "batch"], required=True)
    parser.add_argument("--num-concurrent", type=int, required=True)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--test-runs", type=int, default=3)
    parser.add_argument("--kv-paged", action="store_true")
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-cache-mb", type=int, default=0,
                        help="Paged pool budget in MiB; 0 derives it from the "
                             "contiguous arena reservation")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable prefix caching so every run does a full "
                             "prefill; otherwise the identical warmup prompt is "
                             "reused and the test run only pays decode, which "
                             "makes the contiguous arm look 4x faster than paged")
    parser.add_argument("--json-out", help="Write rank 0's single measurement here")
    parser.add_argument("--log-dir", default="/tmp/qwen_concurrent_tp4")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    nccl = f"{log_dir}/nccl_{args.mode}_{args.num_concurrent}_{int(time.time())}.id"

    try:
        if not Path(PYTHON).exists():
            print(f"[launcher] {PYTHON} not found; set BENCH_PYTHON", file=sys.stderr)
            return 1

        procs: list[subprocess.Popen] = []
        for rank in range(4):
            cmd = build_rank_cmd(args, rank, nccl)
            log_path = log_dir / f"r{rank}.log"
            env = dict(os.environ)
            # The benchmark reads TP_RANK/TP_WORLD from the environment; ensure
            # each process sees the same TP world but its own rank. The NCCL id
            # path is handed to the engine by argument (--backend-option) and by
            # env; the first rank to mount creates the rendezvous file.
            # Pin each rank to one GPU; _native_rank_device() renumbers that to
            # device 0 for this process, so no rank-offset stacking happens.
            env["TP_RANK"] = str(rank)
            env["TP_WORLD"] = "4"
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            env["POCKETLLM_NCCL_ID_PATH"] = nccl
            log_handle = open(log_path, "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            procs.append(proc)

        status = 0
        for proc in procs:
            rc = proc.wait()
            if rc != 0:
                status = rc

        for rank in range(4):
            log_path = log_dir / f"r{rank}.log"
            suffix = f" (rank {rank})" if rank else ""
            print(f"\n===== log{suffix} =====")
            with open(log_path) as fh:
                print(fh.read(), end="")
    finally:
        try:
            Path(nccl).unlink(missing_ok=True)
        except OSError:
            pass

    return status


if __name__ == "__main__":
    sys.exit(main())
