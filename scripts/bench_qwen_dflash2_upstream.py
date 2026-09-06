#!/usr/bin/env python3
"""Benchmark DFlash2 with the upstream single-request workload shape.

This intentionally remains separate from bench_qwen_dflash2.py: upstream's
published 2.67--3.43x figures are aggregate completion tok/s over 128 dataset
requests at concurrency one, not a fixed 128-token decode microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DATASET_FIXTURES = {
    "gsm8k": "openai/gsm8k",
    "math500": "HuggingFaceH4/MATH-500",
    "humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
    "mt-bench": "HuggingFaceH4/mt_bench_prompts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dflash2", required=True)
    parser.add_argument("--binary", default="cpp_engine/build/pocketllm_engine")
    parser.add_argument("--dataset", choices=sorted(DATASET_FIXTURES), default="gsm8k")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--work-dir", default=".tmp/qwen_dflash2_upstream")
    parser.add_argument("--tokenizer-python", default="/home/lvyufeng/miniconda3/envs/deepseek/bin/python")
    parser.add_argument("--max-context", type=int, default=32768)
    return parser.parse_args()


def build_fixture(args: argparse.Namespace, path: Path) -> list[list[int]]:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved, list) and len(saved) >= args.num_prompts:
            return [[int(x) for x in row] for row in saved[:args.num_prompts]]
        print(f"fixture cache at {path} holds {len(saved)} prompts but "
              f"{args.num_prompts} were requested; rebuilding", flush=True)
    chat_code = r'''
from transformers import AutoTokenizer
import json, sys
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print(json.dumps(tok.apply_chat_template(
    [{"role": "user", "content": sys.argv[2]}],
    tokenize=True, add_generation_prompt=True, reasoning_effort="xhigh")))
'''
    def chat_prompt(content: str) -> list[int]:
        result = subprocess.run(
            [args.tokenizer_python, "-c", chat_code, args.ckpt, content],
            capture_output=True, text=True, check=True,
        )
        return [int(x) for x in json.loads(result.stdout)]

    def build_chat_rows(raw: list[str]) -> list[list[int]]:
        return [chat_prompt(item) for item in raw]

    def save_rows(rows: list[list[int]]) -> list[list[int]]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
        return rows

    def load_raw(raw_path: Path) -> list[str] | None:
        if not raw_path.exists():
            return None
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        # A cache holding fewer prompts than requested must be refetched rather
        # than silently truncating the run to the cached count.
        if len(raw) < args.num_prompts:
            return None
        return [str(item) for item in raw[:args.num_prompts]]

    raw_path = path.with_suffix(".raw.json")
    raw = load_raw(raw_path)
    if raw is not None and args.dataset == "gsm8k":
        return save_rows(build_chat_rows(raw))
    if args.dataset == "gsm8k":
        raw_result = subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error",
             "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"],
            capture_output=True, text=True, check=True,
        )
        raw = [json.loads(line)["question"] +
               "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
               for line in raw_result.stdout.splitlines()[:args.num_prompts]]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        return save_rows(build_chat_rows(raw))
    raise RuntimeError("automatic fixture download currently supports gsm8k; cache another dataset manually")

    # Kept below as a compatibility fallback for future locally cached datasets.
    code = r'''
import json, sys
from transformers import AutoTokenizer
from urllib.request import urlopen
name, ckpt, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
urls = {"gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"}
if name not in urls:
    raise RuntimeError("unsupported dataset")
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
rows = []
for line in urlopen(urls[name], timeout=60):
    row = json.loads(line)
    text = f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    rows.append(tok.encode(text, add_special_tokens=False))
    if len(rows) >= count: break
print(json.dumps(rows))
'''
    result = subprocess.run(
        [args.tokenizer_python, "-c", code, args.dataset, args.ckpt,
         str(args.num_prompts)], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build upstream dataset fixture; install the datasets "
            f"package or provide a cached fixture at {path}: {result.stderr.strip()}"
        )
    return save_rows([[int(x) for x in row] for row in json.loads(result.stdout)])
    if args.dataset == "gsm8k" and path.with_suffix(".raw.json").exists():
        raw = json.loads(path.with_suffix(".raw.json").read_text(encoding="utf-8"))
        rows = [chat_prompt(str(item)) for item in raw[:args.num_prompts]]
        path.write_text(json.dumps(rows), encoding="utf-8")
        return rows
    if args.dataset == "gsm8k":
        raw_path = path.with_suffix(".raw.json")
        raw_result = subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error",
             "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"],
            capture_output=True, text=True, check=True,
        )
        raw = [json.loads(line)["question"] +
               "\\nPlease reason step by step, and put your final answer within \\boxed{{}}."
               for line in raw_result.stdout.splitlines()[:args.num_prompts]]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        rows = [chat_prompt(str(item)) for item in raw]
        path.write_text(json.dumps(rows), encoding="utf-8")
        return rows
    code = r'''
import json, sys
from transformers import AutoTokenizer
from urllib.request import urlopen
name, ckpt, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
urls = {
 "gsm8k": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}
if name not in urls:
    raise RuntimeError("automatic fixture download currently supports gsm8k; cache another dataset manually")
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
rows = []
for line in urlopen(urls[name], timeout=60):
    row = json.loads(line)
    text = f"{row['question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    ids = tok.encode(text, add_special_tokens=False)
    if ids:
        rows.append(ids)
    if len(rows) >= count:
        break
print(json.dumps(rows))
'''
    result = subprocess.run(
        [args.tokenizer_python, "-c", code, args.dataset, args.ckpt,
         str(args.num_prompts)], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build upstream dataset fixture; install the datasets "
            f"package in {args.tokenizer_python} or provide a cached fixture "
            f"at {path}: {result.stderr.strip()}"
        )
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or len(rows) != args.num_prompts:
        raise RuntimeError(f"fixture produced {len(rows)} rows, need {args.num_prompts}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return [[int(x) for x in row] for row in rows]


def parse_log(path: Path) -> tuple[dict[str, Any], list[int]]:
    runtime: dict[str, Any] | None = None
    tokens: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("qwen_runtime=1 "):
            runtime = {}
            for key, value in re.findall(r"(\w+)=([^\s]+)", line):
                try:
                    runtime[key] = int(value)
                except ValueError:
                    try:
                        runtime[key] = float(value)
                    except ValueError:
                        runtime[key] = value
        match = re.match(r"generate_step=(\d+) token=(-?\d+)", line)
        if match:
            tokens.append(int(match.group(2)))
    if runtime is None:
        raise RuntimeError(f"missing runtime telemetry in {path}")
    return runtime, tokens


def run_request(args: argparse.Namespace, ckpt: Path, prompt: list[int], mode: str,
                index: int, work_dir: Path) -> dict[str, Any]:
    case_dir = work_dir / mode / f"request_{index:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    token_path = case_dir / "prompt.txt"
    token_path.write_text(",".join(str(x) for x in prompt), encoding="ascii")
    id_path = case_dir / "nccl.id"
    if id_path.exists():
        id_path.unlink()
    processes = []
    logs = []
    started = time.monotonic()
    try:
        for rank in range(args.tp_world):
            command = [str(Path(args.binary).resolve()), "--ckpt", str(ckpt),
                       "--tp-world", str(args.tp_world), "--tp-rank", str(rank),
                       "--device", "0", "--nccl-id-path", str(id_path),
                       "--token-ids-file", str(token_path), "--generate-token", "123",
                       "--max-new-tokens", str(args.max_new_tokens), "--max-context",
                       str(max(args.max_context, len(prompt) + args.max_new_tokens)),
                       "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
                       "--kv-cache-dtype", args.kv_cache_dtype, "--smoke-layers", "0",
                       "--resident-bench"]
            if mode == "dflash2":
                command += ["--qwen-dflash2", str(args.dflash2)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.devices.split(",")[rank]
            log = (case_dir / f"rank{rank}.log").open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
        statuses = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"{mode} request {index} failed: {statuses}")
    parsed = [parse_log(case_dir / f"rank{rank}.log") for rank in range(args.tp_world)]
    if any(parsed[0][1] != item[1] for item in parsed[1:]):
        raise RuntimeError(f"{mode} rank parity failed at request {index}")
    runtime = parsed[0][0]
    return {"mode": mode, "index": index, "prompt_tokens": len(prompt),
            "runtime": runtime, "tokens": parsed[0][1],
            "elapsed_process_wall_seconds": time.monotonic() - started,
            "rank_runtime": [item[0] for item in parsed]}


def main() -> int:
    args = parse_args()
    if args.tp_world != len(args.devices.split(",")):
        raise SystemExit("--devices must contain one device per TP rank")
    root = Path(args.work_dir).resolve()
    prompts = build_fixture(args, root / f"{args.dataset}_prompts.json")
    output = {"dataset": args.dataset, "num_prompts": len(prompts), "results": []}
    for index, prompt in enumerate(prompts):
        plain = run_request(args, Path(args.ckpt), prompt, "plain", index, root)
        dflash = run_request(args, Path(args.ckpt), prompt, "dflash2", index, root)
        if plain["tokens"] != dflash["tokens"]:
            raise RuntimeError(f"cross-mode parity failed at request {index}")
        plain_wall = plain["runtime"]["wall"]
        dflash_wall = dflash["runtime"]["wall"]
        output["results"].append({"plain": plain, "dflash2": dflash,
                                  "speedup": plain_wall / dflash_wall})
        print(f"request={index + 1}/{len(prompts)} prompt_tokens={len(prompt)} "
              f"plain={plain_wall:.4f}s dflash2={dflash_wall:.4f}s "
              f"speedup={plain_wall / dflash_wall:.4f} parity=PASS", flush=True)
    (root / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    plain_total = sum(item["plain"]["runtime"]["wall"] for item in output["results"])
    dflash_total = sum(item["dflash2"]["runtime"]["wall"] for item in output["results"])
    print(f"aggregate_speedup={plain_total / dflash_total:.4f} "
          f"plain_wall={plain_total:.4f} dflash2_wall={dflash_total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
