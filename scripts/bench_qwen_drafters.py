#!/usr/bin/env python3
"""Compare Qwen speculative drafters across upstream datasets.

Runs plain decode plus any of native MTP, external DSpark, and external
DFlash2 over the same real chat-template prompts, at concurrency one, and
reports per-dataset acceptance and speedup for each drafter.

Every drafter is verified against the plain token stream for exact greedy
parity, so a speedup here is never bought with a different output.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

# Each entry yields the raw prompt strings for a dataset. Fetching goes through
# plain HTTP rather than the datasets package so the bench has no extra
# dependency in the inference environment.
DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
               "master/grade_school_math/data/test.jsonl",
        "suffix": "\nPlease reason step by step, and put your final answer "
                  "within \\boxed{}.",
        "field": "question",
    },
    "math500": {
        "url": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
               "resolve/main/test.jsonl",
        "suffix": "\nPlease reason step by step, and put your final answer "
                  "within \\boxed{}.",
        "field": "problem",
    },
    "humaneval": {
        "url": "https://github.com/openai/human-eval/raw/master/data/"
               "HumanEval.jsonl.gz",
        "prefix": "Complete the following Python function. Return the full "
                  "function including the signature.\n\n```python\n",
        "suffix": "\n```",
        "field": "prompt",
        "gzip": True,
    },
    "mbpp": {
        "url": "https://raw.githubusercontent.com/google-research/"
               "google-research/master/mbpp/mbpp.jsonl",
        "suffix": "\nWrite the solution in Python.",
        "field": "text",
    },
    "mt-bench": {
        "url": "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
               "fastchat/llm_judge/data/mt_bench/question.jsonl",
        # mt-bench rows carry a multi-turn list; the first turn is the prompt.
        "field": "turns",
        "first_of_list": True,
    },
}

DRAFTERS = ("mtp", "dspark", "dflash2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dspark", help="external DSpark checkpoint directory")
    parser.add_argument("--dflash2", help="external DFlash2 checkpoint directory")
    parser.add_argument("--binary", default="cpp_engine/build/pocketllm_engine")
    parser.add_argument(
        "--drafters", default="mtp,dspark,dflash2",
        help="comma-separated subset of mtp,dspark,dflash2")
    parser.add_argument(
        "--datasets", default="gsm8k,math500,humaneval,mbpp,mt-bench",
        help=f"comma-separated subset of {','.join(sorted(DATASETS))}")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--tp-world", type=int, default=4)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--work-dir", default=".tmp/qwen_drafters")
    parser.add_argument(
        "--tokenizer-python",
        default="/home/lvyufeng/miniconda3/envs/deepseek/bin/python")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--mtp-tokens", type=int, default=1)
    parser.add_argument("--mtp-adaptive", action="store_true")
    parser.add_argument(
        "--keep-going", action="store_true",
        help="record a failed request and continue instead of aborting the run")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def fetch_raw(name: str, count: int, cache: Path) -> list[str]:
    """Return `count` raw prompt strings, caching the decoded rows on disk."""
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        # A short cache must be refetched rather than silently shrinking the run.
        if len(saved) >= count:
            return [str(item) for item in saved[:count]]

    spec = DATASETS[name]
    with urllib.request.urlopen(spec["url"], timeout=120) as response:
        payload = response.read()
    if spec.get("gzip"):
        payload = gzip.decompress(payload)

    rows: list[str] = []
    for line in payload.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        value = row[spec["field"]]
        if spec.get("first_of_list"):
            value = value[0]
        rows.append(f"{spec.get('prefix', '')}{value}{spec.get('suffix', '')}")
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(
            f"{name} yielded {len(rows)} prompts but {count} were requested")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


CHAT_CODE = r'''
import json, sys
from transformers import AutoTokenizer
ckpt = sys.argv[1]
prompts = json.loads(sys.stdin.read())
tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
rows = [tok.apply_chat_template([{"role": "user", "content": item}],
                                tokenize=True, add_generation_prompt=True,
                                reasoning_effort="xhigh")
        for item in prompts]
print(json.dumps(rows))
'''


def tokenize_chat(args: argparse.Namespace, prompts: list[str]) -> list[list[int]]:
    """Apply the real chat template; benchmark protocol depends on it."""
    result = subprocess.run(
        [args.tokenizer_python, "-c", CHAT_CODE, args.ckpt],
        input=json.dumps(prompts), capture_output=True, text=True, check=True)
    return [[int(x) for x in row] for row in json.loads(result.stdout)]


def build_fixture(args: argparse.Namespace, name: str, root: Path) -> list[list[int]]:
    token_cache = root / f"{name}_prompts.json"
    if token_cache.exists():
        saved = json.loads(token_cache.read_text(encoding="utf-8"))
        if isinstance(saved, list) and len(saved) >= args.num_prompts:
            return [[int(x) for x in row] for row in saved[:args.num_prompts]]
    raw = fetch_raw(name, args.num_prompts, root / f"{name}_prompts.raw.json")
    rows = tokenize_chat(args, raw)
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    token_cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


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


def mode_command(args: argparse.Namespace, mode: str) -> list[str]:
    if mode == "plain":
        return []
    if mode == "mtp":
        command = ["--qwen-mtp-tokens", str(args.mtp_tokens)]
        if args.mtp_adaptive:
            command.append("--qwen-mtp-adaptive")
        return command
    if mode == "dspark":
        return ["--qwen-dspark", str(args.dspark)]
    if mode == "dflash2":
        return ["--qwen-dflash2", str(args.dflash2)]
    raise RuntimeError(f"unknown mode {mode}")


def run_request(args: argparse.Namespace, prompt: list[int], mode: str,
                dataset: str, index: int, root: Path) -> dict[str, Any]:
    case_dir = root / dataset / mode / f"request_{index:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    token_path = case_dir / "prompt.txt"
    token_path.write_text(",".join(str(x) for x in prompt), encoding="ascii")
    id_path = case_dir / "nccl.id"
    if id_path.exists():
        id_path.unlink()

    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    started = time.monotonic()
    try:
        for rank in range(args.tp_world):
            command = [
                str(Path(args.binary).resolve()), "--ckpt", str(args.ckpt),
                "--tp-world", str(args.tp_world), "--tp-rank", str(rank),
                "--device", "0", "--nccl-id-path", str(id_path),
                "--token-ids-file", str(token_path), "--generate-token", "123",
                "--max-new-tokens", str(args.max_new_tokens),
                "--max-context",
                str(max(args.max_context, len(prompt) + args.max_new_tokens)),
                "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
                "--kv-cache-dtype", args.kv_cache_dtype,
                "--smoke-layers", "0", "--resident-bench",
                "--qwen-temperature", str(args.temperature),
                "--qwen-top-p", str(args.top_p),
                "--qwen-top-k", str(args.top_k),
                "--qwen-seed", str(args.seed),
            ] + mode_command(args, mode)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.devices.split(",")[rank]
            log = (case_dir / f"rank{rank}.log").open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env))
        statuses = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
    if any(status != 0 for status in statuses):
        raise RuntimeError(f"{dataset}/{mode} request {index} failed: {statuses}")

    parsed = [parse_log(case_dir / f"rank{rank}.log")
              for rank in range(args.tp_world)]
    if any(parsed[0][1] != item[1] for item in parsed[1:]):
        raise RuntimeError(f"{dataset}/{mode} rank parity failed at request {index}")
    return {"mode": mode, "index": index, "prompt_tokens": len(prompt),
            "runtime": parsed[0][0], "tokens": parsed[0][1],
            "elapsed_process_wall_seconds": time.monotonic() - started,
            "rank_runtime": [item[0] for item in parsed]}


def acceptance(runtime: dict[str, Any]) -> dict[str, Any]:
    """Speculative telemetry shares the mtp_* field names across drafters."""
    proposed = runtime.get("mtp_proposed_drafts", 0) or 0
    correct = runtime.get("mtp_correct_drafts", 0) or 0
    return {
        "accept_length": runtime.get("spec_accept_length", 0),
        "draft_match_rate": (correct / proposed) if proposed else 0.0,
        "proposed_drafts": proposed,
        "correct_drafts": correct,
        "verify_count": runtime.get("mtp_verify_count", 0),
        "rollback_count": runtime.get("mtp_rollback_count", 0),
        "draft_seconds": runtime.get("mtp_draft_seconds", 0.0),
        "verify_seconds": runtime.get("mtp_verify_seconds", 0.0),
        "replay_seconds": runtime.get("mtp_replay_seconds", 0.0),
    }


def main() -> int:
    args = parse_args()
    if args.tp_world != len(args.devices.split(",")):
        raise SystemExit("--devices must contain one device per TP rank")

    drafters = [item for item in args.drafters.split(",") if item]
    for item in drafters:
        if item not in DRAFTERS:
            raise SystemExit(f"unknown drafter {item}; choose from {DRAFTERS}")
    if "dspark" in drafters and not args.dspark:
        raise SystemExit("--dspark is required to benchmark the dspark drafter")
    if "dflash2" in drafters and not args.dflash2:
        raise SystemExit("--dflash2 is required to benchmark the dflash2 drafter")

    datasets = [item for item in args.datasets.split(",") if item]
    for item in datasets:
        if item not in DATASETS:
            raise SystemExit(f"unknown dataset {item}; choose from {sorted(DATASETS)}")

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "num_prompts": args.num_prompts, "max_new_tokens": args.max_new_tokens,
        "drafters": drafters, "datasets": datasets, "results": [],
    }

    for dataset in datasets:
        prompts = build_fixture(args, dataset, root)
        for index, prompt in enumerate(prompts):
            entry: dict[str, Any] = {"dataset": dataset, "index": index,
                                     "prompt_tokens": len(prompt)}
            try:
                plain = run_request(args, prompt, "plain", dataset, index, root)
            except RuntimeError as error:
                if not args.keep_going:
                    raise
                print(f"dataset={dataset} request={index + 1}/{len(prompts)} "
                      f"plain FAILED: {error}", flush=True)
                entry["plain_error"] = str(error)
                output["results"].append(entry)
                continue
            entry["plain"] = plain
            plain_wall = plain["runtime"]["wall"]
            plain_decode = plain["runtime"]["decode_seconds"]

            for mode in drafters:
                try:
                    run = run_request(args, prompt, mode, dataset, index, root)
                except RuntimeError as error:
                    if not args.keep_going:
                        raise
                    print(f"dataset={dataset} request={index + 1}/{len(prompts)} "
                          f"{mode} FAILED: {error}", flush=True)
                    entry[mode] = {"error": str(error)}
                    continue
                # Greedy parity is the precondition for reporting any speedup:
                # a drafter that changes the output is not a faster decoder.
                # In sampled mode (temperature > 0), every step draws a fresh
                # uniform, so drafter and plain paths diverge by design.
                parity = (run["tokens"] == plain["tokens"]
                          if args.temperature <= 1e-5 else None)
                stats = acceptance(run["runtime"])
                wall = run["runtime"]["wall"]
                decode = run["runtime"]["decode_seconds"]
                entry[mode] = {
                    "run": run, "parity": parity, "acceptance": stats,
                    "wall_speedup": plain_wall / wall if wall else 0.0,
                    "decode_speedup": plain_decode / decode if decode else 0.0,
                }
                parity_str = ("PASS" if parity else "FAIL") if parity is not None else "N/A"
                print(f"dataset={dataset} request={index + 1}/{len(prompts)} "
                      f"mode={mode} wall={wall:.4f}s "
                      f"speedup={plain_wall / wall:.4f} "
                      f"decode_speedup={plain_decode / decode if decode else 0:.4f} "
                      f"accept_length={stats['accept_length']} "
                      f"match_rate={stats['draft_match_rate']:.4f} "
                      f"parity={parity_str}", flush=True)
            output["results"].append(entry)
            (root / "results.json").write_text(
                json.dumps(output, indent=2), encoding="utf-8")

    print()
    print(f"{'dataset':<10} {'drafter':<8} {'wall':>7} {'decode':>7} "
          f"{'acc_len':>8} {'match':>7} {'parity':>7}")
    summary: dict[str, Any] = {}
    for dataset in datasets:
        rows = [item for item in output["results"] if item["dataset"] == dataset]
        for mode in drafters:
            runs = [item[mode] for item in rows
                    if mode in item and "error" not in item[mode]]
            if not runs:
                print(f"{dataset:<10} {mode:<8} {'n/a':>7}")
                continue
            plains = [item["plain"] for item in rows
                      if mode in item and "error" not in item[mode]]
            plain_wall = sum(item["runtime"]["wall"] for item in plains)
            plain_decode = sum(item["runtime"]["decode_seconds"] for item in plains)
            mode_wall = sum(item["run"]["runtime"]["wall"] for item in runs)
            mode_decode = sum(
                item["run"]["runtime"]["decode_seconds"] for item in runs)
            # Aggregate acceptance is draft-weighted, not a mean of per-request
            # rates, so long requests are not down-weighted to match short ones.
            proposed = sum(item["acceptance"]["proposed_drafts"] for item in runs)
            correct = sum(item["acceptance"]["correct_drafts"] for item in runs)
            accept_len = sum(
                item["acceptance"]["accept_length"] for item in runs) / len(runs)
            passes = sum(1 for item in runs if item["parity"] is True)
            parity_checked = sum(1 for item in runs if item["parity"] is not None)
            entry = {
                "requests": len(runs),
                "wall_speedup": plain_wall / mode_wall if mode_wall else 0.0,
                "decode_speedup": plain_decode / mode_decode if mode_decode else 0.0,
                "mean_accept_length": accept_len,
                "draft_match_rate": (correct / proposed) if proposed else 0.0,
                "parity_pass": passes, "parity_total": parity_checked,
            }
            summary.setdefault(dataset, {})[mode] = entry
            parity_display = (f"{passes}/{parity_checked}" if parity_checked > 0
                            else "N/A")
            print(f"{dataset:<10} {mode:<8} {entry['wall_speedup']:>7.4f} "
                  f"{entry['decode_speedup']:>7.4f} {accept_len:>8.3f} "
                  f"{entry['draft_match_rate']:>7.4f} "
                  f"{parity_display:>7}")
    output["summary"] = summary
    (root / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
