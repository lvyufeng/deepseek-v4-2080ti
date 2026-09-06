#!/usr/bin/env python3
"""
Benchmark concurrent request throughput for batch scheduler.

This script tests the throughput improvement of batch mode over serial mode
with concurrent requests. Target improvements:
- 2 concurrent: ≥1.7× baseline
- 4 concurrent: ≥3.0× baseline
- 8 concurrent: ≥4.5× baseline
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from queue import Queue
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pocketllm import EngineArgs, LLM, SamplingParams


class ConcurrentBenchmark:
    def __init__(self, llm: LLM, prompt_tokens: List[int], sampling_params: SamplingParams):
        self.llm = llm
        self.prompt_tokens = prompt_tokens
        self.sampling_params = sampling_params
        self.results_queue = Queue()

    def run_request(self, request_id: int, prompt_tokens: List[int]):
        """Run a single generation request."""
        start = time.perf_counter()
        try:
            result = self.llm.generate(
                [prompt_tokens],
                sampling_params=self.sampling_params
            )
            end = time.perf_counter()

            self.results_queue.put({
                "request_id": request_id,
                "success": True,
                "latency": end - start,
                "ttft": result[0].timings.ttft_seconds,
                "tokens": len(result[0].token_ids),
                "finish_reason": result[0].finish_reason,
            })
        except Exception as e:
            end = time.perf_counter()
            self.results_queue.put({
                "request_id": request_id,
                "success": False,
                "error": str(e),
                "latency": end - start,
            })

    def run_concurrent(self, num_requests: int, prompt_tokens: List[int] | None = None) -> List[Dict[str, Any]]:
        """Run multiple requests concurrently.

        ``prompt_tokens`` may be supplied per call.  This is what makes the
        measurement honest: the native engine keeps a slot's KV across runs, so
        re-submitting byte-identical prompts means every run after the first
        only pays decode and the numbers are not comparable across paged and
        contiguous modes (contiguous reuses, paged re-allocates).  Each call
        here uses a distinct prompt so both arms pay a full prefill every run.
        """
        if prompt_tokens is None:
            prompt_tokens = self.prompt_tokens
        threads = []

        # Start all threads
        start_time = time.perf_counter()
        for i in range(num_requests):
            thread = threading.Thread(target=self.run_request, args=(i, prompt_tokens))
            thread.start()
            threads.append(thread)

        # Wait for all to complete
        for thread in threads:
            thread.join()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Collect results
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())

        results.sort(key=lambda x: x["request_id"])

        return results, total_time


def benchmark_concurrent(
    checkpoint: str,
    enable_batching: bool,
    num_concurrent: int,
    max_tokens: int = 128,
    warmup_runs: int = 1,
    test_runs: int = 3,
    prompt_length: int = 32,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    max_model_len: int = 4096,
    backend_options: dict | None = None,
):
    """Benchmark concurrent request throughput."""

    mode = "batch" if enable_batching else "serial"
    print(f"\n{'='*60}")
    print(f"Benchmarking: {mode.upper()} mode, {num_concurrent} concurrent requests")
    print(f"{'='*60}")

    # Create LLM with specified mode
    extra_options = dict(backend_options) if backend_options else {}
    # enable_prefix_caching is a top-level EngineArgs field, not a backend
    # option, even though the CLI surface exposes it as --backend-option.
    # The native engine (qwen_engine.cpp) honours this directly; an identical
    # warmup prompt is otherwise prefix-reused, so the test run only pays decode
    # and the contiguous arm looks ~4x too fast.  Pop it out of the option dict
    # before merging so it does not get passed down twice.
    prefix_caching = bool(extra_options.pop("enable_prefix_caching", True))
    args = EngineArgs(
        model=checkpoint,
        backend="cpp",
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        max_model_len=max_model_len,
        enable_prefix_caching=prefix_caching,
        backend_options={
            "enable_batching": enable_batching,
            # Size the arena to this level's concurrency rather than a fixed 8.
            # max(8, n) made the 2-concurrent run reserve 8 slots' worth of KV
            # cache, so the low levels paid memory they never used.
            "max_batch_size": num_concurrent if enable_batching else 1,
            # kv_paged / kv_block_size / kv_cache_bytes and anything else come
            # from --backend-option KEY=VALUE (JSON-typed, like the server CLI).
            **extra_options,
        }
    )

    llm = LLM(args)

    # A nonzero TensorParallel rank is a worker: the engine is constructed here
    # (so it joins the warmup_tp() barrier with rank 0), then this process
    # enters the native command channel and does no driving of its own.
    # run_worker() blocks until rank 0 sends shutdown, which happens when rank
    # 0's LLM.close() for *this* level runs.  On return, close the engine here
    # too so the worker does not linger across levels, then leave the function.
    if tensor_parallel_rank > 0:
        try:
            llm.backend.run_worker()
            print(f"[TP rank {tensor_parallel_rank}] worker loop exited for "
                  f"{mode} mode, {num_concurrent} concurrent")
        finally:
            llm.close()
        return None

    # Check capabilities
    caps = llm.backend.capabilities
    print(f"Backend: {caps.name}")
    print(f"Scheduler: {caps.details.get('scheduler')}")
    print(f"Supports batch: {caps.supports_batch}")
    print(f"Max batch size: {caps.details.get('max_batch_size', 1)}")

    # Realized KV footprint.  kv_cache_bytes() reports the arena/pool budget the
    # engine committed at construction; kv_paged_blocks is nonzero for paged.
    # This is what makes the two arms directly comparable: the paged pool can be
    # bounded below the contiguous arena's `batch * max_context` reservation and
    # still admit the same requests, because the solver only allocates blocks for
    # tokens actually used.
    engine = getattr(llm.backend, "_engine", None)
    if engine is not None:
        kv_bytes_attr = getattr(engine, "kv_cache_bytes", None)
        if kv_bytes_attr is not None:
            kv_bytes_value = kv_bytes_attr() if callable(kv_bytes_attr) else kv_bytes_attr
            try:
                kv_bytes_value = int(kv_bytes_value)
                print(f"KV cache bytes: {kv_bytes_value} ({kv_bytes_value / (1024 * 1024):.1f} MiB)")
            except (TypeError, ValueError):
                print(f"KV cache bytes: {kv_bytes_value}")
        for name in ("kv_total_blocks", "kv_free_blocks"):
            attr = getattr(engine, name, None)
            if attr is not None:
                try:
                    print(f"KV {name}: {attr() if callable(attr) else attr}")
                except Exception:
                    pass
    else:
        print("KV cache bytes: (no native engine exposed)")

    # A silent fall back to the serial path would be reported as a batch result
    # and make the comparison meaningless, so fail loudly instead.
    actually_batching = bool(getattr(llm.backend, "_batching_enabled", False))
    if actually_batching != enable_batching:
        llm.close()
        raise SystemExit(
            f"requested enable_batching={enable_batching} but backend reports "
            f"{actually_batching}; refusing to report this as a {mode} measurement"
        )

    # Create test configuration
    # A real, tokenizer-produced prompt rather than the synthetic id range
    # [1..n]: a model fed arbitrary vocab ids almost always emits EOS on the
    # first step, which truncated every run to 2 tokens and made the timing
    # meaningless (~10x too fast).  Natural text also exercises the KV cache
    # through the same code path a server would.  The prompt is built to
    # `prompt_length` tokens by repeating a natural sentence.
    sampler = getattr(llm.backend, "_tokenizer", None)
    if getattr(sampler, "encode", None) is not None:
        sentence = "The quick brown fox jumps over the lazy dog and the party never ends. "
        encoding = sampler.encode(sentence)
        if not encoding:
            raise SystemExit("tokenizer returned no tokens for the filler sentence")
        prompt_tokens = []
        while len(prompt_tokens) < prompt_length:
            prompt_tokens.extend(encoding)
        prompt_tokens = prompt_tokens[:prompt_length]
    else:
        # Fallback for a backend without a tokenizer: raw ids are better than a
        # crash, but the numbers will not be meaningful.
        prompt_tokens = list(range(1, prompt_length + 1))
    # ignore_eos makes the model run the full max_tokens budget instead of
    # stopping at whatever token it happens to emit first (which is what the
    # synthetic prompt triggered).  This mirrors the native parity driver,
    # which sets req.sampling.ignore_eos = true for exactly this reason.
    sampling = SamplingParams(
        max_tokens=max_tokens, temperature=0.0, extra={"ignore_eos": True}
    )

    print(f"\nConfiguration:")
    print(f"  Prompt length: {prompt_length} tokens")
    print(f"  Max new tokens: {max_tokens}")
    print(f"  Concurrent requests: {num_concurrent}")
    print(f"  Warmup runs: {warmup_runs}")
    print(f"  Test runs: {test_runs}")

    benchmark = ConcurrentBenchmark(llm, prompt_tokens, sampling)

    # Every measurement run gets a *distinct* prompt of the same length.  The
    # native engine keeps a batch slot's KV across runs, so resubmitting the
    # identical prompt means runs 2..N only pay decode; only the contiguous arm
    # can reuse that (paged re-allocates), which is exactly what produced the
    # false ~10x gap above.  Prepend a unique marker token and shave the tail so
    # the length is held constant.
    def _prompt_for(run_index: int) -> List[int]:
        marker = [1_000_000 + run_index]
        return marker + prompt_tokens[: max(0, prompt_length - len(marker))]

    all_total_times = []
    all_request_latencies = []
    failures = 0
    try:
        # Warmup
        print(f"\nWarmup ({warmup_runs} runs)...")
        for i in range(warmup_runs):
            results, total_time = benchmark.run_concurrent(num_concurrent, _prompt_for(i))
            successful = sum(1 for r in results if r["success"])
            detail = "; ".join(
                f"req{r['request_id']}={r.get('tokens', '?')}tok/{r.get('finish_reason', '?')}"
                for r in results
            )
            print(f"  Run {i+1}: {total_time:.3f}s ({successful}/{num_concurrent} successful) [{detail}]")

        # Benchmark
        print(f"\nBenchmark ({test_runs} runs)...")

        for i in range(test_runs):
            results, total_time = benchmark.run_concurrent(
                num_concurrent, _prompt_for(warmup_runs + i)
            )
            all_total_times.append(total_time)

            successful = sum(1 for r in results if r["success"])
            failures += num_concurrent - successful
            avg_latency = sum(r["latency"] for r in results if r["success"]) / max(successful, 1)
            all_request_latencies.extend([r["latency"] for r in results if r["success"]])

            for failed in (r for r in results if not r["success"]):
                print(f"    request {failed['request_id']} failed: {failed.get('error')}")

            detail = "; ".join(
                f"req{r['request_id']}={r.get('tokens', '?')}tok/{r.get('finish_reason', '?')}"
                for r in results if r["success"]
            )
            print(f"  Run {i+1}: {total_time:.3f}s total, {avg_latency:.3f}s avg latency ({successful}/{num_concurrent} successful) [{detail}]")
    finally:
        # Each concurrency level builds its own LLM.  Without an explicit close
        # the engine is only reclaimed at interpreter exit, so rank 0 never sends
        # the shutdown command and every previous level's TP workers stay alive
        # holding their share of device memory.  Six levels deep that is six
        # resident copies of the model.
        llm.close()

    # A level that dropped requests is not comparable against one that did not.
    if failures:
        raise SystemExit(
            f"{failures} request(s) failed in {mode} mode at {num_concurrent} "
            "concurrent; results would be misleading"
        )

    # Statistics
    avg_total_time = sum(all_total_times) / len(all_total_times)
    min_total_time = min(all_total_times)
    max_total_time = max(all_total_times)

    avg_request_latency = sum(all_request_latencies) / len(all_request_latencies)

    # Throughput calculation
    total_requests = num_concurrent * test_runs
    throughput = total_requests / sum(all_total_times)  # requests per second

    print(f"\n{'='*60}")
    print(f"Results Summary ({mode.upper()} mode, {num_concurrent} concurrent)")
    print(f"{'='*60}")
    print(f"Total Time (wall-clock for {num_concurrent} concurrent):")
    print(f"  Average: {avg_total_time:.3f}s")
    print(f"  Min:     {min_total_time:.3f}s")
    print(f"  Max:     {max_total_time:.3f}s")
    print(f"\nPer-Request Latency:")
    print(f"  Average: {avg_request_latency:.3f}s")
    print(f"\nThroughput:")
    print(f"  {throughput:.2f} requests/sec")

    return {
        "mode": mode,
        "num_concurrent": num_concurrent,
        "avg_total_time": avg_total_time,
        "avg_request_latency": avg_request_latency,
        "throughput": throughput,
        "total_times": all_total_times,
    }


def parse_backend_options(items) -> dict:
    """Turn --backend-option KEY=VALUE pairs into a typed dict.

    VALUE is JSON-parsed so numbers, booleans, and nested values stay typed; a
    bare string that is not valid JSON stays a string, matching the server CLI.
    """
    options: dict = {}
    for item in items or []:
        key, sep, raw = str(item).partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--backend-option expects KEY=VALUE, got {item!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        options[key.strip()] = value
    return options


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark concurrent request throughput"
    )
    parser.add_argument(
        "checkpoint",
        help="Path to Qwen3.5 checkpoint directory"
    )
    # Building a second engine in one process costs the *second* one about 10%,
    # whichever mode it is -- measured with four serial engines in a row, where
    # only #1 was slow (0.735s vs 0.672s) at identical clocks.  Comparing modes
    # inside one process therefore charges that to whichever ran second.  The
    # driver (run_phase35_benchmarks.sh) gives each measurement its own process
    # and this flag emits the one result for it to collect.
    parser.add_argument(
        "--single",
        choices=["serial", "batch"],
        help="Measure only this mode, then write --json-out and exit"
    )
    parser.add_argument(
        "--json-out",
        help="Write the --single measurement here as JSON"
    )
    parser.add_argument(
        "--backend-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Backend option (JSON-typed VALUE). May repeat, e.g. "
             "--backend-option kv_paged=True --backend-option kv_block_size=16"
    )
    parser.add_argument(
        "--num-concurrent",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Number of concurrent requests to test (default: 2 4 8)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per request (default: 128)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs per concurrency level (default: 1)"
    )
    parser.add_argument(
        "--test-runs",
        type=int,
        default=3,
        help="Number of test runs per concurrency level (default: 3)"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=32,
        help="Prompt length in tokens (default: 32)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max sequence length; bounds per-slot KV cache (default: 4096)"
    )

    args = parser.parse_args()
    backend_options = parse_backend_options(args.backend_option)
    # Set by the TP launcher; each process runs this script with a distinct rank.
    # From_env also honours this, but we pass it explicitly so the worker branch
    # and EngineArgs agree even when TP_WORLD is unset in a single-rank run.
    tensor_parallel_rank = int(os.environ.get("TP_RANK", "0"))
    tensor_parallel_size = max(args.tensor_parallel_size, int(os.environ.get("TP_WORLD", "1")))

    print("="*60)
    print("Concurrent Request Throughput Benchmark")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Tensor parallel rank: {tensor_parallel_rank}")
    if backend_options:
        print(f"Backend options: {backend_options}")

    if args.single:
        if len(args.num_concurrent) != 1:
            raise SystemExit("--single takes exactly one --num-concurrent level")
        result = benchmark_concurrent(
            args.checkpoint,
            enable_batching=args.single == "batch",
            num_concurrent=args.num_concurrent[0],
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
            prompt_length=args.prompt_length,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            max_model_len=args.max_model_len,
            backend_options=backend_options,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(result, indent=2))
        return 0

    # Expected improvements
    targets = {
        2: 1.7,
        4: 3.0,
        8: 4.5,
    }

    all_results = {}

    # Benchmark each concurrency level
    for num_concurrent in args.num_concurrent:
        # Serial mode (baseline)
        serial_result = benchmark_concurrent(
            args.checkpoint,
            enable_batching=False,
            num_concurrent=num_concurrent,
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
            prompt_length=args.prompt_length,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            max_model_len=args.max_model_len,
            backend_options=backend_options,
        )

        # Batch mode
        batch_result = benchmark_concurrent(
            args.checkpoint,
            enable_batching=True,
            num_concurrent=num_concurrent,
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs,
            prompt_length=args.prompt_length,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            max_model_len=args.max_model_len,
            backend_options=backend_options,
        )

        all_results[num_concurrent] = {
            "serial": serial_result,
            "batch": batch_result,
        }

    # Final comparison
    print(f"\n{'='*60}")
    print("Final Comparison")
    print(f"{'='*60}")

    all_pass = True
    for num_concurrent in args.num_concurrent:
        results = all_results[num_concurrent]
        serial = results["serial"]
        batch = results["batch"]

        throughput_ratio = batch["throughput"] / serial["throughput"]
        target = targets.get(num_concurrent, 1.0)

        print(f"\n{num_concurrent} Concurrent Requests:")
        print(f"  Serial throughput: {serial['throughput']:.2f} req/s")
        print(f"  Batch throughput:  {batch['throughput']:.2f} req/s")
        print(f"  Improvement:       {throughput_ratio:.2f}× (target: ≥{target}×)")

        if throughput_ratio >= target:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL (gap: {target - throughput_ratio:.2f}×)")
            all_pass = False

    print(f"\n{'='*60}")
    print("Overall Verdict")
    print(f"{'='*60}")
    if all_pass:
        print("✅ All throughput targets met!")
    else:
        print("❌ Some throughput targets not met")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
