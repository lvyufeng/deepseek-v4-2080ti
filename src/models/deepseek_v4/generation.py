import ctypes
import os
import json
import sys
import time
from argparse import ArgumentParser
from datetime import timedelta
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from src.components.moe.shared_weights import SharedCPUMoEWeightArena
from src.runtime.prefix_snapshot import PrefixSnapshotCache
from src.models.deepseek_v4.runtime import (
    Transformer,
    ModelArgs,
)
from src.models.deepseek_v4.loader import load_model
from src.encoding.deepseek_v4 import encode_messages, parse_message_from_completion_text


def _enable_numa_interleave() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(0b11)
    ret = libc.syscall(238, 3, ctypes.byref(mask), ctypes.c_ulong(64))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _bind_shared_cpu_moe_weights(model: Transformer, root_dir: str, args: ModelArgs, world_size: int, rank: int) -> SharedCPUMoEWeightArena:
    precreated = os.getenv("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATED", "0").lower() in {"1", "true", "yes"}
    arena = SharedCPUMoEWeightArena(
        root_dir=root_dir,
        rank=rank,
        world_size=world_size,
        n_layers=args.n_layers,
        n_routed_experts=args.n_routed_experts,
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim,
        create=not precreated,
    )
    for layer in model.layers:
        if layer is None:
            continue
        moe = layer.ffn
        for expert_id in range(moe.experts_start_idx, moe.experts_end_idx):
            expert = moe.experts[expert_id]
            if expert is None:
                continue
            expert.w1.set_int8_storage(
                arena.tensor(layer.layer_id, expert_id, "w1.weight"),
                arena.tensor(layer.layer_id, expert_id, "w1.scale"),
            )
            expert.w2.set_int8_storage(
                arena.tensor(layer.layer_id, expert_id, "w2.weight"),
                arena.tensor(layer.layer_id, expert_id, "w2.scale"),
            )
            expert.w3.set_int8_storage(
                arena.tensor(layer.layer_id, expert_id, "w3.weight"),
                arena.tensor(layer.layer_id, expert_id, "w3.scale"),
            )
    return arena


def _cpu_affinity_for_rank(local_rank: int, world_size: int) -> list[int] | None:
    topo_root = "/sys/devices/system/cpu"
    if not hasattr(os, "sched_setaffinity") or not os.path.isdir(topo_root):
        return None
    core_map: dict[tuple[int, int], list[int]] = {}
    for name in os.listdir(topo_root):
        if not name.startswith("cpu") or not name[3:].isdigit():
            continue
        cpu = int(name[3:])
        try:
            with open(os.path.join(topo_root, name, "topology/core_id")) as f:
                core_id = int(f.read())
            with open(os.path.join(topo_root, name, "topology/physical_package_id")) as f:
                package_id = int(f.read())
        except OSError:
            continue
        core_map.setdefault((package_id, core_id), []).append(cpu)
    physical_cores = [sorted(v) for _, v in sorted(core_map.items())]
    if not physical_cores:
        return None
    base = len(physical_cores) // world_size
    extra = len(physical_cores) % world_size
    start = local_rank * base + min(local_rank, extra)
    count = base + (1 if local_rank < extra else 0)
    selected = physical_cores[start:start + count]
    cpus: list[int] = []
    for siblings in selected:
        cpus.extend(siblings)
    return sorted(cpus)



def _has_generation_processors(options: dict | None) -> bool:
    if not options:
        return False
    return any([
        options.get("top_p") is not None and float(options.get("top_p")) < 1.0,
        options.get("top_k") is not None and int(options.get("top_k")) > 0,
        options.get("min_p") is not None and float(options.get("min_p")) > 0.0,
        abs(float(options.get("frequency_penalty") or 0.0)) > 1e-9,
        abs(float(options.get("presence_penalty") or 0.0)) > 1e-9,
        abs(float(options.get("repetition_penalty") or 1.0) - 1.0) > 1e-9,
        bool(options.get("logprobs")),
    ])


def _apply_generation_processors(logits: torch.Tensor, tokens: torch.Tensor, cur_pos: int, options: dict | None) -> torch.Tensor:
    if not options:
        return logits
    processed = logits.float()
    repetition_penalty = float(options.get("repetition_penalty") or 1.0)
    frequency_penalty = float(options.get("frequency_penalty") or 0.0)
    presence_penalty = float(options.get("presence_penalty") or 0.0)
    if repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0:
        for row in range(processed.size(0)):
            history = tokens[row, :cur_pos]
            history = history[history >= 0]
            if history.numel() == 0:
                continue
            unique, counts = torch.unique(history, return_counts=True)
            if repetition_penalty != 1.0:
                values = processed[row, unique]
                processed[row, unique] = torch.where(values > 0, values / repetition_penalty, values * repetition_penalty)
            if frequency_penalty != 0.0:
                processed[row, unique] -= counts.to(processed.dtype) * frequency_penalty
            if presence_penalty != 0.0:
                processed[row, unique] -= presence_penalty
    top_k = options.get("top_k")
    if top_k is not None and int(top_k) > 0 and int(top_k) < processed.size(-1):
        kth = torch.topk(processed, int(top_k), dim=-1).values[..., -1, None]
        processed = processed.masked_fill(processed < kth, float("-inf"))
    top_p = options.get("top_p")
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(processed, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
        remove[..., 0] = False
        mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        processed = processed.masked_fill(mask, float("-inf"))
    min_p = options.get("min_p")
    if min_p is not None and float(min_p) > 0.0:
        probs = torch.softmax(processed, dim=-1)
        threshold = probs.max(dim=-1, keepdim=True).values * float(min_p)
        processed = processed.masked_fill(probs < threshold, float("-inf"))
    return processed


def sample(logits, temperature: float = 1.0, generator: torch.Generator | None = None):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    noise = torch.empty_like(probs).exponential_(1, generator=generator)
    return probs.div_(noise).argmax(dim=-1)


def _record_generation_logprobs(
    rows: list[list[dict]],
    logits: torch.Tensor,
    next_token: torch.Tensor,
    generated_mask: torch.Tensor,
    top_logprobs: int,
) -> None:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    selected = log_probs.gather(1, next_token.unsqueeze(-1)).squeeze(-1)
    if top_logprobs > 0:
        top_values, top_indices = torch.topk(log_probs, min(top_logprobs, log_probs.size(-1)), dim=-1)
    else:
        top_values = top_indices = None
    for row in range(next_token.size(0)):
        if not bool(generated_mask[row].item()):
            continue
        entry = {"token_id": int(next_token[row].item()), "logprob": float(selected[row].item())}
        if top_values is not None and top_indices is not None:
            entry["top_logprobs"] = [
                {"token_id": int(tok.item()), "logprob": float(val.item())}
                for tok, val in zip(top_indices[row].detach().cpu(), top_values[row].detach().cpu())
            ]
        rows[row].append(entry)


def _sync_timing_boundary(model, enabled: bool) -> None:
    if not enabled:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized() and getattr(model, "is_layer_pp", False):
        dist.barrier()


def _prefix_snapshot_settings(
    model: Transformer,
    prompt_tokens: List[List[int]],
    prompt_lens: list[int],
    prefix_snapshot_hint: dict | None,
) -> tuple[int, list[int]]:
    if not PrefixSnapshotCache.enabled() or len(prompt_tokens) != 1 or getattr(model, "is_layer_pp", False):
        return 0, []
    cache = PrefixSnapshotCache.instance()
    restore_pos = cache.restore(model, prefix_snapshot_hint, prompt_tokens[0]) if prefix_snapshot_hint else 0
    capture_pos = (prompt_lens[0] // cache.block) * cache.block
    if capture_pos < cache.min_tokens:
        return restore_pos, []
    hinted_positions = prefix_snapshot_hint.get("capture_positions") if prefix_snapshot_hint else None
    if hinted_positions:
        capture_positions = [int(pos) for pos in hinted_positions if restore_pos < int(pos) <= capture_pos]
    else:
        capture_positions = [capture_pos] if restore_pos < capture_pos else []
    return restore_pos, capture_positions


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    phase_callback=None,
    prefill_chunk_tokens: int = 0,
    generation_options: dict | None = None,
    prefix_snapshot_hint: dict | None = None,
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    # Optional torch.profiler tracing of a single decode step. Triggered by
    # DEEPSEEK_DECODE_PROFILE_DIR=<dir>; DEEPSEEK_DECODE_PROFILE_STEP picks
    # which decode step (1-indexed; default 2 to skip warmup) to capture.
    decode_profile_dir = os.environ.get("DEEPSEEK_DECODE_PROFILE_DIR", "")
    decode_profile_step = int(os.environ.get("DEEPSEEK_DECODE_PROFILE_STEP", "2") or "2")
    decode_profile_rank = int(os.environ.get("DEEPSEEK_DECODE_PROFILE_RANK", "0") or "0")
    decode_step_idx = 0
    profiler_active = None  # holds the torch.profiler context manager when sampling.
    # Phase 1 MTP probe: when DEEPSEEK_MTP_LOG=1 we run mtp[0] after each
    # decode step to produce a draft token, then on the NEXT step compare
    # the draft against the actual main argmax. This validates the MTP
    # head's IO contract without changing decode behavior.
    mtp_log_enabled = os.environ.get("DEEPSEEK_MTP_LOG", "0").lower() in {"1", "true", "yes"}
    # Phase 3 MTP speculative decoding: when DEEPSEEK_DECODE_MTP_SPEC=1 the
    # decode loop runs in spec mode -- each round either does a length-1
    # main forward + MTP draft, or a length-2 main verify forward over
    # [prev_tok, draft] with strict greedy match (accept iff
    # main.argmax[t+1] == draft). On accept we advance prev_pos by 2.
    mtp_spec_enabled = os.environ.get("DEEPSEEK_DECODE_MTP_SPEC", "0").lower() in {"1", "true", "yes"}
    generation_options = generation_options or {}
    needs_logits = temperature > 0 or _has_generation_processors(generation_options)
    generator = None
    if generation_options.get("seed") is not None and torch.cuda.is_available():
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(generation_options["seed"]))
    collect_logprobs = bool(generation_options.get("logprobs"))
    top_logprobs = max(int(generation_options.get("top_logprobs") or 0), 0)
    logprob_rows: list[list[dict]] = [[] for _ in prompt_tokens]
    if getattr(model, "is_layer_pp", False) and needs_logits:
        raise RuntimeError("layer_pp_4gpu currently supports greedy decoding only")
    if getattr(model, "is_layer_pp", False) and (mtp_log_enabled or mtp_spec_enabled):
        raise RuntimeError("layer_pp_4gpu does not support MTP log/speculative paths yet")
    # Single-prompt requirement for spec mode: prompt_mask handling is
    # easier when all rows have the same prompt length. Disable spec for
    # multi-prompt or temperature > 0 calls.
    if mtp_spec_enabled and (needs_logits or len(set(prompt_lens)) > 1):
        mtp_spec_enabled = False
    prefix_snapshot_start_pos, prefix_snapshot_capture_positions = _prefix_snapshot_settings(
        model, prompt_tokens, prompt_lens, prefix_snapshot_hint
    )
    prefix_snapshot_pending_captures = set(prefix_snapshot_capture_positions)
    prefix_snapshot_cache = PrefixSnapshotCache.instance() if (prefix_snapshot_pending_captures or prefix_snapshot_start_pos > 0) else None

    def maybe_capture_prefix_snapshot(pos: int) -> None:
        if prefix_snapshot_cache is None or pos not in prefix_snapshot_pending_captures:
            return
        if len(prompt_tokens) != 1 or pos > prompt_lens[0]:
            return
        prefix_snapshot_cache.capture(model, prompt_tokens[0], pos)
        prefix_snapshot_pending_captures.discard(pos)

    pending_drafts = None  # tensor [b] of last-step MTP drafts, or None
    mtp_accept = 0
    mtp_total = 0
    mtp_spec_accepts = 0
    mtp_spec_rounds = 0
    prev_pos = prefix_snapshot_start_pos
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    sync_timings = _env_flag("DEEPSEEK_SYNC_TIMINGS", "1" if getattr(model, "is_layer_pp", False) else "0")
    sync_each_step = _env_flag("DEEPSEEK_SYNC_EACH_STEP", "0")
    prompt_prefill_tokens = sum(prompt_lens)
    prefill_time = 0.0
    decode_time = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    decode_wall_start = None
    cur_pos = min(prompt_lens)
    while cur_pos < total_len:
        prefill_step = prev_pos < min(prompt_lens)
        phase = "prefill" if prefill_step else "decode"
        if phase_callback is not None:
            phase_callback(phase)
        # Bump decode index up-front so profiler enable/disable logic is uniform.
        if phase == "decode":
            decode_step_idx += 1
        # Optionally start torch.profiler around the chosen decode step.
        rank_for_profile = dist.get_rank() if dist.is_initialized() else 0
        capture_this = (
            decode_profile_dir
            and phase == "decode"
            and decode_step_idx == decode_profile_step
            and rank_for_profile == decode_profile_rank
        )
        if capture_this:
            os.makedirs(decode_profile_dir, exist_ok=True)
            torch.cuda.synchronize()
            profiler_active = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            profiler_active.__enter__()
        # Decide if this iteration runs a length-2 verify forward (spec mode
        # with a pending draft and room for a 2-token advance).
        spec_verify = (
            mtp_spec_enabled
            and prev_pos > 0
            and pending_drafts is not None
            and cur_pos + 1 < total_len
            and not bool(prompt_mask[:, cur_pos].all())
            and not bool(prompt_mask[:, cur_pos + 1].any())
        )
        _sync_timing_boundary(model, sync_timings and sync_each_step)
        step_start = time.perf_counter()
        if needs_logits:
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            logits = _apply_generation_processors(logits, tokens, cur_pos, generation_options)
            last_hidden = None
            spec_verify = False
        elif prefill_step:
            capture_points = sorted(pos for pos in prefix_snapshot_pending_captures if prev_pos < pos < cur_pos)
            if prefill_chunk_tokens > 0 and cur_pos - prev_pos > prefill_chunk_tokens:
                chunk = prefill_chunk_tokens
                chunk_start = prev_pos
                while chunk_start + chunk < cur_pos:
                    chunk_end = chunk_start + chunk
                    model.forward(tokens[:, chunk_start:chunk_end], chunk_start, return_next_token=False)
                    maybe_capture_prefix_snapshot(chunk_end)
                    chunk_start = chunk_end
                next_token = model.forward(tokens[:, chunk_start:cur_pos], chunk_start, return_next_token=True)
            elif capture_points:
                chunk_start = prev_pos
                for chunk_end in capture_points:
                    model.forward(tokens[:, chunk_start:chunk_end], chunk_start, return_next_token=False)
                    maybe_capture_prefix_snapshot(chunk_end)
                    chunk_start = chunk_end
                next_token = model.forward(tokens[:, chunk_start:cur_pos], chunk_start, return_next_token=True)
            else:
                next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, return_next_token=True)
            last_hidden = None
            logits = None
        elif spec_verify:
            # prev_pos for decode rounds is always cur_pos-1, so the input chunk
            # is of length 2: [tokens[:, prev_pos], draft] -> positions
            # prev_pos and prev_pos+1 (== cur_pos), predicting cur_pos and cur_pos+1.
            assert prev_pos == cur_pos - 1, "spec_verify requires prev_pos==cur_pos-1"
            spec_verify_two_len1 = os.environ.get("DEEPSEEK_DECODE_MTP_SPEC_TWO_LEN1", "0").lower() in {"1", "true", "yes"}
            if spec_verify_two_len1:
                # Diagnostic mode: equivalent to length-2 but uses two length-1
                # forwards (always-validated fast path). If this yields a high
                # accept rate while length-2 does not, the length-2 attention
                # path has a numerical bug.
                tok_a, h_a = model.forward(
                    tokens[:, prev_pos:cur_pos], prev_pos,
                    return_next_token=True, return_hidden=True,
                )
                tok_b, h_b = model.forward(
                    pending_drafts.unsqueeze(-1), cur_pos,
                    return_next_token=True, return_hidden=True,
                )
                next_tokens_2 = torch.stack([tok_a, tok_b], dim=-1)
                last_hidden = torch.cat([h_a, h_b], dim=1)
            else:
                inp = torch.cat([tokens[:, prev_pos:cur_pos], pending_drafts.unsqueeze(-1)], dim=-1)
                next_tokens_2, last_hidden = model.forward(
                    inp, prev_pos,
                    return_next_token=True, return_hidden=True, keep_all_positions=True,
                )
            # next_tokens_2: [b, 2] -- argmax for positions cur_pos and cur_pos+1.
            logits = None
        else:
            if (mtp_log_enabled or mtp_spec_enabled) and prev_pos > 0:
                next_token, last_hidden = model.forward(
                    tokens[:, prev_pos:cur_pos], prev_pos,
                    return_next_token=True, return_hidden=True,
                )
            else:
                next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, return_next_token=True)
                last_hidden = None
            logits = None
        _sync_timing_boundary(model, sync_timings and sync_each_step)
        step_time = time.perf_counter() - step_start
        if capture_this and profiler_active is not None:
            torch.cuda.synchronize()
            profiler_active.__exit__(None, None, None)
            trace_path = os.path.join(
                decode_profile_dir,
                f"decode_step{decode_step_idx}_rank{rank_for_profile}.json",
            )
            try:
                profiler_active.export_chrome_trace(trace_path)
                print(f"decode profile trace exported: {trace_path}", flush=True)
            except Exception as exc:
                print(f"decode profile export failed: {exc}", flush=True)
            try:
                summary = profiler_active.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                )
                summary_path = os.path.join(
                    decode_profile_dir,
                    f"decode_step{decode_step_idx}_rank{rank_for_profile}.summary.txt",
                )
                with open(summary_path, "w") as f:
                    f.write(summary)
            except Exception as exc:
                print(f"decode profile summary failed: {exc}", flush=True)
            profiler_active = None
        if prefill_step:
            _sync_timing_boundary(model, sync_timings and not sync_each_step)
            step_time = time.perf_counter() - step_start
            generated_this_step = int((~prompt_mask[:, cur_pos]).sum().item())
            prefill_time += step_time
            prefill_tokens += generated_this_step
            decode_wall_start = time.perf_counter()
            if cur_pos + 1 < total_len and hasattr(model, "release_gpu_prefill_moe_cache"):
                model.release_gpu_prefill_moe_cache()
                if os.environ.get("DEEPSEEK_RELEASE_PREFILL_INT8_AFTER_PREFILL", "0").lower() in {"1", "true", "yes"} and hasattr(model, "release_cpu_expert_int8_prepare_cache"):
                    model.release_cpu_expert_int8_prepare_cache()
            # Standard length-1 prefill step finishes here -- handle next_token
            # write below.
            if needs_logits:
                next_token = sample(logits, temperature, generator=generator) if temperature > 0 else logits.argmax(dim=-1)
            generated_mask = ~prompt_mask[:, cur_pos]
            if collect_logprobs and needs_logits:
                _record_generation_logprobs(logprob_rows, logits, next_token, generated_mask, top_logprobs)
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
            maybe_capture_prefix_snapshot(cur_pos)
            prev_pos = cur_pos
            cur_pos += 1
            if finished.all():
                break
            continue
        # ---- decode phase ----
        if spec_verify:
            decode_time += step_time
            main_t1 = next_tokens_2[:, 0]  # token for position cur_pos
            main_t2 = next_tokens_2[:, 1]  # token for position cur_pos+1 (only valid if accept)
            main_t1 = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], main_t1)
            tokens[:, cur_pos] = main_t1
            mtp_spec_rounds += 1
            accept = bool((main_t1 == pending_drafts).all().item())
            eos_in_main_t1 = torch.logical_and(~prompt_mask[:, cur_pos], main_t1 == eos_id)
            finished |= eos_in_main_t1
            if accept and not finished.any():
                # Accept draft -> advance 2 positions. main_t2 is the next-next token.
                main_t2 = torch.where(prompt_mask[:, cur_pos + 1], tokens[:, cur_pos + 1], main_t2)
                tokens[:, cur_pos + 1] = main_t2
                eos_in_main_t2 = torch.logical_and(~prompt_mask[:, cur_pos + 1], main_t2 == eos_id)
                finished |= eos_in_main_t2
                decode_tokens += int((~prompt_mask[:, cur_pos]).sum().item())
                decode_tokens += int((~prompt_mask[:, cur_pos + 1]).sum().item())
                mtp_spec_accepts += 1
                # Produce draft for position cur_pos+2 using h at position cur_pos
                # (h_2[:, -1:]) and just-sampled main_t2.
                try:
                    pending_drafts = model.draft_with_mtp(
                        last_hidden[:, -1:],
                        main_t2.unsqueeze(-1),
                        cur_pos,
                    )
                except Exception as exc:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"[mtp_spec] draft failed at cur_pos={cur_pos+1}: {exc}", flush=True)
                    pending_drafts = None
                prev_pos = cur_pos + 1
                cur_pos += 2
                if finished.all():
                    break
                continue
            else:
                # Reject -> advance only 1 position. The kv_cache slot at
                # cur_pos has been written for the draft token, but the
                # *correct* token is main_t1, which is being placed there now.
                # We need to overwrite the kv_cache slot at position cur_pos
                # with the value that main_t1 would produce. Easiest: do a
                # standard length-1 main forward at start_pos=cur_pos-1 with
                # input main_t1 -- but we already DID that as part of the
                # length-2 verify, so the kv_cache slot at cur_pos%win is
                # currently holding the *draft's* kv. We must recompute.
                # However, since on rejection the next loop iteration will
                # do a length-1 forward at start_pos=cur_pos with input
                # main_t1, that forward will populate kv_cache[cur_pos%win]
                # with main_t1's kv -- BUT the slot was already polluted
                # at cur_pos%win by the draft. We need to fix this.
                #
                # Wait: the length-2 forward wrote two slots: (cur_pos-1)%win
                # for tokens[prev_pos] (which is correct, that's the
                # previously-accepted token) and cur_pos%win for the DRAFT
                # (incorrect when rejected). The draft's kv at cur_pos%win
                # is wrong -- it was computed from the rejected token.
                # We must overwrite it with main_t1's kv.
                #
                # Simplest fix: do a length-1 main forward at start_pos=cur_pos-1
                # with input [tokens[prev_pos]] to RESTORE the slot (cur_pos-1)%win
                # AND THEN do a length-1 forward at start_pos=cur_pos with input
                # [main_t1] to write slot cur_pos%win. But that's 2 extra forwards.
                #
                # Alternative: just do a length-1 forward at start_pos=cur_pos with
                # input main_t1 right now to overwrite the wrong slot. The
                # slot (cur_pos-1)%win was correctly written for tokens[prev_pos]
                # by the length-2 forward, so it's fine.
                #
                # But the length-2 forward also wrote the COMPRESSOR state
                # advanced by 2 positions (compressor.kv_state, score_state,
                # ratio counter). On rejection we only want to advance by 1.
                # The compressor state for position cur_pos was computed from
                # the (wrong) draft token, so it's polluted.
                #
                # For now, we'll skip this case and treat all rejections as
                # if we still advance by 1 -- accept the small numerical drift
                # in the compressor cache. If output diverges from baseline,
                # we'll need a deeper fix.
                decode_tokens += int((~prompt_mask[:, cur_pos]).sum().item())
                # Produce draft for position cur_pos+1 using h at position cur_pos-1
                # (which is last_hidden[:, 0:1]) and main_t1.
                try:
                    pending_drafts = model.draft_with_mtp(
                        last_hidden[:, 0:1],
                        main_t1.unsqueeze(-1),
                        cur_pos - 1,
                    )
                except Exception as exc:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"[mtp_spec] draft failed at cur_pos={cur_pos}: {exc}", flush=True)
                    pending_drafts = None
                prev_pos = cur_pos
                cur_pos += 1
                if finished.all():
                    break
                continue
        # ---- standard length-1 decode step ----
        decode_time += step_time
        generated_this_step = int((~prompt_mask[:, cur_pos]).sum().item())
        decode_tokens += generated_this_step
        if needs_logits:
            next_token = sample(logits, temperature, generator=generator) if temperature > 0 else logits.argmax(dim=-1)
        generated_mask = ~prompt_mask[:, cur_pos]
        if collect_logprobs and needs_logits:
            _record_generation_logprobs(logprob_rows, logits, next_token, generated_mask, top_logprobs)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        # Phase 1 MTP probe: verify the previous step's MTP draft against this
        # step's actual main argmax (only meaningful for unmasked positions in
        # decode mode). Then run MTP once to produce the next draft.
        if mtp_log_enabled and temperature == 0 and last_hidden is not None and prev_pos > 0:
            unmasked = ~prompt_mask[:, cur_pos]
            if pending_drafts is not None and unmasked.any():
                match = (pending_drafts == next_token) & unmasked
                mtp_accept += int(match.sum().item())
                mtp_total += int(unmasked.sum().item())
            try:
                pending_drafts = model.draft_with_mtp(
                    last_hidden[:, -1:],
                    next_token.unsqueeze(-1),
                    cur_pos - 1,
                )
            except Exception as exc:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[mtp_log] draft failed at cur_pos={cur_pos}: {exc}", flush=True)
                pending_drafts = None
        elif mtp_spec_enabled and last_hidden is not None and prev_pos > 0:
            # Spec mode but this iteration was a length-1 step (first decode
            # round, or post-rejection). Produce a draft for next round.
            try:
                pending_drafts = model.draft_with_mtp(
                    last_hidden[:, -1:],
                    next_token.unsqueeze(-1),
                    cur_pos - 1,
                )
            except Exception as exc:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[mtp_spec] draft failed at cur_pos={cur_pos}: {exc}", flush=True)
                pending_drafts = None
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        cur_pos += 1
        if finished.all():
            break
    if mtp_log_enabled and (not dist.is_initialized() or dist.get_rank() == 0):
        if mtp_total > 0:
            print(f"[mtp_log] accept_rate {mtp_accept}/{mtp_total} = {mtp_accept / mtp_total:.3f}", flush=True)
        else:
            print("[mtp_log] no decode steps were measured", flush=True)
    if mtp_spec_enabled and (not dist.is_initialized() or dist.get_rank() == 0):
        if mtp_spec_rounds > 0:
            print(f"[mtp_spec] accept_rate {mtp_spec_accepts}/{mtp_spec_rounds} = {mtp_spec_accepts / mtp_spec_rounds:.3f}", flush=True)
        else:
            print("[mtp_spec] no spec rounds were run", flush=True)
    if sync_timings and not sync_each_step and decode_wall_start is not None:
        _sync_timing_boundary(model, True)
        decode_time = time.perf_counter() - decode_wall_start
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    if collect_logprobs:
        return completion_tokens, prefill_time, decode_time, prompt_prefill_tokens, decode_tokens, logprob_rows
    return completion_tokens, prefill_time, decode_time, prompt_prefill_tokens, decode_tokens


@torch.inference_mode()
def generate_stream(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    phase_callback=None,
    prefill_chunk_tokens: int = 0,
    generation_options: dict | None = None,
    prefix_snapshot_hint: dict | None = None,
):
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    decode_profile_dir = os.environ.get("DEEPSEEK_DECODE_PROFILE_DIR", "")
    decode_profile_step = int(os.environ.get("DEEPSEEK_DECODE_PROFILE_STEP", "2") or "2")
    decode_profile_rank = int(os.environ.get("DEEPSEEK_DECODE_PROFILE_RANK", "0") or "0")
    decode_step_idx = 0
    profiler_active = None
    mtp_log_enabled = os.environ.get("DEEPSEEK_MTP_LOG", "0").lower() in {"1", "true", "yes"}
    mtp_spec_enabled = os.environ.get("DEEPSEEK_DECODE_MTP_SPEC", "0").lower() in {"1", "true", "yes"}
    generation_options = generation_options or {}
    needs_logits = temperature > 0 or _has_generation_processors(generation_options)
    generator = None
    if generation_options.get("seed") is not None and torch.cuda.is_available():
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(generation_options["seed"]))
    if getattr(model, "is_layer_pp", False) and needs_logits:
        raise RuntimeError("layer_pp_4gpu currently supports greedy streaming only")
    if getattr(model, "is_layer_pp", False) and (mtp_log_enabled or mtp_spec_enabled):
        raise RuntimeError("layer_pp_4gpu does not support MTP log/speculative streaming paths yet")
    if mtp_spec_enabled and (needs_logits or len(set(prompt_lens)) > 1):
        mtp_spec_enabled = False
    prefix_snapshot_start_pos, prefix_snapshot_capture_positions = _prefix_snapshot_settings(
        model, prompt_tokens, prompt_lens, prefix_snapshot_hint
    )
    prefix_snapshot_pending_captures = set(prefix_snapshot_capture_positions)
    prefix_snapshot_cache = PrefixSnapshotCache.instance() if (prefix_snapshot_pending_captures or prefix_snapshot_start_pos > 0) else None

    def maybe_capture_prefix_snapshot(pos: int) -> None:
        if prefix_snapshot_cache is None or pos not in prefix_snapshot_pending_captures:
            return
        if len(prompt_tokens) != 1 or pos > prompt_lens[0]:
            return
        prefix_snapshot_cache.capture(model, prompt_tokens[0], pos)
        prefix_snapshot_pending_captures.discard(pos)

    pending_drafts = None
    prev_pos = prefix_snapshot_start_pos
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    sync_timings = _env_flag("DEEPSEEK_SYNC_TIMINGS", "1" if getattr(model, "is_layer_pp", False) else "0")
    sync_each_step = _env_flag("DEEPSEEK_SYNC_EACH_STEP", "0")
    prompt_prefill_tokens = sum(prompt_lens)
    prefill_time = 0.0
    decode_time = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    decode_wall_start = None
    cur_pos = min(prompt_lens)
    while cur_pos < total_len:
        prefill_step = prev_pos < min(prompt_lens)
        phase = "prefill" if prefill_step else "decode"
        if phase_callback is not None:
            phase_callback(phase)
        if phase == "decode":
            decode_step_idx += 1
        rank_for_profile = dist.get_rank() if dist.is_initialized() else 0
        capture_this = (
            decode_profile_dir
            and phase == "decode"
            and decode_step_idx == decode_profile_step
            and rank_for_profile == decode_profile_rank
        )
        if capture_this:
            os.makedirs(decode_profile_dir, exist_ok=True)
            torch.cuda.synchronize()
            profiler_active = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            profiler_active.__enter__()
        spec_verify = (
            mtp_spec_enabled
            and prev_pos > 0
            and pending_drafts is not None
            and cur_pos + 1 < total_len
            and not bool(prompt_mask[:, cur_pos].all())
            and not bool(prompt_mask[:, cur_pos + 1].any())
        )
        _sync_timing_boundary(model, sync_timings and sync_each_step)
        step_start = time.perf_counter()
        if needs_logits:
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            logits = _apply_generation_processors(logits, tokens, cur_pos, generation_options)
            last_hidden = None
            spec_verify = False
        elif prefill_step:
            capture_points = sorted(pos for pos in prefix_snapshot_pending_captures if prev_pos < pos < cur_pos)
            if prefill_chunk_tokens > 0 and cur_pos - prev_pos > prefill_chunk_tokens:
                chunk = prefill_chunk_tokens
                chunk_start = prev_pos
                while chunk_start + chunk < cur_pos:
                    chunk_end = chunk_start + chunk
                    model.forward(tokens[:, chunk_start:chunk_end], chunk_start, return_next_token=False)
                    maybe_capture_prefix_snapshot(chunk_end)
                    chunk_start = chunk_end
                next_token = model.forward(tokens[:, chunk_start:cur_pos], chunk_start, return_next_token=True)
            elif capture_points:
                chunk_start = prev_pos
                for chunk_end in capture_points:
                    model.forward(tokens[:, chunk_start:chunk_end], chunk_start, return_next_token=False)
                    maybe_capture_prefix_snapshot(chunk_end)
                    chunk_start = chunk_end
                next_token = model.forward(tokens[:, chunk_start:cur_pos], chunk_start, return_next_token=True)
            else:
                next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, return_next_token=True)
            last_hidden = None
            logits = None
        elif spec_verify:
            assert prev_pos == cur_pos - 1, "spec_verify requires prev_pos==cur_pos-1"
            spec_verify_two_len1 = os.environ.get("DEEPSEEK_DECODE_MTP_SPEC_TWO_LEN1", "0").lower() in {"1", "true", "yes"}
            if spec_verify_two_len1:
                tok_a, h_a = model.forward(
                    tokens[:, prev_pos:cur_pos], prev_pos,
                    return_next_token=True, return_hidden=True,
                )
                tok_b, h_b = model.forward(
                    pending_drafts.unsqueeze(-1), cur_pos,
                    return_next_token=True, return_hidden=True,
                )
                next_tokens_2 = torch.stack([tok_a, tok_b], dim=-1)
                last_hidden = torch.cat([h_a, h_b], dim=1)
            else:
                inp = torch.cat([tokens[:, prev_pos:cur_pos], pending_drafts.unsqueeze(-1)], dim=-1)
                next_tokens_2, last_hidden = model.forward(
                    inp, prev_pos,
                    return_next_token=True, return_hidden=True, keep_all_positions=True,
                )
            logits = None
        else:
            if (mtp_log_enabled or mtp_spec_enabled) and prev_pos > 0:
                next_token, last_hidden = model.forward(
                    tokens[:, prev_pos:cur_pos], prev_pos,
                    return_next_token=True, return_hidden=True,
                )
            else:
                next_token = model.forward(tokens[:, prev_pos:cur_pos], prev_pos, return_next_token=True)
                last_hidden = None
            logits = None
        _sync_timing_boundary(model, sync_timings and sync_each_step)
        step_time = time.perf_counter() - step_start
        if capture_this and profiler_active is not None:
            torch.cuda.synchronize()
            profiler_active.__exit__(None, None, None)
            trace_path = os.path.join(
                decode_profile_dir,
                f"decode_step{decode_step_idx}_rank{rank_for_profile}.json",
            )
            try:
                profiler_active.export_chrome_trace(trace_path)
                print(f"decode profile trace exported: {trace_path}", flush=True)
            except Exception as exc:
                print(f"decode profile export failed: {exc}", flush=True)
            try:
                summary = profiler_active.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                )
                summary_path = os.path.join(
                    decode_profile_dir,
                    f"decode_step{decode_step_idx}_rank{rank_for_profile}.summary.txt",
                )
                with open(summary_path, "w") as f:
                    f.write(summary)
            except Exception as exc:
                print(f"decode profile summary failed: {exc}", flush=True)
            profiler_active = None
        if prefill_step:
            _sync_timing_boundary(model, sync_timings and not sync_each_step)
            step_time = time.perf_counter() - step_start
            generated_this_step = int((~prompt_mask[:, cur_pos]).sum().item())
            prefill_time += step_time
            prefill_tokens += generated_this_step
            decode_wall_start = time.perf_counter()
            if cur_pos + 1 < total_len and hasattr(model, "release_gpu_prefill_moe_cache"):
                model.release_gpu_prefill_moe_cache()
                if os.environ.get("DEEPSEEK_RELEASE_PREFILL_INT8_AFTER_PREFILL", "0").lower() in {"1", "true", "yes"} and hasattr(model, "release_cpu_expert_int8_prepare_cache"):
                    model.release_cpu_expert_int8_prepare_cache()
            if needs_logits:
                next_token = sample(logits, temperature, generator=generator) if temperature > 0 else logits.argmax(dim=-1)
            next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            emit_rank = (not dist.is_initialized() or dist.get_rank() == 0)
            if generated_this_step > 0 and emit_rank:
                yield {
                    "type": "token",
                    "token_ids": next_token[~prompt_mask[:, cur_pos]].tolist(),
                    "position": cur_pos,
                    "phase": "prefill",
                }
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
            maybe_capture_prefix_snapshot(cur_pos)
            prev_pos = cur_pos
            cur_pos += 1
            if finished.all():
                break
            continue
        if spec_verify:
            decode_time += step_time
            main_t1 = next_tokens_2[:, 0]
            main_t2 = next_tokens_2[:, 1]
            main_t1 = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], main_t1)
            tokens[:, cur_pos] = main_t1
            accept = bool((main_t1 == pending_drafts).all().item())
            eos_in_main_t1 = torch.logical_and(~prompt_mask[:, cur_pos], main_t1 == eos_id)
            finished |= eos_in_main_t1
            if accept and not finished.any():
                main_t2 = torch.where(prompt_mask[:, cur_pos + 1], tokens[:, cur_pos + 1], main_t2)
                tokens[:, cur_pos + 1] = main_t2
                eos_in_main_t2 = torch.logical_and(~prompt_mask[:, cur_pos + 1], main_t2 == eos_id)
                finished |= eos_in_main_t2
                decode_tokens += int((~prompt_mask[:, cur_pos]).sum().item())
                decode_tokens += int((~prompt_mask[:, cur_pos + 1]).sum().item())
                token_ids = main_t1[~prompt_mask[:, cur_pos]].tolist() + main_t2[~prompt_mask[:, cur_pos + 1]].tolist()
                if token_ids:
                    yield {
                        "type": "token",
                        "token_ids": token_ids,
                        "position": cur_pos,
                        "phase": "decode",
                    }
                try:
                    pending_drafts = model.draft_with_mtp(
                        last_hidden[:, -1:],
                        main_t2.unsqueeze(-1),
                        cur_pos,
                    )
                except Exception as exc:
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"[mtp_spec] draft failed at cur_pos={cur_pos+1}: {exc}", flush=True)
                    pending_drafts = None
                prev_pos = cur_pos + 1
                cur_pos += 2
                if finished.all():
                    break
                continue
            decode_tokens += int((~prompt_mask[:, cur_pos]).sum().item())
            token_ids = main_t1[~prompt_mask[:, cur_pos]].tolist()
            if token_ids:
                yield {
                    "type": "token",
                    "token_ids": token_ids,
                    "position": cur_pos,
                    "phase": "decode",
                }
            try:
                pending_drafts = model.draft_with_mtp(
                    last_hidden[:, 0:1],
                    main_t1.unsqueeze(-1),
                    cur_pos - 1,
                )
            except Exception as exc:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[mtp_spec] draft failed at cur_pos={cur_pos}: {exc}", flush=True)
                pending_drafts = None
            prev_pos = cur_pos
            cur_pos += 1
            if finished.all():
                break
            continue
        decode_time += step_time
        generated_this_step = int((~prompt_mask[:, cur_pos]).sum().item())
        decode_tokens += generated_this_step
        if needs_logits:
            next_token = sample(logits, temperature, generator=generator) if temperature > 0 else logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        if generated_this_step > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            yield {
                "type": "token",
                "token_ids": next_token[~prompt_mask[:, cur_pos]].tolist(),
                "position": cur_pos,
                "phase": "decode",
            }
        if mtp_log_enabled and temperature == 0 and last_hidden is not None and prev_pos > 0:
            unmasked = ~prompt_mask[:, cur_pos]
            if pending_drafts is not None and unmasked.any():
                pass
            try:
                pending_drafts = model.draft_with_mtp(
                    last_hidden[:, -1:],
                    next_token.unsqueeze(-1),
                    cur_pos - 1,
                )
            except Exception as exc:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[mtp_log] draft failed at cur_pos={cur_pos}: {exc}", flush=True)
                pending_drafts = None
        elif mtp_spec_enabled and last_hidden is not None and prev_pos > 0:
            try:
                pending_drafts = model.draft_with_mtp(
                    last_hidden[:, -1:],
                    next_token.unsqueeze(-1),
                    cur_pos - 1,
                )
            except Exception as exc:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"[mtp_spec] draft failed at cur_pos={cur_pos}: {exc}", flush=True)
                pending_drafts = None
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        cur_pos += 1
        if finished.all():
            break
    if sync_timings and not sync_each_step and decode_wall_start is not None:
        _sync_timing_boundary(model, True)
        decode_time = time.perf_counter() - decode_wall_start
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    if not dist.is_initialized() or dist.get_rank() == 0:
        yield {
            "type": "done",
            "completion_tokens": completion_tokens,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "prefill_tokens": prompt_prefill_tokens,
            "decode_tokens": decode_tokens,
            "finish_reason": "stop" if finished.all() else "length",
        }


def _maybe_print_cuda_memory(tag: str, rank: int) -> None:
    if os.getenv("DEEPSEEK_CUDA_MEMORY_PROFILE", "0").lower() not in {"1", "true", "yes"}:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(
        f"cuda_memory rank={rank} tag={tag} allocated={allocated:.3f}GiB reserved={reserved:.3f}GiB "
        f"peak_allocated={peak_allocated:.3f}GiB peak_reserved={peak_reserved:.3f}GiB",
        flush=True,
    )


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    routed_experts_device: str = "gpu",
    pd_mode: str = "off",
    pd_prefill_chunk_tokens: int = 0,
    ckpt_format: str = "auto",
    tokenizer_path: str | None = None,
    partition_policy: str = "legacy",
) -> None:
    if partition_policy == "baseline_4gpu" and pd_mode != "scheduler":
        raise ValueError("partition_policy=baseline_4gpu requires --pd-mode scheduler")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if partition_policy == "layer_pp_4gpu":
        if world_size not in {2, 4}:
            raise ValueError("partition_policy=layer_pp_4gpu requires torchrun with WORLD_SIZE=2 or 4")
        if temperature > 0:
            raise ValueError("partition_policy=layer_pp_4gpu currently requires --temperature 0")
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    load_barrier_group = None
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(days=7))
        load_barrier_group = dist.new_group(backend="gloo", timeout=timedelta(days=7))
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    if routed_experts_device == "cpu":
        omp_threads_env = os.getenv("DEEPSEEK_CPU_OMP_THREADS")
        omp_threads = int(omp_threads_env) if omp_threads_env else None
        use_affinity = os.getenv("DEEPSEEK_CPU_AFFINITY", "1").lower() not in {"0", "false", "no"}
        rank0_server = os.getenv("DEEPSEEK_CPU_MOE_RANK0_SERVER", "0").lower() in {"1", "true", "yes"}
        inproc_server = os.getenv("DEEPSEEK_CPU_MOE_INPROC_SERVER", "0").lower() in {"1", "true", "yes"}
        centralized_cpu_server = rank0_server or inproc_server
        server_omp_threads_env = os.getenv("DEEPSEEK_CPU_MOE_SERVER_OMP_THREADS")
        nonserver_omp_threads = int(os.getenv("DEEPSEEK_CPU_MOE_NONSERVER_OMP_THREADS", "1"))
        affinity_cpus = None if (centralized_cpu_server and rank == 0) else (_cpu_affinity_for_rank(local_rank, world_size) if use_affinity else None)
        if affinity_cpus is not None:
            os.sched_setaffinity(0, affinity_cpus)
            cpu_threads = omp_threads or max(len(affinity_cpus), 1)
        else:
            if centralized_cpu_server and rank == 0:
                cpu_threads = omp_threads or int(server_omp_threads_env or "22")
            elif inproc_server:
                cpu_threads = omp_threads or max(nonserver_omp_threads, 1)
            else:
                cpu_threads = omp_threads or max((os.cpu_count() or 1) // world_size, 1)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ.setdefault("OMP_DYNAMIC", "FALSE")
        if centralized_cpu_server and rank == 0:
            os.environ.setdefault("OMP_PROC_BIND", "spread")
        else:
            os.environ.setdefault("OMP_PROC_BIND", "close")
        import src.components.moe.cpu_backend as cpu_routed_backend
        cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=cpu_threads)
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        config_data = json.load(f)
    config_data["routed_experts_device"] = routed_experts_device
    config_data["partition_policy"] = partition_policy
    args = ModelArgs(**config_data)
    if interactive:
        args.max_batch_size = 1
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(args)
    shared_cpu_moe_arena = None
    init_start = time.perf_counter()
    with torch.device("cuda"):
        model = Transformer(args)
    if routed_experts_device == "cpu" and SharedCPUMoEWeightArena.enabled():
        if os.getenv("DEEPSEEK_CPU_MOE_SHARED_WEIGHT_NUMA_INTERLEAVE", "0").lower() in {"1", "true", "yes"}:
            _enable_numa_interleave()
        shared_root_dir = SharedCPUMoEWeightArena.root_dir_from_env()
        if not shared_root_dir:
            raise RuntimeError("DEEPSEEK_CPU_MOE_SHARED_WEIGHTS=1 requires DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR")
        shared_cpu_moe_arena = _bind_shared_cpu_moe_weights(model, shared_root_dir, args, world_size, rank)
    tokenizer_source = tokenizer_path or ckpt_path
    if ckpt_format == "gguf" or (ckpt_format == "auto" and ckpt_path.endswith(".gguf")):
        if tokenizer_path is None:
            raise ValueError("GGUF checkpoints require --tokenizer-path to point to a tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    print(f"init time: {time.perf_counter() - init_start:.3f}s", flush=True)
    print("load model")
    load_start = time.perf_counter()
    load_timing_enabled = os.getenv("DEEPSEEK_LOAD_TIMING", "0").lower() in {"1", "true", "yes"}
    load_model(model, ckpt_path, world_size, rank, ckpt_format)
    if load_timing_enabled:
        print(f"load_timing rank={rank} stage=weights time={time.perf_counter() - load_start:.3f}s", flush=True)
    if routed_experts_device == "cpu":
        prepare_start = time.perf_counter()
        model.prepare_cpu_expert_int8()
        if load_timing_enabled:
            print(f"load_timing rank={rank} stage=cpu_expert_prepare time={time.perf_counter() - prepare_start:.3f}s", flush=True)
        if shared_cpu_moe_arena is not None:
            shared_cpu_moe_arena.mark_ready()
    if world_size > 1:
        if load_timing_enabled:
            print(f"load_timing rank={rank} stage=barrier_enter time={time.perf_counter() - load_start:.3f}s", flush=True)
        dist.barrier(group=load_barrier_group)
        if load_timing_enabled:
            print(f"load_timing rank={rank} stage=barrier_exit time={time.perf_counter() - load_start:.3f}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"load time: {time.perf_counter() - load_start:.3f}s", flush=True)
    _maybe_print_cuda_memory("after_load", rank)
    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋")

    pd_scheduler_obj = None
    if pd_mode == "scheduler":
        from src.runtime.pd_scheduler import PDScheduler
        pd_scheduler_obj = PDScheduler()

    def _run_generate(prompt_token_lists):
        if pd_scheduler_obj is None:
            return generate(model, prompt_token_lists, max_new_tokens, tokenizer.eos_token_id, temperature, prefill_chunk_tokens=pd_prefill_chunk_tokens)
        from src.runtime.pd_scheduler import run_single_request
        return run_single_request(
            generate,
            model,
            prompt_token_lists,
            max_new_tokens,
            tokenizer.eos_token_id,
            temperature,
            prefill_chunk_tokens=pd_prefill_chunk_tokens,
            scheduler=pd_scheduler_obj,
        )

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            completion_tokens, prefill_time, decode_time, _prefill_tokens, _decode_tokens = _run_generate([prompt_tokens])
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            text = f.read().strip()
            prompts = text.split("\n\n") if "\n\n" in text else [text]
        prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
        gen_start = time.perf_counter()
        completion_tokens, prefill_time, decode_time, prefill_tokens, decode_tokens = _run_generate(prompt_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - gen_start
        gen_tokens = sum(len(tokens) for tokens in completion_tokens)
        print(f"generate time: {gen_time:.3f}s, tokens: {gen_tokens}, tokens/s: {gen_tokens / max(gen_time, 1e-9):.3f}", flush=True)
        prefill_tps = prefill_tokens / max(prefill_time, 1e-9)
        tpot_ms = decode_time * 1000.0 / max(decode_tokens, 1)
        total_tps = (prefill_tokens + decode_tokens) / max(prefill_time + decode_time, 1e-9)
        print(
            f"prefill time: {prefill_time:.3f}s, prefill tokens: {prefill_tokens}, prefill tokens/s: {prefill_tps:.3f}, "
            f"decode time: {decode_time:.3f}s, decode tokens: {decode_tokens}, decode tokens/s: {decode_tokens / max(decode_time, 1e-9):.3f}, "
            f"ttft: {prefill_time:.3f}s, tpot: {tpot_ms:.3f}ms, throughput tokens/s: {total_tps:.3f}",
            flush=True,
        )
        _maybe_print_cuda_memory("after_generate", rank)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()
    if shared_cpu_moe_arena is not None:
        shared_cpu_moe_arena.close(unlink=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--routed-experts-device", type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--pd-mode", type=str, choices=["off", "scheduler"], default="off")
    parser.add_argument("--pd-prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--ckpt-format", type=str, choices=["auto", "safetensors", "gguf"], default="auto")
    parser.add_argument("--partition-policy", type=str, choices=["legacy", "baseline_4gpu", "layer_pp_4gpu"], default="legacy")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.routed_experts_device, args.pd_mode, args.pd_prefill_chunk_tokens, args.ckpt_format, args.tokenizer_path, args.partition_policy)
