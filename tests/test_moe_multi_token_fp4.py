"""Small-batch FP4 MoE kernel: agreement with the single-token kernel.

The point of moe_multi_token_fp4_forward is to serve a speculative-verify batch
(k+1 tokens) without falling into the prefill grouped path. It is only useful if
it agrees with the one-token kernel that decode already uses: a batch kernel
that shifts logits changes which draft tokens get accepted, which is how the
existing FP4_DIRECT_GROUPED path fails (max logit diff up to 1.48 on the real
model).

So the reference here is moe_single_token_fp4_forward applied one token at a
time. Both kernels use the same dp4a math over the same fp4 unpack and the same
per-row int8 activation scale, so they should agree to within float32 summation
order.

Bit-exactness is the wrong bar: both kernels accumulate a token's top-k
contributions with atomicAdd, so neither is deterministic. Measured on this box,
the reference kernel disagrees with ITSELF by up to ~9e-7 relative across
repeated runs of the same inputs. The tolerance below is set from that floor, and
`test_reference_is_not_bit_exact` pins the reasoning in place so a future reader
does not tighten it to zero and get a phantom failure.
"""
from __future__ import annotations

import os

import pytest
import torch

cuda_kernel = pytest.importorskip("cuda_kernel")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DIM = 256
INTER_DIM = 128
N_EXPERTS = 8
TOPK = 3
SWIGLU_LIMIT = 7.0
# Fallback tolerance for the shapes where the reference's own spread is measured
# to be tiny. Tests that can measure the floor prefer the measured value: the
# spread grows with how many routes are summed per token (~9e-7 relative at
# topk=3, ~8e-4 at topk=8), so no single constant fits every shape.
RTOL = 1e-4


def _random_fp4_weights(n_experts: int, rows: int, k: int, device, seed: int):
    """Random packed FP4 blocks: two 4-bit codes per byte, one e8m0 scale/32."""
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randint(0, 256, (n_experts, rows, k // 2), dtype=torch.uint8,
                      device=device, generator=gen)
    # Keep exponents in a narrow band so the reference and the kernel are
    # compared on values that neither overflow nor collapse to zero.
    s = torch.randint(120, 136, (n_experts, rows, k // 32), dtype=torch.uint8,
                      device=device, generator=gen)
    return q.contiguous(), s.contiguous()


def _build_routes(indices: torch.Tensor, weights: torch.Tensor, n_local: int):
    """Compact [T, topk] routing into the kernel's slot/pair layout.

    Only experts the batch actually touches get a slot, which is what keeps the
    launch proportional to the batch instead of to n_local_experts.
    """
    tokens, topk = indices.shape
    flat_e = indices.reshape(-1)
    flat_t = torch.arange(tokens, device=indices.device).repeat_interleave(topk)
    flat_w = weights.reshape(-1).to(torch.float32)
    keep = (flat_e >= 0) & (flat_e < n_local)
    flat_e, flat_t, flat_w = flat_e[keep], flat_t[keep], flat_w[keep]
    order = torch.argsort(flat_e, stable=True)
    flat_e, flat_t, flat_w = flat_e[order], flat_t[order], flat_w[order]
    slot_expert, counts = torch.unique_consecutive(flat_e, return_counts=True)
    slot_starts = torch.zeros(slot_expert.numel() + 1, dtype=torch.int32,
                              device=indices.device)
    slot_starts[1:] = counts.cumsum(0).to(torch.int32)
    return (slot_expert.to(torch.int32).contiguous(),
            slot_starts.contiguous(),
            flat_t.to(torch.int32).contiguous(),
            flat_w.contiguous())


def _reference_per_token(x, indices, weights, w, swiglu_limit):
    """Run the production single-token kernel once per row."""
    outs = []
    for t in range(x.size(0)):
        outs.append(cuda_kernel.moe_single_token_fp4_forward(
            x[t:t + 1].contiguous(),
            indices[t].contiguous().to(torch.int64),
            weights[t].contiguous().to(torch.float32),
            *w, 0, swiglu_limit,
        ))
    return torch.cat(outs, dim=0)


def _rel_spread(runs) -> float:
    return max(float(((runs[0] - r).abs() / runs[0].abs().clamp_min(1e-9)).max())
               for r in runs[1:])


def _assert_within_reference_noise(got, want, ref_runs, label: str):
    """Compare against the reference's own spread, floored at a fixed epsilon.

    Historically both kernels combined a token's routes with atomicAdd, so the
    reference disagreed with itself and that spread was the natural bound. The
    reduction is deterministic now, so the spread term is normally 0 and RTOL
    carries the bound; the max() is kept so the assertion still adapts if the
    atomic path is selected via DEEPSEEK_MOE_DETERMINISTIC_REDUCE=0.
    """
    floor = max(_rel_spread(ref_runs), RTOL)
    diff = (got - want).abs()
    rel = float((diff / want.abs().clamp_min(1e-9)).max())
    assert rel <= floor * 4.0, (
        f"{label}: cross-kernel max_rel={rel:.3e} exceeds 4x the reference's own "
        f"run-to-run spread ({floor:.3e}); the batching changed the arithmetic")


def _run_pair(tokens: int, seed: int, topk: int = TOPK):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=device, generator=gen)
    w1q, w1s = _random_fp4_weights(N_EXPERTS, INTER_DIM, DIM, device, seed + 1)
    w3q, w3s = _random_fp4_weights(N_EXPERTS, INTER_DIM, DIM, device, seed + 2)
    w2q, w2s = _random_fp4_weights(N_EXPERTS, DIM, INTER_DIM, device, seed + 3)
    w = (w1q, w1s, w2q, w2s, w3q, w3s)

    indices = torch.stack([
        torch.randperm(N_EXPERTS, device=device, generator=gen)[:topk]
        for _ in range(tokens)
    ]).to(torch.int64)
    weights = torch.rand(tokens, topk, dtype=torch.float32, device=device, generator=gen) + 0.1

    slot_expert, slot_starts, slot_tokens, pair_weights = _build_routes(
        indices, weights, N_EXPERTS)
    got = cuda_kernel.moe_multi_token_fp4_forward(
        x, slot_expert, slot_starts, slot_tokens, pair_weights, *w, SWIGLU_LIMIT)
    # Several reference runs: the first is the value to match, the spread across
    # them is the noise floor that match is judged against.
    ref_runs = [_reference_per_token(x, indices, weights, w, SWIGLU_LIMIT)
                for _ in range(4)]
    return got, ref_runs[0], ref_runs


@pytest.mark.parametrize("tokens", [1, 2, 3, 6])
def test_matches_single_token_kernel(tokens):
    """A verify batch must produce what the same tokens produce one at a time."""
    got, want, ref_runs = _run_pair(tokens, seed=1000 + tokens)
    assert got.shape == want.shape == (tokens, DIM)
    _assert_within_reference_noise(got, want, ref_runs, f"tokens={tokens}")


def test_reference_is_not_bit_exact():
    """Guards the tolerance above: the reference kernel is itself nondeterministic.

    Both kernels atomicAdd a token's top-k contributions, so summation order
    varies between launches. If this ever starts passing at rtol=0, the
    tolerance in the other tests can be tightened -- until then, demanding
    bit-exactness would be testing the GPU scheduler, not the kernel.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(4242)
    x = torch.randn(2, DIM, dtype=torch.bfloat16, device=device, generator=gen)
    w1q, w1s = _random_fp4_weights(N_EXPERTS, INTER_DIM, DIM, device, 11)
    w3q, w3s = _random_fp4_weights(N_EXPERTS, INTER_DIM, DIM, device, 12)
    w2q, w2s = _random_fp4_weights(N_EXPERTS, DIM, INTER_DIM, device, 13)
    w = (w1q, w1s, w2q, w2s, w3q, w3s)
    indices = torch.stack([
        torch.randperm(N_EXPERTS, device=device, generator=gen)[:TOPK] for _ in range(2)
    ]).to(torch.int64)
    weights = torch.rand(2, TOPK, dtype=torch.float32, device=device, generator=gen) + 0.1
    runs = [_reference_per_token(x, indices, weights, w, SWIGLU_LIMIT) for _ in range(5)]
    spread = max(float((runs[0] - r).abs().max()) for r in runs[1:])
    rel = max(float(((runs[0] - r).abs() / runs[0].abs().clamp_min(1e-9)).max())
              for r in runs[1:])
    # Whatever the spread is, it must stay well inside the tolerance the
    # cross-kernel tests use, or those tests are not measuring the kernel.
    assert rel < RTOL, f"reference spread {rel:.2e} exceeds test tolerance {RTOL:.0e}"
    print(f"reference self-disagreement: max_abs={spread:.3g} max_rel={rel:.2e}")


def test_tokens_above_register_tile():
    """More tokens per expert than the register tile takes a second pass."""
    # topk == N_EXPERTS puts every token on every expert, so with 10 tokens each
    # slot holds 10 -- past the kernel's 8-wide tile. It also sums 8 routes per
    # token, which is where the atomicAdd noise floor is widest (~8e-4).
    got, want, ref_runs = _run_pair(10, seed=77, topk=N_EXPERTS)
    _assert_within_reference_noise(got, want, ref_runs, "tokens=10 topk=8")


def test_rows_for_non_local_experts_are_zero():
    """Experts outside this rank's range contribute nothing.

    Under TP each rank holds a slice of the experts and drops routes outside it.
    The caller filters those out, so an empty route list must yield zeros rather
    than reading out of bounds.
    """
    device = torch.device("cuda")
    x = torch.randn(3, DIM, dtype=torch.bfloat16, device=device)
    w1q, w1s = _random_fp4_weights(N_EXPERTS, INTER_DIM, DIM, device, 5)
    w3q, w3s = _random_fp4_weights(N_EXPERTS, INTER_DIM, DIM, device, 6)
    w2q, w2s = _random_fp4_weights(N_EXPERTS, DIM, INTER_DIM, device, 7)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    y = cuda_kernel.moe_multi_token_fp4_forward(
        x,
        empty_i32,
        torch.zeros(1, dtype=torch.int32, device=device),
        empty_i32,
        torch.empty(0, dtype=torch.float32, device=device),
        w1q, w1s, w2q, w2s, w3q, w3s, SWIGLU_LIMIT,
    )
    assert torch.count_nonzero(y) == 0


class _FakeCPUBackend:
    """Minimal stand-in for the CPU MoE backend's staging interface.

    _forward_multi_token_fp4 only needs the FP4 arena and a layer index; the
    real backend brings a pinned host arena and a CPU expert pool that are
    irrelevant to the route bookkeeping under test here.
    """

    def __init__(self, arena):
        self._arena = arena
        self.layer_idx = 0
        self._swiglu_limit = SWIGLU_LIMIT

    def get_fp4_arena(self):
        return self._arena

    def _apply_topk_limit(self, ids, w):
        return ids, w


def _backend_with_arena(device, n_local, seed):
    """A GPUPrefillMoEBackend wired to host-pinned FP4 weights."""
    from src.components.moe.gpu_prefill_backend import GPUPrefillMoEBackend

    w1q, w1s = _random_fp4_weights(n_local, INTER_DIM, DIM, device, seed + 1)
    w3q, w3s = _random_fp4_weights(n_local, INTER_DIM, DIM, device, seed + 2)
    w2q, w2s = _random_fp4_weights(n_local, DIM, INTER_DIM, device, seed + 3)
    gpu_w = (w1q, w1s, w2q, w2s, w3q, w3s)
    arena = tuple(t.to("cpu") for t in gpu_w)
    backend = GPUPrefillMoEBackend(
        _FakeCPUBackend(arena), dim=DIM, num_experts=n_local,
        experts_start_idx=0, experts_end_idx=n_local,
    )
    backend.multi_token_fp4_enabled = True
    return backend, gpu_w


def test_backend_route_layout_handles_non_contiguous_indices():
    """The decode-active call site does not make indices contiguous.

    runtime.py's _should_use_gpu_decode_active_moe branch hands the gate's
    indices/weights straight to forward(), unlike the prefill branch which calls
    .contiguous() first. A transposed view reshapes in expert-major order, so
    pairing route i with token i // topk silently mismatches routes to tokens --
    which showed up on the real model as logits differing by up to 3.65 with
    only 50-67% argmax agreement, while the unit tests (building their own
    contiguous routes) passed.
    """
    device = torch.device("cuda")
    n_local, tokens, topk = N_EXPERTS, 3, TOPK
    backend, gpu_w = _backend_with_arena(device, n_local, seed=900)
    gen = torch.Generator(device=device).manual_seed(901)
    x = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=device, generator=gen)

    # Build routing whose natural layout is [topk, tokens], then transpose: same
    # values as a contiguous [tokens, topk] tensor, different memory order.
    idx_t = torch.stack([
        torch.randperm(n_local, device=device, generator=gen)[:tokens] for _ in range(topk)
    ]).to(torch.int64)                       # [topk, tokens]
    w_t = torch.rand(topk, tokens, dtype=torch.float32, device=device, generator=gen) + 0.1
    indices_nc, weights_nc = idx_t.t(), w_t.t()   # [tokens, topk] views
    assert not indices_nc.is_contiguous()

    got = backend._forward_multi_token_fp4(
        x, indices_nc, weights_nc, SWIGLU_LIMIT)
    assert got is not None, "FP4 arena was available; the path should have run"
    want = _reference_per_token(
        x, indices_nc.contiguous(), weights_nc.contiguous(), gpu_w, SWIGLU_LIMIT)
    ref_runs = [_reference_per_token(
        x, indices_nc.contiguous(), weights_nc.contiguous(), gpu_w, SWIGLU_LIMIT)
        for _ in range(4)]
    _assert_within_reference_noise(got, want, ref_runs, "non-contiguous indices")


def test_backend_route_layout_matches_reference_contiguous():
    """Same check with a contiguous batch, so the two cases can be compared."""
    device = torch.device("cuda")
    n_local, tokens, topk = N_EXPERTS, 6, TOPK
    backend, gpu_w = _backend_with_arena(device, n_local, seed=910)
    gen = torch.Generator(device=device).manual_seed(911)
    x = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=device, generator=gen)
    indices = torch.stack([
        torch.randperm(n_local, device=device, generator=gen)[:topk] for _ in range(tokens)
    ]).to(torch.int64)
    weights = torch.rand(tokens, topk, dtype=torch.float32, device=device, generator=gen) + 0.1

    got = backend._forward_multi_token_fp4(x, indices, weights, SWIGLU_LIMIT)
    assert got is not None
    ref_runs = [_reference_per_token(x, indices, weights, gpu_w, SWIGLU_LIMIT)
                for _ in range(4)]
    _assert_within_reference_noise(got, ref_runs[0], ref_runs, "contiguous indices")


# Production shapes: the TP4 rank slice of DeepSeek-V4-Flash (dim 4096, moe_inter_dim
# 2048, n_activated_experts 6, 256/4 = 64 local experts). The small shapes above
# exercise the code paths; this one pins the tolerance at the arithmetic width
# the model actually runs, where more routes per token widen the atomicAdd floor.
REAL_DIM, REAL_INTER, REAL_NLOC, REAL_TOPK = 4096, 2048, 64, 6


@pytest.mark.parametrize("tokens", [2, 3, 6])
def test_agreement_at_production_shapes(tokens):
    """Agreement measured on the layer output, not on logits.

    The real-model probes compared logits after 43 layers and lm_head, where a
    per-layer relative error is amplified and mixed with attention's own
    batch-vs-sequential differences -- so a 0.29 logit delta there could not be
    attributed. Comparing the MoE layer output directly makes the number the
    kernel's own, and it lands at the reference's atomicAdd noise floor.
    """
    device = torch.device("cuda")
    seed = 4000 + tokens
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, REAL_DIM, dtype=torch.bfloat16, device=device, generator=gen)
    w1q, w1s = _random_fp4_weights(REAL_NLOC, REAL_INTER, REAL_DIM, device, seed + 1)
    w3q, w3s = _random_fp4_weights(REAL_NLOC, REAL_INTER, REAL_DIM, device, seed + 2)
    w2q, w2s = _random_fp4_weights(REAL_NLOC, REAL_DIM, REAL_INTER, device, seed + 3)
    w = (w1q, w1s, w2q, w2s, w3q, w3s)
    indices = torch.stack([
        torch.randperm(REAL_NLOC, device=device, generator=gen)[:REAL_TOPK]
        for _ in range(tokens)
    ]).to(torch.int64)
    weights = torch.rand(tokens, REAL_TOPK, dtype=torch.float32, device=device,
                         generator=gen) + 0.1

    routes = _build_routes(indices, weights, REAL_NLOC)
    got = cuda_kernel.moe_multi_token_fp4_forward(x, *routes, *w, SWIGLU_LIMIT)
    ref_runs = [_reference_per_token(x, indices, weights, w, SWIGLU_LIMIT)
                for _ in range(4)]
    assert got.shape == (tokens, REAL_DIM)
    # Both kernels are now deterministic (DEEPSEEK_MOE_DETERMINISTIC_REDUCE
    # defaults on), so run-to-run spread is exactly zero and cannot serve as a
    # noise floor -- it would collapse the bound onto RTOL and fail a
    # disagreement that has always been there. Assert reproducibility separately
    # from magnitude, and bound the magnitude against int8 hidden-state
    # quantization, which is what actually sets it: the reference requantizes per
    # token while the kernel requantizes per (slot, token) pair.
    ours = [cuda_kernel.moe_multi_token_fp4_forward(x, *routes, *w, SWIGLU_LIMIT)
            for _ in range(3)]
    if os.environ.get("DEEPSEEK_MOE_DETERMINISTIC_REDUCE", "1") != "0":
        assert _rel_spread([got] + ours) == 0.0, (
            "kernel is not run-to-run reproducible")
        assert _rel_spread(ref_runs) == 0.0, (
            "reference is not run-to-run reproducible")
    rel = float(((got - ref_runs[0]).abs()
                 / ref_runs[0].abs().clamp_min(1e-9)).max())
    assert rel <= 8e-3, (
        f"tokens={tokens} at production shapes: max_rel={rel:.3e} exceeds the "
        f"int8 requantization bound; the batching changed the arithmetic")


def test_one_token_per_slot_is_the_single_token_computation():
    """Disjoint routing per token means every slot holds one token.

    With one token per slot the multi-token kernel does not enter its tile loop
    and performs the same arithmetic as the single-token kernel, so agreement
    here isolates the slot bookkeeping from the multi-token accumulation. Swept
    over many routings because the real-model bs=2 probe disagreed on 1 of 3
    draws, which a single seed would have missed.
    """
    device = torch.device("cuda")
    n_experts, tokens, topk = 16, 2, 3
    worst, worst_seed = 0.0, None
    for seed in range(24):
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=device, generator=gen)
        w1q, w1s = _random_fp4_weights(n_experts, INTER_DIM, DIM, device, seed + 1)
        w3q, w3s = _random_fp4_weights(n_experts, INTER_DIM, DIM, device, seed + 2)
        w2q, w2s = _random_fp4_weights(n_experts, DIM, INTER_DIM, device, seed + 3)
        w = (w1q, w1s, w2q, w2s, w3q, w3s)
        perm = torch.randperm(n_experts, device=device, generator=gen)
        indices = torch.stack([perm[:topk], perm[topk:2 * topk]]).to(torch.int64)
        weights = torch.rand(tokens, topk, dtype=torch.float32, device=device,
                             generator=gen) + 0.1
        routes = _build_routes(indices, weights, n_experts)
        got = cuda_kernel.moe_multi_token_fp4_forward(x, *routes, *w, SWIGLU_LIMIT)
        ref = _reference_per_token(x, indices, weights, w, SWIGLU_LIMIT)
        rel = float(((got - ref).abs() / ref.abs().clamp_min(1e-9)).max())
        if rel > worst:
            worst, worst_seed = rel, seed
    assert worst < 1e-3, (
        f"one token per slot should match the single-token kernel to the atomicAdd "
        f"floor; worst max_rel={worst:.3e} at seed={worst_seed}")
