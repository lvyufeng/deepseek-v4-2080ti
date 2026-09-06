"""Qwen4-Exp building blocks: norms, RoPE, hyper-connections, PLE, MoE.

Numerics follow the upstream `transformers` implementation exactly (including
where it forces float32), because the tests check bit-level agreement against
goldens captured from it.  Nothing here is TP-aware; sharding happens in
`runtime.py` by handing sliced weights to these modules.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from src.kernels.cuda_loader import load_cuda_kernel
from src.models.qwen4_exp.config import Qwen4ExpTextConfig


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def _dense_fp16_enabled(rows: int) -> bool:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "128")),
    )
    return _env_enabled("POCKETLLM_QWEN4_DENSE_FP16", "0") and rows >= min_rows


def _dense_fp16_fp32_output_enabled(rows: int) -> bool:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "128")),
    )
    return (
        _env_enabled("POCKETLLM_QWEN4_DENSE_FP16_FP32_OUTPUT", "0")
        and rows >= min_rows
    )


def dequant_fp8_block(
    codes: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference dequantization for a block-scaled FP8 matrix."""
    from src.models.qwen4_exp.quant import dequantize_block_fp8

    return dequantize_block_fp8(codes, scale, block, out_dtype)


def _fp16_fp32_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """FP16 tensor-core GEMM with an FP32 accumulator and input dtype output."""
    flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
    output = torch.mm(
        flat,
        weight.to(torch.float16).t(),
        out_dtype=torch.float32,
    )
    return output.to(x.dtype).reshape(
        *x.shape[:-1],
        weight.shape[0],
    )


def prefill_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Use tensor-core GEMM for BF16 projections, including explicit FP16 mode.

    The environment gates are an optimization for BF16 checkpoints.  When the
    whole model was deliberately loaded as FP16, matching FP16 inputs/weights
    are already the native SM75 fast path and must not be cast back to BF16.
    """
    rows = x.numel() // x.shape[-1]
    same_device = weight.device == x.device
    native_fp16 = (
        x.device.type == "cuda"
        and x.dtype is torch.float16
        and weight.dtype is torch.float16
        and same_device
    )
    if native_fp16:
        if _dense_fp16_fp32_output_enabled(rows):
            return _fp16_fp32_linear(x, weight)
        return F.linear(x, weight)

    supported_bf16 = (
        x.device.type == "cuda"
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and same_device
    )
    if supported_bf16 and _dense_fp16_fp32_output_enabled(rows):
        return _fp16_fp32_linear(x, weight)
    if supported_bf16 and _dense_fp16_enabled(rows):
        return F.linear(x.to(torch.float16), weight.to(torch.float16)).to(torch.bfloat16)
    return F.linear(x, weight)


def _hc_fp16_components(rows: int) -> tuple[bool, bool, bool]:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_HC_FP16_MIN_ROWS", "128")),
    )
    if rows < min_rows:
        return False, False, False
    all_enabled = _env_enabled("POCKETLLM_QWEN4_HC_FP16", "0")
    return (
        all_enabled or _env_enabled("POCKETLLM_QWEN4_HC_FP16_MIX_DOWN", "0"),
        all_enabled or _env_enabled("POCKETLLM_QWEN4_HC_FP16_MIX_UP", "0"),
        all_enabled or _env_enabled("POCKETLLM_QWEN4_HC_FP16_INJECT", "0"),
    )


def _hc_fp16_enabled(rows: int) -> bool:
    return any(_hc_fp16_components(rows))


def _hc_fp16_fp32_output_enabled(rows: int) -> bool:
    return (
        _hc_fp16_enabled(rows)
        and _env_enabled("POCKETLLM_QWEN4_HC_FP16_FP32_OUTPUT", "0")
    )


def _hc_cuda_enabled(rows: int) -> bool:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_HC_CUDA_MIN_ROWS", "128")),
    )
    return (
        _env_enabled("POCKETLLM_QWEN4_HC_CUDA", "0")
        and rows >= min_rows
    ) or _hc_fp16_enabled(rows)


def _cuda_hc_activation(
    name: str,
    x: torch.Tensor,
    groups: int,
) -> torch.Tensor | None:
    """Fuse an HC scale-then-activation pair without changing its rounding.

    The reference path divides a BF16 GEMM result by `hc_count` and then applies
    the activation in BF16.  Both kernels keep the divide and the activation in
    float32 and round once on store, which is bit-identical to the reference on
    every value measured so far, so this stays on the exact path.
    """
    if (
        not _hc_cuda_enabled(x.numel() // x.shape[-1])
        or x.device.type != "cuda"
        or x.dtype is not torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, name):
        return None
    return getattr(ext, name)(x, groups)


def _cuda_grouped_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor | None:
    if (
        not _hc_cuda_enabled(x.numel() // x.shape[-1])
        or x.device.type != "cuda"
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.dtype is not x.dtype
        or weight.device != x.device
        or x.shape[-1] != weight.numel()
        or x.shape[-1] % group_size != 0
    ):
        return None
    ext = load_cuda_kernel()
    if ext is None or not hasattr(ext, "qwen4_exp_grouped_rms_norm"):
        return None
    return ext.qwen4_exp_grouped_rms_norm(x, weight, group_size, eps)


class RMSNorm:
    """RMSNorm with upstream's `(1 + weight)` convention and optional grouping.

    `group_size` normalizes each contiguous group independently, which is how
    the hyper-connection norms treat their `hc_count * hidden_size` input: each
    residual stream gets its own statistics.
    """

    def __init__(self, weight: torch.Tensor, eps: float, *, group_size: int | None = None) -> None:
        self.weight = weight
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            output = _cuda_grouped_rms_norm(
                x,
                self.weight,
                self.group_size,
                self.eps,
            )
            if output is not None:
                return output
        in_dtype = x.dtype
        h = x.to(torch.float32)
        if self.group_size is not None:
            h = h.reshape(*h.shape[:-1], -1, self.group_size)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            h = h.flatten(-2)
        # Qwen stores the norm scale centered on zero: the effective gain is
        # (1 + weight), and the multiply happens in float32 before the cast back.
        return (h * (1.0 + self.weight.to(torch.float32))).to(in_dtype)


class RMSNormGated:
    """Per-head RMSNorm followed by a gate activation (GatedDeltaNet output).

    Upstream multiplies by `weight` (not `1 + weight`) here and applies the gate
    activation in float32.
    """

    def __init__(self, weight: torch.Tensor, eps: float, *, activation: str = "silu") -> None:
        self.weight = weight
        self.eps = eps
        self.activation = activation

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(in_dtype)
        act = torch.sigmoid if self.activation == "sigmoid" else F.silu
        h = h * act(gate.to(torch.float32))
        return h.to(in_dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
):
    """Partial RoPE: rotate the leading `cos.shape[-1]` dims, pass the rest."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_out = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    if k is None:
        return q_out
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    k_out = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_out, k_out


class MRoPE:
    """Interleaved 3D (text/height/width) rotary embedding.

    For text-only inference all three sections carry the same positions, so the
    interleaving is a no-op numerically, but we keep the general path so image
    and video position ids work later without a second code path.
    """

    def __init__(self, config: Qwen4ExpTextConfig, device: torch.device, *, rope_theta: float | None = None) -> None:
        dim = config.rotary_dim
        theta = config.rope_theta if rope_theta is None else rope_theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.mrope_section = list(config.mrope_section)
        self.attention_scaling = 1.0

    def __call__(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (3, batch, seq) or (batch, seq). Returns cos/sin (batch, seq, rotary_dim)."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        inv_freq = self.inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq.float() @ pos).transpose(2, 3)  # (3, batch, seq, dim/2)
        freqs = self._interleave(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(dtype), (emb.sin() * self.attention_scaling).to(dtype)

    def _interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        """Rearrange chunked [TTT..HHH..WWW] frequencies into [THWTHW..TT]."""
        out = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            out[..., idx] = freqs[dim, ..., idx]
        return out


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------


class GatedResidual:
    """Mixes the `hc_count` residual streams into one block input.

    The streams stay separate across the whole network; each block reads a
    low-rank-gated mean of them and writes its output back into every stream
    scaled by a learned per-stream injection weight.  With `block_inject_weight`
    absent (the final mixer) only the mixed input is returned.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        hc_norm: torch.Tensor,
        mix_down: torch.Tensor,
        mix_up: torch.Tensor,
        block_inject: torch.Tensor | None,
    ) -> None:
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_norm = RMSNorm(hc_norm, config.rms_norm_eps, group_size=config.hidden_size)
        self.mix_down = mix_down
        self.mix_up = mix_up
        self.block_inject = block_inject

    def __call__(self, hyper_input: torch.Tensor):
        normed = self.hc_norm(hyper_input)
        rows = hyper_input.numel() // hyper_input.shape[-1]
        fp16_down, fp16_up, fp16_inject = _hc_fp16_components(rows)
        fp16_supported = (
            normed.device.type == "cuda"
            and normed.dtype is torch.bfloat16
            and self.mix_down.dtype is torch.bfloat16
            and self.mix_up.dtype is torch.bfloat16
            and (
                self.block_inject is None
                or self.block_inject.dtype is torch.bfloat16
            )
        )
        if not fp16_supported:
            fp16_down = fp16_up = fp16_inject = False

        fp32_output = (
            fp16_supported and _hc_fp16_fp32_output_enabled(rows)
        )
        compute_input_fp16 = (
            normed.to(torch.float16)
            if (fp16_down or fp16_inject) and not fp32_output
            else None
        )
        if fp16_down:
            if fp32_output:
                down = F.silu(
                    _fp16_fp32_linear(normed, self.mix_down)
                    / self.hc_count
                )
            else:
                assert compute_input_fp16 is not None
                down = F.silu(
                    F.linear(
                        compute_input_fp16,
                        self.mix_down.to(torch.float16),
                    )
                    / self.hc_count
                )
        else:
            raw_down = F.linear(normed, self.mix_down)
            down = _cuda_hc_activation(
                "qwen4_exp_hc_silu",
                raw_down,
                self.hc_count,
            )
            if down is None:
                down = F.silu(raw_down / self.hc_count)

        if fp16_up:
            if fp32_output:
                w = torch.sigmoid(
                    _fp16_fp32_linear(
                        down.to(normed.dtype),
                        self.mix_up,
                    )
                )
            else:
                up_input = (
                    down
                    if down.dtype is torch.float16
                    else down.to(torch.float16)
                )
                w = torch.sigmoid(
                    F.linear(up_input, self.mix_up.to(torch.float16))
                ).to(normed.dtype)
        else:
            up_input = (
                down
                if down.dtype is normed.dtype
                else down.to(normed.dtype)
            )
            w = torch.sigmoid(F.linear(up_input, self.mix_up))

        w = w.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (w * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject is None:
            return mixed
        if fp16_inject:
            if fp32_output:
                inject = 2 * torch.sigmoid(
                    _fp16_fp32_linear(normed, self.block_inject)
                    / self.hc_count
                )
            else:
                assert compute_input_fp16 is not None
                inject = 2 * torch.sigmoid(
                    F.linear(
                        compute_input_fp16,
                        self.block_inject.to(torch.float16),
                    )
                    / self.hc_count
                )
                inject = inject.to(normed.dtype)
        else:
            raw_inject = F.linear(normed, self.block_inject)
            inject = _cuda_hc_activation(
                "qwen4_exp_hc_inject_gate",
                raw_inject,
                self.hc_count,
            )
            if inject is None:
                inject = 2 * torch.sigmoid(raw_inject / self.hc_count)
        return mixed, hyper_input, inject


def inject_into_streams(
    block_output: torch.Tensor, hyper_input: torch.Tensor, injection_weights: torch.Tensor
) -> torch.Tensor:
    """Scatter a block's output back into all hyper-connection streams."""
    groups = injection_weights.shape[-1]
    if (
        _hc_cuda_enabled(
            block_output.numel() // block_output.shape[-1]
        )
        and block_output.device.type == "cuda"
        and block_output.is_contiguous()
        and hyper_input.is_contiguous()
        and injection_weights.is_contiguous()
        and block_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and hyper_input.dtype is block_output.dtype
        and injection_weights.dtype is block_output.dtype
        and block_output.device == hyper_input.device == injection_weights.device
        and hyper_input.shape[:-1] == block_output.shape[:-1]
        and injection_weights.shape[:-1] == block_output.shape[:-1]
        and hyper_input.shape[-1] == groups * block_output.shape[-1]
    ):
        ext = load_cuda_kernel()
        if ext is not None and hasattr(ext, "qwen4_exp_inject"):
            return ext.qwen4_exp_inject(
                block_output,
                hyper_input,
                injection_weights,
                groups,
            )
    injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


# ---------------------------------------------------------------------------
# PLE hashed n-gram embeddings
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> torch.Tensor:
    """Odd per-position hash multipliers, derived deterministically from the seed."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


class NGramHasher:
    """Maps token ids to hashed n-gram embedding row ids.

    Rows are looked up in a table whose `total_vocab_size` is the concatenation
    of one distinct-prime slice per (n-gram order, head).  The hash mixes the
    shifted token ids by XOR of per-position multiples, then takes a remainder
    per head.  Token history crossing an EOS boundary is masked to EOS so
    n-grams never straddle documents.
    """

    def __init__(self, config: Qwen4ExpTextConfig, ple_layer_index: int, device: torch.device) -> None:
        # The hash runs on int64 token ids and its output feeds a host-side
        # gather, so it stays on the CPU regardless of the compute device: doing
        # it on the GPU would only add a device round-trip.
        device = torch.device("cpu")
        self.ngram_size = config.ngram_size
        self.context_len = config.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.eos_token_id = config.primary_eos_token_id
        sizes, offsets, padded = config.ngram_head_vocab_sizes(ple_layer_index)
        self.head_vocab_sizes = torch.tensor(sizes, dtype=torch.long, device=device)
        self.head_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.padded_vocab_size = padded
        self.layer_multipliers = build_layer_multipliers(
            config.vocab_size, config.ngram_size, ple_layer_index, config.seed
        ).to(device)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def row_ids(self, token_history: torch.Tensor, out_len: int) -> torch.Tensor:
        """token_history: (batch, context_len + out_len) -> (batch, out_len, ngram_heads)."""
        token_history = token_history.to("cpu", dtype=torch.long)
        shifted = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            sizes = self.head_vocab_sizes[start:end].view(1, 1, -1)
            offsets = self.head_offsets[start:end].view(1, 1, -1)
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(blocks, dim=-1)[:, -out_len:]


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class TopKRouter:
    """Softmax-over-all-experts router with renormalized top-k weights."""

    def __init__(self, weight: torch.Tensor, top_k: int, *, norm_topk_prob: bool = True) -> None:
        self.weight = weight
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states, self.weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_value, indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        return top_value.to(logits.dtype), indices


class DenseMLP:
    """SwiGLU MLP used by the shared expert."""

    def __init__(self, gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(
            prefill_linear(x, self.gate_proj)
        ) * prefill_linear(x, self.up_proj)
        return prefill_linear(hidden, self.down_proj)


def swiglu_expert(
    x: torch.Tensor, gate_up_proj: torch.Tensor, down_proj: torch.Tensor
) -> torch.Tensor:
    """One expert's fused-gate SwiGLU: gate_up is [2*inter, hidden] row-packed."""
    gate, up = F.linear(x, gate_up_proj).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_proj)
